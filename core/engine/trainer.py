import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
import datetime
import torch.utils.data as data
import os
from torchvision import transforms

from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.distributed import *

from core.utils.loss import get_segmentation_loss
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric
from core.utils.tbwriter import TensorboardWriter as TBWriter
from core.utils.metric_logger import MetricLogger
import shutil


class Trainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        if get_rank() == 0:
            TBWriter.init(
                os.path.join(args.project_dir, args.task_dir, "tbevents")
            )
        self.device = torch.device(args.device)

        self.meters = MetricLogger(delimiter="  ")
        # image transform
        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        # dataset and dataloader
        data_kwargs = {
            "transform": input_transform,
            "base_size": args.base_size,
            "crop_size": args.crop_size,
            "root": args.dataroot,
        }
        train_dataset = get_segmentation_dataset(
            args.dataset, split="train", mode="train", **data_kwargs
        )
        val_dataset = get_segmentation_dataset(
            args.dataset, split="val", mode="val", **data_kwargs
        )
        args.iters_per_epoch = len(train_dataset) // (
            args.num_gpus * args.batch_size
        )
        args.max_iters = args.epochs * args.iters_per_epoch

        train_sampler = make_data_sampler(
            train_dataset, shuffle=True, distributed=args.distributed
        )
        train_batch_sampler = make_batch_data_sampler(
            train_sampler, args.batch_size, args.max_iters
        )
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(
            val_sampler, args.batch_size
        )

        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.workers,
            pin_memory=True,
        )
        self.val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=args.workers,
            pin_memory=True,
        )

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(
            model=args.model,
            dataset=args.dataset,
            backbone=args.backbone,
            aux=args.aux,
            jpu=args.jpu,
            norm_layer=BatchNorm2d,
        ).to(self.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert (
                    ext == ".pkl" or ".pth"
                ), "Sorry only .pth and .pkl files supported."
                print("Resuming training, loading {}...".format(args.resume))
                self.model.load_state_dict(
                    torch.load(
                        args.resume, map_location=lambda storage, loc: storage
                    )
                )

        # create criterion
        self.criterion = get_segmentation_loss(
            args.model,
            use_ohem=args.use_ohem,
            aux=args.aux,
            aux_weight=args.aux_weight,
            ignore_index=-1,
        ).to(self.device)

        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        if hasattr(self.model, "pretrained"):
            params_list.append(
                {"params": self.model.pretrained.parameters(), "lr": args.lr}
            )
        if hasattr(self.model, "exclusive"):
            for module in self.model.exclusive:
                params_list.append(
                    {
                        "params": getattr(self.model, module).parameters(),
                        "lr": args.lr * args.lr_scale,
                    }
                )
        self.optimizer = torch.optim.SGD(
            params_list,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(
            self.optimizer,
            max_iters=args.max_iters,
            power=0.9,
            warmup_factor=args.warmup_factor,
            warmup_iters=args.warmup_iters,
            warmup_method=args.warmup_method,
        )

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
            )

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0

    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = (
            self.args.log_iter,
            self.args.val_epoch * self.args.iters_per_epoch,
        )
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        self.logger.info(
            "Start training, Total Epochs: {:d} = Total Iterations {:d}".format(
                epochs, max_iters
            )
        )

        self.model.train()
        end = time.time()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1
            self.lr_scheduler.step()
            data_time = time.time() - end

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss_dict = self.criterion(outputs, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            batch_time = time.time() - end
            end = time.time()
            self.meters.update(
                data_time=data_time, batch_time=batch_time, loss=losses_reduced
            )

            eta_seconds = ((time.time() - start_time) / iteration) * (
                max_iters - iteration
            )
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                self.logger.info(
                    self.meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=(self.meters),
                        lr=self.optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated()
                        / 1024.0
                        / 1024.0,
                    )
                )
                if is_main_process():
                    # write train loss and lr
                    TBWriter.write_scalar(
                        ["train/loss", "train/lr", "train/mem"],
                        [
                            losses_reduced,
                            self.optimizer.param_groups[0]["lr"],
                            torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        ],
                        iter=iteration,
                    )
                    # write time
                    TBWriter.write_scalars(
                        ["train/time"],
                        [self.meters.get_metric(["data_time", "batch_time"])],
                        iter=iteration,
                    )

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                pixAcc, mIoU = self.validation()
                reduced_pixAcc = reduce_tensor(pixAcc)
                reduced_mIoU = reduce_tensor(mIoU)
                new_pred = (reduced_pixAcc + reduced_mIoU) / 2
                new_pred = float(new_pred.cpu().numpy())

                if new_pred > self.best_pred:
                    is_best = True
                    self.best_pred = new_pred

                if is_main_process():
                    TBWriter.write_scalar(
                        ["val/PixelACC", "val/mIoU"],
                        [
                            reduced_pixAcc.cpu().numpy(),
                            reduced_mIoU.cpu().numpy(),
                        ],
                        iter=iteration,
                    )
                    save_checkpoint(self.model, self.args, is_best)
                synchronize()
                self.model.train()

        if is_main_process():
            save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(
            datetime.timedelta(seconds=total_training_time)
        )
        self.logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters
            )
        )

    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            # pixAcc, mIoU = self.metric.get()
            # logger.info(
            # "Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            # i + 1, pixAcc, mIoU
            # )
            # )
        pixAcc, mIoU = self.metric.get()

        return (
            torch.tensor(pixAcc).to(self.device),
            torch.tensor(mIoU).to(self.device),
        )


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.join(
        os.path.expanduser(args.project_dir), args.task_dir, "ckpts"
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = "{}_{}_{}.pth".format(args.model, args.backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = "{}_{}_{}_best_model.pth".format(
            args.model, args.backbone, args.dataset
        )
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)

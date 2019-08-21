import argparse
import datetime
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
from core.utils.distributed import get_rank, synchronize
from core.utils.logger import setup_logger
from core.engine.trainer import Trainer

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation Training With Pytorch"
    )
    # model and dataset
    parser.add_argument(
        "--model",
        type=str,
        default="fcn",
        choices=[
            "fcn32s",
            "fcn16s",
            "fcn8s",
            "fcn",
            "psp",
            "deeplabv3",
            "deeplabv3_plus",
            "danet",
            "denseaspp",
            "bisenet",
            "encnet",
            "dunet",
            "icnet",
            "enet",
            "ocnet",
            "ccnet",
            "psanet",
            "cgnet",
            "espnet",
            "lednet",
            "dfanet",
        ],
        help="model name (default: fcn32s)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=[
            "vgg16",
            "resnet18",
            "resnet50",
            "resnet101",
            "resnet152",
            "densenet121",
            "densenet161",
            "densenet169",
            "densenet201",
        ],
        help="backbone name (default: vgg16)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pascal_voc",
        choices=["pascal_voc", "pascal_aug", "ade20k", "citys", "sbu"],
        help="dataset name (default: pascal_voc)",
    )
    parser.add_argument("--dataroot", type=str, help="root to dataset")
    parser.add_argument(
        "--base-size", type=int, default=520, help="base image size"
    )
    parser.add_argument(
        "--crop-size", type=int, default=480, help="crop image size"
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=4,
        metavar="N",
        help="dataloader threads",
    )
    # training hyper params
    parser.add_argument("--jpu", action="store_true", default=False, help="JPU")
    parser.add_argument(
        "--use-ohem",
        type=bool,
        default=False,
        help="OHEM Loss for cityscapes dataset",
    )
    parser.add_argument(
        "--aux", action="store_true", default=False, help="Auxiliary loss"
    )
    parser.add_argument(
        "--aux-weight", type=float, default=0.4, help="auxiliary loss weight"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--ngpus", type=int, default=2, metavar="N", help="number of gpus"
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        metavar="N",
        help="start epochs (default:0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--lr_scale",
        type=float,
        default=10,
        metavar="LRSCALE",
        help="learning rate scale for layers except in backbone(default: 10)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        metavar="M",
        help="w-decay (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-policy",
        type=str,
        choices=["multi_step", "poly", "cosine"],
        default="cosine",
        help="policy to adjust learning rate (default: cosine)",
    )
    # warmup settings
    parser.add_argument(
        "--warmup-iters", type=int, default=0, help="warmup iters"
    )
    parser.add_argument(
        "--warmup-factor",
        type=float,
        default=1.0 / 3,
        help="lr = warmup_factor * lr",
    )
    parser.add_argument(
        "--warmup-method", type=str, default="linear", help="method of warmup"
    )
    # args for poly policy
    parser.add_argument(
        "--poly-power", type=float, default=0.9, help="power for poly policy"
    )
    # args for cosine anealing policy
    parser.add_argument(
        "--eta-min", type=float, default=0.0, help="eta min for cosie policy"
    )
    # args for multi step policy
    parser.add_argument(
        "--multistep-milestones",
        default=None,
        type=int,
        nargs="+",
        help="steps for learning rate decay",
    )
    parser.add_argument(
        "--multistep-gamma",
        default=0.1,
        type=float,
        help="decay rate between milestones",
    )
    # cuda setting
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    # checkpoint and log
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="put the path to resuming file if needed",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Directory for specific project(model and dataset)",
    )
    parser.add_argument(
        "--task-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoint models, tbevents and log",
    )
    parser.add_argument(
        "--save-epoch",
        type=int,
        default=1,
        help="save model every checkpoint-epoch",
    )
    parser.add_argument(
        "--log-iter", type=int, default=10, help="print log every log-iter"
    )
    # evaluation only
    parser.add_argument(
        "--val-epoch",
        type=int,
        default=1,
        help="run validation every val-epoch",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        default=False,
        help="skip validation during training",
    )
    args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            "coco": 30,
            "pascal_aug": 80,
            "pascal_voc": 50,
            "pcontext": 80,
            "ade20k": 160,
            "citys": 120,
            "sbu": 160,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {
            "coco": 0.004,
            "pascal_aug": 0.001,
            "pascal_voc": 0.0001,
            "pcontext": 0.001,
            "ade20k": 0.01,
            "citys": 0.01,
            "sbu": 0.001,
        }
        args.lr = lrs[args.dataset.lower()] * args.ngpus
    return args


if __name__ == "__main__":
    args = parse_args()

    # reference maskrcnn-benchmark
    num_gpus = (
        int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    )
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    args.lr = args.lr * num_gpus

    logger = setup_logger(
        "semantic_segmentation",
        os.path.join(args.project_dir, args.task_dir),
        get_rank(),
        filename="{}_{}_{}_log.txt".format(
            args.model, args.backbone, args.dataset
        ),
    )
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args, logger)
    trainer.train()
    torch.cuda.empty_cache()


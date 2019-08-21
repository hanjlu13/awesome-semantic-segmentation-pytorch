#!/bin/bash

export NGPUS=$1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --model psp \
                                                                    --backbone resnet50 \
                                                                    --aux --aux-weight 0.5 \
                                                                    --dataroot $2 \
                                                                    --project-dir $3 \
                                                                    --ngpus $NGPUS \
                                                                    --task-dir $4 \
                                                                    --warmup-iters 500 \
                                                                    --dataset ade20k \
                                                                    --lr-policy cosine \
                                                                    --lr 0.001 --lr_scale 10 \
                                                                    --epochs 80
#!/bin/bash

export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --model psp \
                                                                    --backbone resnet50 \
                                                                    --aux --aux-weight 0.5 \
                                                                    --dataroot /home/hancock/training_data/ade20k/ \
                                                                    --project-dir /home/hancock/data/training_records/seg \
                                                                    --ngpus 2 \
                                                                    --task-dir psp_resnet50_ade20k \
                                                                    --warmup-iters 500 \
                                                                    --dataset ade20k \
                                                                    --lr 0.001 \
                                                                    --epochs 80
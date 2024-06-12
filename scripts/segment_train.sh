#!/bin/bash

# Train Segmentation Model

python3 train/segment_train.py \
        --root /mnt/linux_store/Chair_Processed/03001627/ \
        --segment_dir /mnt/linux_store/Segment_Maps/03001627/ \
        --batch_size 24 \
        --num_workers 10 \
        --wandb True \
        --learning_rate 1e-4 \
        --subset_size 100000 \
        --wandb_project chair_segmentation \
        --num_iterations 200000
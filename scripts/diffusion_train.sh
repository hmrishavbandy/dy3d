#!/bin/bash


# Train Diffusion Model

python3 train/diffusion_train.py \
        --max_train_steps 1000000 \
        --root_folder /home/hmrishav/dy3d/ \
        --spaghetti_model /home/hmrishav/dy3d/model \
        --use_wandb True \
        --learning_rate 1e-4 \
        --eval_interval 10000
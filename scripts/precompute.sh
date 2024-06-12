#!/bin/bash


# Precompute Encodings

python3 train/precompute_encodings.py \
        --root_folder /home/hmrishav/dy3d/ \
        --dset_folder /mnt/linux_store/Chair_Processed/03001627/ \
        --batch_size 256 \
        --dataloader_num_workers 10 \

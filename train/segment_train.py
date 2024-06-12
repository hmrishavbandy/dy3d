
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dset_segment import ShapeNet_Segment
import os
import wandb
from tqdm import tqdm
import torchvision
import torchmetrics
import models.resnet_model as resnet_model
import numpy as np


import argparse

import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ViT model on ShapeNet Sketches")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_iterations", type=int, default=2000000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=2000)
    parser.add_argument("--checkpoint_dir", type=str, default="./encoder_ckpt")
    parser.add_argument("--checkpoint_file", type=str, default="segmentation_aligned.pth")
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="segment_train")
    parser.add_argument("--subset_size", type=int, default=11060)

    parser.add_argument("--root", type=str, help="Path to the sketch directory of the dataset",
                        default="/mnt/linux_store/Chair_Processed/03001627/")
                        
    parser.add_argument("--segment_dir", type=str, help="Path to the segment-maps directory of the dataset",
                        default="/mnt/linux_store/Segment_Maps/03001627/")

    return parser.parse_args()
    

def main():
    
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    use_wandb = args.wandb
    if use_wandb:
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)
    

    encoder = resnet_model.resnet18()
    encoder.fc = nn.Identity()
    encoder.avgpool = nn.Identity()
    encoder.cuda()
    decoder = resnet_model.SharedSegmentDecoder()
    decoder.cuda()
    
    model = nn.Sequential(encoder, decoder)
    
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
    ])
    
    preprocess_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
    ])
   
    params_to_update = model.parameters()
   

    optimizer = torch.optim.Adam(
        params_to_update,
        lr=args.learning_rate)

    train_dataset = ShapeNet_Segment(
        root = args.root, segment_dir = args.segment_dir,
        train = True, inversions=True, 
        view_= 'all',  preprocessor = preprocess)
    test_dataset = ShapeNet_Segment(
        root = args.root, segment_dir = args.segment_dir,
        train = False, inversions=True, 
        view_ = 'all', preprocessor = preprocess_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=args.num_workers)
    if args.subset_size <= len(test_dataset):
        print("Will eval on subset of size {} from test set of size {}".format(args.subset_size, len(test_dataset)))
        test_subset = torch.utils.data.Subset(test_dataset, range(args.subset_size))
    else:
        print("Will eval on full test set of size {}".format(len(test_dataset)))
        test_subset = test_dataset
    
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False, num_workers=args.num_workers)
    
    progress_bar = tqdm(range(args.num_iterations))
    
    
    
    gen_train = iter(train_loader)
    val_iterations = len(test_loader)
    gen_test = iter(test_loader)
    
    
    
    best_iou = 0
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(15.0).cuda())
    jaccard = torchmetrics.JaccardIndex(task="binary", threshold = 0.5).cuda()
  
    for k in progress_bar:

       
        model.train()
        try: 
           x, seg, f_name = next(gen_train)
        except:
            gen_train = iter(train_loader)
            x, seg, f_name = next(gen_train)
        
        x = x.cuda()
        seg = seg.cuda()
        seg_pred = model(x)
        
        iou = sum([jaccard(seg_pred[:, i, :, :], seg[:, i, :, :]) for i in range(16)]) / 16
        mean_train_iou = iou.item()
        
        loss = bce_criterion(seg_pred, seg)
        
        
        loss_out = loss
        loss_out.backward()
        
        progress_bar.set_description("Loss {:.3f}, IoU {:.3f}".format(loss.item(), mean_train_iou))
        progress_bar.update(1)
        
        optimizer.step()
        optimizer.zero_grad()
        
        if use_wandb:
            wandb.log({"train_loss": loss.item()})
            wandb.log({"train_iou": mean_train_iou})

        if k % args.checkpoint_interval == 0 and k > 0:
            model.eval()
            with torch.no_grad():
                iou_eval = []
                for iter_val in tqdm(range(val_iterations)):
                    try:
                        x,seg, f_name = next(gen_test)
                    except:
                        gen_test = iter(test_loader)
                        x,seg, f_name = next(gen_test)

                    x = x.cuda()
                    seg = seg.cuda()

                    
                    seg_pred = model(x)

                    iou = sum([jaccard(seg_pred[:, i, :, :], seg[:, i, :, :]) for i in range(16)]) / 16
                    mean_iou = iou.item()

                    iou_eval.append(mean_iou)
                    
                print(f"Validation IOU: {np.mean(iou_eval)}")
                
                if use_wandb:
                    wandb.log({"val_iou": np.mean(iou_eval)})

                iou_test = np.mean(iou_eval)

            if iou_test > best_iou:

                best_iou = iou_test
                print("Saving best model")
                
                torch.save(
                    model.state_dict(), 
                    os.path.join(args.checkpoint_dir, 
                                 args.checkpoint_file))

if __name__ == "__main__":
    main()
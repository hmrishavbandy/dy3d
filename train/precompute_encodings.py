import argparse
import os
import sys
from tqdm.auto import tqdm

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import resnet_model
from data.dset_diffusion import ShapeNet_Diffuse
from utils.base_utils import set_seed

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--root_folder",
        type=str,
        default="/mnt/linux_store/ldm_final/",
        help="Root directory for the project"
    )
    parser.add_argument(
        "--dset_folder",
        type=str,
        default="/mnt/linux_store/Chair_Processed/03001627/",
        help="Dataset directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for data loading"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args

def create_data_loaders(root_folder, dset_folder, batch_size, num_workers, preprocess):
    train_dataset = ShapeNet_Diffuse(
        base_dir=root_folder,
        dset_dir=dset_folder,
        train=True,
        inversions=True,
        view_='all',
        preprocessor=preprocess
    )
    test_dataset = ShapeNet_Diffuse(
        base_dir=root_folder,
        dset_dir=dset_folder,
        train=False,
        inversions=True,
        view_='all',
        preprocessor=preprocess
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    return train_dataloader, test_dataloader

def save_encoding(encoder, dataloader, save_path, device):
    encoder.eval()
    with torch.no_grad():
        for iteration, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            images, labels = batch[0].to(device), batch[1].to(device)
            img_encoding = encoder(images).view(images.size(0), 16, 256)
            out_enc = {
                "img_enc": img_encoding.cpu(),
                "label": labels.cpu(),
            }
            torch.save(out_enc, os.path.join(save_path, f"{iteration}.pt"), _use_new_zipfile_serialization=True)

def main(args):
    set_seed(42)
    
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = resnet_model.resnet18()
    encoder.fc = nn.Identity()
    encoder.avgpool = nn.Identity()
    encoder = encoder.to(device)
    
    decoder = resnet_model.SharedSegmentDecoder().to(device)
    
    model = nn.Sequential(encoder, decoder)
    model.load_state_dict(torch.load(os.path.join(args.root_folder, "encoder_ckpt", "segmentation_aligned.pth")))
    encoder = model[0].to(device)
    
    train_dataloader, test_dataloader = create_data_loaders(
        args.root_folder, args.dset_folder, args.batch_size, args.dataloader_num_workers, preprocess
    )
    
    os.makedirs(os.path.join(args.root_folder, "encodings", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.root_folder, "encodings", "test"), exist_ok=True)
    
    img_demo = next(iter(train_dataloader))[0][0].to(device)
    torchvision.utils.save_image(img_demo, os.path.join(args.root_folder, "demo_sketch.png"))
    
    save_encoding(encoder, train_dataloader, os.path.join(args.root_folder, "encodings", "train"), device)
    save_encoding(encoder, test_dataloader, os.path.join(args.root_folder, "encodings", "test"), device)

if __name__ == "__main__":
    args = parse_args()
    main(args)

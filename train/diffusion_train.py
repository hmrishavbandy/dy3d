import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from models.ldm import MLPConditionModel
from diffusers import DDPMScheduler, DDIMScheduler
from utils.spaghetti_utils import get_frozen, spaghetti_fwd_mesh
from utils.utils_spaghetti import files_utils
from utils.base_utils import set_seed
import models.resnet_model as resnet_model
import wandb
import pytorch3d
from pytorch3d.io import IO
from pytorch3d.loss import chamfer_distance
import pytorch3d.ops


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps.")
    parser.add_argument("--use_wandb", type=bool, default=False, help="Use wandb for logging")
    parser.add_argument("--root_folder", type=str, default="/mnt/linux_store/ldm_final/", help="Root folder for training")
    parser.add_argument("--spaghetti_model", type=str, default="/mnt/linux_store/ldm_final/model", 
                        help="Path to the Spaghetti Model, download from https://github.com/amirhertz/spaghetti")

    parser.add_argument("--eval_timesteps", type=int, default=50, help="Number of timesteps for evaluation")
    parser.add_argument("--eval_interval", type=int, default=10000, help="Interval for evaluation")
    
    return parser.parse_args(input_args) if input_args else parser.parse_args()


def initialize_models(args):
    spaghetti_model, _ = get_frozen(from_here=args.spaghetti_model)
    spaghetti_model.cuda()

    noise_scheduler = DDPMScheduler(thresholding=False, clip_sample=False)
    eval_scheduler = DDIMScheduler(thresholding=False, clip_sample=False)
    eval_scheduler.set_timesteps(args.eval_timesteps)

    mlp_model = MLPConditionModel().cuda()
    print(f"Num Parameters: {sum(p.numel() for p in mlp_model.parameters()) / 1e6:.2f}M")
    
    encoder = resnet_model.resnet18()
    encoder.fc = nn.Identity()
    encoder.avgpool = nn.Identity()
    model = nn.Sequential(encoder, resnet_model.SharedSegmentDecoder())
    model.load_state_dict(torch.load("encoder_ckpt/segmentation_aligned.pth"))
    
    encoder = model[0].cuda()
    encoder.eval()

    return spaghetti_model, noise_scheduler, eval_scheduler, mlp_model, encoder


def load_dataset():
    dset = np.load("samples_final.npy")
    with open("./data/lists/chairs_list.txt", "r") as f:
        f1 = [line.strip() for line in f]
    with open("./data/lists/sv2_chairs_train.json", "r") as f:
        f_train = json.load(f)["ShapeNetV2"]["03001627"]

    dset_app = [torch.tensor(dset[f1.index(f)]) for f in f_train if f in f1]
    dset = torch.stack(dset_app)

    mean_ = dset.mean(0).cpu().numpy()
    std_ = dset.std(0).cpu().numpy()
    return dset, mean_, std_


def main(args):
    set_seed(42)

    use_wandb = args.use_wandb
    root_folder = args.root_folder
    spaghetti_model, noise_scheduler, eval_scheduler, mlp_model, encoder = initialize_models(args)
    
    dset, mean_, std_ = load_dataset()
    mean_ = torch.tensor(mean_).cuda()
    std_ = torch.tensor(std_).cuda()

    optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500000, gamma=0.7)

    if use_wandb:
        wandb.init(project="diff_cond", config=args)

    progress_bar = tqdm(range(args.max_train_steps), desc="Steps")
    list_all = [torch.load(f"encodings/train/{i}.pt") for i in range(len(os.listdir("encodings/train/")))]

    global_step = 0
    for iteration in progress_bar:
        mlp_model.train()
        data = list_all[iteration % len(list_all)]
        lat_in, encoding = data["label"], data["img_enc"].cuda()

        latents = (lat_in.to(dtype=torch.float32, device="cuda") - mean_) / std_
        noise = torch.randn_like(latents, device="cuda")
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device="cuda").long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        model_pred = mlp_model(noisy_latents, timesteps, encoding).sample
        loss = F.mse_loss(model_pred, noise, reduction="mean")
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        logs = {"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']}
        progress_bar.set_postfix(**logs)
        if use_wandb and iteration % 100 == 0:
            wandb.log(logs)
        
        if iteration % args.eval_interval == 0 and iteration != 0:
            with torch.no_grad():
                evaluate_model(mlp_model, encoder, eval_scheduler, spaghetti_model, mean_, std_, root_folder, use_wandb)
                mlp_model.train()

                torch.save({
                    'mlp_model': mlp_model.state_dict(),
                    'encoder': encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, "mlp_diffusion.pt")

@torch.no_grad()
def evaluate_model(mlp_model, encoder, eval_scheduler, spaghetti_model, mean_, std_, root_folder, use_wandb):
    mlp_model.eval()
    encoder.eval()

    all_test = os.listdir("encodings/test/")
    f = random.choice(all_test)
    data = torch.load(f"encodings/test/{f}")
    lat_in, encoding = data["label"], data["img_enc"].cuda()
    latents = (lat_in.to(dtype=torch.float32).cuda() - mean_) / std_

    noise = torch.randn_like(latents).detach()
    noise_latent = noise.clone().detach()
    cond = encoding.cuda()

    for t in eval_scheduler.timesteps:
        noise_pred = mlp_model(noise_latent, t, cond).sample
        noise_latent = eval_scheduler.step(noise_pred, t, noise_latent).prev_sample

    noise_latent = noise_latent.detach()[:16] * std_ + mean_
    latents = latents.detach()[:16] * std_ + mean_

    mesh, _ = spaghetti_fwd_mesh(spaghetti_model, noise_latent, res=64) # Very low resolution, JUST FOR TESTING -> Use 128/256 for actual results
    mesh_gt, _ = spaghetti_fwd_mesh(spaghetti_model, latents, res=64) # Very low resolution, JUST FOR TESTING -> Use 128/256 for actual results

    chamfer_dist_acc = 0
    try:
        for en, m in enumerate(mesh):
            files_utils.export_mesh(mesh_gt[en], f"{root_folder}meshes/gt_{en}")
            files_utils.export_mesh(mesh[en], f"{root_folder}meshes/pred_{en}")

            mesh_pred = IO().load_mesh(f"{root_folder}meshes/pred_{en}.obj")
            mesh_gt_ = IO().load_mesh(f"{root_folder}meshes/gt_{en}.obj")

            pcl_pred = pytorch3d.ops.sample_points_from_meshes(mesh_pred, 100000)
            pcl_gt = pytorch3d.ops.sample_points_from_meshes(mesh_gt_, 100000)

            """
            Rough chamfer distance calculation, NOT evaluation.
            """
            
            chamfer_dist = chamfer_distance(pcl_pred.cuda(), pcl_gt.cuda())[0]
            chamfer_dist_acc += chamfer_dist.item()

    except Exception as e:
        print("Mesh Error:", e)
        return

    if use_wandb:
        wandb.log({"r-chamfer_dist": chamfer_dist_acc / len(mesh)})
    print("Rough Chamfer Dist:", chamfer_dist_acc / len(mesh))


if __name__ == "__main__":
    args = parse_args()
    print("Parsed arguments.")
    main(args)

# From Spaghetti - Hetz et al.
from utils.utils_spaghetti.custom_types import *
import utils.utils_spaghetti.options as options
from utils.utils_spaghetti import train_utils, files_utils, mcubes_meshing
import torch

def get_frozen(from_here="/vol/research/hmriCV/cvpr_2024/s2c/spaghetti/assets/checkpoints/spaghetti_chairs_large/model"):
    opt = options.Options(tag="chairs_large").load()
    opt.dataset_size = 6755
    model, opt = train_utils.model_lc(opt)
    model.load_state_dict(torch.load(from_here))
    print(f"Loaded model from {from_here}")
    return model, opt


def get_frozen_planes():
    opt = options.Options(tag="airplanes").load()
    # print(opt)
    opt.dataset_size = 1775
    # print(opt.dataset_size)
    # print(opt.dim_z)

    # exit()
    model, opt = train_utils.model_lc(opt)
    model.load_state_dict(torch.load("/vol/research/hmriCV/cvpr_2024/s2c/spaghetti/assets/checkpoints/spaghetti_airplanes/model"))
    print("Loaded model from /vol/research/hmriCV/cvpr_2024/s2c/spaghetti/assets/checkpoints/spaghetti_airplanes/model")
    return model, opt
# from spaghetti_utils import get_mesh

def spaghetti_fwd_occ(model, x, res = 32):
    # model, opt = model
    mid = None
    gmms = None
    device = "cuda"

    # meshing = mcubes_meshing.MarchingCubesMeshing(device, scale=1.)
    z_init = x
    # print(z_init.shape)
    zh_base, gmms = model.decomposition_control.forward_mid(z_init)
    zh, attn_b = model.merge_zh(zh_base, gmms)
    # mesh, occ = get_mesh(zh[0], 32, model)
    meshing = mcubes_meshing.MarchingCubesMeshing(device, scale=1.)
    occ = []
    for i in range(len(zh)):
        occ.append(meshing.get_grid(get_occ_fun(zh[i], model), res=res))
    occ = torch.stack(occ)
    
    return occ


def spaghetti_fwd_mesh(model, x, res = 32):
    # model, opt = model
    mid = None
    gmms = None
    device = "cuda"

    # meshing = mcubes_meshing.MarchingCubesMeshing(device, scale=1.)
    z_init = x
    # print(z_init.shape)
    zh_base, gmms = model.decomposition_control.forward_mid(z_init)
    
    zh, attn_b = model.merge_zh(zh_base, gmms)
    # mesh, occ = get_mesh(zh[0], 32, model)
    meshing = mcubes_meshing.MarchingCubesMeshing(device, scale=1.)
    occ = []
    mesh = []
    for i in range(len(zh)):
        mesh_, occ_=meshing.occ_meshing(get_occ_fun(zh[i], model), res=res, get_occ = True)
        mesh.append(mesh_)
        occ.append(occ_)
        

        
    occ = torch.stack(occ)
    # mesh = torch.stack(mesh)
    
    return mesh, occ


def get_occ_fun(z, model):
    model = model

    def forward(x):
        nonlocal z
        x = x.unsqueeze(0)
        out = model.occupancy_network(x, z)[0, :]
        out = 2 * out.sigmoid_() - 1
        return out

    if z.dim() == 2:
        z = z.unsqueeze(0)
    return forward


def get_new_ids(opt,folder_name, nums_sample):
    names = [int(path[1]) for path in files_utils.collect(f'{opt.cp_folder}/{folder_name}/occ/', '.obj')]
    ids = torch.arange(nums_sample)
    if len(names) == 0:
        return ids + opt.dataset_size
    return ids + max(max(names) + 1, opt.dataset_size)

def get_mesh(z, res, model):
        device = "cuda"
        meshing = mcubes_meshing.MarchingCubesMeshing(device, scale=1.)
        mesh, occ = meshing.occ_meshing(get_occ_fun(z, model), res=res, get_occ = True)
        
        return mesh, occ


def main():
    opt = options.Options(tag="chairs_large")
    opt.dataset_size = 6755
    num_samples = 1
    output_name = "test"
    with torch.no_grad():

        opt = opt
        model= train_utils.model_lc(opt)
        model, opt = model
        model.eval()
        mid = None
        gmms = None
        device = "cuda"
        meshing = mcubes_meshing.MarchingCubesMeshing(device, scale=1.)
        z_init = model.get_random_embeddings(num_samples)
        zh_base, gmms = model.decomposition_control(z_init)
        print(gmms.shape)
        zh, attn_b = model.merge_zh(zh_base, gmms)
        numbers = get_new_ids(opt, output_name, num_samples)
        for i in range(len(zh)):
            mesh, occ = get_mesh(zh[i], 64, model)
        # print(mesh.shape)
        print(occ.shape)
        

if __name__ == '__main__':
    main()

import os
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from math import log10
import numpy as np
import random
import torch.nn as nn

import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision
import sys
sys.path.append('..')

import model.model as model
import trainer.train_step as train_step
import trainer.saver as saver
import dataset

import os
import shutil
from tqdm import tqdm
import numpy
import random

import importlib

import argparse
import json
torch.manual_seed(20202464)



def to_psnr(output, gt):
    mse = F.mse_loss(output, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].detach().permute(0, 2, 3, 1).cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].detach().permute(0, 2, 3, 1).cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list

@torch.no_grad()
def main (args):
    out_dir = args["outdir"]
    num = args["index"]
    out_dir = os.path.join(out_dir, str(num))

    args["device"] = 'cpu' if args["cpu"] else 'cuda'

    f = getattr(model, args["model"])()
    ckpt_name = os.path.join(out_dir, "checkpoints", "final.tar" if args["final"] else "checkpoint.tar" if args["latest"] else "best_val.tar")
    if args["cpu"]:
        ckpt = torch.load(ckpt_name, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(ckpt_name)
    f.load_state_dict(ckpt["f"])
    if not args["cpu"]:
        f = f.cuda()

    test_dataset_module = getattr(dataset, args["testdataset"])
    test_dataset = test_dataset_module(root=args["dataroot"], mode='test', )
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False)

    prog_bar = tqdm(test_loader)
    f.eval()

    psnr_list = []
    ssim_list = []

    elapsed_time = 0
    num_running = 0
    import time
    with open(os.path.join(out_dir, 'quantitative_results.txt'), 'w') as outfile:
        outfile.write(f'{ckpt_name}\n')
        for batchIdx, (hazy, clear, img_name) in enumerate(prog_bar):
            # try:
            hazy = hazy.to(args["device"])
            clear = clear.to(args["device"])

            start = time.process_time()

            T, A, J = f(hazy)

            elapsed = time.process_time() - start
            elapsed_time += elapsed
            num_running += 1

            psnr_list.extend(to_psnr(J, clear))
            ssim_list.extend(to_ssim_skimage(J, clear))
    
            prog_bar.set_description(f'psnr {to_psnr(J, clear)} ssmi {to_ssim_skimage(J, clear)}, run {elapsed} ')

            outfile.write(f'batch {batchIdx} ({img_name[0].split("/")[-1]}),  psnr {to_psnr(J, clear)[0]},  ssmi {to_psnr(J, clear)[0]}\n')

        avr_psnr = sum(psnr_list) / len(psnr_list)
        avr_ssim = sum(ssim_list) / len(ssim_list) 

        print(f"psnr: {avr_psnr}, ssim: {avr_ssim}, avg_running: {elapsed_time / num_running}")

        outfile.write(f'avg psnr: {avr_psnr}, avg ssim: {avr_ssim}, avg_running: {elapsed_time / num_running}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--config_lambda', type=str, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--final', action='store_true', default=False)
    parser.add_argument('--latest', action='store_true', default=False)
    parser.add_argument('--use_bean', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--testdataset', type=str, default='RESIDEStandardTestDataset')
    args = parser.parse_args()
    args = vars(args)
    
    with open('config.json', 'r') as f:
        opt = json.load(f)["options"][args['config']]
        for key in opt:
            args[key] = opt[key]

    if args["use_bean"] == True:
        args["outdir"] = os.path.join("/Bean/log/gwangjin/CVF-dehazing/", args["config"], args["config_lambda"])
        args["dataroot"] = os.path.join("/Bean/data/gwangjin/RESIDE_standard")
    else:
        args["outdir"] = os.path.join("/Jarvis/workspace/gwangjin/dehazing/cvf-results/", args["config"], args["config_lambda"])
        args["dataroot"] = os.path.join("/data1/gwangjin/dehazing_bench/RESIDE_standard")


    print(args)

    main(args)

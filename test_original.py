
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

    args["device"] = 'cpu' if args["cpu"] else 'cuda'

    test_dataset_module = getattr(dataset, args["testdataset"])
    test_dataset = test_dataset_module(root=args["dataroot"], mode='test', )
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False)

    prog_bar = tqdm(test_loader)
    psnr_list = []
    ssim_list = []

    with open(os.path.join(out_dir, 'reside_standard_hazy_clear_comparison.txt'), 'w') as outfile:
        outfile.write(f'\n')
        for batchIdx, (hazy, clear, img_name) in enumerate(prog_bar):
            # try:
            hazy = hazy.to(args["device"])
            clear = clear.to(args["device"])

            psnr_list.extend(to_psnr(hazy, clear))
            ssim_list.extend(to_ssim_skimage(hazy, clear))
    
            prog_bar.set_description(f'psnr {to_psnr(hazy, clear)} ssmi {to_ssim_skimage(hazy, clear)}')

            outfile.write(f'batch {batchIdx} ({img_name[0].split("/")[-1]}),  psnr {to_psnr(hazy, clear)[0]},  ssmi {to_ssim_skimage(hazy, clear)[0]}\n')

        avr_psnr = sum(psnr_list) / len(psnr_list)
        avr_ssim = sum(ssim_list) / len(ssim_list) 

        print(f"psnr: {avr_psnr}, ssim: {avr_ssim} ")

        outfile.write(f'avg psnr: {avr_psnr}, avg ssim: {avr_ssim}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--latest', type=bool, default=False)
    parser.add_argument('--use_bean', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--testdataset', type=str, default='RESIDEStandardTestDataset')
    args = parser.parse_args()
    args = vars(args)
    

    args["outdir"] = os.path.join("/Jarvis/workspace/gwangjin/dehazing/cvf-results/")
    args["dataroot"] = os.path.join("/data1/gwangjin/dehazing_bench/RESIDE_standard")

    print(args)

    main(args)

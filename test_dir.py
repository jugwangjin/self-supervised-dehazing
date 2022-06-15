import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision
import sys
sys.path.append('..')

import model.model as model

import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
import argparse
from torchvision import transforms
import json
eps = 1e-7
def h(img):
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,2]==img.max(1)[0] ]
    hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,1]==img.max(1)[0] ]
    hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

    hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    hue = hue/6

    return hue

@torch.no_grad()
def main(args):
    out_dir = args["outdir"]
    num = args["index"]
    out_dir = os.path.join(out_dir, str(num))

    f = getattr(model, args["model"])()
    ckpt_name = os.path.join(out_dir, "checkpoints", "final.tar" if args["final"] else "checkpoint.tar" if args["latest"] else "best_val.tar")
    if args["cpu"]:
        ckpt = torch.load(ckpt_name, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(ckpt_name)
    f.load_state_dict(ckpt["f"])
    if not args["cpu"]:
        f = f.cuda()

    f.eval()
    del ckpt

    img_out = os.path.join(out_dir, 'test_dir_out', args["outname"])
    os.makedirs(img_out, exist_ok=True)
    import tqdm

    elapsed_time = 0
    num_running = 0
    import time

    for fn in tqdm.tqdm(sorted(os.listdir(args["img_dir"]))):
        try:
            img_name = os.path.join(args["img_dir"], fn)

            img = Image.open(img_name).convert("RGB")
            img = torchvision.transforms.ToTensor()(img)
            img = img.unsqueeze(0)
            if not args["cpu"]:
                img = img.cuda()

            start = time.process_time()
            T, A, J = f(img)
            elapsed = time.process_time() - start

            rec = J * T + A * (1 - T)

            torchvision.utils.save_image(img[0], os.path.join(img_out, f'{fn}_input.png'))
            torchvision.utils.save_image(J[0], os.path.join(img_out, f'{fn}_J.png'))
            torchvision.utils.save_image(T[0], os.path.join(img_out, f'{fn}_T.png'))
            torchvision.utils.save_image(A[0].repeat(1, 50, 50), os.path.join(img_out, f'{fn}_A.png'))
            torchvision.utils.save_image(rec[0], os.path.join(img_out, f'{fn}_reconstruct_input.png'))

            with open(os.path.join(img_out, f'{fn}_running_time.txt'), 'w') as fil:
                fil.write(f'elapsed {elapsed} secs for an image with size of {img.shape}')
        except Exception as e:
            print(e)
            pass

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--config_lambda', type=str, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--outname', type=str, default='dir_out')
    parser.add_argument('--final', action='store_true', default=False)
    parser.add_argument('--latest', type=bool, default=False)
    parser.add_argument('--use_bean', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true', default=False)
    args = parser.parse_args()
    args = vars(args)
    
    with open('config.json', 'r') as f:
        opt = json.load(f)["options"][args['config']]
        for key in opt:
            args[key] = opt[key]

    if args["use_bean"] == True:
        args["outdir"] = os.path.join("/Bean/log/gwangjin/CVF-dehazing/", args["config"], args["config_lambda"])
        args["dataroot"] = os.path.join("/Bean/data/gwangjin/", args["dataroot"])
    else:
        args["outdir"] = os.path.join("/Jarvis/workspace/gwangjin/dehazing/cvf-results/", args["config"], args["config_lambda"])
        args["dataroot"] = os.path.join("/data1/gwangjin/dehazing_bench/", args["dataroot"])


    print(args)

    main(args)

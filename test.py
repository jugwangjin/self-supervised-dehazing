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

def main(args):
    out_dir = args["outdir"]
    num = args["index"]
    out_dir = os.path.join(out_dir, str(num))

    f = getattr(model, args["model"])()
    if args["cpu"]:
        ckpt = torch.load(os.path.join(out_dir, "checkpoints", "checkpoint.tar" if args["latest"] else "best_val.tar"), map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(os.path.join(out_dir, "checkpoints", "checkpoint.tar" if args["latest"] else "best_val.tar"))
    f.load_state_dict(ckpt["f"])
    if not args["cpu"]:
        f = f.cuda()

    out_name = args["outname"] if args["outname"] is not None else args["img"].split(".")[0].split("/")[-1]
    img_out = os.path.join(out_dir, 'testing_outputs', out_name)
    os.makedirs(img_out, exist_ok=True)

    maxpool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    if not args["cpu"]:
        maxpool = maxpool.cuda()



    img = Image.open(args["img"]).convert("RGB")
    img = torchvision.transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    if not args["cpu"]:
        img = img.cuda()



    T, A, J = f(img)
    A_T, A_A, A_J = f(A.expand_as(img))
    J_T, J_A, J_J = f(J)



    rec = J * T + A * (1 - T)

    J_rgb_max = torch.amax(J, dim=1, keepdim=True)
    J_rgb_min = torch.amin(J, dim=1, keepdim=True)
    J_HSV_S = (J_rgb_max - J_rgb_min + 1e-4) / (J_rgb_max + 1e-4)
    J_HSV_V = J_rgb_max
    J_HSV_H = h(J)

    img_rgb_max = torch.amax(img, dim=1, keepdim=True)
    img_rgb_min = torch.amin(img, dim=1, keepdim=True)
    img_HSV_S = (img_rgb_max - img_rgb_min + 1e-4) / (img_rgb_max + 1e-4)
    img_HSV_V = img_rgb_max
    img_HSV_H = h(img)


    A_ = torch.amax(img, dim=(2,3), keepdim=True)
    min_patch = -maxpool(-img / A_.clamp(min=1e-1, max=0.95))
    dcp = 1 - torch.amin(min_patch, dim=1, keepdim=True)
    dcp = dcp.clamp(0, 1)

    torchvision.utils.save_image(img[0], os.path.join(img_out, f'input.png'))
    torchvision.utils.save_image(J[0], os.path.join(img_out, f'J.png'))
    torchvision.utils.save_image(T[0], os.path.join(img_out, f'T.png'))
    torchvision.utils.save_image(A[0].repeat(1, 50, 50), os.path.join(img_out, f'A.png'))
    torchvision.utils.save_image(rec[0], os.path.join(img_out, 'reconstruct_input.png'))
    torchvision.utils.save_image(J_HSV_S[0].repeat(3,1,1), os.path.join(img_out, 'J_S.png'))
    torchvision.utils.save_image(J_HSV_V[0].repeat(3,1,1), os.path.join(img_out, 'J_V.png'))
    torchvision.utils.save_image(J_HSV_H[0].repeat(3,1,1), os.path.join(img_out, 'J_H.png'))
    torchvision.utils.save_image(img_HSV_S[0].repeat(3,1,1), os.path.join(img_out, 'input_S.png'))
    torchvision.utils.save_image(img_HSV_V[0].repeat(3,1,1), os.path.join(img_out, 'input_V.png'))
    torchvision.utils.save_image(img_HSV_H[0].repeat(3,1,1), os.path.join(img_out, 'input_H.png'))
    torchvision.utils.save_image(dcp[0].repeat(3,1,1), os.path.join(img_out, 'input_dcp.png'))

    T_aug = (torch.amax(T, dim=(2,3), keepdim=True) - T + torch.amin(T, dim=(2,3), keepdim=True))
    torchvision.utils.save_image(T_aug[0], os.path.join(img_out, f'T_aug1.png'))
    T_aug = (T + torch.randn(T.size(0), 1, 1, 1, device=T.device) * 0.3 + torch.randn(T.size(0), T.size(1), 1, 1, device=T.device) * 0.1).clamp(0.05, 1)
    torchvision.utils.save_image(T_aug[0], os.path.join(img_out, f'T_aug2.png'))
    T_aug = (T * (1 + torch.randn(T.size(0), 1, 1, 1, device=T.device) * 0.3) + torch.randn(T.size(0), T.size(1), 1, 1, device=T.device) * 0.1).clamp(0.05, 1)
    torchvision.utils.save_image(T_aug[0], os.path.join(img_out, f'T_aug3.png'))

    A_ = torch.amax(J, dim=(2,3), keepdim=True)
    min_patch = -maxpool(-J / A_.clamp(min=1e-1, max=0.95))
    dcp = 1 - torch.amin(min_patch, dim=1, keepdim=True)
    dcp = dcp.clamp(0, 1)

    torchvision.utils.save_image(dcp[0].repeat(3,1,1), os.path.join(img_out, 'J_dcp.png'))

    scat_J = (img - A * (1 - T.clamp(min=1e-2))) / T.clamp(min=1e-2)
    torchvision.utils.save_image(scat_J[0].clamp(0, 1), os.path.join(img_out, f'scattering_out.png'))



    rec = J_J * J_T + J_A * (1 - J_T)

    torchvision.utils.save_image(rec[0], os.path.join(img_out, f'J_cycle_reconstruct_J.png'))
    torchvision.utils.save_image(J_J[0], os.path.join(img_out, f'J_cycle_J.png'))
    torchvision.utils.save_image(J_T[0], os.path.join(img_out, f'J_cycle_T.png'))
    torchvision.utils.save_image(J_A[0].repeat(1, 50, 50), os.path.join(img_out, f'J_cycle_A.png'))


    rec = A_J * A_T + A_A * (1 - A_T)

    torchvision.utils.save_image(rec[0], os.path.join(img_out, f'A_cycle_reconstruct_A.png'))
    torchvision.utils.save_image(A_J[0], os.path.join(img_out, f'A_cycle_J.png'))
    torchvision.utils.save_image(A_T[0], os.path.join(img_out, f'A_cycle_T.png'))
    torchvision.utils.save_image(A_A[0].repeat(1, 50, 50), os.path.join(img_out, f'A_cycle_A.png'))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--config_lambda', type=str, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--outname', type=str, default=None)
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

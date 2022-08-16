
import torch.nn.functional as F
import torch
import os
import torchvision

class SaveAMap():
    def __init__(self):
        self.eps = 1e-7
        self.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        pass
    
    def __str(self):
        print("Saver with A Map")

    def get_uv(self, img):
        r = img[:, 0:1]
        g = img[:, 1:2]
        b = img[:, 2:3]

        delta = 0.5
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = (b - y) * 0.564 + delta
        v = (r - y) * 0.713 + delta

        return torch.cat((u, v, torch.zeros_like(u)), dim=1)

    def h(self, img):
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6

        return hue

    def save_image(self, img, T, A, clean, idx, batchIdx, out_dir, valbatchsize):

        rec = clean * T + A * (1 - T)

        J_rgb_max = torch.amax(clean, dim=1, keepdim=True)
        J_rgb_min = torch.amin(clean, dim=1, keepdim=True)
        J_HSV_S = (J_rgb_max - J_rgb_min + 1e-4) / (J_rgb_max + 1e-4)
        J_HSV_V = J_rgb_max
        J_HSV_H = self.h(clean)

        img_rgb_max = torch.amax(img, dim=1, keepdim=True)
        img_rgb_min = torch.amin(img, dim=1, keepdim=True)
        img_HSV_S = (img_rgb_max - img_rgb_min + 1e-4) / (img_rgb_max + 1e-4)
        img_HSV_V = img_rgb_max
        img_HSV_H = self.h(img)

        A_ = torch.amax(img, dim=(2,3), keepdim=True)

        min_patch = -self.maxpool(-img / A_.clamp(min=1e-1, max=0.95))
        dcp = 1 - torch.amin(min_patch, dim=1, keepdim=True)
        dcp = dcp.clamp(0, 1)

        J_uv = self.get_uv(clean)
        img_uv = self.get_uv(img)


        torchvision.utils.save_image(clean[idx], os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_clean.png'))
        torchvision.utils.save_image(img[idx], os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_img.png'))
        torchvision.utils.save_image(T[idx], os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_T.png'))
        torchvision.utils.save_image(A[idx], os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_A.png'))
        torchvision.utils.save_image(rec[idx], os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_reconstuct.png'))
        torchvision.utils.save_image(J_HSV_S[idx].repeat(3,1,1), os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_clean_S.png'))
        torchvision.utils.save_image(J_HSV_V[idx].repeat(3,1,1), os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_clean_V.png'))
        torchvision.utils.save_image(J_HSV_H[idx].repeat(3,1,1), os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_clean_H.png'))
        torchvision.utils.save_image(J_uv[idx], os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_clean_uv.png'))
        torchvision.utils.save_image(img_HSV_S[idx].repeat(3,1,1), os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_img_S.png'))
        torchvision.utils.save_image(img_HSV_V[idx].repeat(3,1,1), os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_img_V.png'))
        torchvision.utils.save_image(img_HSV_H[idx].repeat(3,1,1), os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_img_H.png'))
        torchvision.utils.save_image(img_uv[idx], os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_img_uv.png'))
        torchvision.utils.save_image(dcp[idx].repeat(3,1,1), os.path.join(out_dir, 'results', f'{batchIdx * valbatchsize + idx}_dcp.png'))
    
    
class SaveAConst(SaveAMap):
    def __init__(self):
        super().__init__()
        pass

    def __str(self):
        print("Saver with A Const")

    def save_image(self, img, T, A, clean, idx, batchIdx, out_dir, valbatchsize):
        A = A.expand_as(img)
        super().save_image(img, T, A, clean, idx, batchIdx, out_dir, valbatchsize)
import torch
import torchvision
import random
from .utils import LocalVariance, BoxBlur

class TrainStep(torch.nn.Module):
    def __init__(self, f, args):
        super().__init__()
        self.args = args
        self.f = f
        self.blur = BoxBlur()
        self.var = LocalVariance(channels=3, kernel_size=9)

        # critreion
        if args["loss"] == "sl1":
            self.sl1 = torch.nn.SmoothL1Loss()
        elif args["loss"] == "l1":
            self.sl1 = torch.nn.L1Loss()
        elif args["loss"] == "mse":
            self.sl1 = torch.nn.MSELoss()
        else:
            raise Exception("only sl1, l1, mse is allowed")

        # lambdas
        self.lambdas = args["lambdas"]

        # for transmission - dcp
        self.maxpool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.large_boxblur = BoxBlur(kernel_size=5)
        self.large_boxblur_1 = BoxBlur(channels=1, kernel_size=5)

        # for exclusion loss
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)
        self.sigmoid = torch.nn.Sigmoid()

    def prior_loss(self, T, A, J, img):
        J_s, J_v = self.get_sv(J)
        img_s, img_v = self.get_sv(img)

        J_uv = self.get_uv(J)
        img_uv = self.get_uv(img)
        
        A_ = torch.amax(img, dim=(2,3), keepdim=True).clamp(min=0.5)
        dcp = self.dcp(img, A_)
        
        L_prior = self.lambdas["T_DCP"] * self.Sl1((self.large_boxblur(T) - self.large_boxblur_1(dcp))) + \
                self.lambdas["J_TV"] * self.Sl1(self.TV(J)) +\
                self.lambdas["J_pixel_intensity"] * self.Sl1((J - img).clamp(min=0)) +\
                self.lambdas["J_value"] * self.Sl1((J_v - img_v).clamp(min=0)) +\
                self.lambdas["J_saturation"] * self.Sl1((img_s - J_s).clamp(min=0)) +\
                self.lambdas["J_hue"] * self.Sl1(J_uv - img_uv) +\
                self.lambdas["J_var"] * self.Sl1((self.var(img) - self.var(J)).clamp(min=0))
        return L_prior

    def clean_loss(self, T, A, J, img):
        clean_T, clean_A, clean_J = self.f(J)

        L_clean = self.Sl1(J - clean_J) + self.Sl1(clean_T - 1)
        return L_clean

    def T_zero_loss(self, T, A, J, img):
        T_zero_T, T_zero_A, T_zero_J = self.f(A)

        L_T_zero = self.Sl1(T_zero_T) + self.Sl1(T_zero_A - A) + self.Sl1(T_zero_J - A)
        return L_T_zero

    def augmentation_loss(self, T, A, J, img):
        T_augs = []
        # T_augs.append(T)
        T_augs.append((torch.amax(T, dim=(2,3), keepdim=True) - T + torch.amin(T, dim=(2,3), keepdim=True)))
        T_augs.append((T + torch.randn(T.size(0), 1, 1, 1, device=T.device) * 0.3 + torch.randn(T.size(0), T.size(1), 1, 1, device=T.device) * 0.1).clamp(0.05, 1))
        T_augs.append((T * (1 + torch.randn(T.size(0), 1, 1, 1, device=T.device) * 0.3) + torch.randn(T.size(0), T.size(1), 1, 1, device=T.device) * 0.1).clamp(0.05, 1))
        
        L_aug = 0
        for T_aug in T_augs:
            A_aug = (A + (torch.randn(A.size(0), 1, 1, 1, device=A.device)*0.2) + torch.randn(A.size(0), A.size(1), 1, 1, device=A.device)*0.1).clamp(0.33, 1)
            img_aug = J * T_aug + A_aug * (1 - T_aug)
            aug_T, aug_A, aug_J = self.f(img_aug)
            L_aug = L_aug + (self.Sl1(aug_J - J) + self.Sl1(aug_T - T_aug) +\
                            self.Sl1(aug_A - A_aug))
        return L_aug

    def regularization_loss(self, T, A, J, img):

        L_reg = self.lambdas["A_hint"] * self.Sl1(A - torch.amax(img, dim=(2,3), keepdim=True)) + \
                self.lambdas["T_smooth"] * self.Sl1(T - self.blur(T)) +\
                self.lambdas["T_gray"] * self.Sl1(T - T.mean(dim=1, keepdim=True)) +\
                self.lambdas["J_idt"] * self.Sl1(J - img)
        return L_reg

    def recon_loss(self, T, A, J, img):
        L_recon = self.Sl1(img - (J * T + A * (1 - T)))
        return L_recon

    # def forward(self, img, return_package = False):
    def forward(self, img):
        T, A, J = self.f(img)

        L_recon = self.recon_loss(T, A, J, img)
        L_prior = self.prior_loss(T, A, J, img)
        L_clean = self.clean_loss(T, A, J, img)
        L_T_zero = self.T_zero_loss(T, A.expand_as(img), J, img)
        L_aug = self.augmentation_loss(T, A, J, img)
        L_reg = self.regularization_loss(T, A, J, img)

        L_total = self.lambdas["recon"] * L_recon + L_prior + self.lambdas["clean"] * L_clean +\
                     self.lambdas["aug"] * L_aug + L_reg + self.lambdas["T_zero"] * L_T_zero
        L_total = torch.nan_to_num(L_total, nan=0, posinf=0, neginf=0)
        # if return_package:
        return L_total, {"L_rec": L_recon,  "L_p": L_prior, "L_c": L_clean, 
                        "L_a": L_aug, "L_r": L_reg, "L_Tz": L_T_zero}
        # return L_total
    '''
    loss helpers
    '''
    def Sl1(self, x):
        return self.sl1(x, torch.zeros_like(x))

    def dcp(self, img, A):
        min_patch = -self.maxpool(-img / A.clamp(min=1e-1, max=0.95))
        T_rough = 1 - torch.amin(min_patch, dim=1, keepdim=True)
        return T_rough.clamp(5e-2, 1)

    def TV(self, x):
        TV_x = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        TV_y = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()

        return TV_x + TV_y

    def get_sv(self, img, eps=1e-2):
        img = img.clamp(0 ,1)
        img_max = img.max(1)[0]
        img_min = img.min(1)[0]

        saturation = ( img_max - img_min ) / ( img_max + eps )

        value = img_max
        return saturation, value

    def get_uv(self, img):
        r = img[:, 0:1]
        g = img[:, 1:2]
        b = img[:, 2:3]

        delta = 0.5
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = (b - y) * 0.564 + delta
        v = (r - y) * 0.713 + delta

        return torch.cat((u, v), dim=1)
    '''
    loss helpers end
    '''

    '''
    exclusion loss helper
    from: https://github.com/yossigandelsman/DoubleDIP/blob/master/net/losses.py
    '''
    def get_gradients(self, img1, img2, levels=3):
        gradx_loss = []
        grady_loss = []

        for l in range(levels):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady
    '''
    exclusion loss helper end
    '''

class TrainStep_Semi(TrainStep):
    def __init__(self, f, args):
        super().__init__(f, args)

    def recon_loss(self, T, A, J, img, clear_img=None):
        if clear_img is None:
            L_recon = self.Sl1(img - (J * T + A * (1 - T)))
        else:
            L_recon = self.Sl1(img - (clear_img * T + A * (1 - T))) + self.Sl1(J - clear_img)
        return L_recon

    # def forward(self, img, clear_img = None, return_package = False):
    def forward(self, img, clear_img = None):
        T, A, J = self.f(img)

        L_recon = self.recon_loss(T, A, J, img, clear_img = clear_img)
        
        if clear_img is not None:
            J = clear_img

        L_prior = self.prior_loss(T, A, J, img)
        L_clean = self.clean_loss(T, A, J, img)
        L_T_zero = self.T_zero_loss(T, A.expand_as(img), J, img)
        L_aug = self.augmentation_loss(T, A, J, img)
        L_reg = self.regularization_loss(T, A, J, img)

        L_total = self.lambdas["recon"] * L_recon + L_prior + self.lambdas["clean"] * L_clean +\
                     self.lambdas["aug"] * L_aug + L_reg + self.lambdas["T_zero"] * L_T_zero
        L_total = torch.nan_to_num(L_total, nan=0, posinf=0, neginf=0)
        # if return_package:
        return L_total, {"L_rec": L_recon,  "L_p": L_prior, "L_c": L_clean, 
                        "L_a": L_aug, "L_r": L_reg, "L_Tz": L_T_zero}
        # return L_total



class TrainStep_weighted_dcp(TrainStep):
    def __init__(self, f, args):
        super().__init__(f, args)

    def prior_loss(self, T, A, J, img):
        J_s, J_v = self.get_sv(J)
        img_s, img_v = self.get_sv(img)

        J_uv = self.get_uv(J)
        img_uv = self.get_uv(img)
        
        A_ = torch.amax(img, dim=(2,3), keepdim=True).clamp(min=0.5)
        dcp = self.dcp(img, A_)
        
        L_prior = self.lambdas["T_DCP"] * self.Sl1((self.large_boxblur(T) - self.large_boxblur_1(dcp)) * dcp) + \
                self.lambdas["J_TV"] * self.Sl1(self.TV(J)) +\
                self.lambdas["J_pixel_intensity"] * self.Sl1((J - img).clamp(min=0)) +\
                self.lambdas["J_value"] * self.Sl1((J_v - img_v).clamp(min=0)) +\
                self.lambdas["J_saturation"] * self.Sl1((img_s - J_s).clamp(min=0)) +\
                self.lambdas["J_hue"] * self.Sl1(J_uv - img_uv) +\
                self.lambdas["J_var"] * self.Sl1((self.var(img) - self.var(J)).clamp(min=0))

        return L_prior

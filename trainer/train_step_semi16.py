import torch
import torchvision
import random


from .utils import guidedfilter2d



class LocalVariance(torch.nn.Module):
    def __init__(self, channels = 3, kernel_size = 5):
        super().__init__()
        assert isinstance(kernel_size, int)
        self.kernel_size = kernel_size
        kernel = torch.ones(kernel_size, kernel_size) / (kernel_size**2)
        kernel = kernel.unsqueeze(0).unsqueeze(0) # shape of 1 * 1 * kernel_size * kernel_size
        kernel = kernel * torch.eye(channels).unsqueeze(2).unsqueeze(2) # size of channels * channels * kernel_size * kernel_size
        # kernel = kernel.to(device)
        # kernel -> channelwise local mean filter
        self.pad = [self.kernel_size//2]*4
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        E_x = torch.nn.functional.conv2d(torch.nn.functional.pad(x, self.pad, mode='reflect'), self.kernel, stride=1)
        E_x2 = torch.pow(E_x, 2)
        E2_x = torch.nn.functional.conv2d(torch.nn.functional.pad(torch.pow(x, 2), self.pad, mode='reflect'), self.kernel, stride=1)
        var_x = E2_x - E_x2
        return torch.clamp(var_x, min=1e-7)

class BoxBlur(torch.nn.Module):
    def __init__(self, channels=3, kernel_size=3):
        super().__init__()
        kernel = torch.ones(kernel_size, kernel_size) / (kernel_size ** 2)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel * torch.eye(channels).unsqueeze(2).unsqueeze(2)
        self.pad = [kernel_size//2] * 4
        self.register_buffer('kernel', kernel)
    
    def forward(self, x):
        return torch.nn.functional.conv2d(torch.nn.functional.pad(x, self.pad, mode='reflect'), self.kernel, stride=1)


class TrainStep_Enhance(torch.nn.Module):
    def __init__(self, f, lambda_aug=1, lambda_prior=1):
        super().__init__()
        self.f = f
        self.blur = BoxBlur()
        self.sl1 = torch.nn.L1Loss()

        self.lambda_aug = lambda_aug
        self.lambda_prior=0.1
        self.maxpool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.large_boxblur = BoxBlur(kernel_size=5)
        self.large_boxblur_1 = BoxBlur(channels=1, kernel_size=5)

        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)
        self.sigmoid = torch.nn.Sigmoid()

        self.get_max = torch.nn.AdaptiveMaxPool2d(output_size=(1,1))
        

    def Sl1(self, x):
        return self.sl1(x, torch.zeros_like(x))

    def g(self, T, A, clean):
        return T * clean + A * (1 - T)

    def dcp(self, img, A):
        min_patch = -self.maxpool(-img / A.clamp(min=1e-1))
        T_rough = 1 - torch.amin(min_patch, dim=1, keepdim=True)
        return T_rough.clamp(min=5e-2)

    def bcp(self, img, A):
        A = A.mean(dim=1, keepdim=True)
        T_rough = (torch.amax(self.maxpool(img), dim=1, keepdim=True) - A) / (1 - A).clamp(min=1e-2)
        return T_rough.clamp(min=5e-2)

    def TV(self, x):
        TV_x = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        TV_y = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()

        return TV_x + TV_y

    def get_sv(self, img, eps=1e-7):
        img_max = img.max(1)[0]
        img_min = img.min(1)[0]

        saturation = ( img_max - img_min ) / ( img_max + eps )
        saturation[ img_max==0 ] = 0

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
    exclusion loss part
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

    def exclusion_loss(self, img1, img2, levels=3):
        gradx_loss, grady_loss = self.get_gradients(img1, img2, levels=levels)
        loss_gradxy = sum(gradx_loss) / (levels * 9) + sum(grady_loss) / (levels * 9)
        return loss_gradxy / 2.0
    '''
    exclusion loss part END
    '''

    def prior_loss(self, T, A, J, img):
        J_s, J_v = self.get_sv(J)
        # J_uv = self.get_uv(J)
        # img_uv = self.get_uv(img)

        # instead of guided filtering which requires large computation, blur  dcp output 
        # dcp = self.large_boxblur_1(self.dcp(img))
        A_ = torch.amax(img, dim=(2,3), keepdim=True)
        dcp = self.dcp(img, A_)
        # bcp = self.bcp(img, A_)
        # L_prior = self.Sl1((self.large_boxblur(T) - dcp)) + \
        L_prior = 1 * self.Sl1((self.large_boxblur(T) - self.large_boxblur_1(dcp))) + \
                1e-1 * self.Sl1(J_s - J_v) +\
                1e-1 * self.Sl1(self.TV(J)) +\
                self.Sl1((J - img).clamp(min=0))
                # 1 * self.Sl1(T - T.mean(dim=1, keepdim=True))/
                # 1e-2 * self.Sl1(J.amin(dim=1, keepdim=True)) +\
                # 1e-2 * self.Sl1(J.amax(dim=1, keepdim=True) - 1) +\
        # self.Sl1((self.large_boxblur(T) - self.large_boxblur_1(bcp))) + \
                # 1e-2 * self.Sl1(J_uv - img_uv) +\
        return L_prior

    def clean_loss(self, J):
        clean_T, clean_A, clean_J = self.f(J)

        L_clean = self.Sl1(J - clean_J)
        return L_clean

    def augmentation_loss(self, T, A, J, img):
        T_augs = []
        # T_augs.append(torch.ones_like(T))
        T_augs.append(T)
        # T_augs.append(1-T*0.95)
        T_augs.append((torch.amax(T, dim=(2,3), keepdim=True) - T + torch.amin(T, dim=(2,3), keepdim=True)))
        T_augs.append((T + torch.randn(T.size(0), 1, 1, 1, device=T.device) * 0.3 + torch.randn(T.size(0), T.size(1), 1, 1, device=T.device) * 0.05).clamp(0.05, 1))
        T_augs.append((T * (1 + torch.randn(T.size(0), 1, 1, 1, device=T.device) * 0.3) + torch.randn(T.size(0), T.size(1), 1, 1, device=T.device) * 0.05).clamp(0.05, 1))
        
        L_aug = 0
        for T_aug in T_augs:
            A_aug = (A + (torch.randn(A.size(0), 1, 1, 1, device=A.device)*0.2) + torch.randn(A.size(0), A.size(1), 1, 1, device=A.device)*0.05).clamp(0.33, 1)
            img_aug = J * T_aug + A_aug * (1 - T_aug)
            aug_T, aug_A, aug_J = self.f(img_aug)
            L_aug = L_aug + (self.Sl1(aug_J - J) + self.Sl1(aug_T - T_aug) +\
                            self.Sl1(aug_A - A_aug)) / len(T_augs)
        return L_aug

    def regularization_loss(self, T, A, J, img):

        L_reg = self.Sl1(A - self.blur(A)) + \
                self.Sl1(A - torch.amax(img, dim=(2,3), keepdim=True)) + \
                self.exclusion_loss(A, T) +\
                self.Sl1(T - T.mean(dim=1, keepdim=True))
        return L_reg
    
    def recon_loss(self, T, A, J, img):
        L_recon = self.Sl1(img - self.g(T, A, J))
        return L_recon

    def forward(self, img, return_package = False):
        T, A, J = self.f(img)

        L_recon = self.recon_loss(T, A, J, img)
        L_prior = self.prior_loss(T, A, J, img)
        L_clean = self.clean_loss(J)
        L_aug = self.augmentation_loss(T, A, J, img)
        L_reg = self.regularization_loss(T, A, J, img)

        L_total = L_recon + self.lambda_prior * L_prior + L_clean + self.lambda_aug * L_aug + L_reg 
        if return_package:
            return L_total, {"L_rec": L_recon,  "L_p": L_prior, "L_c": L_clean, "L_a": L_aug, "L_r": L_reg}
        return L_total
 

class TrainStep_Enhance_AConst(TrainStep_Enhance):
    def __init__(self, f, lambda_aug=1, lambda_prior=1):
        super().__init__(f, lambda_aug, lambda_prior)

    def regularization_loss(self, T, A, J, img):

        L_reg = self.Sl1(A - torch.amax(img, dim=(2,3), keepdim=True)) + \
                1e-2 * self.Sl1(T - self.blur(T))
        return L_reg

    def forward(self, img, return_package = False):
        T, A, J = self.f(img)

        L_recon = self.recon_loss(T, A, J, img)
        L_prior = self.prior_loss(T, A, J, img)
        L_clean = self.clean_loss(J)
        L_aug = self.augmentation_loss(T, A, J, img)
        L_reg = self.regularization_loss(T, A, J, img)

        L_total = L_recon + self.lambda_prior * L_prior + L_clean + self.lambda_aug * L_aug + L_reg 
        if return_package:
            return L_total, {"L_rec": L_recon,  "L_p": L_prior, "L_c": L_clean, "L_a": L_aug, "L_r": L_reg}
        return L_total


class TrainStep_Enhance_Semi(TrainStep_Enhance):
    def __init__(self, f, lambda_aug=1, lambda_prior=1):
        super().__init__(f, lambda_aug, lambda_prior)
    
    def recon_loss(self, T, A, J, img, clear_img=None):
        if clear_img is None:
            L_recon = self.Sl1(img - self.g(T, A, J))
        else:
            L_recon = self.Sl1(img - self.g(T, A, clear_img)) + self.Sl1(J - clear_img)
        return L_recon
        
    def forward(self, img, clear_img=None, return_package = False):
        T, A, J = self.f(img)
        L_recon = self.recon_loss(T, A, J, img, clear_img)
        if clear_img is not None:
            J = clear_img

        L_prior = self.prior_loss(T, A, J, img)
        L_clean = self.clean_loss(J)
        L_aug = self.augmentation_loss(T, A, J, img)
        L_reg = self.regularization_loss(T, A, J, img)

        L_total = L_recon + self.lambda_prior * L_prior + L_clean + self.lambda_aug * L_aug + L_reg 
        if return_package:
            return L_total, {"L_rec": L_recon,  "L_p": L_prior, "L_c": L_clean, "L_a": L_aug, "L_r": L_reg}
        return L_total
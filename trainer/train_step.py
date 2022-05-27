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

class TrainStep(torch.nn.Module):
    def __init__(self, f, lambda_aug=0.4):
        super().__init__()
        self.f = f
        self.blur = BoxBlur()
        self.variance = LocalVariance()
        self.lambda_aug = lambda_aug
        self.l1 = torch.nn.L1Loss()

    def L2Norm(self, x):
        return self.l1(x, torch.zeros_like(x))
        return torch.sqrt(torch.mean(torch.pow(x, 2)) + 1e-3)

    def g(self, noise_D, noise_I, clean):
        return clean + clean * noise_D + noise_I

    @torch.no_grad()
    def augment(self, noise_D, noise_I, clean):
        B, C, H, W = noise_D.shape
        T = noise_D + 1
        TAug = (torch.amax(T, dim=(1,2,3), keepdim=True) - T + torch.amin(T, dim=(1,2,3), keepdim=True) - 1)

        # smoothed_noise_I = self.blur(noise_I)
        # captureNoise = noise_I - smoothed_noise_I

        # A = (smoothed_noise_I / (- noise_D))

        return TAug.detach(), noise_I.detach()

    def decompose(self, noise_D, noise_I, clean):
        B, C, H, W = noise_D.shape
        T = noise_D + 1
        TAug = (torch.amax(T, dim=(1,2,3), keepdim=True) - T + torch.amin(T, dim=(1,2,3), keepdim=True) - 1)
        smoothed_noise_I = self.blur(noise_I)
        captureNoise = noise_I - smoothed_noise_I

        A = 0
        # A = (smoothed_noise_I / (- noise_D))

        return T, A, captureNoise

    def forward(self, img, return_package = False):
        noise_D, noise_I, clean = self.f(img)

        L_con = self.L2Norm(img - self.g(noise_D, noise_I, clean))

        clean_noise_D, clean_noise_I, clean_clean = self.f(clean)
        dep_noise_D, dep_noise_I, dep_clean = self.f(self.g(noise_D, torch.zeros_like(noise_I), clean))
        indep_noise_D, indep_noise_I, indep_clean = self.f(noise_I)

        L_id = self.L2Norm(clean_clean - clean) + self.L2Norm(dep_clean - clean) + \
                self.L2Norm(dep_noise_D - noise_D) + self.L2Norm(indep_noise_I - noise_I)

        L_zero = self.L2Norm(clean_noise_D) + self.L2Norm(clean_noise_I) + \
                self.L2Norm(indep_clean) + self.L2Norm(dep_noise_I)

        aug_noise_D_target, aug_noise_I_target = self.augment(noise_D, noise_I, clean)

        aug_noise_D, aug_noise_I, aug_clean = self.f(self.g(aug_noise_D_target, aug_noise_I_target, clean))

        L_aug = self.L2Norm(aug_clean - clean) + self.L2Norm(aug_noise_D - aug_noise_D_target) + \
                self.L2Norm(aug_noise_I - aug_noise_I_target)

        L_reg = self.L2Norm(self.variance(img) - self.variance(noise_I))
                # self.L2Norm(A - self.blur(A)) + 0.1 * self.L2Norm(T - self.blur(T))
        L_total = L_con + L_id + L_zero + L_reg + self.lambda_aug * L_aug


        if return_package:
            return L_total, {"L_c": L_con, "L_i": L_id, "L_z": L_zero, "L_r": L_reg, "L_a": L_aug}

        if torch.any(torch.isnan(L_total)) and L_total.get_device()==0:
            print('da', img.mean(), noise_D.mean(), noise_I.mean(), clean.mean())
            print('cle', clean_noise_D.mean(), clean_noise_I.mean(), clean_clean.mean())
            print('dep', dep_noise_D.mean(), dep_noise_I.mean(), dep_clean.mean())
            print('indep', indep_noise_D.mean(), indep_noise_I.mean(), indep_clean.mean())
            print('lo', L_con.mean(), L_id.mean(), L_zero.mean(), L_aug.mean(), L_reg.mean())

        return L_total
 

class TrainStepSM(TrainStep):
    def __init__(self, f, lambda_aug=0.1):
        super().__init__(f, lambda_aug)
        self.sl1 = torch.nn.SmoothL1Loss()

    def L2Norm(self, x):
        return self.sl1(x, torch.zeros_like(x))
        return torch.sqrt(torch.mean(torch.pow(x, 2)) + 1e-6)


    def g(self, noise_D, noise_I, noise_C, clean):
        return clean + clean * noise_D + noise_I + noise_C

    def forward(self, img, return_package = False):
        noise_D, noise_I, noise_C, clean = self.f(img)
        T = (noise_D + 1).clamp(0, 1)
        A = (noise_I / (1 - T)).clamp(0, 1)

        L_con = self.L2Norm(img - self.g(noise_D, noise_I, noise_C, clean))
        
        clean_noise_D, clean_noise_I, clean_noise_C, clean_clean = self.f(clean.detach())
        dep_noise_D, dep_noise_I, dep_noise_C, dep_clean = self.f((clean + noise_D * clean).detach())
        indep_noise_D, indep_noise_I, indep_noise_C, indep_clean = self.f((noise_I + noise_C).detach())

        L_id = self.L2Norm(clean_clean - clean) + self.L2Norm(dep_clean - clean) + \
                self.L2Norm(dep_noise_D - noise_D) + self.L2Norm(indep_noise_I - noise_I) + self.L2Norm(indep_noise_C - noise_C)

        L_zero = self.L2Norm(clean_noise_D) + self.L2Norm(clean_noise_I) + self.L2Norm(clean_noise_C) + \
                self.L2Norm(indep_clean) + self.L2Norm(dep_noise_I) + self.L2Norm(dep_noise_C)

        with torch.no_grad():
            T_aug = (torch.amax(T, dim=(1,2,3), keepdim=True) - T + torch.amin(T, dim=(1,2,3), keepdim=True)).detach()
            A_aug = (A * (torch.rand(A.size(0), A.size(1), 1, 1, device=A.device) * 0.2 + 0.9)).clamp(0, 1).detach()  # scale A by 0.9 ~ 1.1
        
        L_aug = 0
        T_augs = [T, T_aug]
        A_augs = [A, A_aug]
        C_coeff = [1, -1]
        for T_aug in T_augs:
            for A_aug in A_augs:
                with torch.no_grad():
                    aug_noise_D_input = T_aug - 1
                    aug_noise_I_input = A_aug * (1 - T_aug)
                    aug_noise_C_input = (noise_C * random.choice(C_coeff))
                    img_aug = self.g(aug_noise_D_input, aug_noise_I_input, aug_noise_C_input, clean).detach()
                aug_noise_D, aug_noise_I, aug_noise_C, aug_clean = self.f(img_aug)
                L_aug = L_aug + self.L2Norm(aug_clean - clean) + self.L2Norm(aug_noise_D - aug_noise_D_input) +\
                                self.L2Norm(aug_noise_C - aug_noise_C_input) + self.L2Norm(aug_noise_I - aug_noise_I_input)

        L_reg = self.L2Norm(torch.sqrt(self.variance(img)+1e-4) - torch.sqrt(self.variance(noise_C)+1e-4)) + \
                1e-3 * self.L2Norm(T - self.blur(T)) + 1e-1 * self.L2Norm(A - self.blur(A))

        L_total = L_con + L_id + L_zero + L_reg + self.lambda_aug * L_aug
        if return_package:
            return L_total, {"L_c": L_con, "L_i": L_id, "L_z": L_zero, "L_r": L_reg, "L_a": L_aug}
        return L_total
 
 
class TrainStepExplicitSM(TrainStep):
    def __init__(self, f, lambda_aug=0.1):
        super().__init__(f, lambda_aug)
        self.sl1 = torch.nn.SmoothL1Loss()

    def L2Norm(self, x):
        # return self.sl1(x, torch.zeros_like(x))
        return torch.sqrt(torch.mean(torch.pow(x, 2)) + 1e-4)


    def g(self, T, A, C, clean):
        return T * clean + A * (1 - T) + C

    def forward(self, img, return_package = False):
        T, A, C, clean = self.f(img)

        L_con = self.L2Norm(img - self.g(T, A, C, clean))
        
        clean_T, clean_A, clean_C, clean_clean = self.f(clean.detach())

        T0_T, T0_A, T0_C, T0_clean = self.f((A + C).detach())

        A0_T, A0_A, A0_C, A0_clean = self.f((clean * T + C).detach())

        J0_T, J0_A, J0_C, J0_clean = self.f((A * (1 - T) + C).detach())

        L_id = self.L2Norm(clean - clean_clean) + self.L2Norm(clean - A0_clean) +\
                self.L2Norm(T - A0_T) + self.L2Norm(T - J0_T) +\
                self.L2Norm(A - T0_A) + self.L2Norm(A - J0_A) +\
                self.L2Norm(C - A0_C) + self.L2Norm(C - T0_C) + self.L2Norm(C - J0_C)

        L_zero = self.L2Norm(J0_clean) +\
                self.L2Norm(clean_T - 1) + self.L2Norm(T0_T) +\
                self.L2Norm(A0_A) +\
                self.L2Norm(clean_C)
                
        with torch.no_grad():
            T_aug = (torch.amax(T, dim=(1,2,3), keepdim=True) - T + torch.amin(T, dim=(1,2,3), keepdim=True)).detach()
            A_aug = (A * (torch.rand(A.size(0), A.size(1), 1, 1, device=A.device) * 0.2 + 0.9)).clamp(0, 1).detach()  # scale A by 0.9 ~ 1.1
        
        L_aug = 0
        T_augs = [T, T_aug]
        A_augs = [A, A_aug]
        C_coeff = [1, -1]
        for T_aug in T_augs:
            for A_aug in A_augs:
                with torch.no_grad():
                    C_aug = C * random.choice(C_coeff)
                    img_aug = self.g(T_aug, A_aug, C_aug, clean).detach()
                aug_T, aug_A, aug_C, aug_clean = self.f(img_aug)
                L_aug = L_aug + self.L2Norm(aug_clean - clean) + self.L2Norm(aug_T - T_aug) +\
                                self.L2Norm(aug_C - C_aug) + self.L2Norm(aug_A - A_aug)

        L_reg = self.L2Norm(torch.sqrt(self.variance(img)+1e-4) - torch.sqrt(self.variance(C)+1e-4)) + \
                1e-3 * self.L2Norm(A - self.blur(A))

        L_total = L_con + L_id + L_zero + L_reg + self.lambda_aug * L_aug
        if return_package:
            return L_total, {"L_con": L_con, "L_id": L_id, "L_zero": L_zero, "L_reg": L_reg, "L_aug": L_aug}
        return L_total
 
class TrainStepJAT(TrainStep):
    def __init__(self, f, lambda_aug=0.05):
        super().__init__(f, lambda_aug)
        self.l1 = torch.nn.L1Loss()

    def L2Norm(self, x):
        return self.l1(x, torch.zeros_like(x))

    def g(self, T, A, clean):
        return T * clean + A * (1 - T)

    def forward(self, img, return_package = False):
        T, A, J = self.f(img)

        L_con = self.L2Norm(img - self.g(T, A, J))
        
        clean_T, clean_A, clean_J = self.f(J)

        A0_T, A0_A, A0_J = self.f(J * T)

        T0_T, T0_A, T0_J = self.f(A)

        J0_T, J0_A, J0_J = self.f(A * (1 - T))

        L_id = self.L2Norm(J - clean_J) +\
                self.L2Norm(J - A0_J) + self.L2Norm(T - A0_T) +\
                self.L2Norm(A - T0_A) + \
                self.L2Norm(A - J0_A) + self.L2Norm(T - J0_T)

        L_zero = self.L2Norm(clean_T - 1) + \
                self.L2Norm(A0_A) +\
                self.L2Norm(T0_T) + self.L2Norm(T0_J) +\
                self.L2Norm(J0_J)

        T_augs = []
        T_augs.append(T)
        T_augs.append(1-T)
        T_augs.append(T**2)
        # T_augs.append((1-T)**2)
        T_augs.append((torch.amax(T, dim=(1,2,3), keepdim=True) - T + torch.amin(T, dim=(1,2,3), keepdim=True)))
        
        L_aug = 0
        for T_aug in T_augs:
            A_aug = (A * (torch.randn(A.size(0), A.size(1), 1, 1, device=A.device)*0.05 + 1) + (torch.randn(A.size(0), A.size(1), 1, 1, device=A.device)*0.05)).clamp(0, 1)
            img_aug = J * T_aug + A_aug * (1 - T_aug)
            aug_T, aug_A, aug_J = self.f(img_aug)
            L_aug = L_aug + self.L2Norm(aug_J - J) + self.L2Norm(aug_T - T_aug) +\
                            self.L2Norm(aug_A - A_aug)

        L_reg = 1e-3 * self.L2Norm(A - self.blur(A)) + 1e-7 * self.L2Norm(T - self.blur(T))

        L_total = L_con + L_id + L_zero + L_reg + self.lambda_aug * L_aug
        if return_package:
            return L_total, {"L_con": L_con, "L_id": L_id, "L_zero": L_zero, "L_reg": L_reg, "L_aug": L_aug}
        return L_total




class TrainStepJAT_RESIDE(TrainStep):
    def __init__(self, f, lambda_aug=0.05):
        super().__init__(f, lambda_aug)
        self.l1 = torch.nn.L1Loss()

    def L2Norm(self, x):
        return self.l1(x, torch.zeros_like(x))

    def g(self, T, A, clean):
        return T * clean + A * (1 - T)


    def forward(self, img, return_package = False):
        T, A, J = self.f(img)

        L_con = self.L2Norm(img - self.g(T, A, J))
        
        clean_T, clean_A, clean_J = self.f(J)

        A0_T, A0_A, A0_J = self.f(J * T)

        T0_T, T0_A, T0_J = self.f(A)

        J0_T, J0_A, J0_J = self.f(A * (1 - T))

        L_id = self.L2Norm(J - clean_J) +\
                self.L2Norm(J - A0_J) + self.L2Norm(T - A0_T) +\
                self.L2Norm(A - T0_A) + \
                self.L2Norm(A - J0_A) + self.L2Norm(T - J0_T)

        L_zero = self.L2Norm(clean_T - 1) + \
                self.L2Norm(A0_A) +\
                self.L2Norm(T0_T) + self.L2Norm(T0_J) +\
                self.L2Norm(J0_J)

        T_augs = []
        T_augs.append(T)
        T_augs.append(1-T)
        T_augs.append(T**2)
        # T_augs.append(torch.sqrt(T))
        T_augs.append((torch.amax(T, dim=(1,2,3), keepdim=True) - T + torch.amin(T, dim=(1,2,3), keepdim=True)))
        
        L_aug = 0
        for T_aug in T_augs:
            A_aug = (A * (torch.randn(A.size(0), A.size(1), 1, 1, device=A.device)*0.05 + 1) + (torch.randn(A.size(0), A.size(1), 1, 1, device=A.device)*0.05)).clamp(0, 1)
            img_aug = J * T_aug + A_aug * (1 - T_aug)
            aug_T, aug_A, aug_J = self.f(img_aug)
            L_aug = L_aug + self.L2Norm(aug_J - J) + self.L2Norm(aug_T - T_aug) +\
                            self.L2Norm(aug_A - A_aug)

        L_reg = 1e-3 * self.L2Norm(A - self.blur(A)) + 1e-7 * self.L2Norm(T - self.blur(T))

        L_total = L_con + L_id + L_zero + L_reg + self.lambda_aug * L_aug
        if return_package:
            return L_total, {"L_con": L_con, "L_id": L_id, "L_zero": L_zero, "L_reg": L_reg, "L_aug": L_aug}
        return L_total
 

class TrainStep_Enhance(TrainStep):
    def __init__(self, f, lambda_aug=1, lambda_prior=1):
        super().__init__(f, lambda_aug)
        self.sl1 = torch.nn.SmoothL1Loss()
        self.lambda_prior=0.1
        self.maxpool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.large_boxblur = BoxBlur(kernel_size=5)
        self.large_boxblur_1 = BoxBlur(channels=1, kernel_size=5)

    def Sl1(self, x):
        return self.sl1(x, torch.zeros_like(x))

    def g(self, T, A, clean):
        return T * clean + A * (1 - T)

    def dcp(self, img, A):
        min_patch = -self.maxpool(-img / A.clamp(min=1e-1))
        T_rough = 1 - torch.amin(min_patch, dim=1, keepdim=True)
        return T_rough

    def TV(self, x):
        TV_x = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        TV_y = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()

        return TV_x + TV_y

    def get_hsv(self, img, eps=1e-7):
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        img_max = img.max(1)[0]
        img_min = img.min(1)[0]

        hue[ img[:,2]==img_max ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img_max - img_min + eps) ) [ img[:,2]==img_max ]
        hue[ img[:,1]==img_max ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img_max - img_min + eps) ) [ img[:,1]==img_max ]
        hue[ img[:,0]==img_max ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img_max - img_min + eps) ) [ img[:,0]==img_max ]) % 6

        hue[img_min==img_max] = 0.0
        hue = hue/6

        saturation = ( img_max - img_min ) / ( img_max + eps )
        saturation[ img_max==0 ] = 0

        value = img_max
        return hue, saturation, value
        
    def get_uv(self, img):
        r = img[:, 0:1]
        g = img[:, 1:2]
        b = img[:, 2:3]

        delta = 0.5
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = (b - y) * 0.564 + delta
        v = (r - y) * 0.713 + delta

        return torch.cat((u, v), dim=1)

    def forward(self, img, return_package = False):
        T, A, J = self.f(img)

        L_recon = self.Sl1(img - self.g(T, A, J))

        J_h, J_s, J_v = self.get_hsv(J)
        J_uv = self.get_uv(J)
        img_uv = self.get_uv(img)

        # instead of guided filtering which requires large computation, blur  dcp output 
        # dcp = self.large_boxblur_1(self.dcp(img))
        dcp = self.dcp(img, A)
        # L_prior = self.Sl1((self.large_boxblur(T) - dcp)) + \
        L_prior = self.Sl1((self.large_boxblur(T) - self.large_boxblur_1(dcp))) + \
                1e-1 * self.Sl1(self.TV(J)) +\
                1e-1 * self.Sl1(J_s - J_v) +\
                1e-1 * self.Sl1(J_uv - img_uv)
                
        clean_T, clean_A, clean_J = self.f(J)

        L_clear = self.Sl1(J - clean_J)

        # A_T, A_A, A_J = self.f(A)

        # L_A = self.Sl1(A_A - A) + self.Sl1(A_J)

        T_augs = []
        T_augs.append(torch.ones_like(T))
        T_augs.append(T)
        T_augs.append(1-T*0.95)
        T_augs.append((torch.amax(T, dim=(2,3), keepdim=True) - T + torch.amin(T, dim=(2,3), keepdim=True)))
        T_augs.append((T + torch.randn(T.size(0), 1, 1, 1, device=T.device) * 0.3).clamp(0.05, 1))
        
        L_aug = 0
        for T_aug in T_augs:
            A_aug = (A + (torch.randn(A.size(0), 1, 1, 1, device=A.device)*0.1)).clamp(0, 1)
            img_aug = J * T_aug + A_aug * (1 - T_aug)
            aug_T, aug_A, aug_J = self.f(img_aug)
            L_aug = L_aug + self.Sl1(aug_J - J) + self.Sl1(aug_T - T_aug) +\
                            self.Sl1(aug_A - A_aug)


        L_reg = self.Sl1(A - self.blur(A)) + \
                1e-4 * self.Sl1(A - 1) + \
                1e-1 * self.Sl1(T - self.blur(T)) 

        L_total = L_recon + self.lambda_prior * L_prior + L_clear + self.lambda_aug * L_aug + L_reg 
        if return_package:
            return L_total, {"L_rec": L_recon,  "L_p": L_prior, "L_c": L_clear, "L_a": L_aug, "L_r": L_reg}
        return L_total
 


 



 
class TrainStep_Enhance_AConst(TrainStep):
    def __init__(self, f, lambda_aug=1, lambda_prior=1):
        super().__init__(f, lambda_aug)
        self.sl1 = torch.nn.SmoothL1Loss()
        self.lambda_prior=0.1
        self.maxpool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.large_boxblur = BoxBlur(kernel_size=5)
        self.large_boxblur_1 = BoxBlur(channels=1, kernel_size=5)

    def Sl1(self, x):
        return self.sl1(x, torch.zeros_like(x))

    def g(self, T, A, clean):
        return T * clean + A * (1 - T)

    def dcp(self, img, A):
        min_patch = -self.maxpool(-img / A.clamp(min=1e-1))
        T_rough = 1 - torch.amin(min_patch, dim=1, keepdim=True)
        return T_rough

    def TV(self, x):
        TV_x = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        TV_y = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()

        return TV_x + TV_y

    def forward(self, img, return_package = False):
        T, A, J = self.f(img)

        L_recon = self.Sl1(img - self.g(T, A, J))
        
        J_rgb_max = torch.amax(J, dim=1, keepdim=True)
        J_rgb_min = torch.amin(J, dim=1, keepdim=True)
        J_HSV_S = (J_rgb_max - J_rgb_min + 1e-4) / (J_rgb_max + 1e-4)
        J_HSV_V = J_rgb_max

        # instead of guided filtering which requires large computation, blur  dcp output 
        dcp = self.dcp(img, A)
        # L_prior = self.Sl1((self.large_boxblur(T) - dcp)) + \
        L_prior = self.Sl1((self.large_boxblur(T) - self.large_boxblur_1(dcp))) + \
                1e-1 * self.Sl1(self.TV(J)) +\
                1e-1 * self.Sl1(J_HSV_S - J_HSV_V)
                
        clean_T, clean_A, clean_J = self.f(J)

        L_clear = self.Sl1(J - clean_J)

        # A_T, A_A, A_J = self.f(A.expand_as(J))

        T_augs = []
        T_augs.append(T)
        T_augs.append(1-T/2)
        T_augs.append((torch.amax(T, dim=(1,2,3), keepdim=True) - T + torch.amin(T, dim=(1,2,3), keepdim=True)))
        T_augs.append((T + torch.randn(T.size(0), 1, 1, 1, device=T.device) * 0.3).clamp(0, 1))
        
        L_aug = 0
        for T_aug in T_augs:
            A_aug = (A + (torch.randn(A.size(0), A.size(1), 1, 1, device=A.device)*0.2)).clamp(0, 1)
            img_aug = J * T_aug + A_aug * (1 - T_aug)
            aug_T, aug_A, aug_J = self.f(img_aug)
            L_aug = L_aug + self.Sl1(aug_J - J) + self.Sl1(aug_T - T_aug) +\
                            self.Sl1(aug_A - A_aug)


        L_reg = 1e-4 * self.Sl1(A - 1) + \
                self.Sl1(T - self.blur(T))


        L_total = L_recon + self.lambda_prior * L_prior + L_clear + self.lambda_aug * L_aug + L_reg 
        if return_package:
            return L_total, {"L_rec": L_recon, "L_p": L_prior, "L_c": L_clear, "L_a": L_aug, "L_r": L_reg}
        return L_total
import torch
import torchvision
import random

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
    def __init__(self, f, lambda_aug=0.1):
        super().__init__()
        self.f = f
        self.blur = BoxBlur()
        self.variance = LocalVariance()
        self.lambda_aug = lambda_aug

    def L2Norm(self, x):
        return torch.sqrt(torch.mean(torch.pow(x, 2)))

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

    def forward(self, img):
        noise_D, noise_I, clean = self.f(img)
        T, A, captureNoise = self.decompose(noise_D, noise_I, clean)

        L_con = self.L2Norm(img - self.g(noise_D, noise_I, clean))

        clean_noise_D, clean_noise_I, clean_clean = self.f(clean)
        dep_noise_D, dep_noise_I, dep_clean = self.f(self.g(noise_D, torch.zeros_like(noise_I), clean).detach())
        indep_noise_D, indep_noise_I, indep_clean = self.f(noise_I.detach())

        L_id = self.L2Norm(clean_clean - clean) + self.L2Norm(dep_clean - clean) + \
                self.L2Norm(dep_noise_D - noise_D) + self.L2Norm(indep_noise_I - noise_I)

        L_zero = self.L2Norm(clean_noise_D) + self.L2Norm(clean_noise_I) + \
                self.L2Norm(indep_clean) + self.L2Norm(dep_noise_I)

        aug_noise_D_target, aug_noise_I_target = self.augment(noise_D, noise_I, clean)

        aug_noise_D, aug_noise_I, aug_clean = self.f(self.g(aug_noise_D_target, aug_noise_I_target, clean).detach())

        L_aug = self.L2Norm(aug_clean - clean) + self.L2Norm(aug_noise_D - aug_noise_D_target) + \
                self.L2Norm(aug_noise_I - aug_noise_I_target)

        L_reg = self.L2Norm(torch.mean(self.variance(img), dim=(1,2,3)) - torch.mean(self.variance(noise_I), dim=(1,2,3))) 
                # self.L2Norm(A - self.blur(A)) + 0.1 * self.L2Norm(T - self.blur(T))
        L_total = L_con + L_id + L_zero + L_reg + self.lambda_aug * L_aug
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

    def L2Norm(self, x):

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

        L_reg = self.L2Norm(self.variance(img) - self.variance(noise_C)) + \
                1e-3 * self.L2Norm(T - self.blur(T)) + 1e-1 * self.L2Norm(A - self.blur(A))

        L_total = L_con + L_id + L_zero + L_reg + self.lambda_aug * L_aug
        if return_package:
            return L_total, {"L_con": L_con, "L_id": L_id, "L_zero": L_zero, "L_reg": L_reg, "L_aug": L_aug}
        return L_total
 
 
class TrainStepExplicitSM(TrainStep):
    def __init__(self, f, lambda_aug=0.1):
        super().__init__(f, lambda_aug)

    def L2Norm(self, x):

        return torch.sqrt(torch.mean(torch.pow(x, 2)) + 1e-6)


    def g(self, T, A, C, clean):
        return T * clean + A * (1 - T) + C

    def forward(self, img, return_package = False):
        T, A, C, clean = self.f(img)

        L_con = self.L2Norm(img - self.g(T, A, C, clean))
        
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

        L_reg = self.L2Norm(self.variance(img) - self.variance(noise_C)) + \
                1e-3 * self.L2Norm(T - self.blur(T)) + 1e-1 * self.L2Norm(A - self.blur(A))

        L_total = L_con + L_id + L_zero + L_reg + self.lambda_aug * L_aug
        if return_package:
            return L_total, {"L_con": L_con, "L_id": L_id, "L_zero": L_zero, "L_reg": L_reg, "L_aug": L_aug}
        return L_total
 
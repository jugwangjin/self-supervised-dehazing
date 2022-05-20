import torch
import torchvision

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
    def __init__(self, channels=3, kernelSize=3):
        super().__init__()
        kernel = torch.ones(kernelSize, kernelSize) / (kernelSize ** 2)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel * torch.eye(channels).unsqueeze(2).unsqueeze(2)
        self.pad = [kernelSize//2] * 4
        self.register_buffer('kernel', kernel)
    
    def forward(self, x):
        return torch.nn.functional.conv2d(torch.nn.functional.pad(x, self.pad, mode='reflect'), self.kernel, stride=1)

class TrainStep(torch.nn.Module):
    def __init__(self, f, lambdaAug=0.1):
        super().__init__()
        self.f = f
        self.blur = BoxBlur()
        self.variance = LocalVariance()
        self.lambdaAug = lambdaAug

    def L2Norm(self, x):
        return torch.sqrt(torch.mean(torch.pow(x, 2)))

    def g(self, noiseD, noiseI, clean):
        return clean + clean * noiseD + noiseI

    @torch.no_grad()
    def augment(self, noiseD, noiseI, clean):
        B, C, H, W = noiseD.shape
        T = noiseD + 1
        TAug = (torch.amax(T, dim=(1,2,3), keepdim=True) - T + torch.amin(T, dim=(1,2,3), keepdim=True) - 1)

        # smoothedNoiseI = self.blur(noiseI)
        # captureNoise = noiseI - smoothedNoiseI

        # A = (smoothedNoiseI / (- noiseD))

        return TAug.detach(), noiseI.detach()

    def decompose(self, noiseD, noiseI, clean):
        B, C, H, W = noiseD.shape
        T = noiseD + 1
        TAug = (torch.amax(T, dim=(1,2,3), keepdim=True) - T + torch.amin(T, dim=(1,2,3), keepdim=True) - 1)
        smoothedNoiseI = self.blur(noiseI)
        captureNoise = noiseI - smoothedNoiseI

        A = 0
        # A = (smoothedNoiseI / (- noiseD))

        return T, A, captureNoise

    def forward(self, img):
        noiseD, noiseI, clean = self.f(img)
        T, A, captureNoise = self.decompose(noiseD, noiseI, clean)

        LCon = self.L2Norm(img - self.g(noiseD, noiseI, clean))

        cleanNoiseD, cleanNoiseI, cleanClean = self.f(clean)
        depNoiseD, depNoiseI, depClean = self.f(self.g(noiseD, torch.zeros_like(noiseI), clean).detach())
        indepNoiseD, indepNoiseI, indepClean = self.f(self.g(noiseD, noiseI, torch.zeros_like(clean)).detach())

        LId = self.L2Norm(cleanClean - clean) + self.L2Norm(depClean - clean) + \
                self.L2Norm(depNoiseD - noiseD) + self.L2Norm(indepNoiseI - noiseI)

        LZero = self.L2Norm(cleanNoiseD) + self.L2Norm(cleanNoiseI) + \
                self.L2Norm(indepClean) + self.L2Norm(depNoiseI)

        augNoiseDTarget, augNoiseITarget = self.augment(noiseD, noiseI, clean)

        augNoiseD, augNoiseI, augClean = self.f(self.g(augNoiseDTarget, augNoiseITarget, clean).detach())

        LAug = self.L2Norm(augClean - clean) + self.L2Norm(augNoiseD - augNoiseDTarget) + \
                self.L2Norm(augNoiseI - augNoiseITarget)

        LReg = self.L2Norm(torch.mean(self.variance(img), dim=(1,2,3)) - torch.mean(self.variance(noiseI), dim=(1,2,3))) 
                # self.L2Norm(A - self.blur(A)) + 0.1 * self.L2Norm(T - self.blur(T))
        LTotal = LCon + LId + LZero + LReg + self.lambdaAug * LAug

        if torch.any(torch.isnan(LTotal)):
            print('da', img.mean(), noiseD.mean(), noiseI.mean(), clean.mean())
            print('cle', cleanNoiseD.mean(), cleanNoiseI.mean(), cleanClean.mean())
            print('dep', depNoiseD.mean(), depNoiseI.mean(), depClean.mean())
            print('indep', indepNoiseD.mean(), indepNoiseI.mean(), indepClean.mean())
            print('lo', LCon.mean(), LId.mean(), LZero.mean(), LAug.mean(), LReg.mean())

        return LTotal
 
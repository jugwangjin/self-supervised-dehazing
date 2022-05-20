import torch
import torchvision

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

class Decompose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blur = BoxBlur()

    def forward(self, noiseD, noiseI, clean):
        B, C, H, W = noiseD.shape
        T = noiseD + 1
        TAug = (torch.amax(T, dim=(1,2,3), keepdim=True) - T + torch.amin(T, dim=(1,2,3), keepdim=True) - 1)
        smoothedNoiseI = self.blur(noiseI)
        captureNoise = noiseI - smoothedNoiseI

        # A = 0/
        A = torch.nan_to_num(smoothedNoiseI / (- noiseD), nan=1.0).clamp(0, 1)

        return T, A, captureNoise
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

    def forward(self, noise_D, noise_I, clean):
        B, C, H, W = noise_D.shape
        T = noise_D + 1
        TAug = (torch.amax(T, dim=(1,2,3), keepdim=True) - T + torch.amin(T, dim=(1,2,3), keepdim=True) - 1)
        smoothed_noise_I = self.blur(noise_I)
        capture_noise = noise_I - smoothed_noise_I

        # A = 0/
        A = torch.nan_to_num(smoothed_noise_I / (- noise_D), nan=1.0).clamp(0, 1)

        return T, A, capture_noise
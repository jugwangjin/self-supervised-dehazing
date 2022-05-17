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

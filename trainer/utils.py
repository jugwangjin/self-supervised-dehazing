import  torch

class LocalVariance(torch.nn.Module):
    def __init__(self, channels = 3, kernel_size = 5):
        super().__init__()
        assert isinstance(kernel_size, int)
        self.kernel_size = kernel_size
        kernel = torch.ones(kernel_size, kernel_size) / (kernel_size**2)
        kernel = kernel.unsqueeze(0).unsqueeze(0) # shape of 1 * 1 * kernel_size * kernel_size
        kernel = kernel * torch.eye(channels).unsqueeze(2).unsqueeze(2) # size of channels * channels * kernel_size * kernel_size
        self.pad = [self.kernel_size//2]*4
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        E_x = torch.nn.functional.conv2d(torch.nn.functional.pad(x, self.pad, mode='replicate'), self.kernel, stride=1)
        E_x2 = torch.pow(E_x, 2)
        E2_x = torch.nn.functional.conv2d(torch.nn.functional.pad(torch.pow(x, 2), self.pad, mode='replicate'), self.kernel, stride=1)
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
        return torch.nn.functional.conv2d(torch.nn.functional.pad(x, self.pad, mode='replicate'), self.kernel, stride=1)

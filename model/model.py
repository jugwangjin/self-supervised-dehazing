# https://github.com/Reyhanehne/CVF-SID_PyTorch/blob/main/src/model/model.py

if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from base.base_model import BaseModel

class GenClean(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(GenClean, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0))
        self.genclean = nn.Sequential(*layers)
        # for m in self.genclean:
        #     if isinstance(m, nn.Conv2d):
        #        nn.init.xavier_uniform(m.weight)
        #        nn.init.constant(m.bias, 0)

    def forward(self, x):
        out = self.genclean(x)
        return out


class GenNoise(nn.Module):
    def __init__(self, NLayer=10, FSize=64):
        super(GenNoise, self).__init__()
        kernel_size = 3
        padding = 1
        m = [nn.Conv2d(3, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'),
             nn.ReLU(inplace=True)]             
        for i in range(NLayer-1):
            m.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            m.append(nn.ReLU(inplace=True))        
        self.body = nn.Sequential(*m)
        
        gen_noise_w = []
        for i in range(4):
            gen_noise_w.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            gen_noise_w.append(nn.ReLU(inplace=True))
        gen_noise_w.append(nn.Conv2d(FSize, 3, kernel_size=1, padding=0))       
        self.gen_noise_w = nn.Sequential(*gen_noise_w)

        gen_noise_b = []
        for i in range(4):
            gen_noise_b.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            gen_noise_b.append(nn.ReLU(inplace=True))
        gen_noise_b.append(nn.Conv2d(FSize, 3, kernel_size=1, padding=0))       
        self.gen_noise_b = nn.Sequential(*gen_noise_b)
        
        # for m in self.body:
        #     if isinstance(m, nn.Conv2d):
        #        nn.init.xavier_uniform(m.weight)
        #        nn.init.constant(m.bias, 0)
        # for m in self.gen_noise_w:
        #     if isinstance(m, nn.Conv2d):
        #        nn.init.xavier_uniform(m.weight)
        #        nn.init.constant(m.bias, 0)  
        # for m in self.gen_noise_b:
        #     if isinstance(m, nn.Conv2d):
        #        nn.init.xavier_uniform(m.weight)
        #        nn.init.constant(m.bias, 0)
	       
    def forward(self, x, weights=None, test=False):
        noise = self.body(x)
        noise_w = self.gen_noise_w(noise)
        noise_b = self.gen_noise_b(noise)       
          
        m_w = torch.mean(torch.mean(noise_w,-1),-1).unsqueeze(-1).unsqueeze(-1)
        noise_w = noise_w-m_w      
        m_b = torch.mean(torch.mean(noise_b,-1),-1).unsqueeze(-1).unsqueeze(-1)
        noise_b = noise_b-m_b

        return noise_w, noise_b
 

class GenCleanSM(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(GenCleanSM, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
        layers.append(nn.PReLU(features))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            layers.append(nn.InstanceNorm2d(features))
            layers.append(nn.PReLU(features))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0))
        self.genclean = nn.Sequential(*layers)
        # for m in self.genclean:
        #     if isinstance(m, nn.Conv2d):
        #        nn.init.xavier_uniform(m.weight)
        #        nn.init.constant(m.bias, 0)

    def forward(self, x):
        out = self.genclean(x)
        return out

class GenNoiseSM(nn.Module):
    def __init__(self, NLayer=8, FSize=64):
        super(GenNoiseSM, self).__init__()
        kernel_size = 3
        padding = 1
        m = [nn.Conv2d(3, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'),
             nn.PReLU(FSize)]             
        for i in range(NLayer-1):
            m.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            m.append(nn.InstanceNorm2d(FSize))
            m.append(nn.PReLU(FSize))        
        self.body = nn.Sequential(*m)
        
        gen_T = []
        for i in range(4):
            gen_T.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            gen_T.append(nn.InstanceNorm2d(FSize))
            gen_T.append(nn.PReLU(FSize))
        gen_T.append(nn.Conv2d(FSize, 3, kernel_size=1, padding=0))       
        self.gen_T = nn.Sequential(*gen_T)

        gen_A = []
        for i in range(4):
            gen_A.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            gen_A.append(nn.InstanceNorm2d(FSize))
            gen_A.append(nn.PReLU(FSize))
        gen_A.append(nn.Conv2d(FSize, 3, kernel_size=1, padding=0))       
        self.gen_A = nn.Sequential(*gen_A)

        gen_C = []
        for i in range(4):
            gen_C.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            gen_C.append(nn.InstanceNorm2d(FSize))
            gen_C.append(nn.PReLU(FSize))
        gen_C.append(nn.Conv2d(FSize, 3, kernel_size=1, padding=0))       
        self.gen_C = nn.Sequential(*gen_C)
        
        # for m in self.body:
        #     if isinstance(m, nn.Conv2d):
        #        nn.init.xavier_uniform(m.weight)
        #        nn.init.constant(m.bias, 0)
        # for m in self.gen_T:
        #     if isinstance(m, nn.Conv2d):
        #        nn.init.xavier_uniform(m.weight)
        #        nn.init.constant(m.bias, 0)  
        # for m in self.gen_A:
        #     if isinstance(m, nn.Conv2d):
        #        nn.init.xavier_uniform(m.weight)
        #        nn.init.constant(m.bias, 0)
        # for m in self.gen_C:
        #     if isinstance(m, nn.Conv2d):
        #        nn.init.xavier_uniform(m.weight)
        #        nn.init.constant(m.bias, 0)
	       
    def forward(self, x, weights=None, test=False):
        noise = self.body(x)
        T = self.gen_T(noise)
        A = self.gen_A(noise)       
        C = self.gen_C(noise)       
          
        return T, A, C



class CVFModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.n_colors = 3
        FSize = 64
        self.gen_noise = GenNoise(FSize=FSize)
        self.genclean = GenClean()
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, x, weights=None, test=False):                   

        clean = self.genclean(x)
        noise_w, noise_b = self.gen_noise(x-clean)               
        return noise_w, noise_b, clean



class SMCVFModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.n_colors = 3
        FSize = 64
        self.gen_noise = GenNoiseSM()
        self.genclean = GenCleanSM()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, weights=None, test=False):                   

        clean = self.genclean(x)
        T, A, C = self.gen_noise(x-clean)               
        T = self.sigmoid(T)
        A = self.sigmoid(A).clamp(min=1e-2)
        noise_D = T - 1
        noise_I = A * (1 - T)
        noise_C = C - torch.mean(C, dim=(2,3), keepdim=True)
        return noise_D, noise_I, noise_C, clean


class ExplicitSMCVFModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.n_colors = 3
        FSize = 64
        self.gen_noise = GenNoiseSM()
        self.genclean = GenCleanSM()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, weights=None, test=False):                   

        clean = self.genclean(x)
        T, A, C = self.gen_noise(x-clean)               
        T = self.sigmoid(T)
        A = self.sigmoid(A).clamp(min=1e-2)
        C = C - torch.mean(C, dim=(2,3), keepdim=True)
        return T, A, C, clean
# https://github.com/Reyhanehne/CVF-SID_PyTorch/blob/main/src/model/model.py

if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from base.base_model import BaseModel
import torch.nn.functional as F

class GenClean(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(GenClean, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
        layers.append(nn.PReLU(features))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            layers.append(nn.PReLU(features))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0))
        self.genclean = nn.Sequential(*layers)
        for m in self.genclean:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)

    def forward(self, x):
        out = self.genclean(x) + x
        return out


class GenNoise(nn.Module):
    def __init__(self, NLayer=10, FSize=64):
        super(GenNoise, self).__init__()
        kernel_size = 3
        padding = 1
        m = [nn.Conv2d(3, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'),
             nn.PReLU(FSize)]             
        for i in range(NLayer-1):
            m.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            m.append(nn.PReLU(FSize))        
        self.body = nn.Sequential(*m)
        
        gen_noise_w = []
        for i in range(4):
            gen_noise_w.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            gen_noise_w.append(nn.PReLU(FSize))
        gen_noise_w.append(nn.Conv2d(FSize, 3, kernel_size=1, padding=0))       
        self.gen_noise_w = nn.Sequential(*gen_noise_w)

        gen_noise_b = []
        for i in range(4):
            gen_noise_b.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            gen_noise_b.append(nn.PReLU(FSize))
        gen_noise_b.append(nn.Conv2d(FSize, 3, kernel_size=1, padding=0))       
        self.gen_noise_b = nn.Sequential(*gen_noise_b)
        
        for m in self.body:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)
        for m in self.gen_noise_w:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)  
        for m in self.gen_noise_b:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)
	       
    def forward(self, x, weights=None, test=False):
        noise = self.body(x)
        noise_w = self.gen_noise_w(noise)
        noise_b = self.gen_noise_b(noise)       
          
        # m_w = torch.mean(torch.mean(noise_w,-1),-1).unsqueeze(-1).unsqueeze(-1)
        # noise_w = noise_w-m_w      
        # m_b = torch.mean(torch.mean(noise_b,-1),-1).unsqueeze(-1).unsqueeze(-1)
        # noise_b = noise_b-m_b

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
            # layers.append(nn.InstanceNorm2d(features))
            layers.append(nn.PReLU(features))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0))
        self.genclean = nn.Sequential(*layers)
        for m in self.genclean:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)

    def forward(self, x):
        out = self.genclean(x) + x
        return out

class GenNoiseSM(nn.Module):
    def __init__(self, NLayer=8, FSize=64):
        super(GenNoiseSM, self).__init__()
        kernel_size = 3
        padding = 1
        m = [nn.Conv2d(6, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'),
             nn.PReLU(FSize)]             
        for i in range(NLayer-1):
            m.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            # m.append(nn.InstanceNorm2d(FSize))
            m.append(nn.PReLU(FSize))        
        self.body = nn.Sequential(*m)
        
        gen_T = []
        for i in range(4):
            gen_T.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            # gen_T.append(nn.InstanceNorm2d(FSize))
            gen_T.append(nn.PReLU(FSize))
        gen_T.append(nn.Conv2d(FSize, 3, kernel_size=1, padding=0))       
        self.gen_T = nn.Sequential(*gen_T)

        gen_A = []
        for i in range(4):
            gen_A.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            # gen_A.append(nn.InstanceNorm2d(FSize))
            gen_A.append(nn.PReLU(FSize))
        gen_A.append(nn.Conv2d(FSize, 3, kernel_size=1, padding=0))       
        self.gen_A = nn.Sequential(*gen_A)

        gen_C = []
        for i in range(4):
            gen_C.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
            # gen_C.append(nn.InstanceNorm2d(FSize))
            gen_C.append(nn.PReLU(FSize))
        gen_C.append(nn.Conv2d(FSize, 3, kernel_size=1, padding=0))       
        self.gen_C = nn.Sequential(*gen_C)
        
        for m in self.body:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)
        for m in self.gen_T:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)  
        for m in self.gen_A:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)
        for m in self.gen_C:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)
	       
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
        T, A, C = self.gen_noise(torch.cat((x,clean), dim=1))               
        T = self.sigmoid(T)
        A = self.sigmoid(A).clamp(min=1e-2)
        C = C - torch.mean(C, dim=(2,3), keepdim=True)
        return T, A, C, clean





class GenCleanJAT(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(GenCleanJAT, self).__init__()

        kernel_size = 3
        padding = 1
        features = 64

        class ResidualConvReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, padding_mode='reflect')
                self.relu = nn.PReLU(features)
            def forward(self, x):
                return x + self.relu(self.conv(x))
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))
        layers.append(nn.PReLU(features))
        for _ in range(num_of_layers-2):
            layers.append(ResidualConvReLU())
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0))
        self.genclean = nn.Sequential(*layers)
        for m in self.genclean:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)

    def forward(self, x):
        out = self.genclean(x) 
        return out

class GenNoiseJAT(nn.Module):
    def __init__(self, NLayer=8, FSize=64):
        super(GenNoiseJAT, self).__init__()
        kernel_size = 3
        padding = 1

        class ResidualConvReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(in_channels=FSize, out_channels=FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect')
                self.relu = nn.PReLU(FSize)
            def forward(self, x):
                return x + self.relu(self.conv(x))
        m = [nn.Conv2d(6, FSize, kernel_size=kernel_size, padding=padding, padding_mode='reflect'),
             nn.PReLU(FSize)]             
        for i in range(NLayer-1):
            m.append(ResidualConvReLU())
        self.body = nn.Sequential(*m)
        
        gen_T = []
        for i in range(4):
            gen_T.append(ResidualConvReLU())
        gen_T.append(nn.Conv2d(FSize, 3, kernel_size=1, padding=0))       
        self.gen_T = nn.Sequential(*gen_T)

        gen_A = []
        for i in range(4):
            gen_A.append(ResidualConvReLU())
        gen_A.append(nn.Conv2d(FSize, 3, kernel_size=1, padding=0))       
        self.gen_A = nn.Sequential(*gen_A)

        for m in self.body:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)
        for m in self.gen_T:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)  
        for m in self.gen_A:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.5)
               nn.init.constant(m.bias, 0)
	       
    def forward(self, x, weights=None, test=False):
        noise = self.body(x)
        T = self.gen_T(noise)
        A = self.gen_A(noise)        
          
        return T, A


class JAT_CVFModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.n_colors = 3
        FSize = 64
        self.gen_noise = GenNoiseJAT()
        self.genclean = GenCleanJAT()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, weights=None, test=False):                   

        clean = self.genclean(x)
        clean = self.sigmoid(clean)
        T, A = self.gen_noise(torch.cat((x,clean), dim=1))               
        T = self.sigmoid(T).clamp(min=1e-3, max=1-1e-3)
        A = self.sigmoid(A).clamp(min=1e-2)
        return T, A, clean


class EnhancementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_downs = 2
        self.downscale = 2**self.num_downs

        self.conv1_1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(32),)
        self.conv2_1 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1, padding_mode='reflect'), nn.InstanceNorm2d(64), nn.LeakyReLU(),)
        self.conv2_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(64), nn.LeakyReLU(),)   
        self.conv2_3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(64),)
        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, padding_mode='reflect'), nn.InstanceNorm2d(128), nn.LeakyReLU(),)
        self.conv3_2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(128),)

        class ResBlocks(nn.Module):
            def __init__(self, channels):
                super().__init__()
                conv1 = nn.Sequential(
                                    nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),
                                    nn.InstanceNorm2d(channels),
                                    nn.LeakyReLU()
                            )
                conv2 = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect')
                self.conv = nn.Sequential(conv1, conv2)

            def forward(self, x):
                return x + self.conv(x)

        # resblocks = [ResBlocks(128) for _ in range(16)] # semi 16
        resblocks = [ResBlocks(128) for _ in range(16)]
        self.resblocks = nn.Sequential(*resblocks)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv4_1 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(128))
        self.conv5_1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(64))
        self.relu5_1 = nn.LeakyReLU()
        self.conv5_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(64), nn.LeakyReLU())
        self.conv5_3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(64), nn.LeakyReLU())
        self.conv6_1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(32))
        self.relu6_1 = nn.LeakyReLU()
        self.conv6_2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(32), nn.LeakyReLU())
        self.conv6_3 = nn.Sequential(nn.Conv2d(32, 9, 3, 1, 1, padding_mode='reflect'))

        self.sigmoid = nn.Sigmoid() # for T and A

    def pad(self, x):
        x_padx = ((self.downscale) - (x.size(2)%(self.downscale))) % (self.downscale)
        x_pady = ((self.downscale) - (x.size(3)%(self.downscale))) % (self.downscale)
        x = F.pad(x, [x_pady//2, (x_pady - x_pady//2), 
                    x_padx // 2, (x_padx - x_padx//2)], mode='reflect')

        return x, (x_padx, x_pady)

    def unpad(self, x, pad):

        x = x[:, :, pad[0] // 2 : x.size(2) - (pad[0] - pad[0] // 2), 
                pad[1] // 2 : x.size(3) - (pad[1] - pad[1] // 2)]
        return x             

    def forward(self, x):
        x_, pad = self.pad(x)

        feat1_1 = self.conv1_1(x_)
        feat2_1 = self.conv2_1(feat1_1)
        feat2_2 = self.conv2_2(feat2_1)
        feat2_3 = self.conv2_3(feat2_2)
        feat3_1 = self.conv3_1(feat2_3)
        feat3_2 = self.conv3_2(feat3_1)

        resblocks_out = self.resblocks(feat3_2) + feat3_2 # long range residual term

        feat4_1 = self.conv4_1(resblocks_out)
        feat4_1 = feat4_1 + feat3_2
        feat4_1 = self.upsample(feat4_1)
        feat5_1 = self.conv5_1(feat4_1)
        feat5_1 = feat5_1 + feat2_3
        feat5_1 = self.relu5_1(feat5_1)
        feat5_2 = self.conv5_2(feat5_1)
        feat5_3 = self.conv5_3(feat5_2)
        feat5_3 = self.upsample(feat5_3)
        feat6_1 = self.conv6_1(feat5_3)
        feat6_1 = feat6_1 + feat1_1
        feat6_1 = self.relu6_1(feat6_1)
        feat6_2 = self.conv6_2(feat6_1)
        feat6_3 = self.conv6_3(feat6_2)

        out = self.unpad(feat6_3, pad)

        T, A, J = torch.split(out, 3, dim=1)

        T = self.sigmoid(T)
        A = self.sigmoid(A)
        J = J + x

        return T, A, J




        
class EnhancementAConst(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_downs = 2
        self.downscale = 2**self.num_downs

        self.conv1_1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(32),)
        self.conv2_1 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1, padding_mode='reflect'), nn.InstanceNorm2d(64), nn.LeakyReLU(),)
        self.conv2_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(64), nn.LeakyReLU(),)   
        self.conv2_3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(64),)
        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, padding_mode='reflect'), nn.InstanceNorm2d(128), nn.LeakyReLU(),)
        self.conv3_2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(128),)

        class ResBlocks(nn.Module):
            def __init__(self, channels):
                super().__init__()
                conv1 = nn.Sequential(
                                    nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),
                                    nn.InstanceNorm2d(channels),
                                    nn.LeakyReLU()
                            )
                conv2 = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect')
                self.conv = nn.Sequential(conv1, conv2)

            def forward(self, x):
                return x + self.conv(x)

        resblocks = [ResBlocks(128) for _ in range(16)]
        self.resblocks = nn.Sequential(*resblocks)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv4_1 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(128))
        self.conv5_1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(64))
        self.relu5_1 = nn.LeakyReLU()
        self.conv5_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(64), nn.LeakyReLU())
        self.conv5_3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(64), nn.LeakyReLU())
        self.conv6_1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(32))
        self.relu6_1 = nn.LeakyReLU()
        self.conv6_2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1, padding_mode='reflect'), nn.InstanceNorm2d(32), nn.LeakyReLU())
        self.conv6_3 = nn.Sequential(nn.Conv2d(32, 6, 3, 1, 1, padding_mode='reflect'))

        self.sigmoid = nn.Sigmoid() # for T and A

        self.conv_a = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1),
                                    nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Conv2d(128, 3, 1, 1))

    def pad(self, x):
        x_padx = ((self.downscale) - (x.size(2)%(self.downscale))) % (self.downscale)
        x_pady = ((self.downscale) - (x.size(3)%(self.downscale))) % (self.downscale)
        x = F.pad(x, [x_pady//2, (x_pady - x_pady//2), 
                    x_padx // 2, (x_padx - x_padx//2)], mode='reflect')

        return x, (x_padx, x_pady)

    def unpad(self, x, pad):

        x = x[:, :, pad[0] // 2 : x.size(2) - (pad[0] - pad[0] // 2), 
                pad[1] // 2 : x.size(3) - (pad[1] - pad[1] // 2)]
        return x             

    def forward(self, x):
        x_, pad = self.pad(x)

        feat1_1 = self.conv1_1(x_)
        feat2_1 = self.conv2_1(feat1_1)
        feat2_2 = self.conv2_2(feat2_1)
        feat2_3 = self.conv2_3(feat2_2)
        feat3_1 = self.conv3_1(feat2_3)
        feat3_2 = self.conv3_2(feat3_1)

        resblocks_out = self.resblocks(feat3_2) + feat3_2 # long range residual term

        A = self.conv_a(resblocks_out)

        feat4_1 = self.conv4_1(resblocks_out)
        feat4_1 = feat4_1 + feat3_2
        feat4_1 = self.upsample(feat4_1)
        feat5_1 = self.conv5_1(feat4_1)
        feat5_1 = feat5_1 + feat2_3
        feat5_1 = self.relu5_1(feat5_1)
        feat5_2 = self.conv5_2(feat5_1)
        feat5_3 = self.conv5_3(feat5_2)
        feat5_3 = self.upsample(feat5_3)
        feat6_1 = self.conv6_1(feat5_3)
        feat6_1 = feat6_1 + feat1_1
        feat6_1 = self.relu6_1(feat6_1)
        feat6_2 = self.conv6_2(feat6_1)
        feat6_3 = self.conv6_3(feat6_2)

        out = self.unpad(feat6_3, pad)

        T, J = torch.split(out, 3, dim=1)

        T = self.sigmoid(T)
        A = self.sigmoid(A)
        J = J + x

        return T, A, J
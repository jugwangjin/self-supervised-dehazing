# https://github.com/Reyhanehne/CVF-SID_PyTorch/blob/main/src/model/model.py

if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.utils import BoxBlur



class UNetBL_NB(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_downs=5
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        class ResBlocks(nn.Module):
            def __init__(self, channels):
                super().__init__()
                conv1 = nn.Sequential(
                                    nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='replicate'),
                                    nn.PReLU(channels)
                            )
                conv2 = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='replicate')
                self.conv = nn.Sequential(conv1, conv2)

            def forward(self, x):
                return x + self.conv(x)

        self.down = nn.Sequential(nn.ReplicationPad2d(1), nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

        self.l1_encoder = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, padding_mode='replicate'),
                                        nn.PReLU(32),
                                        )

        self.l2_encoder = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(64),
                                        )

        self.l3_encoder = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(128),
                                        )

        self.l4_encoder = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(256),
                                        )

        self.l5_encoder = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(256),
                                        )

        self.l4_skip = nn.Conv2d(256, 128, 1)
        self.l3_skip = nn.Conv2d(128, 64, 1)
        self.l2_skip = nn.Conv2d(64, 32, 1)
        self.l1_skip = nn.Conv2d(32, 16, 1)

        self.l5_up = nn.Sequential(self.up, nn.Conv2d(256, 256, 3, 1, 1))
        self.l4_up = nn.Sequential(self.up, nn.Conv2d(256, 256, 3, 1, 1))
        self.l3_up = nn.Sequential(self.up, nn.Conv2d(128, 128, 3, 1, 1))
        self.l2_up = nn.Sequential(self.up, nn.Conv2d(64, 64, 3, 1, 1))
        self.l1_up = nn.Sequential(self.up, nn.Conv2d(32, 32, 3, 1, 1))

        self.l5_decoder = nn.Sequential(*[ResBlocks(256) for _ in range(2)])
        
        self.l4_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(384) for _ in range(2)]),
                                        nn.Conv2d(384, 256, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(256),
                                        )

        self.l3_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(320) for _ in range(2)]),
                                        nn.Conv2d(320, 128, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(128),
                                        )
        
        self.l2_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(160) for _ in range(2)]),
                                        nn.Conv2d(160, 64, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(64),
                                        )
        
        self.l1_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(80) for _ in range(2)]),
                                        nn.Conv2d(80, 32, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(32),
                                        )

        self.out_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(35) for _ in range(2)]),
                                        nn.Conv2d(35, 6, 3, 1, 1, padding_mode='replicate'))
        

        self.conv_A = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                                    nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Conv2d(128, 3, 1, 1),
                                    nn.Sigmoid())

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.1)
               nn.init.constant(m.bias, 0)


    def forward(self, x):
        x, pad = self.pad(x)

        enc_l1 = self.l1_encoder(x)
        

        enc_l2 = self.l2_encoder(self.down(enc_l1))
        

        enc_l3 = self.l3_encoder(self.down(enc_l2))
        

        enc_l4 = self.l4_encoder(self.down(enc_l3))
        

        enc_l5 = self.l5_encoder(self.down(enc_l4))


        enc_l5 = self.l5_decoder(enc_l5)

        dec_l4_inp = torch.cat((self.l4_skip(enc_l4), self.l5_up(enc_l5)), dim=1)
        
        dec_l4 = self.l4_decoder(dec_l4_inp)
        

        dec_l3_inp = torch.cat((self.l3_skip(enc_l3), self.l4_up(dec_l4)), dim=1)
        
        dec_l3 = self.l3_decoder(dec_l3_inp)


        dec_l2_inp = torch.cat((self.l2_skip(enc_l2), self.l3_up(dec_l3)), dim=1)
        
        dec_l2 = self.l2_decoder(dec_l2_inp)


        dec_l1_inp = torch.cat((self.l1_skip(enc_l1), self.l2_up(dec_l2)), dim=1)
        
        dec_l1 = self.l1_decoder(dec_l1_inp)

        out = self.l1_up(dec_l1)

        out = self.out_decoder(torch.cat((x, out), dim=1))

        T, J = torch.split(out, 3, dim=1)
        J = J + x
        J = self.unpad(J, pad)
        T = self.unpad(T, pad)
        T = self.sigmoid(T)
        A = self.conv_A(enc_l5)

        return T, A, J


    def pad(self, x):
        x_padx = ((2**self.num_downs) - (x.size(2)%(2**self.num_downs))) % (2**self.num_downs)
        x_pady = ((2**self.num_downs) - (x.size(3)%(2**self.num_downs))) % (2**self.num_downs)
        x = F.pad(x, [x_pady//2, (x_pady - x_pady//2), 
                    x_padx // 2, (x_padx - x_padx//2)], mode='replicate')

        return x, (x_padx, x_pady)

    def unpad(self, x, pad):

        x = x[:, :, pad[0] // 2 : x.size(2) - (pad[0] - pad[0] // 2), 
                pad[1] // 2 : x.size(3) - (pad[1] - pad[1] // 2)]
        return x     







        




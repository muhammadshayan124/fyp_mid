
import torch
import torch.nn as nn
from models.blocks import ResidualBlock

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])
        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output = nn.Conv2d(256, 3, 9, padding=4)

    def forward(self, x):
        x = self.init(x)
        res = self.res_blocks(x)
        x = self.mid_conv(res)
        x = x + res
        x = self.upsample(x)
        return torch.tanh(self.output(x))

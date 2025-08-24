
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_f, out_f, stride, bn=True):
            layers = [nn.Conv2d(in_f, out_f, 3, stride, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(3, 64, 1, False),
            *block(64, 64, 2),
            *block(64, 128, 1),
            *block(128, 128, 2),
            *block(128, 256, 1),
            *block(256, 256, 2),
            *block(256, 512, 1),
            *block(512, 512, 2),
            nn.Flatten(),
            nn.Linear(512 * 16 * 16, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

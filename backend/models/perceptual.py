
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from config import DEVICE

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(DEVICE)

    def forward(self, sr, hr):
        return F.mse_loss(self.vgg(sr), self.vgg(hr))

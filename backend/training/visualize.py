
import numpy as np
import torch
from matplotlib import pyplot as plt
from training.metrics import calculate_psnr, calculate_ssim
from config import DEVICE

def show_sample(generator, dataloader):
    generator.eval()
    lr, hr = next(iter(dataloader))
    lr, hr = lr.to(DEVICE), hr.to(DEVICE)
    with torch.no_grad():
        sr = generator(lr)
    lr, sr, hr = lr[0].cpu(), sr[0].cpu(), hr[0].cpu()

    psnr_val = calculate_psnr(sr, hr)
    ssim_val = calculate_ssim(sr, hr)

    def imshow(img, title, idx):
        img = (img + 1) / 2
        img = np.clip(img.numpy().transpose(1, 2, 0), 0, 1)
        plt.subplot(1, 3, idx)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    plt.figure(figsize=(12, 4))
    imshow(lr, "Low Res", 1)
    imshow(sr, f"Super Res\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}", 2)
    imshow(hr, "Ground Truth", 3)
    plt.show()
    return psnr_val, ssim_val

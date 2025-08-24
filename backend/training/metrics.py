
import torch
import numpy as np
import cv2
from math import log10
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    img1 = (img1 + 1) * 127.5
    img2 = (img2 + 1) * 127.5
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * log10((255.0 ** 2) / mse)

def calculate_ssim(img1, img2):
    img1 = (img1 + 1) * 127.5
    img2 = (img2 + 1) * 127.5
    img1_np = img1.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    img2_np = img2.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
    return ssim(img1_gray, img2_gray, data_range=255)

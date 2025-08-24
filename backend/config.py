
import os
import torch
import random
import numpy as np

# =========================
# CONFIG
# =========================
HR_SIZE = 256          # training crop size for HR
SCALE = 4
LR_SIZE = HR_SIZE // SCALE
BATCH_SIZE = 16
EPOCHS = 100
GAUSSIAN_KERNEL = (5, 5)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 20           # Early stopping patience
DATASET_PATH = "/content/drive/MyDrive/satellite_images"
OUTPUT_DIR = "/content/drive/MyDrive/srgan_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

print("🚀 Using device:", DEVICE)
if DEVICE == "cuda":
    try:
        print("🧠 GPU Name:", torch.cuda.get_device_name(0))
    except Exception:
        pass

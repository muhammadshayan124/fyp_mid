
import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from config import HR_SIZE, LR_SIZE, GAUSSIAN_KERNEL

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

class FolderImageDataset(Dataset):
    def __init__(self, root_folder):
        self.paths = []
        for root, _, files in os.walk(root_folder):
            for f in files:
                if f.lower().endswith(IMG_EXTS):
                    self.paths.append(os.path.join(root, f))
        print(f"📸 Found {len(self.paths)} images in {root_folder}")
        if len(self.paths) == 0:
            raise ValueError("No images found in dataset folder!")

    def __len__(self):
        return len(self.paths)

    def _safe_read_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        img = self._safe_read_rgb(self.paths[idx])

        if random.random() > 0.5:
            img = cv2.flip(img, 1)

        h, w, _ = img.shape
        if h >= HR_SIZE and w >= HR_SIZE:
            top = random.randint(0, h - HR_SIZE)
            left = random.randint(0, w - HR_SIZE)
            hr = img[top:top + HR_SIZE, left:left + HR_SIZE]
        else:
            hr = cv2.resize(img, (HR_SIZE, HR_SIZE), interpolation=cv2.INTER_CUBIC)

        lr = cv2.GaussianBlur(hr, GAUSSIAN_KERNEL, 0)
        lr = cv2.resize(lr, (LR_SIZE, LR_SIZE), interpolation=cv2.INTER_AREA)

        hr = ((hr / 127.5) - 1).astype(np.float32)
        lr = ((lr / 127.5) - 1).astype(np.float32)

        hr = torch.from_numpy(hr.transpose(2, 0, 1))
        lr = torch.from_numpy(lr.transpose(2, 0, 1))

        return lr, hr

"""PyTorch Dataset for SAR despeckling with synthetic speckle injection."""
import torch
from torch.utils.data import Dataset
import numpy as np


def add_speckle(img, L=1):
    """Add Gamma-distributed multiplicative speckle noise.
    
    Args:
        img: Clean amplitude image
        L: Number of looks (L=1 is fully developed speckle)
    """
    noise = np.random.gamma(L, 1.0 / L, img.shape).astype(np.float32)
    return img * noise


class SARDataset(Dataset):
    """SAR dataset with scene-conditioned embeddings.
    
    Args:
        patches: Clean SAR amplitude patches (N, H, W), float32, [0,1]
        embeddings: CLIP scene embeddings (N, 512)
        scale: Downscale factor for super-resolution
        L: Number of looks for synthetic speckle
    """
    def __init__(self, patches, embeddings, scale=2, L=1):
        self.patches = patches
        self.embeddings = embeddings
        self.scale = scale
        self.L = L

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        clean = self.patches[idx]
        h, w = clean.shape
        target = torch.from_numpy(clean).unsqueeze(0)

        noisy = add_speckle(clean, self.L)
        lr_h, lr_w = h // self.scale, w // self.scale
        noisy_lr = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)
        noisy_lr = torch.nn.functional.interpolate(
            noisy_lr, size=(lr_h, lr_w), mode='bilinear', align_corners=False)
        noisy_lr = noisy_lr.squeeze(0)

        embed = torch.from_numpy(self.embeddings[idx]).float()
        return noisy_lr, target, embed

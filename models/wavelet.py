"""Differentiable 2D Discrete Wavelet Transform layers for SAR frequency decomposition."""
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


class DWT2d(nn.Module):
    """Decomposes input into 4 wavelet subbands (LL, LH, HL, HH).
    
    The LL subband captures coarse scene structure while LH/HL/HH
    subbands isolate high-frequency content where speckle dominates.
    This gives the network explicit frequency separation before learning.
    """
    def __init__(self, wave='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode='zero')

    def forward(self, x):
        ll, (detail,) = self.dwt(x)
        lh, hl, hh = detail[:, :, 0], detail[:, :, 1], detail[:, :, 2]
        return torch.cat([ll, lh, hl, hh], dim=1)


class IDWT2d(nn.Module):
    """Inverse DWT — reconstructs image from wavelet subbands."""
    def __init__(self, wave='haar'):
        super().__init__()
        self.idwt = DWTInverse(wave=wave, mode='zero')

    def forward(self, ll, lh, hl, hh):
        detail = torch.stack([lh, hl, hh], dim=2)
        return self.idwt((ll, [detail]))

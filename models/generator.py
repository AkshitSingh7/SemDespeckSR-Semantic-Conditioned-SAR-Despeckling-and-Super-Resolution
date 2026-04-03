"""Scene-conditioned generator with wavelet preprocessing and AdaIN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavelet import DWT2d
from .rrdb import RRDB


class AdaIN(nn.Module):
    """Adaptive Instance Normalization — injects scene embedding into features.
    
    Enables the network to adapt its despeckling behavior based on scene type
    (e.g., smooth aggressively for water, preserve detail for urban areas).
    """
    def __init__(self, nf, embed_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(nf, affine=False)
        self.fc = nn.Linear(embed_dim, nf * 2)

    def forward(self, x, embed):
        h = self.fc(embed)
        gamma, beta = h.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return self.norm(x) * (1 + gamma) + beta


class ConditionedRRDB(nn.Module):
    """RRDB block with AdaIN scene conditioning."""
    def __init__(self, nf=64, gc=32, embed_dim=512):
        super().__init__()
        self.rrdb = RRDB(nf, gc)
        self.adain = AdaIN(nf, embed_dim)

    def forward(self, x, embed):
        out = self.rrdb(x)
        out = self.adain(out, embed)
        return out * 0.2 + x


class SARWaveSRNet(nn.Module):
    """Joint SAR despeckling and super-resolution generator.
    
    Pipeline: Input → DWT → 4-channel subbands → Conditioned RRDB trunk → Upsample → Output
    
    Args:
        in_nc: Input channels (1 for SAR amplitude)
        out_nc: Output channels
        nf: Number of feature channels
        nb: Number of RRDB blocks
        gc: Growth channels in dense blocks
        scale: Super-resolution scale factor (2 or 4)
        wave: Wavelet type ('haar' or 'db4')
        embed_dim: Scene embedding dimension (512 for CLIP)
    """
    def __init__(self, in_nc=1, out_nc=1, nf=64, nb=16, gc=32, scale=2,
                 wave='haar', embed_dim=512):
        super().__init__()
        self.scale = scale
        self.dwt = DWT2d(wave=wave)
        self.conv_first = nn.Conv2d(in_nc * 4, nf, 3, 1, 1)

        self.trunk = nn.ModuleList([
            ConditionedRRDB(nf, gc, embed_dim) for _ in range(nb)
        ])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        total_up = scale * 2  # DWT halves spatial dims
        self.upsamples = nn.ModuleList()
        while total_up > 1:
            self.upsamples.append(nn.Conv2d(nf, nf * 4, 3, 1, 1))
            total_up //= 2

        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, scene_embed):
        x_wt = self.dwt(x)
        feat = self.lrelu(self.conv_first(x_wt))

        trunk_input = feat
        for block in self.trunk:
            feat = block(feat, scene_embed)

        feat = self.trunk_conv(feat) + trunk_input

        for up_conv in self.upsamples:
            feat = self.lrelu(F.pixel_shuffle(up_conv(feat), 2))

        return self.conv_last(self.lrelu(self.conv_hr(feat)))

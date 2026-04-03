"""PatchGAN discriminator for adversarial training."""
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_nc=1, nf=64):
        super().__init__()

        def block(in_c, out_c, stride=2, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 3, stride, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_nc, nf, stride=2, bn=False),
            *block(nf, nf * 2),
            *block(nf * 2, nf * 4),
            *block(nf * 4, nf * 8, stride=1),
            nn.Conv2d(nf * 8, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.model(x)

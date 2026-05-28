"""U-Net discriminator with spectral normalization, after Real-ESRGAN.

Single-channel (Y) input, per-pixel real/fake logits. The per-pixel output gives
dense feedback that helps the generator restore local high-frequency detail
without hallucinating large-scale structure. Spectral norm stabilizes the GAN.

Used only at training (Stage 2 fine-tune); never deployed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class UNetDiscriminatorSN(nn.Module):
    def __init__(self, in_ch: int = 1, num_feat: int = 32, skip: bool = True) -> None:
        super().__init__()
        self.skip = skip
        sn = spectral_norm
        self.conv0 = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.conv1 = sn(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = sn(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = sn(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = sn(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = sn(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = sn(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = sn(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = sn(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), 0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), 0.2, inplace=True)

        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), 0.2, inplace=True)
        if self.skip:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), 0.2, inplace=True)
        if self.skip:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), 0.2, inplace=True)
        if self.skip:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), 0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), 0.2, inplace=True)
        out = self.conv9(out)
        return out

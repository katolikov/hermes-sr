"""EDSR-baseline ×2 teacher for knowledge distillation.

Architecture matches the official EDSR-baseline (16 ResBlocks, 64 features,
rgb_range=255, mean-shift) so the pretrained weights from
huggingface eugenesiow/edsr-base load directly. ~38 dB on Set5 ×2.

forward() returns (sr_rgb, body_feat):
  sr_rgb    : (N, 3, sH, sW) in [0, 1]
  body_feat : (N, 64, H, W) LR-resolution feature for feature distillation.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, n_feats: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class EDSRTeacher(nn.Module):
    def __init__(self, n_resblocks: int = 16, n_feats: int = 64, scale: int = 2, rgb_range: int = 255) -> None:
        super().__init__()
        self.rgb_range = rgb_range
        # DIV2K RGB mean (the official EDSR mean-shift)
        mean = torch.tensor([0.4488, 0.4371, 0.4040]) * rgb_range
        self.sub_mean = nn.Conv2d(3, 3, 1)
        self.add_mean = nn.Conv2d(3, 3, 1)
        self.sub_mean.weight.data.copy_(torch.eye(3).view(3, 3, 1, 1))
        self.sub_mean.bias.data.copy_(-mean)
        self.add_mean.weight.data.copy_(torch.eye(3).view(3, 3, 1, 1))
        self.add_mean.bias.data.copy_(mean)

        self.head = nn.Sequential(nn.Conv2d(3, n_feats, 3, padding=1))
        body = [_ResBlock(n_feats) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            nn.Sequential(nn.Conv2d(n_feats, scale * scale * n_feats, 3, padding=1), nn.PixelShuffle(scale)),
            nn.Conv2d(n_feats, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, 3, H, W) in [0, 1]."""
        x = x * self.rgb_range
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res = res + x
        feat = res
        out = self.tail(res)
        out = self.add_mean(out)
        return (out / self.rgb_range).clamp(0.0, 1.0), feat


def load_edsr_teacher(weights_path: str | Path, device: torch.device) -> EDSRTeacher:
    """Load the HF super-image EDSR-base x2 checkpoint into EDSRTeacher."""
    model = EDSRTeacher(n_resblocks=16, n_feats=64, scale=2, rgb_range=255)
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    # strip the "module." prefix from the DataParallel-saved checkpoint
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"EDSR teacher missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"EDSR teacher unexpected keys: {unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)

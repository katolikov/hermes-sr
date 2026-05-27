"""Set5 / SIDD evaluation — Y-channel PSNR and SSIM with the standard s-pixel border crop.

For Mode A: bicubic-downsample HR images by `upscale`, super-resolve, compare on Y.
For Mode B: HR images are downsampled then noise-augmented; the network is fed
the (Y, sigma) two-channel input.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from hermes_sr.data.augment import (
    add_gaussian_noise,
    rgb_to_y,
    robust_mad_estimator,
)


def _pil_to_y(path: Path) -> torch.Tensor:
    import numpy as np

    with Image.open(path) as im:
        rgb = torch.from_numpy(np.asarray(im.convert("RGB")).transpose(2, 0, 1)).float() / 255.0
    return rgb_to_y(rgb)


def _psnr(pred: torch.Tensor, target: torch.Tensor, border: int) -> float:
    if border > 0:
        pred = pred[..., border:-border, border:-border]
        target = target[..., border:-border, border:-border]
    mse = ((pred - target) ** 2).mean().clamp(min=1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def _ssim(pred: torch.Tensor, target: torch.Tensor, border: int) -> float:
    # Small windowed SSIM implementation; sufficient for the MVP. Constants per
    # Wang et al., 2004 with L=1 (images already in [0,1]).
    if border > 0:
        pred = pred[..., border:-border, border:-border]
        target = target[..., border:-border, border:-border]
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    window = 11
    kernel = torch.ones(1, 1, window, window, device=pred.device, dtype=pred.dtype) / (window * window)
    pad = window // 2
    mu_x = F.conv2d(pred, kernel, padding=pad)
    mu_y = F.conv2d(target, kernel, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(pred * pred, kernel, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=pad) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad) - mu_xy
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float((num / den).mean())


@torch.no_grad()
def evaluate_set5(
    model: torch.nn.Module,
    set5_root: Optional[str],
    upscale: int,
    device: torch.device,
    mode: str = "A",
    noise_sigma: float = 0.0,
) -> tuple[float, float]:
    """Returns (PSNR, SSIM). If set5_root is missing, runs on a small synthetic set."""
    images: list[torch.Tensor] = []
    if set5_root is not None:
        root = Path(os.path.expanduser(set5_root))
        # Common layouts: <root>/Set5/HR/*.png, <root>/Set5/*.png, or <root>/*.png
        candidates: list[Path] = []
        for sub in [root / "Set5" / "HR", root / "Set5", root]:
            if sub.exists():
                candidates = sorted([p for p in sub.glob("*.png") if "LR" not in p.name.upper()])
                if candidates:
                    break
        for p in candidates[:5]:
            images.append(_pil_to_y(p))
    if not images:
        # Synthetic five-image set: deterministic per-index random textures
        for i in range(5):
            torch.manual_seed(1729 + i)
            rgb = torch.rand(3, 256, 256)
            rgb = F.avg_pool2d(rgb[None], 3, stride=1, padding=1).squeeze(0)
            images.append(rgb_to_y(rgb))

    model.eval()
    psnrs: list[float] = []
    ssims: list[float] = []
    for y_hr in images:
        y_hr = y_hr.to(device)
        # Pad to a multiple of upscale to keep dimensions clean
        _, H, W = y_hr.shape
        H -= H % upscale
        W -= W % upscale
        y_hr = y_hr[:, :H, :W]
        # Bicubic downsample then upsample with the network
        y_lr = F.interpolate(y_hr[None], scale_factor=1.0 / upscale, mode="bicubic", align_corners=False).squeeze(0).clamp(0.0, 1.0)
        if mode == "B":
            if noise_sigma > 0.0:
                y_lr = add_gaussian_noise(y_lr, noise_sigma).clamp(0.0, 1.0)
            sigma_est = robust_mad_estimator(y_lr[None]).squeeze(0)
            _, h, w = y_lr.shape
            sigma_chan = (sigma_est.view(1, 1, 1).expand(1, h, w)) / 255.0
            y_in = torch.cat([y_lr, sigma_chan], dim=0)
        else:
            y_in = y_lr

        y_pred, _ = model(y_in[None], h_prev=None, y_prev=None)
        y_pred = y_pred.clamp(0.0, 1.0).squeeze(0)
        psnrs.append(_psnr(y_pred[None], y_hr[None], border=upscale))
        ssims.append(_ssim(y_pred[None], y_hr[None], border=upscale))

    model.train()
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)

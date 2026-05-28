"""Sensor-buffer video super-resolution dataset.

Synthesizes temporal Y-plane pairs as they'd come off a phone camera YUV buffer:
  HR image -> two motion-shifted crops (prev, curr) -> per-crop sensor
  degradation (optical blur + downsample + Poisson-Gaussian noise) -> noisy LR Y.

The two frames of a pair share noise PARAMETERS (same ISO within a short clip)
but get INDEPENDENT noise realizations (sensor noise is temporally uncorrelated).
Each LR frame is returned as a 2-channel tensor: [noisy Y, broadcast MAD-sigma],
matching the Mode-B noise-aware input. The HR target is the clean curr crop.

HR sources: DIV2K + Flickr2K (falls back to synthetic textures if missing).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from hermes_sr.data.augment import make_temporal_pair, rgb_to_y, robust_mad_estimator
from hermes_sr.data.degrade import degrade_y, sample_noise_params
from hermes_sr.data.div2k import _pil_to_array


def _gather_hr(roots: list[str]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        r = Path(os.path.expanduser(root))
        for sub in ("DIV2K_train_HR", "Flickr2K/Flickr2K_HR", "Flickr2K_HR"):
            d = r / sub
            if d.exists():
                paths.extend(sorted(d.glob("*.png")))
    return paths


class SensorVideoDataset(Dataset):
    def __init__(
        self,
        roots: list[str],
        patch_hr: int = 96,
        upscale: int = 2,
        max_shift_lr: int = 4,
        blur_max: float = 0.8,
    ) -> None:
        self.patch_hr = patch_hr
        self.upscale = upscale
        self.max_shift_lr = max_shift_lr
        self.blur_max = blur_max
        self.images = _gather_hr(roots)

    def __len__(self) -> int:
        return max(len(self.images), 800)

    def _load_y(self, idx: int) -> torch.Tensor:
        if not self.images:
            torch.manual_seed(idx * 2654435761 % (2**31))
            import torch.nn.functional as F
            base = F.avg_pool2d(torch.rand(3, 256, 256)[None], 3, stride=1, padding=1).squeeze(0)
            return rgb_to_y(base)
        path = self.images[idx % len(self.images)]
        with Image.open(path) as im:
            arr = torch.from_numpy(_pil_to_array(im.convert("RGB"))).float() / 255.0
        return rgb_to_y(arr)

    def __getitem__(self, idx: int) -> dict:
        y = self._load_y(idx)
        # clean motion-shifted crops (prev, curr) + known LR-pixel shift
        _lp, _lc, hr_prev, hr_curr, (dx_lr, dy_lr) = make_temporal_pair(
            y, patch_hr=self.patch_hr, upscale=self.upscale, max_shift_lr=self.max_shift_lr
        )
        # shared sensor params for the clip, independent noise per frame
        shot, read = sample_noise_params()
        blur = float(torch.empty(()).uniform_(0.0, self.blur_max))
        lr_prev = degrade_y(hr_prev, self.upscale, shot, read, blur_sigma=blur)
        lr_curr = degrade_y(hr_curr, self.upscale, shot, read, blur_sigma=blur)

        sig_prev = robust_mad_estimator(lr_prev[None]).squeeze(0)
        sig_curr = robust_mad_estimator(lr_curr[None]).squeeze(0)
        _, h, w = lr_prev.shape
        chan_prev = (sig_prev.view(1, 1, 1).expand(1, h, w)) / 255.0
        chan_curr = (sig_curr.view(1, 1, 1).expand(1, h, w)) / 255.0

        return {
            "lr_prev": torch.cat([lr_prev, chan_prev], dim=0),
            "lr_curr": torch.cat([lr_curr, chan_curr], dim=0),
            "hr_curr": hr_curr,
            "shift_lr": torch.tensor([dx_lr, dy_lr], dtype=torch.float32),
        }

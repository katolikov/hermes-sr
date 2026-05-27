"""Mode B noise-aware training pairs.

The spec calls for the SIDD-medium subset but uses a simplified Gaussian
degradation in the MVP: clean Y patches are augmented with synthetic noise at
sigma ~ U(0, 50) on a 0–255 scale, then the dataloader concatenates the MAD
sigma estimate as a second channel.

If the SIDD root is missing, falls back to DIV2K HR images (also synthetic if
those are missing) so the loop can be smoke-tested without real data.

TODO(SIDD-real): swap the synthetic Gaussian path for actual SIDD noisy/clean
pairs and the Zurich RAW-to-RGB integration that the design doc mentions but
which is out of scope here.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from hermes_sr.data.augment import (
    add_gaussian_noise,
    make_temporal_pair,
    rgb_to_y,
    robust_mad_estimator,
)
from hermes_sr.data.div2k import _pil_to_array


class SIDDTrainset(Dataset):
    def __init__(
        self,
        root: str,
        patch_hr: int = 96,
        upscale: int = 3,
        max_shift_lr: int = 4,
        sigma_range: tuple[float, float] = (0.0, 50.0),
        fallback_root: str | None = None,
    ) -> None:
        self.root = Path(os.path.expanduser(root))
        self.patch_hr = patch_hr
        self.upscale = upscale
        self.max_shift_lr = max_shift_lr
        self.sigma_range = sigma_range

        clean_dir = self.root / "SIDD_Medium_Srgb" / "Data"
        if clean_dir.exists():
            # SIDD layout: <scene_id>/<*_GT_*.PNG> (ground-truth clean) and noisy variants
            self.images = sorted(clean_dir.rglob("*_GT_*.PNG"))
        else:
            self.images = []

        if fallback_root is not None:
            fb = Path(os.path.expanduser(fallback_root)) / "DIV2K_train_HR"
            if fb.exists():
                self.fallback_images = sorted(fb.glob("*.png"))
            else:
                self.fallback_images = []
        else:
            self.fallback_images = []

    def __len__(self) -> int:
        if self.images:
            return len(self.images)
        if self.fallback_images:
            return len(self.fallback_images)
        return 800

    def _load_image(self, idx: int) -> torch.Tensor:
        pool = self.images or self.fallback_images
        if not pool:
            torch.manual_seed(idx * 2246822519 % (2**31))
            base = torch.rand(3, 256, 256)
            base = F.avg_pool2d(base[None], 3, stride=1, padding=1).squeeze(0)
            return base
        path = pool[idx % len(pool)]
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = torch.from_numpy(_pil_to_array(im)).float() / 255.0
        return arr

    def __getitem__(self, idx: int) -> dict:
        rgb = self._load_image(idx)
        y = rgb_to_y(rgb)
        lr_prev_clean, lr_curr_clean, _hr_prev, hr_curr, (dx_lr, dy_lr) = make_temporal_pair(
            y, patch_hr=self.patch_hr, upscale=self.upscale, max_shift_lr=self.max_shift_lr
        )
        # Synthetic noise: a different sigma per pair gives the model a wide
        # input distribution to learn the noise channel against.
        sigma = float(torch.empty(()).uniform_(*self.sigma_range))
        lr_prev_noisy = add_gaussian_noise(lr_prev_clean, sigma).clamp(0.0, 1.0)
        lr_curr_noisy = add_gaussian_noise(lr_curr_clean, sigma).clamp(0.0, 1.0)

        sigma_prev = robust_mad_estimator(lr_prev_noisy[None]).squeeze(0)
        sigma_curr = robust_mad_estimator(lr_curr_noisy[None]).squeeze(0)

        # Broadcast sigma to (1, H, W) and concat as channel 1
        _, h, w = lr_prev_noisy.shape
        sigma_chan_prev = sigma_prev.view(1, 1, 1).expand(1, h, w) / 255.0
        sigma_chan_curr = sigma_curr.view(1, 1, 1).expand(1, h, w) / 255.0

        lr_prev_in = torch.cat([lr_prev_noisy, sigma_chan_prev], dim=0)
        lr_curr_in = torch.cat([lr_curr_noisy, sigma_chan_curr], dim=0)

        return {
            "lr_prev": lr_prev_in,
            "lr_curr": lr_curr_in,
            "hr_curr": hr_curr,
            "shift_lr": torch.tensor([dx_lr, dy_lr], dtype=torch.float32),
            "sigma_true": torch.tensor(sigma, dtype=torch.float32),
        }

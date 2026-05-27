"""DIV2K training dataset.

Looks for a `<root>/DIV2K_train_HR/*.png` directory. On-the-fly downsamples HR
images to LR (×2 for Mode A) using bicubic interpolation. Returns Y-channel
temporal pairs (prev, curr) for the recurrent path.

If the root path does not exist, a synthetic fallback dataset is returned with
random images — useful for smoke-testing the training loop without downloading
several gigabytes of real data.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from hermes_sr.data.augment import rgb_to_y, make_temporal_pair


class DIV2KTrainset(Dataset):
    def __init__(
        self,
        root: str,
        patch_hr: int = 96,
        upscale: int = 2,
        max_shift_lr: int = 4,
    ) -> None:
        self.root = Path(os.path.expanduser(root))
        self.patch_hr = patch_hr
        self.upscale = upscale
        self.max_shift_lr = max_shift_lr

        hr_dir = self.root / "DIV2K_train_HR"
        if hr_dir.exists():
            self.images = sorted(hr_dir.glob("*.png"))
        else:
            self.images = []

    def __len__(self) -> int:
        # 800 real images cycle if available; otherwise emit a synthetic dataset
        # of fixed length so the training loop can run.
        return max(len(self.images), 800)

    def _load_image(self, idx: int) -> torch.Tensor:
        if not self.images:
            # Synthetic fallback: deterministic per-index random image with some
            # smooth structure so SR has a non-trivial signal to learn.
            torch.manual_seed(idx * 2654435761 % (2**31))
            base = torch.rand(3, 256, 256)
            base = F.avg_pool2d(base[None], 3, stride=1, padding=1).squeeze(0)
            return base
        path = self.images[idx % len(self.images)]
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = torch.from_numpy(_pil_to_array(im)).float() / 255.0
        return arr  # (3, H, W)

    def __getitem__(self, idx: int) -> dict:
        rgb = self._load_image(idx)
        y = rgb_to_y(rgb)  # (1, H, W)
        lr_prev, lr_curr, _hr_prev, hr_curr, (dx_lr, dy_lr) = make_temporal_pair(
            y, patch_hr=self.patch_hr, upscale=self.upscale, max_shift_lr=self.max_shift_lr
        )
        return {
            "lr_prev": lr_prev,
            "lr_curr": lr_curr,
            "hr_curr": hr_curr,
            "shift_lr": torch.tensor([dx_lr, dy_lr], dtype=torch.float32),
        }


def _pil_to_array(im: Image.Image):
    import numpy as np

    return np.asarray(im).transpose(2, 0, 1)

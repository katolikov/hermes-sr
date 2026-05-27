"""Dataset-side helpers: RGB→Y, MAD noise-sigma estimate, synthetic temporal pairs."""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ITU-R BT.601 luma coefficients
_BT601 = torch.tensor([0.299, 0.587, 0.114])


def rgb_to_y(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (..., 3, H, W) in [0, 1]. Returns Y in [0, 1] with shape (..., 1, H, W)."""
    w = _BT601.to(device=rgb.device, dtype=rgb.dtype).view(3, 1, 1)
    return (rgb * w).sum(dim=-3, keepdim=True)


def add_gaussian_noise(y: torch.Tensor, sigma_per_255: float) -> torch.Tensor:
    """Add Gaussian noise to Y in [0, 1]. sigma is on a 0–255 scale."""
    sigma = sigma_per_255 / 255.0
    return y + torch.randn_like(y) * sigma


def robust_mad_estimator(y: torch.Tensor) -> torch.Tensor:
    """Estimate per-image noise sigma from a Laplacian-MAD on Y in [0, 1].

    y: (N, 1, H, W). Returns (N,) sigma estimates on a 0–255 scale.
    """
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=y.device,
        dtype=y.dtype,
    ).view(1, 1, 3, 3)
    response = F.conv2d(y, kernel, padding=1)
    flat = response.flatten(start_dim=1)
    median = flat.median(dim=1, keepdim=True).values
    mad = (flat - median).abs().median(dim=1).values
    # 1.4826 * MAD ≈ sigma of the Laplacian response; divide by sqrt(6) for the
    # noise multiplier of a discrete 3x3 Laplacian (||k||_2^2 = 6).
    sigma = mad * 1.4826 / (6.0 ** 0.5)
    return sigma * 255.0


def make_temporal_pair(
    hr_image: torch.Tensor,
    patch_hr: int,
    upscale: int,
    max_shift_lr: int = 4,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Crop two HR patches offset by a random integer LR-pixel shift, downsample to LR.

    hr_image: (1, H, W) Y-channel image in [0, 1].
    Returns (lr_prev, lr_curr, hr_prev, hr_curr, (dx_lr, dy_lr)).
    The shift is in LR pixels; the HR shift is dx_lr * upscale.
    """
    _, H, W = hr_image.shape
    margin_lr = max_shift_lr * upscale
    # Random shift in LR pixels — HR shift must keep both crops inside the image
    if rng is None:
        dx_lr = int(torch.randint(-max_shift_lr, max_shift_lr + 1, (1,)).item())
        dy_lr = int(torch.randint(-max_shift_lr, max_shift_lr + 1, (1,)).item())
    else:
        dx_lr = int(torch.randint(-max_shift_lr, max_shift_lr + 1, (1,), generator=rng).item())
        dy_lr = int(torch.randint(-max_shift_lr, max_shift_lr + 1, (1,), generator=rng).item())
    dx_hr = dx_lr * upscale
    dy_hr = dy_lr * upscale

    # Origin for prev crop (must leave room for both crops plus the shift)
    max_y = H - patch_hr - margin_lr * 2
    max_x = W - patch_hr - margin_lr * 2
    assert max_y > 0 and max_x > 0, "Image too small for the requested patch size and shift"
    y0 = int(torch.randint(margin_lr, margin_lr + max_y, (1,)).item())
    x0 = int(torch.randint(margin_lr, margin_lr + max_x, (1,)).item())

    hr_prev = hr_image[:, y0:y0 + patch_hr, x0:x0 + patch_hr]
    hr_curr = hr_image[:, y0 + dy_hr:y0 + dy_hr + patch_hr, x0 + dx_hr:x0 + dx_hr + patch_hr]

    lr_prev = F.interpolate(hr_prev[None], scale_factor=1.0 / upscale, mode="bicubic", align_corners=False).squeeze(0)
    lr_curr = F.interpolate(hr_curr[None], scale_factor=1.0 / upscale, mode="bicubic", align_corners=False).squeeze(0)
    lr_prev = lr_prev.clamp(0.0, 1.0)
    lr_curr = lr_curr.clamp(0.0, 1.0)

    return lr_prev, lr_curr, hr_prev, hr_curr, (dx_lr, dy_lr)

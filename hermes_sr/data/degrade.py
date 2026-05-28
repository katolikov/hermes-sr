"""Realistic sensor-buffer degradation for Y-plane video super-resolution.

Models the LR Y plane that comes out of a phone camera ISP buffer (YUV 4:2:0):
  HR Y  -> optional optical blur -> downsample -> Poisson-Gaussian sensor noise -> LR Y

No JPEG: the input is a raw-ish sensor buffer, not a compressed file. The noise
is the physically-correct Poisson (photon shot, signal-dependent) + Gaussian
(read) model, with parameters drawn over a wide range so one model covers
daylight (near-clean) through low-light (heavy noise).

All ops on float tensors in [0, 1].
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, radius: int) -> torch.Tensor:
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Isotropic Gaussian blur (optical PSF). y: (C, H, W)."""
    if sigma <= 0:
        return y
    radius = max(1, int(math.ceil(3 * sigma)))
    k1d = _gaussian_kernel1d(sigma, radius).to(y.device, y.dtype)
    c = y.shape[0]
    kx = k1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    y = F.conv2d(y[None], kx, padding=(0, radius), groups=c)
    y = F.conv2d(y, ky, padding=(radius, 0), groups=c).squeeze(0)
    return y


def poisson_gaussian_noise(
    y: torch.Tensor, shot_gain: float, read_sigma: float
) -> torch.Tensor:
    """Add Poisson (shot) + Gaussian (read) noise to y in [0, 1].

    shot_gain: larger -> stronger signal-dependent noise (effectively higher ISO).
               variance of the shot component at signal level s is shot_gain * s.
    read_sigma: standard deviation of the signal-independent read noise.
    """
    y = y.clamp(min=0.0)
    if shot_gain > 0:
        # chi = photons per unit signal; var(Poisson(y*chi)/chi) = y/chi = y*shot_gain
        chi = 1.0 / shot_gain
        noisy = torch.poisson(y * chi) / chi
    else:
        noisy = y.clone()
    if read_sigma > 0:
        noisy = noisy + torch.randn_like(y) * read_sigma
    return noisy.clamp(0.0, 1.0)


def sample_noise_params(
    rng: torch.Generator | None = None,
    shot_log_range: tuple[float, float] = (-10.0, -3.7),
    read_log_range: tuple[float, float] = (-8.0, -3.5),
) -> tuple[float, float]:
    """Draw (shot_gain, read_sigma) log-uniformly over a wide sensor range.

    Defaults give effective Y noise sigma roughly [~1/255 .. ~32/255], i.e.
    near-clean daylight through heavy low-light. Read and shot are correlated by
    sharing the draw so high-shot pairs with high-read (realistic high-ISO).
    """
    if rng is None:
        u = torch.rand(1).item()
    else:
        u = torch.rand(1, generator=rng).item()
    shot_gain = float(math.exp(shot_log_range[0] + u * (shot_log_range[1] - shot_log_range[0])))
    # read tracks shot with a little independent jitter
    v = 0.7 * u + 0.3 * (torch.rand(1).item() if rng is None else torch.rand(1, generator=rng).item())
    read_sigma = float(math.exp(read_log_range[0] + v * (read_log_range[1] - read_log_range[0])))
    return shot_gain, read_sigma


def degrade_y(
    hr_y: torch.Tensor,
    scale: int,
    shot_gain: float,
    read_sigma: float,
    blur_sigma: float = 0.0,
) -> torch.Tensor:
    """HR Y (1,H,W) -> LR Y (1,H/scale,W/scale) with optical blur + sensor noise.

    Order: blur -> bicubic downsample -> Poisson-Gaussian noise. (Noise last so
    it lands at the sensor resolution, as in a real buffer.)
    """
    y = hr_y
    if blur_sigma > 0:
        y = gaussian_blur(y, blur_sigma)
    y = F.interpolate(y[None], scale_factor=1.0 / scale, mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1)
    y = poisson_gaussian_noise(y, shot_gain, read_sigma)
    return y


def effective_sigma(y: torch.Tensor, shot_gain: float, read_sigma: float, n: int = 4096) -> float:
    """Empirical noise sigma at the mean signal level — for sanity/reporting."""
    s = float(y.mean())
    var = shot_gain * s + read_sigma ** 2
    return math.sqrt(max(var, 0.0))

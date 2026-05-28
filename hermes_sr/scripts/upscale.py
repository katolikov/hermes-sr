"""Run HERMES-SR on a real image.

Usage:
    python -m hermes_sr.scripts.upscale --in path/to/image.png --out out.png \\
        [--mode A|B] [--ckpt path/to/deploy.pt] [--simulate-lr] \\
        [--noise-sigma 25]

The model is Y-channel only; Cb/Cr are bicubic-upsampled separately and the
result is recomposed to RGB so the output is full-color.

If --simulate-lr is set, the script first bicubic-downsamples the input by the
mode's upscale factor (×2 for A, ×3 for B), then super-resolves — useful for
visually comparing model vs bicubic on a known HR image. Without that flag, the
input is treated as the LR image and super-resolved directly.

For Mode B, --noise-sigma adds synthetic Gaussian noise (on a 0–255 scale) to
the LR input before super-resolving — demonstrates the noise-aware path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hermes_sr.data.augment import add_gaussian_noise, robust_mad_estimator
from hermes_sr.model import HermesConfig, HermesSR


# BT.601 RGB ↔ YCbCr matrices (values in [0, 1], full-range)
_RGB2Y = torch.tensor([0.299, 0.587, 0.114])
_RGB2CB = torch.tensor([-0.168736, -0.331264, 0.5])
_RGB2CR = torch.tensor([0.5, -0.418688, -0.081312])


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """rgb in [0,1] of shape (3,H,W) → ycbcr in [0,1] of shape (3,H,W).
    Cb/Cr are centered at 0.5."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack([y, cb, cr], dim=0)


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """Inverse of rgb_to_ycbcr."""
    y, cb, cr = ycbcr[0], ycbcr[1], ycbcr[2] - 0.5
    cb = cb - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], dim=0).clamp(0.0, 1.0)


def _load_ckpt(ckpt_path: Path, mode: str) -> tuple[HermesSR, HermesConfig]:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = HermesConfig(**state["config"])
    if cfg.mode != mode:
        raise ValueError(f"checkpoint mode {cfg.mode!r} != requested {mode!r}")
    model = HermesSR(cfg, reparameterized=state.get("reparameterized", False))
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, cfg


def _save_png(arr: torch.Tensor, path: Path) -> None:
    arr = (arr.clamp(0.0, 1.0) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


def upscale(
    in_path: Path,
    out_path: Path,
    mode: str,
    ckpt_path: Path,
    simulate_lr: bool,
    noise_sigma: float,
    save_intermediates: bool,
) -> None:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model, cfg = _load_ckpt(ckpt_path, mode)
    model = model.to(device)
    s = cfg.upscale

    with Image.open(in_path) as im:
        rgb = torch.from_numpy(np.asarray(im.convert("RGB")).transpose(2, 0, 1).copy()).float() / 255.0
    rgb = rgb.to(device)

    if simulate_lr:
        # Treat input as HR, bicubic-downsample to LR, then super-resolve
        _, H, W = rgb.shape
        H -= H % s
        W -= W % s
        rgb_hr = rgb[:, :H, :W]
        rgb_lr = F.interpolate(rgb_hr[None], scale_factor=1.0 / s, mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1)
        if save_intermediates:
            _save_png(rgb_hr, out_path.with_name(out_path.stem + "_hr_reference" + out_path.suffix))
    else:
        rgb_lr = rgb
        rgb_hr = None

    if mode == "B" and noise_sigma > 0:
        # Add noise channel-wise on RGB, then re-derive Y
        rgb_lr = add_gaussian_noise(rgb_lr, noise_sigma).clamp(0.0, 1.0)

    ycbcr_lr = rgb_to_ycbcr(rgb_lr)
    y_lr = ycbcr_lr[:1]  # (1, H, W)
    cb_lr = ycbcr_lr[1:2]
    cr_lr = ycbcr_lr[2:3]

    # Save the LR input for reference
    if save_intermediates:
        _save_png(rgb_lr, out_path.with_name(out_path.stem + "_lr_input" + out_path.suffix))

    # Bicubic baseline (no model)
    rgb_bicubic = F.interpolate(rgb_lr[None], scale_factor=s, mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1)
    if save_intermediates:
        _save_png(rgb_bicubic, out_path.with_name(out_path.stem + "_bicubic" + out_path.suffix))

    # Mode B: prepare 2-channel input (Y + sigma)
    if mode == "B":
        sigma_est = robust_mad_estimator(y_lr[None]).squeeze(0)
        _, h, w = y_lr.shape
        sigma_chan = sigma_est.view(1, 1, 1).expand(1, h, w) / 255.0
        net_in = torch.cat([y_lr, sigma_chan], dim=0)
        print(f"[upscale] MAD-estimated sigma: {float(sigma_est):.2f} (on 0-255 scale)")
    else:
        net_in = y_lr

    with torch.no_grad():
        y_hr_pred, _ = model(net_in[None])
    y_hr_pred = y_hr_pred.clamp(0.0, 1.0).squeeze(0)  # (1, sH, sW)

    # Bicubic-upsample chroma (model only handles Y)
    cb_hr = F.interpolate(cb_lr[None], scale_factor=s, mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1)
    cr_hr = F.interpolate(cr_lr[None], scale_factor=s, mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1)

    # Recombine and convert back to RGB
    ycbcr_hr = torch.cat([y_hr_pred, cb_hr, cr_hr], dim=0)
    rgb_hr_pred = ycbcr_to_rgb(ycbcr_hr)
    _save_png(rgb_hr_pred, out_path)

    # Print PSNR if we have a ground-truth HR reference
    if rgb_hr is not None:
        def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
            mse = ((a - b) ** 2).mean().clamp(min=1e-12)
            return float(10.0 * torch.log10(1.0 / mse))

        rgb_hr = rgb_hr.to(device)
        psnr_bic = _psnr(rgb_bicubic, rgb_hr)
        psnr_mdl = _psnr(rgb_hr_pred, rgb_hr)
        print(f"[upscale] RGB PSNR vs HR — bicubic: {psnr_bic:.2f} dB | model: {psnr_mdl:.2f} dB | Δ: {psnr_mdl - psnr_bic:+.2f} dB")

    print(f"[upscale] wrote {out_path}")
    print(f"[upscale] input  {tuple(rgb_lr.shape[-2:])} -> output {tuple(rgb_hr_pred.shape[-2:])} (×{s})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--out", dest="out_path", required=True, type=Path)
    p.add_argument("--mode", choices=["A", "B"], default="A")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="deploy checkpoint path; defaults to checkpoints/hermes_<mode>_deploy.pt")
    p.add_argument("--simulate-lr", action="store_true",
                   help="treat input as HR, downsample to LR first, then super-resolve")
    p.add_argument("--noise-sigma", type=float, default=0.0,
                   help="for Mode B: add Gaussian noise at this sigma (0-255 scale) to LR")
    p.add_argument("--save-intermediates", action="store_true",
                   help="also write *_lr_input, *_bicubic, and (if simulate-lr) *_hr_reference siblings")
    args = p.parse_args()

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = Path("checkpoints") / f"hermes_{args.mode.lower()}_deploy.pt"

    upscale(
        in_path=args.in_path,
        out_path=args.out_path,
        mode=args.mode,
        ckpt_path=ckpt,
        simulate_lr=args.simulate_lr,
        noise_sigma=args.noise_sigma,
        save_intermediates=args.save_intermediates,
    )


if __name__ == "__main__":
    main()

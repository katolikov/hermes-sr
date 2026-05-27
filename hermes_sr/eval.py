"""Set5 / Urban100 / SIDD evaluation — Y-channel PSNR and SSIM with the standard
s-pixel border crop. Library entrypoint (`evaluate_set5`) is used by train.py at
each eval_every interval; CLI entrypoint (`python -m hermes_sr.eval ...`) is
used by the convergence-run shell scripts after training.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from hermes_sr.data.augment import (
    add_gaussian_noise,
    rgb_to_y,
    robust_mad_estimator,
)
from hermes_sr.model import HermesConfig, HermesSR


def _pil_to_y(path: Path) -> torch.Tensor:
    import numpy as np

    with Image.open(path) as im:
        rgb = (
            torch.from_numpy(np.asarray(im.convert("RGB")).transpose(2, 0, 1).copy()).float() / 255.0
        )
    return rgb_to_y(rgb)


def _psnr(pred: torch.Tensor, target: torch.Tensor, border: int) -> float:
    if border > 0:
        pred = pred[..., border:-border, border:-border]
        target = target[..., border:-border, border:-border]
    mse = ((pred - target) ** 2).mean().clamp(min=1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def _ssim(pred: torch.Tensor, target: torch.Tensor, border: int) -> float:
    # Windowed SSIM per Wang et al., 2004 with L=1 (images in [0,1]).
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


_DATASET_LAYOUTS = {
    "set5": ("Set5/HR", "Set5/Set5_HR", "Set5"),
    "urban100": ("Urban100/HR", "Urban100/Urban100_HR", "Urban100/image_SRF_2", "Urban100"),
    "bsds100": ("BSDS100/HR", "BSDS100"),
    "manga109": ("Manga109/HR", "Manga109"),
}


def _resolve_dataset_root(data_root: Path, name: str) -> Optional[Path]:
    candidates = _DATASET_LAYOUTS.get(name.lower(), (name,))
    for sub in candidates:
        p = data_root / sub
        if p.is_dir() and any(p.glob("*.png")):
            return p
    return None


@torch.no_grad()
def evaluate_image_paths(
    model: torch.nn.Module,
    image_paths: Iterable[Path],
    upscale: int,
    device: torch.device,
    mode: str = "A",
    noise_sigma: float = 0.0,
) -> tuple[float, float]:
    """Run model on each image and return mean PSNR, mean SSIM (Y-channel)."""
    model.eval()
    psnrs: list[float] = []
    ssims: list[float] = []
    for p in image_paths:
        y_hr = _pil_to_y(p).to(device)
        _, H, W = y_hr.shape
        H -= H % upscale
        W -= W % upscale
        y_hr = y_hr[:, :H, :W]
        y_lr = (
            F.interpolate(y_hr[None], scale_factor=1.0 / upscale, mode="bicubic", align_corners=False)
            .squeeze(0)
            .clamp(0.0, 1.0)
        )
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


@torch.no_grad()
def evaluate_set5(
    model: torch.nn.Module,
    set5_root: Optional[str],
    upscale: int,
    device: torch.device,
    mode: str = "A",
    noise_sigma: float = 0.0,
) -> tuple[float, float]:
    """Backward-compatible Set5 eval used by train.py. Falls back to synthetic
    textures when the data path is missing so the training loop can smoke-test
    without real Set5."""
    images: list[torch.Tensor] = []
    if set5_root is not None:
        root = Path(os.path.expanduser(set5_root))
        ds_path = _resolve_dataset_root(root, "set5")
        if ds_path is None and root.is_dir():
            # Try the historical fallback: <root>/HR or <root>/*.png
            for sub in (root / "Set5" / "HR", root / "Set5", root):
                if sub.exists():
                    pngs = sorted([p for p in sub.glob("*.png") if "LR" not in p.name.upper()])
                    if pngs:
                        ds_path = sub
                        break
        if ds_path is not None:
            paths = sorted(ds_path.glob("*.png"))[:5]
            if paths:
                return evaluate_image_paths(model, paths, upscale, device, mode, noise_sigma)
    # Synthetic fallback for smoke tests
    import numpy as np
    for i in range(5):
        torch.manual_seed(1729 + i)
        rgb = torch.rand(3, 256, 256)
        rgb = F.avg_pool2d(rgb[None], 3, stride=1, padding=1).squeeze(0)
        images.append(rgb_to_y(rgb))
    model.eval()
    psnrs = []
    ssims = []
    for y_hr in images:
        y_hr = y_hr.to(device)
        _, H, W = y_hr.shape
        H -= H % upscale
        W -= W % upscale
        y_hr = y_hr[:, :H, :W]
        y_lr = (
            F.interpolate(y_hr[None], scale_factor=1.0 / upscale, mode="bicubic", align_corners=False)
            .squeeze(0)
            .clamp(0.0, 1.0)
        )
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


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(description="HERMES-SR full-benchmark evaluation")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--datasets", default="set5", help="comma-separated, e.g. set5,urban100")
    p.add_argument(
        "--noise-sigmas",
        default="",
        help="for Mode B: comma-separated sigmas on 0-255 scale, e.g. 15,25,50",
    )
    p.add_argument("--data-root", default="~/datasets")
    p.add_argument("--output", type=Path, default=None, help="JSON results path")
    args = p.parse_args()

    device = _pick_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = HermesConfig(**ckpt["config"])
    model = HermesSR(cfg, reparameterized=bool(ckpt.get("reparameterized", False)))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    print(f"[eval] loaded {args.checkpoint} (mode {cfg.mode}, ×{cfg.upscale}, "
          f"{sum(p.numel() for p in model.parameters()):,} params, device {device.type})")

    data_root = Path(os.path.expanduser(args.data_root))
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    sigmas: list[float] = []
    if args.noise_sigmas.strip():
        sigmas = [float(s) for s in args.noise_sigmas.split(",") if s.strip()]
    if cfg.mode == "B" and not sigmas:
        sigmas = [25.0]
    if cfg.mode == "A":
        sigmas = [0.0]

    results: dict = {"checkpoint": str(args.checkpoint), "config": asdict(cfg), "metrics": {}}
    for ds_name in datasets:
        ds_path = _resolve_dataset_root(data_root, ds_name)
        if ds_path is None:
            print(f"[eval] WARN: dataset {ds_name!r} not found under {data_root}; skipping")
            continue
        images = sorted(ds_path.glob("*.png"))
        if not images:
            print(f"[eval] WARN: no PNGs in {ds_path}; skipping")
            continue
        for sigma in sigmas:
            psnr, ssim = evaluate_image_paths(model, images, cfg.upscale, device, cfg.mode, sigma)
            key = f"{ds_name}_x{cfg.upscale}"
            if cfg.mode == "B" and sigma > 0:
                key += f"_sigma{int(sigma)}"
            results["metrics"][key] = {"psnr": psnr, "ssim": ssim, "n_images": len(images)}
            print(f"[eval] {key}: {psnr:.3f} dB / {ssim:.4f} SSIM ({len(images)} images)")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] wrote {args.output}")


if __name__ == "__main__":
    main()

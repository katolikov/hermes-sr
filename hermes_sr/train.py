"""Single-entry-point trainer.

Usage:
    python -m hermes_sr.train --config configs/mode_a.json
    python -m hermes_sr.train --config configs/mode_b.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hermes_sr.data import DIV2KTrainset, Flickr2KTrainset, SIDDTrainset
from hermes_sr.eval import evaluate_set5
from hermes_sr.losses import CompositeLoss
from hermes_sr.model import HermesConfig, HermesSR


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_dataset(cfg: dict):
    mode = cfg["mode"]
    upscale = cfg["upscale"]
    patch_hr = cfg.get("patch_size", 96)
    data_root = cfg.get("data_root", "~/datasets")
    if mode == "A":
        primary = DIV2KTrainset(root=data_root, patch_hr=patch_hr, upscale=upscale)
        aux = Flickr2KTrainset(root=data_root, patch_hr=patch_hr, upscale=upscale)
        return torch.utils.data.ConcatDataset([primary, aux])
    if mode == "B":
        return SIDDTrainset(
            root=data_root,
            patch_hr=patch_hr,
            upscale=upscale,
            fallback_root=data_root,
        )
    raise ValueError(f"unknown mode {mode!r}")


def train(cfg_path: str) -> None:
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    device = _pick_device()
    print(f"[hermes_sr.train] device={device.type} config={cfg_path}")

    in_channels = 2 if cfg["mode"] == "B" else 1
    model_cfg = HermesConfig(
        mode=cfg["mode"],
        upscale=cfg["upscale"],
        in_channels=in_channels,
    )
    model = HermesSR(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hermes_sr.train] model params: {n_params:,}")

    loss_fn = CompositeLoss(
        weights=cfg.get("weights"),
        use_perceptual=cfg.get("use_perceptual", True),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 2e-4), betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.get("max_iters", 200_000))

    # AMP only on CUDA — MPS autocast is shaky in 2.10 for some FFT/grid ops
    use_amp = device.type == "cuda" and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    dataset = _build_dataset(cfg)
    num_workers = cfg.get("num_workers", 2 if device.type != "mps" else 0)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    max_iters = cfg.get("max_iters", 200_000)
    eval_interval = cfg.get("eval_interval", 5_000)
    log_interval = cfg.get("log_interval", 100)
    set5_root = cfg.get("set5_root")
    eval_noise = cfg.get("eval_noise_sigma", 0.0)

    # Optional linear warmup on the temporal-loss weight. Disabled by default
    # (warmup_iters=0). If a future run plateaus before crossing bicubic at
    # iter 8K, set "temporal_warmup_iters": 5000 in the config to ramp it
    # linearly from 0 to its configured weight.
    temporal_target = float(loss_fn.weights["temporal"])
    temporal_warmup = int(cfg.get("temporal_warmup_iters", 0))

    out_dir = Path(cfg.get("out_dir", "checkpoints")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    iter_count = 0
    t0 = time.time()
    running_loss = 0.0
    running_parts: dict = {k: 0.0 for k in ("pixel", "perceptual", "spectral", "temporal", "distill")}
    running_n = 0

    while iter_count < max_iters:
        for batch in loader:
            if iter_count >= max_iters:
                break
            lr_prev = batch["lr_prev"].to(device, non_blocking=True)
            lr_curr = batch["lr_curr"].to(device, non_blocking=True)
            hr_curr = batch["hr_curr"].to(device, non_blocking=True)

            if temporal_warmup > 0:
                loss_fn.weights["temporal"] = temporal_target * min(
                    1.0, iter_count / temporal_warmup
                )

            opt.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else _NullCtx()
            with ctx:
                # Previous frame: no recurrent state, no flow; cheap pass for the
                # state hand-off and the temporal-loss reference.
                with torch.no_grad():
                    pred_prev, h_state = model(lr_prev)
                pred_curr, _ = model(lr_curr, h_prev=h_state, y_prev=lr_prev[:, :1])

                # Use the synthetic-known LR shift, lifted to HR pixels, for the
                # temporal loss instead of re-running the diamond estimator.
                shift_lr = batch["shift_lr"].to(device).float()
                flow = _make_uniform_hr_flow(shift_lr, hr_curr.shape[-2:], cfg["upscale"]).to(device)

                total, parts = loss_fn(pred_curr, hr_curr, prev_pred=pred_prev, flow=flow)

            if use_amp:
                scaler.scale(total).backward()
                scaler.step(opt)
                scaler.update()
            else:
                total.backward()
                opt.step()
            sched.step()

            running_loss += float(total.detach())
            for k in running_parts:
                running_parts[k] += float(parts[k].detach())
            running_n += 1
            iter_count += 1

            if iter_count % log_interval == 0:
                dt = time.time() - t0
                ips = iter_count / max(dt, 1e-6)
                pix = running_parts["pixel"] / running_n
                tmp = running_parts["temporal"] / running_n
                perc = running_parts["perceptual"] / running_n
                spec = running_parts["spectral"] / running_n
                print(
                    f"iter {iter_count:>7d} | loss {running_loss / running_n:.4f}"
                    f" | pix {pix:.4f} | tmp {tmp:.4f} | perc {perc:.4f} | spec {spec:.4f}"
                    f" | {ips:.2f} it/s"
                )
                running_loss = 0.0
                running_n = 0
                for k in running_parts:
                    running_parts[k] = 0.0

            if iter_count % eval_interval == 0 or iter_count == max_iters:
                psnr, ssim = evaluate_set5(
                    model,
                    set5_root=set5_root,
                    upscale=cfg["upscale"],
                    device=device,
                    mode=cfg["mode"],
                    noise_sigma=eval_noise,
                )
                print(f"iter {iter_count:>7d} | val PSNR {psnr:.3f} dB | val SSIM {ssim:.4f}")
                ckpt_path = out_dir / f"hermes_{cfg['mode'].lower()}_iter{iter_count}.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": asdict(model_cfg),
                        "reparameterized": False,
                        "iter": iter_count,
                    },
                    ckpt_path,
                )

    print(f"[hermes_sr.train] done in {time.time() - t0:.1f}s, {iter_count} iters")


def _make_uniform_hr_flow(
    shift_lr: torch.Tensor,
    hr_size: tuple[int, int],
    upscale: int,
) -> torch.Tensor:
    """Build a uniform flow field at HR resolution from a per-batch LR shift."""
    n = shift_lr.shape[0]
    h, w = hr_size
    flow = torch.zeros(n, 2, h, w, device=shift_lr.device, dtype=shift_lr.dtype)
    flow[:, 0] = (shift_lr[:, 0] * upscale).view(n, 1, 1)
    flow[:, 1] = (shift_lr[:, 1] * upscale).view(n, 1, 1)
    return flow


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

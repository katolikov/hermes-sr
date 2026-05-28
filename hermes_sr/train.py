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
from typing import Mapping

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
    model_cfg_kwargs: dict = {
        "mode": cfg["mode"],
        "upscale": cfg["upscale"],
        "in_channels": in_channels,
    }
    # Pass-through architecture overrides from the config file if present.
    for key in ("kernel_size", "use_recurrent_state", "anchor_mode",
                "trunk_channels", "ib_channels", "num_blocks",
                "state_channels", "flow_down_factor", "flow_levels",
                "block_type", "ecb_depth_multiplier"):
        if key in cfg:
            model_cfg_kwargs[key] = cfg[key]
    model_cfg = HermesConfig(**model_cfg_kwargs)

    # Warm-start: if init_from is given, mirror the source checkpoint's
    # reparameterized state so the trunk dw layer names line up before loading.
    init_from = cfg.get("init_from")
    init_reparameterized = False
    init_state = None
    if init_from:
        init_path = Path(init_from).expanduser()
        if not init_path.exists():
            raise FileNotFoundError(f"init_from checkpoint not found: {init_path}")
        init_ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        init_reparameterized = bool(init_ckpt.get("reparameterized", False))
        init_state = init_ckpt["state_dict"]
        print(f"[hermes_sr.train] warm-starting from {init_path} (reparameterized={init_reparameterized})")

    model = HermesSR(model_cfg, reparameterized=init_reparameterized).to(device)
    if init_state is not None:
        _load_warm_start(model, init_state)
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
    # Accept eval_every (preferred for convergence configs) with eval_interval as fallback.
    eval_interval = int(cfg.get("eval_every", cfg.get("eval_interval", 5_000)))
    save_interval = int(cfg.get("save_every", eval_interval))
    log_interval = int(cfg.get("log_interval", 100))
    set5_root = cfg.get("set5_root")
    eval_noise = cfg.get("eval_noise_sigma", 0.0)
    ckpt_prefix = cfg.get("ckpt_prefix", f"hermes_{cfg['mode'].lower()}_iter")

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
                # Skip the previous-frame forward when neither the temporal loss
                # nor the recurrent state path needs it. Pure-L1 + static-SR
                # configs save ~50% of per-iter wall time this way.
                need_prev = (
                    float(loss_fn.weights.get("temporal", 0.0)) > 0.0
                    or model.config.use_recurrent_state
                )
                if need_prev:
                    with torch.no_grad():
                        pred_prev, h_state = model(lr_prev)
                    pred_curr, _ = model(lr_curr, h_prev=h_state, y_prev=lr_prev[:, :1])
                    shift_lr = batch["shift_lr"].to(device).float()
                    flow = _make_uniform_hr_flow(shift_lr, hr_curr.shape[-2:], cfg["upscale"]).to(device)
                else:
                    pred_curr, _ = model(lr_curr)
                    pred_prev = None
                    flow = None

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
            if iter_count % save_interval == 0 or iter_count == max_iters:
                ckpt_path = out_dir / f"{ckpt_prefix}{iter_count}.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": asdict(model_cfg),
                        "reparameterized": init_reparameterized,
                        "iter": iter_count,
                    },
                    ckpt_path,
                )

    print(f"[hermes_sr.train] done in {time.time() - t0:.1f}s, {iter_count} iters")


def _load_warm_start(model: torch.nn.Module, src_state: Mapping[str, torch.Tensor]) -> None:
    """Copy compatible tensors from src_state into model.

    Same name + same shape -> direct copy.
    Stem conv weight (target 2-channel, source 1-channel) -> copy source into
    channel 0 of target, leave channel 1 zero. This is the Mode A -> Mode B
    warm-start case (Mode B adds a broadcast noise-sigma channel).
    All other shape/name mismatches are skipped with a printed note.
    """
    dst_state = model.state_dict()
    copied = []
    skipped = []
    for k, dst in dst_state.items():
        src = src_state.get(k)
        if src is None:
            skipped.append(f"{k} (missing in source)")
            continue
        if src.shape == dst.shape:
            dst.copy_(src)
            copied.append(k)
            continue
        # Special case: stem conv weight expanded from 1 to 2 input channels
        if (
            k == "stem.0.weight"
            and src.ndim == 4
            and dst.ndim == 4
            and src.shape[0] == dst.shape[0]
            and src.shape[2:] == dst.shape[2:]
            and src.shape[1] == 1
            and dst.shape[1] == 2
        ):
            dst.zero_()
            dst[:, :1].copy_(src)
            copied.append(f"{k} (stem 1->2ch expand)")
            continue
        skipped.append(f"{k} (shape {tuple(src.shape)} vs {tuple(dst.shape)})")
    print(f"[hermes_sr.train] warm-start: copied {len(copied)} tensors, skipped {len(skipped)}")
    for s in skipped:
        print(f"  skip: {s}")


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

"""Stage 2: GAN fine-tune (Real-ESRGAN-style) for real sensor-video SR.

Warm-starts the generator from a Stage-1 (L1/temporal) checkpoint and fine-tunes
with L1 + VGG perceptual + hinge adversarial (+ temporal for video) against a
U-Net spectral-norm discriminator. The generator architecture/speed is unchanged
(ECB dense-3x3); the discriminator is training-only.

    python -m hermes_sr.train_gan --config configs/sensor_b_x2_gan.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hermes_sr.eval import evaluate_set5
from hermes_sr.gan.discriminator import UNetDiscriminatorSN
from hermes_sr.losses import CompositeLoss
from hermes_sr.model import HermesConfig, HermesSR
from hermes_sr.train import (
    _build_dataset,
    _load_warm_start,
    _make_uniform_hr_flow,
    _pick_device,
)


def train_gan(cfg_path: str) -> None:
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    device = _pick_device()
    print(f"[train_gan] device={device.type} config={cfg_path}")

    # Generator: build from the Stage-1 checkpoint's config, warm-start its weights.
    init_path = Path(cfg["init_from"]).expanduser()
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    model_cfg = HermesConfig(**ckpt["config"])
    G = HermesSR(model_cfg, reparameterized=bool(ckpt.get("reparameterized", False))).to(device)
    _load_warm_start(G, ckpt["state_dict"])
    print(f"[train_gan] generator warm-started from {init_path} "
          f"({sum(p.numel() for p in G.parameters()):,} params)")

    D = UNetDiscriminatorSN(in_ch=1, num_feat=cfg.get("disc_feat", 32)).to(device)
    print(f"[train_gan] discriminator: {sum(p.numel() for p in D.parameters()):,} params")

    # Composite handles pixel + perceptual + temporal; adversarial added here.
    weights = cfg.get("weights", {"pixel": 1.0, "perceptual": 0.1, "spectral": 0.0,
                                   "temporal": 0.25, "distill": 0.0})
    loss_fn = CompositeLoss(weights=weights, use_perceptual=True).to(device)
    lam_gan = float(cfg.get("gan_weight", 0.1))

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.get("lr_g", 1e-4), betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.get("lr_d", 1e-4), betas=(0.9, 0.99))
    max_iters = cfg.get("max_iters", 100_000)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=max_iters)

    ema_decay = float(cfg.get("ema_decay", 0.999))
    ema_state = {k: v.detach().clone() for k, v in G.state_dict().items()} if ema_decay > 0 else None

    loader = DataLoader(
        _build_dataset(cfg), batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 0), drop_last=True,
    )
    out_dir = Path(cfg.get("out_dir", "checkpoints")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_prefix = cfg.get("ckpt_prefix", "gan_iter")
    eval_interval = int(cfg.get("eval_every", 2000))
    save_interval = int(cfg.get("save_every", 5000))
    log_interval = int(cfg.get("log_interval", 200))
    upscale = cfg["upscale"]

    G.train()
    D.train()
    it = 0
    t0 = time.time()
    acc = {"l1": 0.0, "perc": 0.0, "g_adv": 0.0, "d": 0.0, "n": 0}
    while it < max_iters:
        for batch in loader:
            if it >= max_iters:
                break
            lr_prev = batch["lr_prev"].to(device)
            lr_curr = batch["lr_curr"].to(device)
            hr_curr = batch["hr_curr"].to(device)
            shift_lr = batch["shift_lr"].to(device).float()

            # Generator forward (prev pass no-grad to hand off recurrent state)
            if G.config.use_recurrent_state:
                with torch.no_grad():
                    pred_prev, h_state = G(lr_prev)
                pred_curr, _ = G(lr_curr, h_prev=h_state, y_prev=lr_prev[:, :1])
                flow = _make_uniform_hr_flow(shift_lr, hr_curr.shape[-2:], upscale).to(device)
            else:
                pred_curr, _ = G(lr_curr)
                pred_prev, flow = None, None

            # ---- Discriminator step (hinge) ----
            opt_d.zero_grad(set_to_none=True)
            d_real = D(hr_curr)
            d_fake = D(pred_curr.detach())
            loss_d = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            loss_d.backward()
            opt_d.step()

            # ---- Generator step ----
            opt_g.zero_grad(set_to_none=True)
            content, parts = loss_fn(pred_curr, hr_curr, prev_pred=pred_prev, flow=flow)
            g_adv = -D(pred_curr).mean()  # hinge generator
            loss_g = content + lam_gan * g_adv
            loss_g.backward()
            opt_g.step()
            sched_g.step()

            if ema_state is not None:
                with torch.no_grad():
                    gsd = G.state_dict()
                    for k, v in ema_state.items():
                        if v.is_floating_point():
                            v.mul_(ema_decay).add_(gsd[k].detach(), alpha=1 - ema_decay)
                        else:
                            v.copy_(gsd[k])

            acc["l1"] += float(parts["pixel"].detach())
            acc["perc"] += float(parts["perceptual"].detach())
            acc["g_adv"] += float(g_adv.detach())
            acc["d"] += float(loss_d.detach())
            acc["n"] += 1
            it += 1

            if it % log_interval == 0:
                n = acc["n"]
                ips = it / max(time.time() - t0, 1e-6)
                print(f"iter {it:>7d} | l1 {acc['l1']/n:.4f} | perc {acc['perc']/n:.4f}"
                      f" | g_adv {acc['g_adv']/n:+.4f} | d {acc['d']/n:.4f} | {ips:.2f} it/s")
                acc = {k: 0.0 for k in acc}

            if it % eval_interval == 0 or it == max_iters:
                backup = None
                if ema_state is not None:
                    backup = {k: v.detach().clone() for k, v in G.state_dict().items()}
                    G.load_state_dict(ema_state)
                psnr, ssim = evaluate_set5(G, cfg.get("set5_root"), upscale, device,
                                           mode=cfg["mode"], noise_sigma=cfg.get("eval_noise_sigma", 0.0))
                print(f"iter {it:>7d} | val PSNR {psnr:.3f} dB | val SSIM {ssim:.4f} (ema)")
                if it % save_interval == 0 or it == max_iters:
                    torch.save({"state_dict": G.state_dict(), "config": asdict(model_cfg),
                                "reparameterized": bool(ckpt.get("reparameterized", False)),
                                "iter": it}, out_dir / f"{ckpt_prefix}{it}.pt")
                if backup is not None:
                    G.load_state_dict(backup)
            G.train()

    print(f"[train_gan] done in {time.time() - t0:.1f}s, {it} iters")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    train_gan(args.config)


if __name__ == "__main__":
    main()

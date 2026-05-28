"""Consume a trained checkpoint and emit a reparameterized one.

Usage:
    python -m hermes_sr.scripts.reparameterize --in checkpoints/hermes_a_iterN.pt --out checkpoints/hermes_a_deploy.pt

Asserts that the post-reparameterization output on a fixed random input matches
the pre-reparameterization output to within 1e-5 absolute.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from hermes_sr.model import HermesConfig, HermesSR


def reparameterize_and_save(
    in_path: str | Path,
    out_path: str | Path,
    seed: int = 0,
    atol: float = 1e-5,
    spatial: tuple[int, int] = (32, 32),
) -> None:
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    cfg = HermesConfig(**ckpt["config"])
    model = HermesSR(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    torch.manual_seed(seed)
    h, w = spatial
    x = torch.randn(1, cfg.in_channels, h, w)
    x = x.abs().clamp(max=1.0)
    with torch.no_grad():
        y_before, h_before = model(x)
        model.reparameterize()
        y_after, h_after = model(x)

    max_y = (y_before - y_after).abs().max().item()
    assert max_y < atol, f"reconstruction diverged after reparam: max diff {max_y:.3e} >= {atol}"
    if h_before is not None and h_after is not None:
        max_h = (h_before - h_after).abs().max().item()
        assert max_h < atol, f"state diverged after reparam: max diff {max_h:.3e} >= {atol}"
        print(f"[reparameterize] equivalence ok (max y diff {max_y:.3e}, max h diff {max_h:.3e})")
    else:
        print(f"[reparameterize] equivalence ok (max y diff {max_y:.3e}, no recurrent state)")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "reparameterized": True,
        },
        out_path,
    )
    print(f"[reparameterize] saved deployment checkpoint -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args()
    reparameterize_and_save(args.in_path, args.out_path, seed=args.seed, atol=args.atol)


if __name__ == "__main__":
    main()

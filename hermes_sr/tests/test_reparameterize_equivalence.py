"""Reparameterization must preserve the forward output to within fp32 tolerance."""
from __future__ import annotations

import torch

from hermes_sr.model import HermesConfig, HermesSR


def _check_equivalence(mode: str, upscale: int, in_channels: int, atol: float = 1e-5) -> None:
    cfg = HermesConfig(mode=mode, upscale=upscale, in_channels=in_channels)
    torch.manual_seed(42)
    model = HermesSR(cfg).eval()
    x = torch.randn(2, in_channels, 24, 24).abs().clamp(max=1.0)

    with torch.no_grad():
        y_before, h_before = model(x)
        model.reparameterize()
        y_after, h_after = model(x)

    max_y = (y_before - y_after).abs().max().item()
    max_h = (h_before - h_after).abs().max().item()
    assert max_y < atol, f"{mode}: y diff {max_y:.3e} >= {atol}"
    assert max_h < atol, f"{mode}: h diff {max_h:.3e} >= {atol}"


def test_reparameterize_equivalence_mode_a() -> None:
    _check_equivalence(mode="A", upscale=2, in_channels=1)


def test_reparameterize_equivalence_mode_b() -> None:
    _check_equivalence(mode="B", upscale=3, in_channels=2)

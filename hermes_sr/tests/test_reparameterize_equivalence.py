"""Reparameterization must preserve the forward output to within fp32 tolerance."""
from __future__ import annotations

import torch

from hermes_sr.model import HermesConfig, HermesSR


def _check_equivalence(cfg: HermesConfig, atol: float = 1e-5) -> None:
    torch.manual_seed(42)
    model = HermesSR(cfg).eval()
    x = torch.randn(2, cfg.in_channels, 24, 24).abs().clamp(max=1.0)

    with torch.no_grad():
        y_before, h_before = model(x)
        model.reparameterize()
        y_after, h_after = model(x)

    max_y = (y_before - y_after).abs().max().item()
    assert max_y < atol, f"{cfg.mode} k={cfg.kernel_size}: y diff {max_y:.3e} >= {atol}"
    if h_before is not None and h_after is not None:
        max_h = (h_before - h_after).abs().max().item()
        assert max_h < atol, f"{cfg.mode} k={cfg.kernel_size}: h diff {max_h:.3e} >= {atol}"


def test_reparameterize_equivalence_mode_a() -> None:
    # Default block_type="ecb": branches collapse to a dense 3x3.
    _check_equivalence(HermesConfig(mode="A", upscale=2, in_channels=1))


def test_reparameterize_equivalence_mode_b() -> None:
    _check_equivalence(HermesConfig(mode="B", upscale=3, in_channels=2))


def test_reparameterize_equivalence_hermes_block() -> None:
    # The depthwise-large-kernel block (block_type="hermes"), 7x7.
    _check_equivalence(
        HermesConfig(mode="A", upscale=2, in_channels=1, block_type="hermes", kernel_size=7)
    )


def test_reparameterize_equivalence_mvp_legacy() -> None:
    # MVP legacy: hermes block, 11x11, recurrent-state path active, bicubic anchor.
    _check_equivalence(
        HermesConfig(mode="A", upscale=2, in_channels=1, block_type="hermes",
                     kernel_size=11, use_recurrent_state=True, anchor_mode="bicubic")
    )

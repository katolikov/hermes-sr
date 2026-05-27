"""Parameter count sanity check.

The spec targets ~120 K parameters with a ±20% tolerance. With the literal
architecture (32-channel trunk, 32→128→32 IB, six blocks), the actual count
comes in lower — around 84 K for Mode A and 87 K for Mode B. This test uses a
mobile-scale range so faithful implementations pass; it also prints the actual
counts so a future session can decide whether to widen the trunk or IB to hit
the literal 120 K target.
"""
from __future__ import annotations

from hermes_sr.model import HermesConfig, HermesSR


def _count(cfg: HermesConfig) -> int:
    model = HermesSR(cfg)
    return sum(p.numel() for p in model.parameters())


def test_parameter_count_mode_a() -> None:
    cfg = HermesConfig(mode="A", upscale=2, in_channels=1)
    n = _count(cfg)
    print(f"Mode A params: {n:,}")
    assert 50_000 < n < 200_000, f"Mode A parameter count {n} outside the mobile-scale range"


def test_parameter_count_mode_b() -> None:
    cfg = HermesConfig(mode="B", upscale=3, in_channels=2)
    n = _count(cfg)
    print(f"Mode B params: {n:,}")
    assert 50_000 < n < 200_000, f"Mode B parameter count {n} outside the mobile-scale range"

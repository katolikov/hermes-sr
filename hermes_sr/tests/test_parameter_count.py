"""Parameter count sanity check.

Deployment-meaningful count is the INFERENCE (reparameterized) count, so the
test reparameterizes first. ECB blocks have a large multi-branch training-time
footprint that collapses to a single dense 3x3 per block at inference — the
inference count is what lands on the device.

Target: a small mobile-scale network. Inference count asserted in [30K, 120K]
for both modes; training count printed for reference.
"""
from __future__ import annotations

import copy

from hermes_sr.model import HermesConfig, HermesSR


def _counts(cfg: HermesConfig) -> tuple[int, int]:
    model = HermesSR(cfg)
    train_n = sum(p.numel() for p in model.parameters())
    rep = copy.deepcopy(model)
    rep.reparameterize()
    inf_n = sum(p.numel() for p in rep.parameters())
    return train_n, inf_n


def test_parameter_count_mode_a() -> None:
    train_n, inf_n = _counts(HermesConfig(mode="A", upscale=2, in_channels=1))
    print(f"Mode A params: train {train_n:,}, inference {inf_n:,}")
    assert 30_000 < inf_n < 120_000, f"Mode A inference count {inf_n} outside mobile range"


def test_parameter_count_mode_b() -> None:
    train_n, inf_n = _counts(HermesConfig(mode="B", upscale=3, in_channels=2))
    print(f"Mode B params: train {train_n:,}, inference {inf_n:,}")
    assert 30_000 < inf_n < 120_000, f"Mode B inference count {inf_n} outside mobile range"

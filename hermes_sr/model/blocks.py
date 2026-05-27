"""HERMES trunk block.

Training-mode path:
    dw_out = dw11(x) + dw5(x) + dw3(x) + dw1(x) + x        # five parallel depthwise branches
    ib_out = ib2(relu6(ib1(dw_out)))                       # 1x1 -> ReLU6 -> 1x1 inverted bottleneck
    return x + gate * (ib_out [+ simple_gate(dw_out)])     # learned scalar gate on residual

After reparameterize(), the five DW branches collapse into a single 11x11
depthwise conv with summed weights. Equivalence is bit-exact up to float
rounding (verified in tests/test_reparameterize_equivalence.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HermesBlock(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        ib_channels: int = 128,
        use_simple_gate: bool = False,
        reparameterized: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.ib_channels = ib_channels
        self.use_simple_gate = use_simple_gate
        self._reparameterized = reparameterized

        if reparameterized:
            self.dw = nn.Conv2d(channels, channels, 11, padding=5, groups=channels)
        else:
            self.dw11 = nn.Conv2d(channels, channels, 11, padding=5, groups=channels)
            self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
            self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
            self.dw1 = nn.Conv2d(channels, channels, 1, groups=channels)
            # identity branch is x; no parameters

        self.ib1 = nn.Conv2d(channels, ib_channels, 1)
        self.ib2 = nn.Conv2d(ib_channels, channels, 1)

        self.gate = nn.Parameter(torch.ones(()))

        if use_simple_gate:
            assert channels % 2 == 0
            self.sg_proj = nn.Conv2d(channels // 2, channels, 1)

    def _dw_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._reparameterized:
            return self.dw(x)
        return self.dw11(x) + self.dw5(x) + self.dw3(x) + self.dw1(x) + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dw_out = self._dw_forward(x)
        ib_out = self.ib2(F.relu6(self.ib1(dw_out)))
        if self.use_simple_gate:
            a, b = dw_out.chunk(2, dim=1)
            ib_out = ib_out + self.sg_proj(a * b)
        return x + self.gate * ib_out

    @torch.no_grad()
    def reparameterize(self) -> None:
        if self._reparameterized:
            return
        c = self.channels

        w11 = self.dw11.weight.clone()
        # Center-pad each smaller kernel into an 11x11 footprint and sum
        w5 = F.pad(self.dw5.weight, [3, 3, 3, 3])
        w3 = F.pad(self.dw3.weight, [4, 4, 4, 4])
        w1 = F.pad(self.dw1.weight, [5, 5, 5, 5])
        # Identity branch as a depthwise kernel: 1 at center, zero elsewhere
        w_id = torch.zeros_like(w11)
        w_id[:, :, 5, 5] = 1.0

        w_merged = w11 + w5 + w3 + w1 + w_id
        # Identity branch carries no bias; the four convs each do
        b_merged = self.dw11.bias.clone() + self.dw5.bias + self.dw3.bias + self.dw1.bias

        merged = nn.Conv2d(c, c, 11, padding=5, groups=c)
        merged.weight.data.copy_(w_merged)
        merged.bias.data.copy_(b_merged)

        # Move to the same device/dtype as the originals
        merged = merged.to(device=w_merged.device, dtype=w_merged.dtype)

        self.dw = merged
        del self.dw11, self.dw5, self.dw3, self.dw1
        self._reparameterized = True

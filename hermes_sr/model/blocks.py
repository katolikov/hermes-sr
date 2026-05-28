"""HERMES trunk block.

Training-mode path:
    dw_out = sum of parallel depthwise branches (main + smaller + identity)
    ib_out = ib2(relu6(ib1(dw_out)))                       # 1x1 -> ReLU6 -> 1x1 inverted bottleneck
    return x + gate * (ib_out [+ simple_gate(dw_out)])     # learned scalar gate on residual

After reparameterize(), the depthwise branches collapse into a single kernel_size x kernel_size
depthwise conv with summed weights. Equivalence is bit-exact up to float rounding.

kernel_size parameterization: for kernel_size >= 7, branches are
[kernel_size, 5, 3, 1, identity]. For kernel_size == 5: [5, 3, 1, identity].
This trades the spec's hard-coded 11x11 max for mobile-NPU friendliness — Apple
ANE caps depthwise at 7x7, Qualcomm HTP is optimized for 3x3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _branch_sizes_for(kernel_size: int) -> list[int]:
    """Smaller branches paired with the main kernel_size branch."""
    candidates = [5, 3, 1]
    return [c for c in candidates if c < kernel_size]


class HermesBlock(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        ib_channels: int = 128,
        use_simple_gate: bool = False,
        reparameterized: bool = False,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        assert kernel_size >= 3, "kernel_size must be >= 3"
        self.channels = channels
        self.ib_channels = ib_channels
        self.use_simple_gate = use_simple_gate
        self._reparameterized = reparameterized
        self.kernel_size = kernel_size
        self._branch_sizes = _branch_sizes_for(kernel_size)
        pad = kernel_size // 2

        if reparameterized:
            self.dw = nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=channels)
        else:
            self.dw_main = nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=channels)
            for b in self._branch_sizes:
                setattr(
                    self,
                    f"dw{b}",
                    nn.Conv2d(channels, channels, b, padding=b // 2, groups=channels),
                )
            # identity branch is x itself; no parameters

        self.ib1 = nn.Conv2d(channels, ib_channels, 1)
        self.ib2 = nn.Conv2d(ib_channels, channels, 1)

        self.gate = nn.Parameter(torch.ones(()))

        if use_simple_gate:
            assert channels % 2 == 0
            self.sg_proj = nn.Conv2d(channels // 2, channels, 1)

    def _dw_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._reparameterized:
            return self.dw(x)
        out = self.dw_main(x)
        for b in self._branch_sizes:
            out = out + getattr(self, f"dw{b}")(x)
        out = out + x  # identity
        return out

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
        k = self.kernel_size
        pad = k // 2

        w_merged = self.dw_main.weight.clone()
        b_merged = self.dw_main.bias.clone()

        # Center-pad each smaller-kernel weight into a k x k footprint and add
        for b in self._branch_sizes:
            conv = getattr(self, f"dw{b}")
            pad_each = (k - b) // 2
            w_padded = F.pad(conv.weight, [pad_each, pad_each, pad_each, pad_each])
            w_merged = w_merged + w_padded
            b_merged = b_merged + conv.bias

        # Identity branch as a depthwise kernel: 1 at center, zero elsewhere
        w_id = torch.zeros_like(self.dw_main.weight)
        w_id[:, :, pad, pad] = 1.0
        w_merged = w_merged + w_id

        merged = nn.Conv2d(c, c, k, padding=pad, groups=c)
        merged.weight.data.copy_(w_merged)
        merged.bias.data.copy_(b_merged)
        merged = merged.to(device=w_merged.device, dtype=w_merged.dtype)

        self.dw = merged
        del self.dw_main
        for b in self._branch_sizes:
            delattr(self, f"dw{b}")
        self._reparameterized = True

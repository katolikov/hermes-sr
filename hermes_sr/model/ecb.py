"""Edge-oriented Convolution Block (ECB), after Zhang et al. ECBSR (ACM MM 2021).

Training time: five parallel branches summed (+ optional identity):
  1. plain 3x3
  2. 1x1 -> 3x3 (expand-then-conv)
  3. 1x1 -> fixed Sobel-Dx depthwise
  4. 1x1 -> fixed Sobel-Dy depthwise
  5. 1x1 -> fixed Laplacian depthwise

Inference: all branches collapse to a SINGLE plain 3x3 conv via rep_params().
This gives a dense 3x3 at deploy (best NPU utilization — same op class as ABPN)
while the edge-prior branches steer high-frequency reconstruction during training.

Equivalence of forward(x) vs the reparameterized 3x3 is verified to fp32
tolerance in tests/test_reparameterize_equivalence.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqConv3x3(nn.Module):
    """A 1x1 conv followed by a 3x3 op that collapses to one equivalent 3x3 conv.

    seq_type:
      'conv1x1-conv3x3'  : 1x1 (inp->mid) then learnable 3x3 (mid->out)
      'conv1x1-sobelx'   : 1x1 (inp->out) then fixed scaled Sobel-Dx depthwise 3x3
      'conv1x1-sobely'   : 1x1 (inp->out) then fixed scaled Sobel-Dy depthwise 3x3
      'conv1x1-laplacian': 1x1 (inp->out) then fixed scaled Laplacian depthwise 3x3
    """

    def __init__(self, seq_type: str, inp: int, out: int, depth_multiplier: int = 1) -> None:
        super().__init__()
        self.seq_type = seq_type
        self.inp = inp
        self.out = out

        # conv0 (the 1x1) carries NO bias: a biased 1x1 followed by a padded 3x3
        # is not exactly reparameterizable (zero-padding != bias-padding at the
        # border). The block keeps bias on the main 3x3 and on the edge branches'
        # depthwise, so expressivity is unaffected.
        if seq_type == "conv1x1-conv3x3":
            mid = int(out * depth_multiplier)
            self.mid = mid
            self.conv0 = nn.Conv2d(inp, mid, 1, bias=False)
            self.conv1 = nn.Conv2d(mid, out, 3, padding=1)
        elif seq_type in ("conv1x1-sobelx", "conv1x1-sobely", "conv1x1-laplacian"):
            self.conv0 = nn.Conv2d(inp, out, 1, bias=False)
            # learnable per-channel scale + bias on the fixed edge kernel
            self.scale = nn.Parameter(torch.randn(out, 1, 1, 1) * 1e-3)
            self.bias = nn.Parameter(torch.zeros(out))
            mask = torch.zeros(out, 1, 3, 3)
            if seq_type == "conv1x1-sobelx":
                base = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
            elif seq_type == "conv1x1-sobely":
                base = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
            else:  # laplacian
                base = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
            mask[:, 0, :, :] = base
            self.register_buffer("mask", mask)
        else:
            raise ValueError(f"unknown seq_type {seq_type!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq_type == "conv1x1-conv3x3":
            y = self.conv0(x)
            return self.conv1(y)
        y = self.conv0(x)
        kernel = self.scale * self.mask  # (out, 1, 3, 3)
        return F.conv2d(y, kernel, bias=self.bias, padding=1, groups=self.out)

    @torch.no_grad()
    def rep_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (weight, bias) of the single equivalent 3x3 conv."""
        if self.seq_type == "conv1x1-conv3x3":
            k0 = self.conv0.weight  # (mid, inp, 1, 1), no bias
            k1 = self.conv1.weight  # (out, mid, 3, 3)
            b1 = self.conv1.bias    # (out,)
            # Fuse 1x1 into the 3x3: conv the 3x3 kernel by the 1x1 kernel
            merged_w = F.conv2d(k1, k0.permute(1, 0, 2, 3))  # (out, inp, 3, 3)
            return merged_w, b1.clone()

        # edge branches: 1x1 (out,inp), no bias, then fixed depthwise edge kernel
        k0 = self.conv0.weight  # (out, inp, 1, 1), no bias
        edge = self.scale * self.mask  # (out, 1, 3, 3)
        # For output channel o: fused[o, i] = k0[o, i] * edge[o]
        merged_w = k0[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1) * edge  # (out,inp,1,1)*(out,1,3,3)->(out,inp,3,3)
        merged_b = self.bias.clone()
        return merged_w, merged_b


class ECB(nn.Module):
    """Edge-oriented Convolution Block: five parallel branches collapsing to one 3x3."""

    def __init__(
        self,
        inp: int,
        out: int,
        depth_multiplier: float = 2.0,
        with_idt: bool = True,
        reparameterized: bool = False,
    ) -> None:
        super().__init__()
        self.inp = inp
        self.out = out
        self.with_idt = with_idt and (inp == out)
        self._reparameterized = reparameterized

        if reparameterized:
            self.rep = nn.Conv2d(inp, out, 3, padding=1)
        else:
            self.conv3x3 = nn.Conv2d(inp, out, 3, padding=1)
            self.conv1x1_3x3 = SeqConv3x3("conv1x1-conv3x3", inp, out, depth_multiplier)
            self.conv1x1_sbx = SeqConv3x3("conv1x1-sobelx", inp, out)
            self.conv1x1_sby = SeqConv3x3("conv1x1-sobely", inp, out)
            self.conv1x1_lpl = SeqConv3x3("conv1x1-laplacian", inp, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._reparameterized:
            return self.rep(x)
        y = (
            self.conv3x3(x)
            + self.conv1x1_3x3(x)
            + self.conv1x1_sbx(x)
            + self.conv1x1_sby(x)
            + self.conv1x1_lpl(x)
        )
        if self.with_idt:
            y = y + x
        return y

    @torch.no_grad()
    def reparameterize(self) -> None:
        if self._reparameterized:
            return
        w = self.conv3x3.weight.clone()
        b = self.conv3x3.bias.clone()
        for branch in (self.conv1x1_3x3, self.conv1x1_sbx, self.conv1x1_sby, self.conv1x1_lpl):
            bw, bb = branch.rep_params()
            w = w + bw
            b = b + bb
        if self.with_idt:
            # identity as a 3x3 kernel with 1 at center on the diagonal channel
            idt = torch.zeros_like(w)
            for c in range(self.out):
                idt[c, c, 1, 1] = 1.0
            w = w + idt

        rep = nn.Conv2d(self.inp, self.out, 3, padding=1)
        rep.weight.data.copy_(w)
        rep.bias.data.copy_(b)
        rep = rep.to(device=w.device, dtype=w.dtype)
        self.rep = rep
        del self.conv3x3, self.conv1x1_3x3, self.conv1x1_sbx, self.conv1x1_sby, self.conv1x1_lpl
        self._reparameterized = True

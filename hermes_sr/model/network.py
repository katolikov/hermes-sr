"""Full HERMES-SR network.

Mode A: Y-only input (1ch),  ×s reconstruction, 6 plain blocks.
Mode B: Y+noise-sigma input (2ch), ×s reconstruction, 6 blocks with SimpleGate
        residual in the last two.

The trunk fuses a warped 16-channel recurrent state with the stem output before
six HERMES blocks, then emits both an HR Y prediction (PixelShuffle + bicubic
anchor) and the next-frame state.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hermes_sr.model.blocks import HermesBlock
from hermes_sr.model.flow import DiamondSearchFlow, grid_warp


@dataclass
class HermesConfig:
    mode: str = "A"            # "A" or "B"
    upscale: int = 2
    in_channels: int = 1       # 1 for Mode A, 2 for Mode B (Y + noise-sigma)
    trunk_channels: int = 32
    ib_channels: int = 128
    num_blocks: int = 6
    state_channels: int = 16
    flow_down_factor: int = 8
    flow_levels: int = 2
    quantize_aware: bool = False  # TODO(QAT): structural hook only; not yet active


class HermesSR(nn.Module):
    def __init__(self, config: HermesConfig, reparameterized: bool = False) -> None:
        super().__init__()
        self.config = config
        self._reparameterized = reparameterized

        c = config.trunk_channels
        s = config.upscale
        sc = config.state_channels

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, c, 3, padding=1),
            nn.ReLU6(inplace=True),
        )
        self.state_fuse = nn.Conv2d(c + sc, c, 1)

        self.blocks = nn.ModuleList()
        for i in range(config.num_blocks):
            use_sg = (config.mode == "B") and (i >= config.num_blocks - 2)
            self.blocks.append(
                HermesBlock(
                    channels=c,
                    ib_channels=config.ib_channels,
                    use_simple_gate=use_sg,
                    reparameterized=reparameterized,
                )
            )

        self.state_head = nn.Conv2d(c, sc, 1)
        self.recon_head = nn.Conv2d(c, s * s, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(s)

        self.flow_estimator = DiamondSearchFlow(
            down_factor=config.flow_down_factor,
            num_levels=config.flow_levels,
        )

        if config.quantize_aware:
            # TODO(QAT): instantiate FakeQuantize stubs on the inputs/outputs of each
            # conv when a later session activates QAT. The architecture is already
            # free of ops that wouldn't quantize (no softmax, no in-place sigmoid,
            # no Python control flow inside a module forward except the no-grad
            # flow estimator, which is replaced at deploy).
            pass

    def forward(
        self,
        y_lr: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        y_prev: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one frame.

        y_lr:    (N, in_channels, H, W) — Y plane, or Y+sigma for Mode B
        h_prev:  (N, state_channels, H, W) or None (None means zeros)
        y_prev:  (N, 1, H, W) Y plane of previous frame, or None (None disables flow)
        """
        n, _, h, w = y_lr.shape
        device = y_lr.device

        if h_prev is None:
            h_prev = torch.zeros(n, self.config.state_channels, h, w, device=device, dtype=y_lr.dtype)
        if y_prev is None:
            flow = torch.zeros(n, 2, h, w, device=device, dtype=y_lr.dtype)
        else:
            # Use only the Y channel to estimate motion
            y_only_prev = y_prev[:, :1]
            y_only_curr = y_lr[:, :1]
            flow = self.flow_estimator(y_only_prev, y_only_curr)

        h_warped = grid_warp(h_prev, flow)

        x = self.stem(y_lr)
        x = self.state_fuse(torch.cat([x, h_warped], dim=1))
        for block in self.blocks:
            x = block(x)

        h_new = self.state_head(x)
        residual = self.pixel_shuffle(self.recon_head(x))

        # Bicubic anchor — Y channel only, no learnable params; kept in fp32 for stability
        y_only_lr = y_lr[:, :1]
        with torch.amp.autocast(device_type=device.type, enabled=False):
            y_anchor = F.interpolate(
                y_only_lr.float(),
                scale_factor=self.config.upscale,
                mode="bicubic",
                align_corners=False,
            )
        y_hr = residual + y_anchor.to(residual.dtype)
        return y_hr, h_new

    def reparameterize(self) -> None:
        for block in self.blocks:
            block.reparameterize()
        self._reparameterized = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    cfg_a = HermesConfig(mode="A", upscale=2, in_channels=1)
    cfg_b = HermesConfig(mode="B", upscale=3, in_channels=2)
    m_a = HermesSR(cfg_a)
    m_b = HermesSR(cfg_b)
    n_a = count_parameters(m_a)
    n_b = count_parameters(m_b)
    print(f"Mode A (×{cfg_a.upscale}, in={cfg_a.in_channels}): {n_a:,} parameters")
    print(f"Mode B (×{cfg_b.upscale}, in={cfg_b.in_channels}): {n_b:,} parameters")
    # Spec target is ~120K with ±20% tolerance. The literal architecture (32-ch
    # trunk, 128-ch IB) comes in lower than the target; see the parameter-count
    # test for the asserted range.

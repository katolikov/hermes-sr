"""Full HERMES-SR network.

Two trunk block types, selected by HermesConfig.block_type:
  "ecb"    : Edge-oriented Convolution Blocks (ECBSR-style). Each block collapses
             to a dense 3x3 conv at inference — best NPU utilization (ABPN class),
             with Sobel/Laplacian edge priors during training. Default.
  "hermes" : the MVP depthwise-large-kernel block with inverted bottleneck.

Mode A: Y-only input (1ch), ×s. Mode B: Y+noise-sigma input (2ch), ×s.

Trunk path:
  stem -> (optional) state_fuse with warped recurrent state -> N blocks
  -> recon head + PixelShuffle + anchor upsample = HR Y prediction
Recurrent state + flow path is optional (use_recurrent_state); off for static SR
and mobile deploy. Anchor is bilinear (NPU-native) or bicubic.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hermes_sr.model.blocks import HermesBlock
from hermes_sr.model.ecb import ECB
from hermes_sr.model.flow import DiamondSearchFlow, grid_warp


@dataclass
class HermesConfig:
    mode: str = "A"
    upscale: int = 2
    in_channels: int = 1
    trunk_channels: int = 32
    ib_channels: int = 128
    num_blocks: int = 6
    block_type: str = "ecb"           # "ecb" | "hermes"
    ecb_depth_multiplier: float = 2.0
    kernel_size: int = 7              # only used by block_type="hermes"
    state_channels: int = 16
    use_recurrent_state: bool = False
    flow_down_factor: int = 8
    flow_levels: int = 2
    anchor_mode: str = "bilinear"     # bilinear | bicubic
    quantize_aware: bool = False


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

        if config.use_recurrent_state:
            self.state_fuse = nn.Conv2d(c + sc, c, 1)
            self.state_head = nn.Conv2d(c, sc, 1)
            self.flow_estimator = DiamondSearchFlow(
                down_factor=config.flow_down_factor,
                num_levels=config.flow_levels,
            )
        else:
            self.state_fuse = None
            self.state_head = None
            self.flow_estimator = None

        self.blocks = nn.ModuleList()
        for i in range(config.num_blocks):
            if config.block_type == "ecb":
                self.blocks.append(
                    ECB(c, c, depth_multiplier=config.ecb_depth_multiplier,
                        with_idt=True, reparameterized=reparameterized)
                )
            elif config.block_type == "hermes":
                use_sg = (config.mode == "B") and (i >= config.num_blocks - 2)
                self.blocks.append(
                    HermesBlock(channels=c, ib_channels=config.ib_channels,
                                use_simple_gate=use_sg, reparameterized=reparameterized,
                                kernel_size=config.kernel_size)
                )
            else:
                raise ValueError(f"unknown block_type {config.block_type!r}")

        self.recon_head = nn.Conv2d(c, s * s, 3, padding=1)
        nn.init.zeros_(self.recon_head.weight)
        nn.init.zeros_(self.recon_head.bias)
        self.pixel_shuffle = nn.PixelShuffle(s)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
            if self.config.block_type == "ecb":
                x = F.relu(x)  # activation after each ECB (ECBSR-style)
        return x

    def forward(
        self,
        y_lr: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        y_prev: Optional[torch.Tensor] = None,
        return_feat: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        n, _, h, w = y_lr.shape
        device = y_lr.device

        x = self.stem(y_lr)

        if self.config.use_recurrent_state:
            if h_prev is None:
                h_prev = torch.zeros(
                    n, self.config.state_channels, h, w, device=device, dtype=y_lr.dtype
                )
            if y_prev is None:
                flow = torch.zeros(n, 2, h, w, device=device, dtype=y_lr.dtype)
            else:
                flow = self.flow_estimator(y_prev[:, :1], y_lr[:, :1])
            h_warped = grid_warp(h_prev, flow)
            x = self.state_fuse(torch.cat([x, h_warped], dim=1))

        feat = self._trunk(x)

        h_new: Optional[torch.Tensor] = None
        if self.config.use_recurrent_state:
            h_new = self.state_head(feat)

        residual = self.pixel_shuffle(self.recon_head(feat))

        y_only_lr = y_lr[:, :1]
        anchor_mode = self.config.anchor_mode
        with torch.amp.autocast(device_type=device.type, enabled=False):
            interp_kwargs: dict = {"scale_factor": self.config.upscale, "mode": anchor_mode}
            if anchor_mode in ("bilinear", "bicubic"):
                interp_kwargs["align_corners"] = False
            y_anchor = F.interpolate(y_only_lr.float(), **interp_kwargs)
        y_hr = residual + y_anchor.to(residual.dtype)

        if return_feat:
            return y_hr, h_new, feat
        return y_hr, h_new

    def reparameterize(self) -> None:
        for block in self.blocks:
            block.reparameterize()
        self._reparameterized = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import copy
    for mode, up, inc in [("A", 2, 1), ("B", 3, 2)]:
        cfg = HermesConfig(mode=mode, upscale=up, in_channels=inc)
        m = HermesSR(cfg)
        train_p = count_parameters(m)
        m_rep = copy.deepcopy(m)
        m_rep.reparameterize()
        inf_p = count_parameters(m_rep)
        print(f"Mode {mode} (×{up}, block={cfg.block_type}): "
              f"train {train_p:,} -> inference {inf_p:,} params")

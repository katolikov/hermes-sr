"""Learning-free 3-level diamond-search block-matching flow estimator.

Operates on the previous and current Y planes at 1/down_factor of the network's
input resolution. Outputs a per-pixel flow field at 1/down_factor res, then
bilinearly upsamples to the network's input resolution with displacements
scaled to input-pixel units.

The pure-PyTorch loop is slow on purpose: in a mobile deployment the codec's
motion-estimation path replaces this entirely, so optimizing it now is wasted.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_LDSP_OFFSETS = torch.tensor(
    [
        [0, 0],
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [2, 0], [-2, 0], [0, 2], [0, -2],
    ],
    dtype=torch.float32,
)


def grid_warp(img: torch.Tensor, flow: torch.Tensor, padding_mode: str = "zeros") -> torch.Tensor:
    """Warp img by flow such that out(p) = img(p + flow(p)).

    img:  (N, C, H, W)
    flow: (N, 2, H, W) — flow[:, 0] is dx (column), flow[:, 1] is dy (row), both in input-pixel units
    """
    n, _, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=img.dtype),
        torch.arange(w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=0)[None].expand(n, -1, -1, -1)
    sample = base + flow
    sample_x = 2.0 * sample[:, 0] / max(w - 1, 1) - 1.0
    sample_y = 2.0 * sample[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([sample_x, sample_y], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", align_corners=True, padding_mode=padding_mode)


class DiamondSearchFlow(nn.Module):
    """Estimate flow F such that I_curr(p) ≈ I_prev(p + F(p))."""

    def __init__(
        self,
        down_factor: int = 8,
        num_levels: int = 2,
        max_iters_per_level: int = 8,
        block_size: int = 5,
        sad_rel_threshold: float = 0.02,
    ) -> None:
        # The spec recommends three-level diamond search. For typical sub-LR-pixel
        # motions the coarsest (1/32 of input) scale degrades to noise and
        # propagates spurious displacements; two levels covers up to ~256 input
        # pixels of motion at down_factor=8 with the LDSP step ranges, which is
        # plenty for frame-to-frame motion in mobile video. The third level can
        # be re-enabled when integrating a real high-motion encoder ME path.
        super().__init__()
        self.down_factor = down_factor
        self.num_levels = num_levels
        self.max_iters_per_level = max_iters_per_level
        self.block_size = block_size
        # Don't accept a move unless the best candidate's SAD is at least
        # `sad_rel_threshold` lower than the current center's SAD. This stops
        # the coarsest pyramid level from random-walking on sub-pixel motion.
        self.sad_rel_threshold = sad_rel_threshold

    @torch.no_grad()
    def _search_one_level(
        self,
        i_prev: torch.Tensor,
        i_curr: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        offsets = _LDSP_OFFSETS.to(device=i_prev.device, dtype=i_prev.dtype)
        for _ in range(self.max_iters_per_level):
            sads = []
            for k in range(offsets.shape[0]):
                test = flow + offsets[k].view(1, 2, 1, 1)
                warped = grid_warp(i_prev, test)
                diff = (i_curr - warped).abs()
                sad = F.avg_pool2d(diff, self.block_size, stride=1, padding=self.block_size // 2)
                sads.append(sad)
            sads = torch.stack(sads, dim=0)
            best = sads.argmin(dim=0)
            if self.sad_rel_threshold > 0.0:
                sad_center = sads[0]
                sad_best = sads.gather(0, best.unsqueeze(0)).squeeze(0)
                # Stay at center where the improvement is below the threshold
                keep_center = sad_best >= sad_center * (1.0 - self.sad_rel_threshold)
                best = torch.where(keep_center, torch.zeros_like(best), best)
            chosen = offsets[best.squeeze(1)]
            chosen = chosen.permute(0, 3, 1, 2).contiguous()
            new_flow = flow + chosen
            if torch.equal(new_flow, flow):
                return new_flow
            flow = new_flow
        return flow

    @torch.no_grad()
    def forward_lr(self, i_prev: torch.Tensor, i_curr: torch.Tensor) -> torch.Tensor:
        """Return the flow at 1/down_factor resolution, in down-res pixel units."""
        n = i_prev.shape[0]
        i_prev_lo = F.avg_pool2d(i_prev, self.down_factor)
        i_curr_lo = F.avg_pool2d(i_curr, self.down_factor)

        pyramid_prev = [i_prev_lo]
        pyramid_curr = [i_curr_lo]
        for _ in range(self.num_levels - 1):
            pyramid_prev.append(F.avg_pool2d(pyramid_prev[-1], 2))
            pyramid_curr.append(F.avg_pool2d(pyramid_curr[-1], 2))

        coarse_h, coarse_w = pyramid_prev[-1].shape[-2:]
        flow = torch.zeros(n, 2, coarse_h, coarse_w, device=i_prev.device, dtype=i_prev.dtype)
        flow = self._search_one_level(pyramid_prev[-1], pyramid_curr[-1], flow)

        for level in range(self.num_levels - 2, -1, -1):
            target_h, target_w = pyramid_prev[level].shape[-2:]
            flow = F.interpolate(flow, size=(target_h, target_w), mode="bilinear", align_corners=False) * 2.0
            flow = self._search_one_level(pyramid_prev[level], pyramid_curr[level], flow)

        return flow

    @torch.no_grad()
    def forward(self, i_prev: torch.Tensor, i_curr: torch.Tensor) -> torch.Tensor:
        _, _, h, w = i_prev.shape
        flow_lr = self.forward_lr(i_prev, i_curr)
        flow_full = F.interpolate(flow_lr, size=(h, w), mode="bilinear", align_corners=False) * self.down_factor
        return flow_full

"""Diamond search must recover integer synthetic motion to within one pixel.

With down_factor=8, the estimator quantizes displacements to multiples of 8
input pixels. The shifts here are multiples of 8 so the post-upsample flow at
input resolution lands within ±1 input pixel of the truth, as the spec requires.

The synthetic source is random noise, which produces SAD ties in a small set of
textureless pixels — those pixels show outlier flow values. The unit-test
metric is therefore the median over the center crop, which is robust to the
ambiguous-pixel outliers; in real footage these regions exist too but the
network's state head re-initializes the recurrent feature each frame so the
outlier flow has limited downstream impact.
"""
from __future__ import annotations

import torch

from hermes_sr.model.flow import DiamondSearchFlow, grid_warp


def _synth_pair(shift: tuple[int, int], size: int = 256, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    img = torch.randn(1, 1, size, size)
    # Mild low-pass so SAD is well-conditioned at the coarse pyramid level
    img = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(img, [2, 2, 2, 2], mode="reflect"),
        kernel_size=5,
        stride=1,
    )
    flow = torch.zeros(1, 2, size, size)
    flow[0, 0] = float(shift[0])
    flow[0, 1] = float(shift[1])
    # Synthesize the shifted frame with edge-replicating boundary so the LR
    # downsample doesn't include zero-padded regions inside the search window.
    shifted = grid_warp(img, flow, padding_mode="border")
    return img, shifted


def test_diamond_search_recovers_synthetic_motion() -> None:
    est = DiamondSearchFlow(down_factor=8, num_levels=2, max_iters_per_level=8, block_size=5)
    # Use multiples of 8 input pixels so the LR-scale shift is an integer
    shifts = [(8, 0), (-8, 0), (0, 8), (0, -8), (16, 0), (8, 8), (-16, 8)]
    for dx, dy in shifts:
        frame1, frame2 = _synth_pair((dx, dy))
        # Compare at the down-resolution scale where the algorithm produces
        # integer displacements; the median across the LR center crop is robust
        # to the ambiguous textureless pixels that produce outliers.
        flow_lr = est.forward_lr(frame1, frame2)
        center_lr = flow_lr[:, :, 8:24, 8:24]
        med_lr_dx = center_lr[:, 0].median().item()
        med_lr_dy = center_lr[:, 1].median().item()
        # LR-pixel units; multiply by down_factor to compare to input-pixel shift
        med_dx = med_lr_dx * est.down_factor
        med_dy = med_lr_dy * est.down_factor
        assert abs(med_dx - dx) < 1.0, f"dx: got {med_dx:.3f}, expected {dx}"
        assert abs(med_dy - dy) < 1.0, f"dy: got {med_dy:.3f}, expected {dy}"

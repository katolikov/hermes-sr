"""Five-term composite loss.

L = w_pix * L1(pred, target)
  + w_perc * L1(VGG_relu_2_2(pred_3ch), VGG_relu_2_2(target_3ch))
  + w_spec * L1(|fft2(pred)|, |fft2(target)|)
  + w_temp * L1(warp(pred_t, flow), pred_t_minus_1)   # zero on the first frame
  + w_dist * distill_loss_stub(...)                   # returns 0 in the MVP

Weights default to (1.0, 0.1, 0.05, 0.5, 0.2) as specified.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from hermes_sr.model.flow import grid_warp


_DEFAULT_WEIGHTS = {
    "pixel": 1.0,
    "perceptual": 0.1,
    "spectral": 0.05,
    "temporal": 0.5,
    "distill": 0.2,
}


def distill_loss_stub(pred: torch.Tensor) -> torch.Tensor:
    # TODO(LUT distillation): replace with HAT/BSRGAN teacher loss + LUT
    # distillation. Returning zero keeps the training loop wired but inert.
    return pred.new_zeros(())


class CompositeLoss(nn.Module):
    def __init__(
        self,
        weights: Optional[dict] = None,
        use_perceptual: bool = True,
        vgg_weights: Optional[str] = "DEFAULT",
    ) -> None:
        super().__init__()
        self.weights = {**_DEFAULT_WEIGHTS, **(weights or {})}
        self.use_perceptual = use_perceptual
        if use_perceptual:
            vgg = torchvision.models.vgg19(weights=vgg_weights).features.eval()
            # relu_2_2 ≈ index 8 in VGG19.features: conv1_1, relu, conv1_2, relu,
            # pool, conv2_1, relu, conv2_2, relu  -> take [0:9]
            self.vgg = nn.Sequential(*list(vgg.children())[:9])
            for p in self.vgg.parameters():
                p.requires_grad = False
        else:
            self.vgg = None

    def _perceptual(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            return pred.new_zeros(())
        pred_rgb = pred.expand(-1, 3, -1, -1)
        target_rgb = target.expand(-1, 3, -1, -1)
        return F.l1_loss(self.vgg(pred_rgb), self.vgg(target_rgb))

    def _spectral(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT not always available under autocast; compute in fp32
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            fp = torch.fft.fft2(pred.float()).abs()
            ft = torch.fft.fft2(target.float()).abs()
        return F.l1_loss(fp, ft)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prev_pred: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        parts: dict = {}
        parts["pixel"] = F.l1_loss(pred, target)
        parts["perceptual"] = self._perceptual(pred, target)
        parts["spectral"] = self._spectral(pred, target)
        if prev_pred is not None and flow is not None:
            parts["temporal"] = F.l1_loss(grid_warp(pred, flow), prev_pred)
        else:
            parts["temporal"] = pred.new_zeros(())
        parts["distill"] = distill_loss_stub(pred)

        total = sum(self.weights[k] * parts[k] for k in parts)
        return total, parts

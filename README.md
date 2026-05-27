# HERMES-SR — research MVP

Y-channel mobile super-resolution. ~85K params, six-block trunk with reparameterizable depthwise large-kernel branches, a learning-free diamond-search flow estimator warping a 16-channel recurrent state, and a PixelShuffle head on a bicubic anchor. Mode A: single-image ×2. Mode B: noise-aware ×3 with a noise-sigma side channel and SimpleGate residual in the last two blocks.

**Status:** MVP architecture verification complete on M1 Pro MPS at 20K iters. 4 of 5 acceptance criteria met. See [RESULTS.md](RESULTS.md) for trajectories, per-image breakdowns, and the Mode B shortfall analysis. Convergence run on CUDA at 200K iters is pending. The design document lives separately.

## Install

```bash
pip install torch torchvision numpy pillow pytest
```

## Train

```bash
python -m hermes_sr.train --config configs/mode_a.json
python -m hermes_sr.train --config configs/mode_b.json
```

Datasets under `~/datasets`. On Apple Silicon set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Tests

```bash
pytest hermes_sr/tests/
```

## Deploy checkpoint

```bash
python -m hermes_sr.scripts.reparameterize \
  --in checkpoints/hermes_a_iter20000.pt --out checkpoints/hermes_a_deploy.pt
```

## Out of scope

LUT distillation, real QAT, ONNX/ENN export, chroma, Zurich RAW, GAN loss.

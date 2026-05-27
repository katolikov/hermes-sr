# HERMES-SR — research MVP

Y-channel mobile super-resolution. ~85K params, reparameterizable depthwise large-kernel trunk, learning-free diamond-search flow estimator warping a 16-channel recurrent state, PixelShuffle head on a bicubic anchor. Mode A: clean ×2. Mode B: noise-aware ×3.

![Mode A butterfly ×2 comparison](docs/butterfly_x2_comparison.png)
![Mode B baby ×3 σ=25 comparison](docs/baby_x3_comparison.png)

Top: Mode A `butterfly` ×2 — **27.23** vs bicubic **26.99 dB** (+0.24). Bottom: Mode B `baby` ×3 σ=25 — **22.22** vs **21.70 dB** (+0.52). L→R: LR / bicubic / model / HR.

**Status:** MVP verified on M1 Pro MPS at 20K iters; 4/5 criteria met. See [RESULTS.md](RESULTS.md). CUDA convergence run pending.

## Install + test

```bash
pip install -e .
pytest hermes_sr/tests/
```

## MVP train (MPS / CPU)

```bash
python -m hermes_sr.train --config configs/mode_a.json
python -m hermes_sr.train --config configs/mode_b.json
```

Datasets under `~/datasets`. On Apple Silicon set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## CUDA convergence run

GPU server with PyTorch + CUDA already set up:

```bash
./run_mode_a_train.sh   # 600K iters, ~3.75 hr on RTX 4090
./run_mode_b_train.sh   # 300K iters, warm-starts from Mode A, ~3.5 hr on RTX 4090
```

Auto-downloads to `~/datasets/`. Deploy ckpts at `checkpoints/hermes_*_deploy_convergence.pt`; results at `results/`.

## Deploy

```bash
python -m hermes_sr.scripts.reparameterize \
  --in checkpoints/<training>.pt --out checkpoints/<deploy>.pt
```

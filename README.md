# HERMES-SR

Mobile Y-plane super-resolution for **real camera sensor buffers** (YUV 4:2:0).
The network upscales the full-res Y plane (where detail lives); U/V are bicubic-
upsampled outside. Trunk is reparameterizable Edge-oriented Convolution Blocks
(ECB) that collapse to a **dense 3×3 at deploy** (NPU-optimal, ~57K params), with
a recurrent feature state warped frame-to-frame for video temporal stability and
a noise-sigma side channel for joint denoise + SR.

![ECB ×2 butterfly](docs/butterfly_x2_comparison.png)

Architecture reference (clean bicubic ×2, Set5, 20K MPS): ECB **35.07 dB** vs depthwise-MVP 32.91 (bicubic 33.11). L→R: LR / bicubic / model / HR.

## Real-world sensor pipeline

Degradation models the actual sensor buffer (`hermes_sr/data/degrade.py`): optical blur → downsample → **Poisson-Gaussian sensor noise** (shot + read, wide ISO range). **No JPEG** — it's a raw buffer. Two-stage Real-ESRGAN-style training:

```
Stage 1 (Net):  ECB + recurrent + L1 + temporal + EMA   → PSNR base
Stage 2 (GAN):  + U-Net disc, hinge + perceptual         → sharp on real photos
```

## Train

MPS / single GPU:
```bash
python -m hermes_sr.train     --config configs/sensor_b_x2.json       # Stage 1
python -m hermes_sr.train_gan --config configs/sensor_b_x2_gan.json   # Stage 2
```

CUDA convergence (downloads data, runs both stages, reparameterizes):
```bash
./run_sensor_train.sh 2    # x2 model
./run_sensor_train.sh 3    # x3 model
```

## Test

```bash
pytest hermes_sr/tests/
```

On Apple Silicon set `PYTORCH_ENABLE_MPS_FALLBACK=1`. Deploy via `hermes_sr.scripts.reparameterize` (dense-3×3, same speed).

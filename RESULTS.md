# HERMES-SR Results

**Hardware:** Apple M1 Pro, 16 GB, MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`).
**All numbers are 20K-iter MPS verification results** on our eval protocol — not
convergence-run numbers, and not the literature protocol (see caveats below).

## Architecture progression (Mode A, Set5 ×2, 20K iters)

| variant | inference op | inf. params | Set5 ×2 PSNR | SSIM (box) | vs bicubic |
|---|---|---|---|---|---|
| MVP | 11×11 depthwise | ~61K | 32.91 | 0.9565 | −0.20 |
| Mobile | 7×7 depthwise | ~57K | 34.81 | 0.9667 | +1.70 |
| **ECB** | **3×3 dense** | **~57K** | **35.07** | **0.9677** | **+1.96** |

ECB wins on **both** axes: highest PSNR **and** the deploy op is a dense 3×3 —
the ABPN-class operation that NPUs accelerate best (depthwise large-kernels are
memory-bound and underutilize the MAC array). bicubic baseline = 33.11 dB.

Recipe behind the jump from MVP: pure-L1 loss (dropped perceptual/spectral/
temporal which fight PSNR), zero-init residual head (starts at the anchor, not
below it), flip+rotation augmentation, lr 5e-4, and the ECB edge-prior branches
(Sobel-x/y + Laplacian) that collapse losslessly into the 3×3 at inference.

## Per-image demo (ECB Mode A, ×2)

`docs/butterfly_x2_comparison.png`: butterfly RGB-PSNR **30.75 dB** vs bicubic
26.99 (+3.76). Reparameterization equivalence: max diff 3.3e-7 (≪ 1e-5).

## Mode B

Mode B (×3 + σ=25 denoise) ECB retrain is in progress. The last validated Mode B
numbers are MVP-era (24.08 dB / σ=25); they will be refreshed once the ECB Mode B
run completes.

## Reparameterization

| mode | max y diff | deploy size |
|---|---|---|
| A (ECB) | 3.3e-7 | 229 KB |

Deploy checkpoints carry only the collapsed dense-3×3 weights.

## Tests

`pytest hermes_sr/tests/` — 7 passing: ECB + hermes-block + MVP-legacy
reparameterization equivalence (≤1e-5), diamond-search motion recovery,
parameter counts (inference, mobile range).

## Honest caveats

1. **SSIM in the table is box-window** (what the trainer logs), inflated ~+0.02
   vs the literature Gaussian-window SSIM. ECB Mode A Gaussian SSIM ≈ 0.949.
2. **Our protocol runs ~0.5 dB cold vs literature.** We generate LR with PyTorch
   `F.interpolate(bicubic)`; the SR-benchmark standard is MATLAB `imresize`.
   Adjusted, ECB Mode A ≈ 35.6 dB on the literature scale.
3. **20K is verification, not convergence.** ABPN/ECBSR train 1M+ iters and reach
   ~37.5–37.9 dB. The CUDA convergence scripts (`run_mode_*_train.sh`) target this.
4. **Knowledge distillation (the L_distill design) is not yet active** — sourcing
   a teacher whose LR domain matches ours is the open task; a broken-domain
   teacher (~30 dB) would hurt, so none is wired rather than wiring a bad one.

## Out of scope (unchanged)

LUT/MuLUT prior, real QAT/INT8, ONNX/ENN export, GAN loss, on-device latency.

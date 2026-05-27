#!/usr/bin/env bash
# HERMES-SR Mode B convergence run: 100K iters ×3 + denoise, warm-started from
# the Mode A deploy checkpoint. Assumes Python + PyTorch + CUDA are already set
# up on this host. Requires a Mode A checkpoint to exist.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
pip install -e . --quiet

# Mode B warm-start prerequisite
INIT_CONV="checkpoints/hermes_a_deploy_convergence.pt"
INIT_MVP="checkpoints/hermes_a_deploy.pt"
if [ -f "$INIT_CONV" ]; then
    INIT_FROM="$INIT_CONV"
elif [ -f "$INIT_MVP" ]; then
    INIT_FROM="$INIT_MVP"
else
    cat <<'EOF'
ERROR: Mode B requires a trained Mode A checkpoint for warm-start.
Run ./run_mode_a_train.sh first.
EOF
    exit 1
fi

mkdir -p ~/datasets logs results checkpoints

DATASETS="$HOME/datasets"

download_zip() {
    local url="$1" zip_path="$2" target_dir="$3"
    if [ -d "$target_dir" ]; then
        return 0
    fi
    echo ">>> downloading $(basename "$target_dir") ..."
    wget --continue -O "$zip_path" "$url"
    unzip -t "$zip_path" >/dev/null || { echo "ERROR: $zip_path failed integrity check"; exit 1; }
    unzip -q "$zip_path" -d "$(dirname "$target_dir")"
    rm -f "$zip_path"
}

# DIV2K train HR (required — used as Mode B fallback when SIDD is unavailable)
download_zip \
    "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" \
    "$DATASETS/DIV2K_train_HR.zip" \
    "$DATASETS/DIV2K_train_HR"

# Set5
if [ ! -d "$DATASETS/Set5/HR" ] && [ ! -d "$DATASETS/Set5/Set5_HR" ]; then
    echo ">>> downloading Set5 ..."
    SET5_ZIP="$DATASETS/Set5.zip"
    wget --continue -O "$SET5_ZIP" \
        "https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip"
    unzip -t "$SET5_ZIP" >/dev/null || { echo "ERROR: Set5 zip failed integrity check"; exit 1; }
    mkdir -p "$DATASETS/Set5"
    unzip -q "$SET5_ZIP" -d "$DATASETS/Set5"
    rm -f "$SET5_ZIP"
fi

# Urban100 (optional eval)
if [ ! -d "$DATASETS/Urban100" ]; then
    echo ">>> downloading Urban100 ..."
    URBAN_ZIP="$DATASETS/Urban100.zip"
    if wget --continue -O "$URBAN_ZIP" \
        "https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip"; then
        if unzip -t "$URBAN_ZIP" >/dev/null 2>&1; then
            mkdir -p "$DATASETS/Urban100"
            unzip -q "$URBAN_ZIP" -d "$DATASETS/Urban100"
        else
            echo "WARN: Urban100 zip failed integrity check; skipping."
        fi
        rm -f "$URBAN_ZIP"
    else
        echo "WARN: Urban100 download failed; eval will skip it."
        rm -f "$URBAN_ZIP"
    fi
fi

# SIDD validation — URL is unstable. We attempt nothing here; Mode B's dataloader
# falls back to DIV2K-with-synthetic-Gaussian (the same regime the MVP used).
echo ">>> SIDD validation download skipped: URL is unstable. Mode B will train"
echo "    on DIV2K with synthetic Gaussian noise augmentation (same as MVP)."

# Convergence config
python -c "
import json
with open('configs/mode_b.json') as f:
    cfg = json.load(f)
cfg['max_iters'] = 100000
cfg['eval_every'] = 5000
cfg['save_every'] = 10000
cfg['ckpt_prefix'] = 'mode_b_convergence_iter_'
cfg['init_from'] = '$INIT_FROM'
cfg['data_root'] = '$HOME/datasets'
cfg['set5_root'] = '$HOME/datasets'
with open('configs/mode_b_convergence.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('configs/mode_b_convergence.json written (init_from=$INIT_FROM)')
"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/mode_b_convergence_${TIMESTAMP}.log"

cat <<EOF
=====================================================
HERMES-SR Mode B Convergence Run
=====================================================
GPU:           ${GPU}
VRAM:          ${VRAM}
Iterations:    100000 (warm-start)
Warm-start:    ${INIT_FROM}
Expected time: ~70 min on RTX 4090, ~40 min on H100, ~2 hr on RTX 3090
Output ckpt:   checkpoints/hermes_b_deploy_convergence.pt
Log:           ${LOG}
=====================================================
Starting in 5 seconds. Ctrl-C to abort.
EOF
sleep 5

trap 'echo "Interrupted. Latest checkpoint under checkpoints/mode_b_convergence_iter_*.pt"; exit 130' INT TERM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m hermes_sr.train --config configs/mode_b_convergence.json 2>&1 | tee "${LOG}"

# Reparameterize the final training checkpoint into the deploy artifact.
python -m hermes_sr.scripts.reparameterize \
    --in checkpoints/mode_b_convergence_iter_100000.pt \
    --out checkpoints/hermes_b_deploy_convergence.pt

# Full evaluation at three noise levels (sigma on 0–255 scale)
python -m hermes_sr.eval \
    --checkpoint checkpoints/hermes_b_deploy_convergence.pt \
    --datasets set5 \
    --noise-sigmas 15,25,50 \
    --data-root "$HOME/datasets" \
    --output results/mode_b_convergence_results.json

DEPLOY_SIZE=$(du -h checkpoints/hermes_b_deploy_convergence.pt | awk '{print $1}')
SUMMARY=$(python -c "
import json
r = json.load(open('results/mode_b_convergence_results.json'))['metrics']
def row(k):
    if k not in r: return f'  {k}: (not in results)'
    return f'  {k}: {r[k][\"psnr\"]:.2f} dB / {r[k][\"ssim\"]:.4f} SSIM'
for k in ('set5_x3_sigma15', 'set5_x3_sigma25', 'set5_x3_sigma50'):
    print(row(k))
")

cat <<EOF
=====================================================
Mode B Convergence Run COMPLETE
=====================================================
Deploy checkpoint: checkpoints/hermes_b_deploy_convergence.pt
File size:         ${DEPLOY_SIZE}
Warm-started from: ${INIT_FROM}

${SUMMARY}

Full results: results/mode_b_convergence_results.json
Training log: ${LOG}
=====================================================
EOF

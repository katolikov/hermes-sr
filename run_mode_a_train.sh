#!/usr/bin/env bash
# HERMES-SR Mode A convergence run: 200K iters ×2 SR from scratch.
# Assumes Python + PyTorch + CUDA are already set up on this host.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate an existing project venv if one is sitting at .venv; otherwise install
# into whatever Python is on PATH. Do not create a venv.
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
pip install -e . --quiet

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

# DIV2K train HR (required)
download_zip \
    "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" \
    "$DATASETS/DIV2K_train_HR.zip" \
    "$DATASETS/DIV2K_train_HR"

# Set5 (required for eval)
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

# Flickr2K (optional augmentation)
if [ ! -d "$DATASETS/Flickr2K" ]; then
    echo ">>> downloading Flickr2K (optional) ..."
    FLICKR_TAR="$DATASETS/Flickr2K.tar"
    if wget --continue -O "$FLICKR_TAR" "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"; then
        if tar -tf "$FLICKR_TAR" >/dev/null 2>&1; then
            tar -xf "$FLICKR_TAR" -C "$DATASETS"
        else
            echo "WARN: Flickr2K tar corrupt; continuing with DIV2K only."
        fi
        rm -f "$FLICKR_TAR"
    else
        echo "WARN: Flickr2K download failed; continuing with DIV2K only."
        rm -f "$FLICKR_TAR"
    fi
fi

# Convergence config — read MVP config, bump iters, point checkpoint prefix at the right name.
python -c "
import json
with open('configs/mode_a.json') as f:
    cfg = json.load(f)
cfg['max_iters'] = 600000
cfg['eval_every'] = 10000
cfg['save_every'] = 25000
cfg['ckpt_prefix'] = 'mode_a_convergence_iter_'
cfg['data_root'] = '$HOME/datasets'
cfg['set5_root'] = '$HOME/datasets'
with open('configs/mode_a_convergence.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('configs/mode_a_convergence.json written')
"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/mode_a_convergence_${TIMESTAMP}.log"

cat <<EOF
=====================================================
HERMES-SR Mode A Convergence Run
=====================================================
GPU:           ${GPU}
VRAM:          ${VRAM}
Iterations:    600000
Expected time: ~3.75 hr on RTX 4090, ~2.25 hr on H100, ~6 hr on RTX 3090
Output ckpt:   checkpoints/hermes_a_deploy_convergence.pt
Log:           ${LOG}
=====================================================
Starting in 5 seconds. Ctrl-C to abort.
EOF
sleep 5

trap 'echo "Interrupted. Latest checkpoint under checkpoints/mode_a_convergence_iter_*.pt"; exit 130' INT TERM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m hermes_sr.train --config configs/mode_a_convergence.json 2>&1 | tee "${LOG}"

# Reparameterize the final training checkpoint into the deploy artifact.
python -m hermes_sr.scripts.reparameterize \
    --in checkpoints/mode_a_convergence_iter_600000.pt \
    --out checkpoints/hermes_a_deploy_convergence.pt

# Full evaluation
python -m hermes_sr.eval \
    --checkpoint checkpoints/hermes_a_deploy_convergence.pt \
    --datasets set5,urban100 \
    --data-root "$HOME/datasets" \
    --output results/mode_a_convergence_results.json

# Final summary
DEPLOY_SIZE=$(du -h checkpoints/hermes_a_deploy_convergence.pt | awk '{print $1}')
SUMMARY=$(python -c "
import json
r = json.load(open('results/mode_a_convergence_results.json'))['metrics']
def row(k):
    if k not in r: return f'  {k}: (not in results)'
    return f'  {k}: {r[k][\"psnr\"]:.2f} dB / {r[k][\"ssim\"]:.4f} SSIM'
print(row('set5_x2'))
print(row('urban100_x2'))
")

cat <<EOF
=====================================================
Mode A Convergence Run COMPLETE
=====================================================
Deploy checkpoint: checkpoints/hermes_a_deploy_convergence.pt
File size:         ${DEPLOY_SIZE}

${SUMMARY}

Full results: results/mode_a_convergence_results.json
Training log: ${LOG}
=====================================================
EOF

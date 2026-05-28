#!/usr/bin/env bash
# HERMES-SR real-world sensor-video SR convergence run for a ready CUDA box.
# Two-stage Real-ESRGAN-style recipe per scale:
#   Stage 1 (Net): ECB Mode B + recurrent state, L1 + temporal + EMA
#   Stage 2 (GAN): warm-start from Stage 1, + U-Net disc, hinge + perceptual
#   -> reparameterize the Stage-2 generator to a dense-3x3 deploy checkpoint
#
# Usage:
#   ./run_sensor_train.sh 2      # x2 model
#   ./run_sensor_train.sh 3      # x3 model
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SCALE="${1:-2}"
if [ "$SCALE" != "2" ] && [ "$SCALE" != "3" ]; then
    echo "ERROR: scale must be 2 or 3 (got '$SCALE')"
    exit 1
fi

STAGE1_ITERS="${STAGE1_ITERS:-300000}"
STAGE2_ITERS="${STAGE2_ITERS:-150000}"

if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
pip install -e . --quiet

mkdir -p ~/datasets logs results checkpoints
DATASETS="$HOME/datasets"

download_zip() {
    local url="$1" zip_path="$2" target_dir="$3"
    if [ -d "$target_dir" ]; then return 0; fi
    echo ">>> downloading $(basename "$target_dir") ..."
    wget --continue -O "$zip_path" "$url"
    unzip -t "$zip_path" >/dev/null || { echo "ERROR: $zip_path integrity check failed"; exit 1; }
    unzip -q "$zip_path" -d "$(dirname "$target_dir")"
    rm -f "$zip_path"
}

# DIV2K (required HR source)
download_zip \
    "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" \
    "$DATASETS/DIV2K_train_HR.zip" "$DATASETS/DIV2K_train_HR"

# Set5 (eval proxy)
if [ ! -d "$DATASETS/Set5/HR" ] && [ ! -d "$DATASETS/Set5/Set5_HR" ]; then
    echo ">>> downloading Set5 ..."
    wget --continue -O "$DATASETS/Set5.zip" \
        "https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip"
    unzip -t "$DATASETS/Set5.zip" >/dev/null || { echo "ERROR: Set5 zip failed"; exit 1; }
    mkdir -p "$DATASETS/Set5"; unzip -q "$DATASETS/Set5.zip" -d "$DATASETS/Set5"; rm -f "$DATASETS/Set5.zip"
fi

# Flickr2K (optional extra HR source for the sensor degradation)
if [ ! -d "$DATASETS/Flickr2K" ]; then
    echo ">>> downloading Flickr2K (optional) ..."
    if wget --continue -O "$DATASETS/Flickr2K.tar" "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"; then
        if tar -tf "$DATASETS/Flickr2K.tar" >/dev/null 2>&1; then
            tar -xf "$DATASETS/Flickr2K.tar" -C "$DATASETS"
        else
            echo "WARN: Flickr2K tar corrupt; continuing with DIV2K only."
        fi
        rm -f "$DATASETS/Flickr2K.tar"
    else
        echo "WARN: Flickr2K download failed; continuing with DIV2K only."
        rm -f "$DATASETS/Flickr2K.tar"
    fi
fi

STAGE1_CKPT="checkpoints/sensor_b_x${SCALE}_iter${STAGE1_ITERS}.pt"
STAGE2_CKPT="checkpoints/sensor_b_x${SCALE}_gan_iter${STAGE2_ITERS}.pt"
DEPLOY_CKPT="checkpoints/hermes_sensor_x${SCALE}_deploy.pt"

# Stage-1 convergence config (derive from the MVP-scale sensor config)
python -c "
import json
c = json.load(open('configs/sensor_b_x${SCALE}.json'))
c['max_iters'] = ${STAGE1_ITERS}
c['eval_every'] = 10000
c['save_every'] = ${STAGE1_ITERS}
c['data_root'] = '$HOME/datasets'
c['set5_root'] = '$HOME/datasets'
json.dump(c, open('configs/sensor_b_x${SCALE}_convergence.json','w'), indent=4)
print('Stage-1 convergence config written')
"

# Stage-2 GAN convergence config (warm-start from the Stage-1 output)
python -c "
import json
c = json.load(open('configs/sensor_b_x${SCALE}_gan.json'))
c['max_iters'] = ${STAGE2_ITERS}
c['eval_every'] = 5000
c['save_every'] = ${STAGE2_ITERS}
c['init_from'] = '${STAGE1_CKPT}'
c['data_root'] = '$HOME/datasets'
c['set5_root'] = '$HOME/datasets'
json.dump(c, open('configs/sensor_b_x${SCALE}_gan_convergence.json','w'), indent=4)
print('Stage-2 GAN convergence config written (init_from=${STAGE1_CKPT})')
"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
TS=$(date +%Y%m%d_%H%M%S)

cat <<EOF
=====================================================
HERMES-SR Sensor-Video SR Convergence — x${SCALE}
=====================================================
GPU:        ${GPU}    VRAM: ${VRAM}
Stage 1:    ${STAGE1_ITERS} iters (ECB Net base: L1 + temporal + EMA)
Stage 2:    ${STAGE2_ITERS} iters (GAN fine-tune: + disc, hinge + perceptual)
Deploy:     ${DEPLOY_CKPT}
=====================================================
Starting in 5 seconds. Ctrl-C to abort.
EOF
sleep 5

trap 'echo "Interrupted. Latest checkpoints under checkpoints/sensor_b_x${SCALE}_*"; exit 130' INT TERM

echo ">>> Stage 1 (Net base) ..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m hermes_sr.train --config "configs/sensor_b_x${SCALE}_convergence.json" \
    2>&1 | tee "logs/sensor_b_x${SCALE}_stage1_${TS}.log"

echo ">>> Stage 2 (GAN fine-tune) ..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m hermes_sr.train_gan --config "configs/sensor_b_x${SCALE}_gan_convergence.json" \
    2>&1 | tee "logs/sensor_b_x${SCALE}_stage2_${TS}.log"

echo ">>> Reparameterize to dense-3x3 deploy checkpoint ..."
python -m hermes_sr.scripts.reparameterize --in "${STAGE2_CKPT}" --out "${DEPLOY_CKPT}"

echo ">>> Evaluation (Set5 proxy, multi-noise) ..."
python -m hermes_sr.eval --checkpoint "${DEPLOY_CKPT}" --datasets set5 \
    --noise-sigmas 15,25,50 --data-root "$HOME/datasets" \
    --output "results/sensor_x${SCALE}_results.json"

cat <<EOF
=====================================================
Sensor-Video SR x${SCALE} COMPLETE
=====================================================
Deploy checkpoint: ${DEPLOY_CKPT}  ($(du -h "${DEPLOY_CKPT}" 2>/dev/null | awk '{print $1}'))
Stage-1 log: logs/sensor_b_x${SCALE}_stage1_${TS}.log
Stage-2 log: logs/sensor_b_x${SCALE}_stage2_${TS}.log
Results:     results/sensor_x${SCALE}_results.json
=====================================================
EOF

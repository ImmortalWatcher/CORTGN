#!/bin/bash
# 下载图片 -> 生成 pkl -> 训练（全程在数据盘）
# 用法: bash deploy/autodl/run_pipeline.sh
# 可选: DOWNLOAD_WORKERS=8 BATCHSIZE=32 GPU=0 bash deploy/autodl/run_pipeline.sh

set -e

DATA="${AUTODL_DATA:-/root/autodl-tmp}"
PROJECT="${DATA}/DAMMFND"
VENV="${PROJECT}/.venv"
export DAMMFND_DATA="${PROJECT}"
export PIP_CACHE_DIR="${DATA}/pip-cache"
export TMPDIR="${DATA}/tmp"

source "${VENV}/bin/activate"
cd "${PROJECT}/src"

WORKERS="${DOWNLOAD_WORKERS:-8}"
BATCH="${BATCHSIZE:-32}"
GPU="${GPU:-0}"
EPOCH="${EPOCH:-50}"

mkdir -p "${PROJECT}/logs"

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  echo "========== 1/3 下载图片到数据盘 weibo/ =========="
  python download_weibo_images.py --weibo-dir "${PROJECT}/weibo" --workers "${WORKERS}" --delay 0.05 \
    2>&1 | tee "${PROJECT}/logs/download.log"
else
  echo "========== 1/3 跳过下载（SKIP_DOWNLOAD=1，使用已有 weibo/*_images）=========="
fi

echo "========== 2/3 生成 MAE + CLIP pkl（需 GPU）=========="
python weibo_build_pkl.py --split train val test \
  2>&1 | tee "${PROJECT}/logs/build_pkl.log"

echo "========== 3/3 训练 =========="
python run_weibo.py --gpu "${GPU}" --batchsize "${BATCH}" --epoch "${EPOCH}" \
  2>&1 | tee "${PROJECT}/logs/train.log"

echo "训练结束。权重: ${PROJECT}/param_model/dammfnd/parameter_dammfnd.pkl"

#!/bin/bash
# pkl 已生成后单独训练
set -e
DATA="${AUTODL_DATA:-/root/autodl-tmp}"
PROJECT="${DATA}/DAMMFND"
export DAMMFND_DATA="${PROJECT}"
source "${PROJECT}/.venv/bin/activate"
cd "${PROJECT}/src"
mkdir -p "${PROJECT}/logs"
GPU="${GPU:-0}"
BATCH="${BATCHSIZE:-32}"
EPOCH="${EPOCH:-50}"
python run_weibo.py --gpu "${GPU}" --batchsize "${BATCH}" --epoch "${EPOCH}" \
  2>&1 | tee "${PROJECT}/logs/train.log"

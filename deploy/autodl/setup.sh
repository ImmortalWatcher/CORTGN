#!/bin/bash
# AutoDL 一键环境：虚拟环境、PyTorch、依赖 均放在数据盘
# 用法: bash deploy/autodl/setup.sh

set -e

# 数据盘（AutoDL 默认；若实例不同可 export AUTODL_DATA=你的路径）
DATA="${AUTODL_DATA:-/root/autodl-tmp}"
PROJECT="${DATA}/DAMMFND"
VENV="${PROJECT}/.venv"

echo "========== AutoDL 环境安装 =========="
echo "数据盘: ${DATA}"
echo "项目目录: ${PROJECT}"

mkdir -p "${PROJECT}"
cd "${PROJECT}"

if [[ ! -d "${PROJECT}/src" ]]; then
  echo "[error] 未找到 ${PROJECT}/src，请先把代码上传到数据盘（见 deploy/autodl/UPLOAD.md）"
  exit 1
fi

# 避免 pip/缓存 撑爆系统盘
export PIP_CACHE_DIR="${DATA}/pip-cache"
export TMPDIR="${DATA}/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi
source "${VENV}/bin/activate"

pip install -U pip wheel

# RTX 50 系需 sm_120，用 cu128；旧卡可改 cu118/cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 项目实际用 cn_clip，不依赖 GitHub openai/CLIP（国内常 clone 失败）
pip install cn_clip transformers pandas ftfy regex tqdm scikit-learn positional_encodings openpyxl timm pillow
# numpy 与 torch 2.8 兼容即可，不 pin 1.23.2
pip install "numpy>=1.23.2,<2.3" 2>/dev/null || pip install "numpy>=1.23.2"

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 预训练放数据盘，链到 src（若已上传）
PRE="${DATA}/pretrained_model/chinese_roberta_wwm_base_ext_pytorch"
if [[ -d "${PRE}" ]]; then
  ln -sfn "${PRE}" "${PROJECT}/src/pretrained_model/chinese_roberta_wwm_base_ext_pytorch"
  echo "已链接 RoBERTa -> src/pretrained_model/"
else
  echo "[warn] 未找到 ${PRE}，请上传 chinese_roberta_wwm_base_ext_pytorch（含 pytorch_model.bin）"
fi

echo ""
echo "完成。以后请先执行:"
echo "  export DAMMFND_DATA=${PROJECT}"
echo "  source ${VENV}/bin/activate"
echo "  cd ${PROJECT}/src"

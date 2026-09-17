#!/bin/bash
echo "========== AutoDL 环境诊断 =========="
echo "主机: $(hostname)"
echo "用户: $(whoami)"
echo "PWD:  $(pwd)"
echo ""
echo "磁盘:"
df -h | grep -E 'Filesystem|/root|autodl|overlay' || df -h
echo ""
DATA="${AUTODL_DATA:-/root/autodl-tmp}"
echo "AUTODL_DATA=${DATA}"
ls -la "${DATA}" 2>/dev/null | head -15 || echo "数据盘目录不可读"
echo ""
PROJ="${DATA}/DAMMFND"
if [[ -d "${PROJ}" ]]; then
  echo "项目 ${PROJ}:"
  ls -la "${PROJ}"
  echo ""
  ls -la "${PROJ}/src" 2>/dev/null | head -8
  ls -la "${PROJ}/weibo" 2>/dev/null | head -8
else
  echo "[缺失] ${PROJ} 不存在，需要先上传代码"
fi
echo ""
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "nvidia-smi 不可用"
fi
echo ""
python3 --version 2>/dev/null || echo "python3 未找到"

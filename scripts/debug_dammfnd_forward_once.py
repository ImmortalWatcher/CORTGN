#!/usr/bin/env python3
"""
用真实 DAMMFNDMODEL 跑 1 次 forward，打印模块一各阶段 shape。

需要：CUDA、BERT/MAE 权重、Chinese-CLIP（首次可能下载）。

用法（项目根目录）：
  set DAMMFND_DEBUG_SHAPES=1
  python scripts/debug_dammfnd_forward_once.py
  python scripts/debug_dammfnd_forward_once.py --batch 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.environ.setdefault("DAMMFND_DEBUG_SHAPES", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=197)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA。请在有 GPU 的环境运行。")

    import cn_clip.clip as clip
    from model.dammfnd import DAMMFNDMODEL

    B = args.batch
    max_len = args.max_len
    bert = str(SRC / "pretrained_model" / "chinese_roberta_wwm_base_ext_pytorch")

    print(f"加载 DAMMFNDMODEL（batch={B}）…")
    model = DAMMFNDMODEL(emb_dim=768, mlp_dims=[384], bert=bert, out_channels=320, dropout=0.2)
    model = model.cuda().eval()

    dummy_texts = ["这是一条测试新闻"] * B
    clip_text = clip.tokenize(dummy_texts).cuda()

    batch_data = {
        "content": torch.randint(100, 30000, (B, max_len), device="cuda"),
        "content_masks": torch.ones(B, max_len, device="cuda"),
        "image": torch.randn(B, 3, 224, 224, device="cuda"),
        "clip_image": torch.randn(B, 3, 224, 224, device="cuda"),
        "clip_text": clip_text,
    }
    batch_data["content_masks"][:, max_len // 2 :] = 0

    print("\n--- 开始 forward（下方 [DEBUG shapes] 为模块一关键张量）---\n")
    with torch.no_grad():
        out = model(**batch_data)
    print(f"\n--- forward 完成 ---")
    print(f"最终预测 fake_news_sigmoid: {tuple(out[0].shape)}  示例值: {out[0][: min(3, B)].cpu().tolist()}")


if __name__ == "__main__":
    main()

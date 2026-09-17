#!/usr/bin/env python3
"""
模块一 shape 调试：用真实项目里的 MaskAttention / cnn_extractor 跑一遍，打印每一步张量形状。

不依赖完整预训练权重，用随机张量模拟 [batch, 197, 768] 等输入。

用法（在项目根目录）：
  python scripts/debug_module1_shapes.py
  python scripts/debug_module1_shapes.py --batch 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from model.layers import MaskAttention, cnn_extractor  # noqa: E402


def header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_batch_dim(batch: int) -> None:
    header("1. 第一维 B（batch_size）是什么意思？")
    print(
        f"""
batch_size = {batch} 表示：这一次 forward 同时处理 {batch} 条新闻（{batch} 个样本）。

张量形状总是 [B, ...] 开头：
  - B=1  → 只处理 1 条新闻（调试时常用）
  - B=64 → 训练时默认一次塞 64 条（见 run_weibo.py --batchsize 64）
  - B 可以是任意正整数，由 DataLoader 的 batch_size 决定

代码里注释写的 [64, 197, 768] 只是「训练时 batch=64」的举例；
你这次调试用的是 B={batch}，所以会是 [{batch}, 197, 768]。
"""
    )


def demo_why_numbers(batch: int, max_len: int, hidden: int) -> None:
    header("2. 197、768、320 这些数字从哪来？")
    print(
        f"""
| 数字 | 含义 | 在本项目里谁定的 |
|------|------|------------------|
| {max_len} | 每条新闻最多多少个 token（词/子词位置） | run_weibo.py / run.py 里 max_len={max_len} |
| {hidden} | BERT 每个 token 的向量维度（隐藏层大小） | RoBERTa-base / BERT-base 标准配置 |
| 320 | TextCNN 输出维度 | 5 个卷积核 × 每个 64 通道 = 5×64=320 |

197 的构成（典型）：
  [CLS] + 正文 token + [SEP] + ...  总长 padding 到 {max_len}

768 的构成：
  预训练语言模型规定「每个词用 768 个数描述」，不是本项目手写的。

320 的构成：
  feature_kernel = {{1:64, 2:64, 3:64, 5:64, 10:64}}  →  5×64=320
"""
    )


def demo_attention(batch: int, max_len: int, hidden: int, device: torch.device) -> torch.Tensor:
    header("3. 注意力之后：197 个词变成了什么？")
    attn = MaskAttention(hidden).to(device)
    text_feature = torch.randn(batch, max_len, hidden, device=device)
    masks = torch.ones(batch, max_len, device=device)
    masks[:, max_len // 2 :] = 0  # 后半段当作 padding，演示 mask 作用

    print(f"输入 text_feature: {tuple(text_feature.shape)}  →  B 条新闻，每条 {max_len} 个词，每词 {hidden} 维")
    print(f"输入 masks:        {tuple(masks.shape)}  →  1=真词，0=padding（模型会忽略）")

    text_atn = attn(text_feature, masks)
    print(f"输出 text_atn:     {tuple(text_atn.shape)}  →  B 条新闻，每条只剩 1 个 {hidden} 维向量")

    print(
        """
结论：197 个词并没有「消失成 197 个东西」，而是加权合成 1 个句子向量。
  - 之前：197 行 × 768 列  （197 个词，每个 768 维）
  - 之后：1 行 × 768 列    （整句话 1 个摘要向量）

类比：197 个证人各自发言（768 维），注意力给每人打分，按权重混合成「一条总结」。

注意（很重要）：
  - text_atn_feature 给门控网络用（后面模块二）
  - TextCNN 吃的是完整的 text_feature [B,197,768]，不是 attention 后的向量！
"""
    )
    return text_atn


def demo_textcnn(batch: int, max_len: int, hidden: int, device: torch.device) -> None:
    header("4. TextCNN（cnn_extractor）逐步 shape 变化")
    feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
    cnn = cnn_extractor(hidden, feature_kernel).to(device)
    x = torch.randn(batch, max_len, hidden, device=device)

    print(f"Step 0 输入 x:              {tuple(x.shape)}")
    x_perm = x.permute(0, 2, 1)
    print(f"Step 1 permute 后:          {tuple(x_perm.shape)}  （Conv1d 要求 [B, 通道, 长度]）")

    conv_shapes = []
    for k, conv in zip(feature_kernel.keys(), cnn.convs):
        y = conv(x_perm)
        conv_shapes.append((k, tuple(y.shape)))
    print("Step 2 五个 Conv1d 各自输出（核宽度 k）：")
    for k, shape in conv_shapes:
        print(f"         kernel={k:2d}  →  {shape}   （{shape[1]} 个通道，长度约 {max_len}-k+1）")

    pooled = []
    for k, conv in zip(feature_kernel.keys(), cnn.convs):
        y = conv(x_perm)
        p = torch.max_pool1d(y, y.shape[-1])
        pooled.append(p)
        print(f"Step 3 kernel={k:2d} max_pool 整条序列 → {tuple(p.shape)}  （每个通道压成 1 个数）")

    out = cnn(x)
    print(f"Step 4 cat 五个池化结果:    {tuple(out.shape)}  （5×64=320）")
    print(
        """
一句话：在 197 个词上滑 5 种不同宽度的窗口，每种抽 64 个特征，再对整条序列取最大值，拼成 320 维。
"""
    )


def demo_textcnn_slide_ascii() -> None:
    header("5. TextCNN 五个卷积核在一句话上怎么滑（示意图）")
    sentence = ["[CLS]", "据", "报道", "某市", "将", "封城", "消息", "不实", "[SEP]", "", "..."]
    print("示例句子（逻辑位置，实际已 tokenize）：")
    print("  " + " | ".join(f"{i}:{w}" for i, w in enumerate(sentence[:9])))
    print(
        r"""
每个卷积核只在「连续 k 个词」上局部看模式，在整条序列上从左滑到右：

kernel=1（看 1 个词）
  [据] [报道] [某市] [将] [封城] ...
   ↑    ↑     ↑     ↑    ↑
  每个位置独立扫

kernel=2（看连续 2 个词）
  [据|报道] [报道|某市] [某市|将] [将|封城] ...
       ↑         ↑          ↑

kernel=3（看连续 3 个词）
  [据|报道|某市] [报道|某市|将] ...
         ↑              ↑

kernel=5
  [据|报道|某市|将|封城] [报道|某市|将|封城|消息] ...

kernel=10
  [据|报道|...共10个词...] 一次看更长短语

每个 kernel 有 64 个「探测器」（64 通道）→ 在序列每个滑动位置得到 64 个数
→ 对整条 197 长度做 max pooling：每个通道只保留「全句最强响应」→ 64 个数
→ 5 个 kernel 拼起来 → 64×5 = 320 维向量（代表这一条新闻的 TextCNN 特征）

Mermaid 流程图（可在支持 Mermaid 的编辑器里预览）：
"""
    )
    print(
        """```mermaid
flowchart TB
    subgraph input [输入一条新闻 BERT 输出]
        T["text_feature [B, 197, 768]"]
    end
    subgraph permute [维度换位]
        P["[B, 768, 197]"]
    end
    subgraph convs [5 个 Conv1d 并行]
        C1["k=1 → [B,64,L1]"]
        C2["k=2 → [B,64,L2]"]
        C3["k=3 → [B,64,L3]"]
        C5["k=5 → [B,64,L5]"]
        C10["k=10 → [B,64,L10]"]
    end
    subgraph pool [每条序列全局 max pool]
        M["5 × [B,64,1]"]
    end
    subgraph out [拼接]
        O["[B, 320]"]
    end
    T --> P
    P --> C1 & C2 & C3 & C5 & C10
    C1 & C2 & C3 & C5 & C10 --> M --> O
```"""
    )


def try_full_model_forward(batch: int) -> None:
    header("6. 尝试完整 DAMMFND forward（需要 GPU + 预训练权重）")
    bert_dir = SRC / "pretrained_model" / "chinese_roberta_wwm_base_ext_pytorch"
    mae_ckpt = SRC / "mae_pretrain_vit_base.pth"
    missing = []
    if not (bert_dir / "pytorch_model.bin").exists() and not list(bert_dir.glob("*.safetensors")):
        missing.append(f"BERT 权重（{bert_dir}）")
    if not mae_ckpt.exists():
        missing.append(f"MAE 权重（{mae_ckpt}）")

    if missing:
        print("跳过完整模型 forward，缺少：")
        for m in missing:
            print(f"  - {m}")
        print("\n上面 1–5 步已用真实 layers.py 代码 + 随机输入跑通。")
        print("补齐权重后设置环境变量再跑：")
        print("  set DAMMFND_DEBUG_SHAPES=1")
        print("  python run_weibo.py --batchsize 2 --epoch 1")
        return

    if not torch.cuda.is_available():
        print("跳过：需要 CUDA")
        return

    os.environ["DAMMFND_DEBUG_SHAPES"] = "1"
    print("权重齐全时可运行 run_weibo.py 并查看 dammfnd.py 内 [DEBUG shapes] 打印。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2, help="调试用 batch，建议 2 或 4")
    parser.add_argument("--max-len", type=int, default=197)
    parser.add_argument("--hidden", type=int, default=768)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    demo_batch_dim(args.batch)
    demo_why_numbers(args.batch, args.max_len, args.hidden)
    demo_attention(args.batch, args.max_len, args.hidden, device)
    demo_textcnn(args.batch, args.max_len, args.hidden, device)
    demo_textcnn_slide_ascii()
    try_full_model_forward(args.batch)


if __name__ == "__main__":
    main()

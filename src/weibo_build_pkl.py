"""
为 weibo 多模态训练生成 MAE / CLIP 两套 pkl，并输出与 pkl 行数对齐的 CSV。

默认目录（可用环境变量 DAMMFND_DATA 覆盖）：
  {DATA}/weibo/train_2_domain.csv
  {DATA}/weibo/nonrumor_images/
  {DATA}/weibo/rumor_images/
  -> {DATA}/weibo/train_loader.pkl, train_clip_loader.pkl, train_aligned.csv

在 src 目录执行：
  python weibo_build_pkl.py --split train
  python weibo_build_pkl.py --split val test
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

_SPLIT_TO_CSV = {
    "train": "train_2_domain.csv",
    "val": "val_2_domain.csv",
    "test": "test_2_domain.csv",
}


def _data_root() -> str:
    return os.environ.get("DAMMFND_DATA", os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))


def _stem_from_url(part: str) -> str | None:
    part = part.strip()
    if not part or part.lower() == "null":
        return None
    return part.split("/")[-1].split(".")[0].split("?")[0].lower()


def load_mae_image_bank(weibo_dir: str) -> dict[str, torch.Tensor]:
    """与 clip_dataloader.read_image 一致：ImageNet 归一化 [3,224,224]。"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    bank: dict[str, torch.Tensor] = {}
    for sub in ("nonrumor_images", "rumor_images"):
        folder = os.path.join(weibo_dir, sub)
        if not os.path.isdir(folder):
            print(f"[warn] 目录不存在: {folder}")
            continue
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            try:
                im = Image.open(path).convert("RGB")
                stem = filename.split(".")[0].lower()
                bank[stem] = transform(im)
            except Exception as e:
                print(f"[warn] MAE 读图失败 {path}: {e}")
    print(f"[mae] 图片库大小: {len(bank)}")
    return bank


def load_clip_image_bank(weibo_dir: str, device: torch.device) -> dict[str, torch.Tensor]:
    """与 clip_data_pre.read_image 一致：Chinese-CLIP preprocess。"""
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name

    clip_cache = os.path.join(_data_root(), "pretrained_model", "cn_clip")
    os.makedirs(clip_cache, exist_ok=True)
    _, preprocess = load_from_name(
        "ViT-B-16", device=device, download_root=clip_cache, use_modelscope=True
    )
    bank: dict[str, torch.Tensor] = {}
    for sub in ("nonrumor_images", "rumor_images"):
        folder = os.path.join(weibo_dir, sub)
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            try:
                im = Image.open(path).convert("RGB")
                stem = filename.split(".")[0].lower()
                t = preprocess(im).unsqueeze(0).to(device)
                bank[stem] = t
            except Exception as e:
                print(f"[warn] CLIP 读图失败 {path}: {e}")
    print(f"[clip] 图片库大小: {len(bank)}")
    return bank


def align_csv_with_images(csv_path: str, mae_bank: dict, clip_bank: dict) -> tuple[pd.DataFrame, list[torch.Tensor], list]:
    df = pd.read_csv(csv_path, encoding="utf-8")
    mae_tensors: list[torch.Tensor] = []
    clip_tensors: list = []
    kept_rows = []

    for i in range(len(df)):
        row = df.iloc[i]
        stem_found = None
        for part in str(row["image_id"]).split("|"):
            stem = _stem_from_url(part)
            if stem and stem in mae_bank and stem in clip_bank:
                stem_found = stem
                break
        if stem_found is None:
            continue
        mae_tensors.append(mae_bank[stem_found])
        clip_tensors.append(clip_bank[stem_found])
        kept_rows.append(row)

    aligned = pd.DataFrame(kept_rows).reset_index(drop=True)
    return aligned, mae_tensors, clip_tensors


def save_split(weibo_dir: str, split: str, device: torch.device) -> None:
    csv_name = _SPLIT_TO_CSV[split]
    csv_path = os.path.join(weibo_dir, csv_name)
    if not os.path.isfile(csv_path):
        print(f"[error] 找不到 {csv_path}")
        sys.exit(1)

    print(f"\n=== 处理 {split}: {csv_path} ===")
    mae_bank = load_mae_image_bank(weibo_dir)
    clip_bank = load_clip_image_bank(weibo_dir, device)
    aligned, mae_list, clip_list = align_csv_with_images(csv_path, mae_bank, clip_bank)

    n_raw = len(pd.read_csv(csv_path, encoding="utf-8"))
    n_keep = len(aligned)
    print(f"CSV 原始 {n_raw} 行 -> 有图且对齐 {n_keep} 行 ({100 * n_keep / max(n_raw, 1):.1f}%)")

    if n_keep == 0:
        print("[error] 没有可对齐样本，请先下载图片到 nonrumor_images / rumor_images")
        sys.exit(1)

    mae_stack = torch.stack(mae_list, dim=0)
    clip_stack = torch.tensor([t.cpu().detach().numpy() for t in clip_list]).squeeze(1)

    aligned_path = os.path.join(weibo_dir, f"{split}_aligned.csv")
    mae_pkl = os.path.join(weibo_dir, f"{split}_loader.pkl")
    clip_pkl = os.path.join(weibo_dir, f"{split}_clip_loader.pkl")

    aligned.to_csv(aligned_path, index=False, encoding="utf-8")
    with open(mae_pkl, "wb") as f:
        pickle.dump(mae_stack, f)
    with open(clip_pkl, "wb") as f:
        pickle.dump(clip_stack, f)

    print(f"  -> {aligned_path}")
    print(f"  -> {mae_pkl}  shape={tuple(mae_stack.shape)}")
    print(f"  -> {clip_pkl}  shape={tuple(clip_stack.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--weibo-dir", default=None, help="默认 $DAMMFND_DATA/weibo")
    args = parser.parse_args()

    data_root = _data_root()
    weibo_dir = args.weibo_dir or os.path.join(data_root, "weibo")
    weibo_dir = os.path.normpath(weibo_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DAMMFND_DATA =", data_root)
    print("weibo_dir    =", weibo_dir)
    print("device       =", device)

    for split in args.split:
        if split not in _SPLIT_TO_CSV:
            print(f"[warn] 未知 split: {split}")
            continue
        save_split(weibo_dir, split, device)


if __name__ == "__main__":
    main()

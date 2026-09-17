"""本地 Chinese-CLIP 加载（ModelScope，避免训练时访问 HuggingFace）。"""
import os


def clip_download_root() -> str:
    data = os.environ.get(
        "DAMMFND_DATA",
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    root = os.path.join(data, "pretrained_model", "cn_clip")
    os.makedirs(root, exist_ok=True)
    return root


def load_cn_clip(device):
    from cn_clip.clip import load_from_name

    return load_from_name(
        "ViT-B-16",
        device=device,
        download_root=clip_download_root(),
        use_modelscope=True,
    )

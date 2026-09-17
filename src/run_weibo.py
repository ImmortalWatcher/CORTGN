"""
在 AutoDL / 自定义数据盘上训练 weibo 多模态（使用 *_aligned.csv 与 weibo/*.pkl）。

环境变量：
  DAMMFND_DATA  项目数据根，默认 ../ 相对 src；AutoDL 建议 /root/autodl-tmp/DAMMFND

用法（在 src 下）：
  export DAMMFND_DATA=/root/autodl-tmp/DAMMFND
  python run_weibo.py --gpu 0 --batchsize 32
"""

import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="dammfnd")
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--max_len", type=int, default=197)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--early_stop", type=int, default=10)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--seed", type=int, default=3074)
parser.add_argument("--gpu", default="0")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--emb_type", default="bert")
parser.add_argument("--save_param_dir", default=None, help="默认 $DAMMFND_DATA/param_model")
args = parser.parse_args()

_DATA = os.environ.get(
    "DAMMFND_DATA",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
)
_SRC = os.path.dirname(os.path.abspath(__file__))
_BERT = os.path.join(_SRC, "pretrained_model", "chinese_roberta_wwm_base_ext_pytorch")
_VOCAB = os.path.join(_BERT, "vocab.txt")
_WEIBO = os.path.join(_DATA, "weibo")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch
from run import Run

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)

emb_dim = 768
vocab_file = _VOCAB
save_param_dir = args.save_param_dir or os.path.join(_DATA, "param_model")

print("DAMMFND_DATA:", _DATA)
print("weibo dir:", _WEIBO)
print("save_param_dir:", save_param_dir)

config = {
    "use_cuda": True,
    "batchsize": args.batchsize,
    "max_len": args.max_len,
    "early_stop": args.early_stop,
    "num_workers": args.num_workers,
    "vocab_file": vocab_file,
    "emb_type": args.emb_type,
    "bert": _BERT,
    "root_path": _WEIBO + os.sep,
    "weight_decay": 5e-5,
    "model": {"mlp": {"dims": [384], "dropout": 0.2}},
    "emb_dim": emb_dim,
    "lr": args.lr,
    "epoch": args.epoch,
    "model_name": args.model_name,
    "seed": args.seed,
    "save_param_dir": save_param_dir,
    "dataset": "weibo",
}


class RunWeibo(Run):
    """使用 weibo/*_aligned.csv 与对应 pkl。"""

    def __init__(self, config):
        super().__init__(config)
        root = config["root_path"]
        self.train_path = os.path.join(root, "train_aligned.csv")
        self.val_path = os.path.join(root, "val_aligned.csv")
        self.test_path = os.path.join(root, "test_aligned.csv")
        self._pkl = {
            "train": (os.path.join(root, "train_loader.pkl"), os.path.join(root, "train_clip_loader.pkl")),
            "val": (os.path.join(root, "val_loader.pkl"), os.path.join(root, "val_clip_loader.pkl")),
            "test": (os.path.join(root, "test_loader.pkl"), os.path.join(root, "test_clip_loader.pkl")),
        }

    def get_dataloader(self, dataset):
        from utils.clip_dataloader import bert_data as weibo_data

        loader = weibo_data(
            max_len=self.max_len,
            batch_size=self.batchsize,
            vocab_file=self.vocab_file,
            category_dict=self.category_dict,
            num_workers=self.num_workers,
        )
        train_loader = loader.load_data(self.train_path, *self._pkl["train"], True)
        val_loader = loader.load_data(self.val_path, *self._pkl["val"], False)
        test_loader = loader.load_data(self.test_path, *self._pkl["test"], False)
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    RunWeibo(config=config).main()

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

# 默认与仓库 layout 一致：…/DAMMFND/data2（由 run_text 根据 config 覆盖）
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.normpath(os.path.join(_SRC_DIR, '..', '..', 'data2'))
DATA_DIR = _DEFAULT_DATA_DIR


def set_data_dir(root_path):
    """root_path 为数据目录（含 train_loader.pkl 的文件夹），如 ../data/ 或 D:/.../data/。"""
    global DATA_DIR
    DATA_DIR = os.path.normpath(os.path.abspath(os.path.expanduser(str(root_path).rstrip('/\\'))))

class TextDataset(Dataset):
    """纯文本数据集 - 适配 data_pre.py 生成的 pkl"""
    
    def __init__(self, pkl_path):
        print(f"Loading pkl: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # 基础字段
        self.token_ids = self.data['token_ids']      # [N, 197]
        self.masks = self.data['masks']              # [N, 197]
        self.labels = self.data['labels']            # [N]
        self.post_ids = self.data.get('post_id', None)

        n = len(self.token_ids)
        if 'category' in self.data:
            self.category_primary = self.data['category']
        else:
            self.category_primary = torch.zeros(n, dtype=torch.long)

        if 'multi_category' in self.data:
            self.multi_cat = self.data['multi_category']
        else:
            # 仅有主领域 id 时构造 [N,9] multi-hot（与 run_text 的 weibo 9 类一致）
            self.multi_cat = torch.zeros(n, 9, dtype=torch.float32)
            for i in range(n):
                c = int(self.category_primary[i].item())
                if 0 <= c < 9:
                    self.multi_cat[i, c] = 1.0

        self.len = n
        print(f"Loaded {self.len} samples")
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        item = {
            'content': self.token_ids[idx],           # 对应 model.forward 的 content
            'content_masks': self.masks[idx],         # 对应 model.forward 的 content_masks
            'label': self.labels[idx],
            'category': self.category_primary[idx],
            'multi_category': self.multi_cat[idx],
        }
        if self.post_ids is not None:
            item['post_id'] = self.post_ids[idx]
        return item

def get_dataloader(pkl_path, batch_size=32, shuffle=True, num_workers=0):
    """创建 DataLoader"""
    dataset = TextDataset(pkl_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    return dataloader

def get_train_loader(batch_size=32):
    return get_dataloader(os.path.join(DATA_DIR, 'train_loader.pkl'), batch_size, shuffle=True)

def get_val_loader(batch_size=32):
    return get_dataloader(os.path.join(DATA_DIR, 'val_loader.pkl'), batch_size, shuffle=False)

def get_test_loader(batch_size=32):
    return get_dataloader(os.path.join(DATA_DIR, 'test_loader.pkl'), batch_size, shuffle=False)

def data2gpu(batch, use_cuda=True):
    """将 batch 数据移到 GPU（与 Trainer.use_cuda 一致）。"""
    if not use_cuda or not torch.cuda.is_available():
        return batch
    if isinstance(batch, dict):
        return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    elif isinstance(batch, list):
        return [data2gpu(item) for item in batch]
    else:
        return batch
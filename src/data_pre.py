import pickle
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import torch
import os

def _init_fn(worker_id):
    np.random.seed(2021)

def df_filter(df_data):
    """过滤无效类别"""
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data

def word2input(texts, vocab_file, max_len):
    """文本转 BERT 输入"""
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(tokenizer.encode(
            text, 
            max_length=max_len, 
            add_special_tokens=True, 
            padding='max_length',
            truncation=True
        ))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i, token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids, masks

class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=2, out_dir=None):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict
        self.out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        self.out_dir = os.path.normpath(self.out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

    def _domain_labels(self, post):
        """与 clip_dataloader 一致：主领域 + 领域列 → multi_category [N, 9]。"""
        cat_col = post['category'].astype(str)
        if '领域' in post.columns:
            domain2 = post['领域'].fillna('无领域').astype(str)
        else:
            domain2 = pd.Series(['无领域'] * len(post))

        category_idx = torch.tensor(
            [self.category_dict[str(c).strip()] for c in cat_col],
            dtype=torch.long,
        )
        categories = []
        for cat1, cat2 in zip(cat_col, domain2):
            c1 = cat1.strip()
            c2 = str(cat2).strip()
            if c2 in ('无领域', 'nan', ''):
                categories.append([self.category_dict[c1]])
            else:
                categories.append([self.category_dict[c1], self.category_dict[c2]])

        num_domains = 9
        labels_multi_domain = torch.zeros(len(categories), num_domains)
        for i, cat_list in enumerate(categories):
            for j in cat_list:
                labels_multi_domain[i, j] = 1.0
        return category_idx, labels_multi_domain

    def load_data_val(self, path, shuffle, text_only=True):
        """加载验证集 - 纯文本"""
        self.data = pd.read_csv(path, encoding='utf-8')
        post = self.data
        
        # ✅ 使用正确的列名
        texts = post['post_text'].fillna('').tolist()
        labels = post['label'].fillna(0).tolist()
        post_id = post['post_id'].tolist()
        category_idx, mul_category = self._domain_labels(post)
        
        # 文本转 BERT 输入
        token_ids, masks = word2input(texts, self.vocab_file, self.max_len)
        
        # 保存数据
        data_dict = {
            'token_ids': token_ids,
            'masks': masks,
            'labels': torch.tensor(labels),
            'category': category_idx,
            'multi_category': mul_category,
            'post_id': post_id
        }
        
        with open(os.path.join(self.out_dir, 'val_loader.pkl'), 'wb') as file:
            pickle.dump(data_dict, file)
        
        print(f"[ok] val: {len(texts)} samples")
        return 1

    def load_data_test(self, path, shuffle, text_only=True):
        """加载测试集 - 纯文本"""
        self.data = pd.read_csv(path, encoding='utf-8')
        post = self.data
        
        # ✅ 使用正确的列名
        texts = post['post_text'].fillna('').tolist()
        labels = post['label'].fillna(0).tolist()
        post_id = post['post_id'].tolist()
        category_idx, mul_category = self._domain_labels(post)
        
        # 文本转 BERT 输入
        token_ids, masks = word2input(texts, self.vocab_file, self.max_len)
        
        # 保存数据
        data_dict = {
            'token_ids': token_ids,
            'masks': masks,
            'labels': torch.tensor(labels),
            'category': category_idx,
            'multi_category': mul_category,
            'post_id': post_id
        }
        
        with open(os.path.join(self.out_dir, 'test_loader.pkl'), 'wb') as file:
            pickle.dump(data_dict, file)
        
        print(f"[ok] test: {len(texts)} samples")
        return 1

    def load_data_train(self, path, shuffle, text_only=True):
        """加载训练集 - 纯文本"""
        self.data = pd.read_csv(path, encoding='utf-8')
        post = self.data
        
        # ✅ 使用正确的列名
        texts = post['post_text'].fillna('').tolist()
        labels = post['label'].fillna(0).tolist()
        post_id = post['post_id'].tolist()
        category_idx, mul_category = self._domain_labels(post)
        
        # 文本转 BERT 输入
        token_ids, masks = word2input(texts, self.vocab_file, self.max_len)
        
        # 保存数据
        data_dict = {
            'token_ids': token_ids,
            'masks': masks,
            'labels': torch.tensor(labels),
            'category': category_idx,
            'multi_category': mul_category,
            'post_id': post_id
        }
        
        with open(os.path.join(self.out_dir, 'train_loader.pkl'), 'wb') as file:
            pickle.dump(data_dict, file)
        
        print(f"[ok] train: {len(texts)} samples")
        return 1

# 类别字典（须与 run_text.py / clip_dataloader 中 weibo 的 9 类一致）
category_dict = {
    "经济": 0,
    "健康": 1,
    "军事": 2,
    "科学": 3,
    "政治": 4,
    "国际": 5,
    "教育": 6,
    "娱乐": 7,
    "社会": 8,
}

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_SRC_DIR, '..', 'data'))
_VOCAB = os.path.join(_SRC_DIR, 'pretrained_model', 'chinese_roberta_wwm_base_ext_pytorch', 'vocab.txt')

# 初始化数据加载器
loader = bert_data(
    max_len=197, 
    batch_size=64, 
    vocab_file=_VOCAB,
    category_dict=category_dict, 
    num_workers=1,
    out_dir=_DATA_DIR,
)

# 处理数据集
print("Running data_pre...")
val_loader = loader.load_data_val(os.path.join(_DATA_DIR, 'val_origin.csv'), True)
test_loader = loader.load_data_test(os.path.join(_DATA_DIR, 'test_origin.csv'), True)
train_loader = loader.load_data_train(os.path.join(_DATA_DIR, 'train_origin.csv'), True)

print("Done. Wrote train/val/test_loader.pkl to:", _DATA_DIR)
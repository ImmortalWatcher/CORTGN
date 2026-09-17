# 在 src 目录下执行: python main_text.py
import os
import argparse
import torch
import numpy as np
import random
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ================= 1. 参数配置 =================
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='dammfnd_text')
parser.add_argument('--dataset', default='weibo')       # weibo21 或 weibo
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=197)
parser.add_argument('--num_workers', type=int, default=0)  # Windows 建议设为 0 避免多进程报错
parser.add_argument('--early_stop', type=int, default=10)
parser.add_argument(
    '--bert',
    default=os.path.join(_SRC, 'pretrained_model', 'chinese_roberta_wwm_base_ext_pytorch'),
    help='HuggingFace 模型名或本地目录（与 data_pre 使用的词表一致时用本地）',
)
parser.add_argument('--root_path', default=os.path.join(_SRC, '..', 'data2') + os.sep)
parser.add_argument('--batchsize', type=int, default=32)   # 文本版本显存占用小，但建议适中
parser.add_argument('--seed', type=int, default=3074)
parser.add_argument('--gpu', default='0')
parser.add_argument('--bert_emb_dim', type=int, default=768)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--w2v_emb_dim', type=int, default=200)
parser.add_argument('--save_param_dir', default='./param_model_text') # 单独保存文本模型参数

args = parser.parse_args()

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 设置随机种子
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

# 设置 Embedding 维度
if args.emb_type == 'bert':
    emb_dim = args.bert_emb_dim
else:
    emb_dim = args.w2v_emb_dim

print('=' * 60)
print('Text-only DAMMFND training')
print(f'lr: {args.lr}; model: {args.model_name}; batchsize: {args.batchsize}; epoch: {args.epoch}')
print(f'gpu: {args.gpu}; emb_dim: {emb_dim}; dataset: {args.dataset}')
print('=' * 60)

# ================= 2. 配置字典 =================
config = {
    'use_cuda': torch.cuda.is_available(),
    'batchsize': args.batchsize,
    'max_len': args.max_len,
    'early_stop': args.early_stop,
    'num_workers': args.num_workers,
    'emb_type': args.emb_type,
    'bert': args.bert,
    'root_path': args.root_path,
    'weight_decay': 5e-5,
    'model': {
        'mlp': {'dims': [384], 'dropout': 0.2}
    },
    'emb_dim': emb_dim,
    'lr': args.lr,
    'epoch': args.epoch,
    'model_name': args.model_name,
    'seed': args.seed,
    'save_param_dir': args.save_param_dir,
    'dataset': args.dataset
}

# ================= 3. 启动运行 =================
if __name__ == '__main__':
    from run_text import Run
    Run(config=config).main()
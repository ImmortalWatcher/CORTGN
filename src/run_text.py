# D:/deep-learning/DAMMFND/src/run_text.py
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from model.dammfnd_text import Trainer as TextTrainer
from utils.text_dataloader import get_train_loader, get_val_loader, get_test_loader, set_data_dir

class Run():
    def __init__(self, config):
        self.configinfo = config

        # 解析配置
        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']
        self.dataset = config['dataset']
        
        # 数据集类别字典配置
        if config['dataset'] == "weibo":
            self.category_dict = {
                "经济": 0, "健康": 1, "军事": 2, "科学": 3, "政治": 4,
                "国际": 5, "教育": 6, "娱乐": 7, "社会": 8
            }
        elif config['dataset'] == "weibo21":
            self.category_dict = {
                "科技": 0, "军事": 1, "教育考试": 2, "灾难事故": 3, "政治": 4,
                "医药健康": 5, "财经商业": 6, "文体娱乐": 7, "社会生活": 8
            }
        else:
            self.category_dict = {}

    def get_dataloader(self, dataset):
        """
        文本版本直接加载预处理好的 pkl 文件
        路径在 utils/text_dataloader.py 中已配置为 D:/deep-learning/DAMMFND/data/
        """
        print(f"Loading dataset: {dataset}")
        set_data_dir(self.root_path)
        train_loader = get_train_loader(batch_size=self.batchsize)
        val_loader = get_val_loader(batch_size=self.batchsize)
        test_loader = get_test_loader(batch_size=self.batchsize)
        return train_loader, val_loader, test_loader

    def main(self):
        # 1. 获取数据加载器
        train_loader, val_loader, test_loader = self.get_dataloader(self.dataset)

        # 2. 初始化 Trainer
        if self.model_name == 'dammfnd_text':
            print("Init TextTrainer...")
            trainer = TextTrainer(
                emb_dim=self.emb_dim,
                mlp_dims=self.mlp_dims,
                bert=self.bert,
                use_cuda=self.use_cuda,
                lr=self.lr,
                train_loader=train_loader,
                dropout=self.dropout,
                weight_decay=self.weight_decay,
                val_loader=val_loader,
                test_loader=test_loader,
                category_dict=self.category_dict,
                early_stop=self.early_stop,
                epoches=self.epoch,
                save_param_dir=os.path.join(self.save_param_dir, self.model_name)
            )
            
            # 3. 开始训练
            results, save_path = trainer.train()
            
            print("=" * 60)
            print("Training done. Model:", save_path)
            print("Test metrics:", results)
            print("=" * 60)
            
            return results
        else:
            print(f"Error: run_text.py only supports dammfnd_text, got: {self.model_name}")
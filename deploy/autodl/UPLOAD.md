# AutoDL 上传清单（注意系统盘 vs 数据盘）

## 盘符说明

| 位置 | 路径 | 容量 | 放什么 |
|------|------|------|--------|
| **系统盘** | `/root/` | 小（约 30G） | 仅临时文件；**不要**放大数据集、venv、图片 |
| **数据盘** | `/root/autodl-tmp/` | 大（可扩容） | **整个项目**、图片、pkl、权重、虚拟环境 |

实例若数据盘路径不同，在 Jupyter 里看「存储」说明，或执行 `df -h`，把下面路径里的 `/root/autodl-tmp` 换成你的数据盘挂载点。

## 你需要上传到数据盘的内容

目标目录：`/root/autodl-tmp/DAMMFND/`

```
DAMMFND/
  requirements.txt
  src/                          # 整个源码目录（含 deploy/）
  weibo/
    train_2_domain.csv          # 三个 CSV 必传
    val_2_domain.csv
    test_2_domain.csv
    （不要传图片，在服务器上用脚本下载）
  pretrained_model/             # 可选：与 src 平级，setup.sh 会链到 src
    chinese_roberta_wwm_base_ext_pytorch/
      vocab.txt
      config.json
      pytorch_model.bin           # 必有大文件，约 400MB+
```

**不要上传：** `.venv`、`__pycache__`、`weibo22/`（除非你要用别的数据）、本地 `.git`（可选）。

## 本机上传命令示例

AutoDL 控制台 → 实例 → SSH 指令，得到 `ssh -p 端口 root@connect.westb.seetacloud.com`。

在 **本机**（PowerShell / WSL）：

```bash
# 端口、地址以控制台为准
PORT=你的端口
HOST=connect.westb.seetacloud.com

# 在服务器建目录
ssh -p $PORT root@$HOST "mkdir -p /root/autodl-tmp/DAMMFND"

# 传代码与 CSV（体积小，走数据盘）
scp -P $PORT -r d:/deep-learning/DAMMFND/src root@$HOST:/root/autodl-tmp/DAMMFND/
scp -P $PORT d:/deep-learning/DAMMFND/requirements.txt root@$HOST:/root/autodl-tmp/DAMMFND/
scp -P $PORT d:/deep-learning/DAMMFND/weibo/train_2_domain.csv root@$HOST:/root/autodl-tmp/DAMMFND/weibo/
scp -P $PORT d:/deep-learning/DAMMFND/weibo/val_2_domain.csv root@$HOST:/root/autodl-tmp/DAMMFND/weibo/
scp -P $PORT d:/deep-learning/DAMMFND/weibo/test_2_domain.csv root@$HOST:/root/autodl-tmp/DAMMFND/weibo/
```

RoBERTa 较大，建议本机先打包再传，或 AutoDL 「公网网盘 / 数据集」下载到数据盘后执行 `setup.sh` 里的软链接。

## 服务器上执行顺序

```bash
cd /root/autodl-tmp/DAMMFND
bash deploy/autodl/setup.sh          # 装环境（数据盘 .venv）
bash deploy/autodl/run_pipeline.sh   # 下图 + pkl + 训练
```

后台长时间训练：

```bash
cd /root/autodl-tmp/DAMMFND
nohup bash deploy/autodl/run_pipeline.sh > logs/pipeline.log 2>&1 &
tail -f logs/pipeline.log
```

## 我还需要你提供的信息（可选）

1. AutoDL 镜像是否 **PyTorch + CUDA**（推荐官方 PyTorch 镜像，省安装时间）
2. GPU 型号与显存（便于定 `batchsize`）
3. 数据盘实际路径（若不是 `/root/autodl-tmp`）

不提供也能先按默认路径跑；显存不足把 `BATCHSIZE=16` 传给 `run_pipeline.sh`。

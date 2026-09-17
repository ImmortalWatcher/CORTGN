# check_pkl.py (更新版)
import pickle
import os

# 使用绝对路径
base_dir = 'D:/deep-learning/DAMMFND'
pkl_path = os.path.join(base_dir, 'data', 'train_loader.pkl')

print(f"📂 检查文件：{pkl_path}")
print(f"文件存在：{os.path.exists(pkl_path)}\n")

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("📋 pkl 文件结构：")
print(f"类型：{type(data)}")

if isinstance(data, dict):
    print(f"键名：{data.keys()}")
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: 形状={v.shape}, 类型={type(v)}")
        elif isinstance(v, list):
            print(f"  {k}: 长度={len(v)}, 类型={type(v[0]) if len(v)>0 else 'empty'}")
        else:
            print(f"  {k}: 类型={type(v)}")
elif isinstance(data, list):
    print(f"长度：{len(data)}")
    if len(data) > 0:
        print(f"第一项类型：{type(data[0])}")
        if isinstance(data[0], dict):
            print(f"第一项键名：{data[0].keys()}")
        elif isinstance(data[0], (list, tuple)):
            print(f"第一项长度：{len(data[0])}")
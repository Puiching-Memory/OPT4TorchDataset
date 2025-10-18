# OPT4TorchDataset
Plug-and-Play Optimal Page Replacement Algorithm (OPT) for Torch Dataset

## What is OPT?

**OPT (Optimal Page Replacement Algorithm)** 是**理论最优**的页面替换算法，也被称为 Bélády's algorithm。
它需要**预知未来**的访问模式来决定当前应该替换缓存中的哪个数据项，从而达到**最大的**缓存命中率。

- OPT需要知道未来的访问序列
- 理论最佳的缓存命中率
- 总是替换"在未来最晚被访问"的数据项

在深度学习训练中，数据访问模式通常是**可预测的**（由采样器决定），这使得我们能够将OPT算法应用于深度学习训练中：
- 提升数据加载速度
- 提高缓存命中率，比LRU、LFU等缓存算法更高效

## OPT的局限性

**在多进程数据加载（`num_workers > 0`）中，OPT 和其他所有缓存方法的收益都显著降低。**  
多个工作进程可以并行预加载数据，这本身就能掩盖数据传输延迟。

## Install
```bash
pip install OPT4TorchDataset
```
> 目前，我们尚未将wheel包推送至pypi，所以该方法无法使用。

## Quick Start

### Method 1: API

```python
from OPT4TorchDataSet.cachelib import generate_precomputed_file, OPTCacheDecorator
from torch.utils.data import DataLoader

# Step 1: 离线生成预计算文件（一次性）
generate_precomputed_file(
    dataset_size=10000,
    total_iterations=100000,
    persist_path="precomputed/my_experiment.pkl",
    random_seed=0,
    replacement=True,
    maxsize=3000
)

# Step 2: 运行时创建缓存装饰器
decorator = OPTCacheDecorator(
    precomputed_path="precomputed/my_experiment.pkl",
    maxsize=3000, # 必须与预计算时的maxsize一致
    total_iter=100000
)

# Step 3: 应用到数据集
dataset = MyDataset()
dataset.__getitem__ = decorator(dataset.__getitem__)

# 使用数据加载器
dataloader = DataLoader(dataset, batch_size=32)
for batch in dataloader:
    pass
```

### Method 2: CLI

```bash

python -m src.OPT4TorchDataSet.cli \
    --dataset-size 10000 \
    --total-iter 100000 \
    --output precomputed/my_experiment.pkl \
    --seed 0
```

- `--dataset-size`: **必需**。数据集大小
- `--total-iter`: **必需**。预计算的总访问次数
- `--output`: **必需**。保存预计算结果的文件路径（.pkl格式）
- `--seed`: 随机种子，确保结果可重现（可选）
- `--no-replacement`: 禁用有放回采样（可选）

## 开发者指南

### 本地开发使用

如果你没有安装包，而是直接使用源码，可以通过以下方式启动CLI：

```bash
# 在项目根目录下
python -m src.OPT4TorchDataSet.cli \
    --total-iter 100000 \
    --output ./precomputed/imagenet_opt.pkl \
    --seed 42
```

### 环境配置

#### 兼容性矩阵

| 操作系统     | CUDA版本 | GPU型号       | SM架构 | 测试状态 |
| ------------ | -------- | ------------- | ------ | -------- |
| Ubuntu 24.04 | 12.8.2   | H800          | sm90   | ✅        |
| Windows 11   | 12.9.1   | NVIDIA 4060Ti | sm89   | ✅        |
| Windows 11   | 13.0.2   | NVIDIA 4060Ti | sm89   | ✅        |

#### 安装步骤

**创建 Conda 环境**
```bash
conda create -n opt4 python=3.14
conda activate opt4
```

**安装 PyTorch**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**安装项目依赖**
```bash
pip install -r requirements.txt
```

#### 可选配置

**安装 Triton（Windows）**
```bash
pip install -U "triton-windows<3.5"
```

**登录 SwanLab**（用于实验追踪）
```bash
swanlab login
```
https://docs.swanlab.cn/guide_cloud/general/quick-start.html

**选择 GPU 设备**
```bash
export CUDA_VISIBLE_DEVICES=2
```

**设置 Hugging Face 镜像**（提升下载速度）
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 或在 Windows 上
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### 构建 Python 包

```bash
pip install build
python -m build
```

## Experiment
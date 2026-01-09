# OPT4TorchDataset
Plug-and-Play Optimal Page Replacement Algorithm (OPT) for Torch Dataset

[English](README.md) | [中文](README_zh.md)

![Experiment Overview A](media/Experiment_Overview_A.svg)
![Experiment Overview B](media/Experiment_Overview_B.svg)


## What is OPT?

**OPT (Optimal Page Replacement Algorithm)** 是**理论最优**的页面替换算法，也被称为 Bélády's algorithm。
它需要**预知未来**的访问模式来决定当前应该替换缓存中的哪个数据项，从而达到**最大的**缓存命中率。

- OPT需要知道未来的访问序列
- 理论最佳的缓存命中率
- 总是替换"在未来最晚被访问"的数据项

在深度学习训练中，数据访问模式通常是**可预测的**（由采样器决定），这使得我们能够将OPT算法应用于深度学习训练中：
- 提升数据加载速度
- 提高缓存命中率，比LRU、LFU等缓存算法更高效

![flow.png](media/伪随机数缓存的一种方法.drawio.svg)

## 多进程加载与分布式支持

**本项目已完成多进程环境（`num_workers > 0`）下的完全适配：**

1.  **Shared OPT Cache**: 通过 `SharedOPTCacheDecorator` 实现跨进程式的高效共享缓存。它利用共享内存技术，确保所有数据加载进程访问同一个缓存池，最大化利用内存并显著降低样本生成的计算开销。
2.  **Picklable Caches**: 所有的传统缓存（LRU, LFU, FIFO, RR）现在均通过 `CachetoolsDecorator` 进行了序列化适配，可以在 Windows 等系统的 `spawn` 模式下正常运行，不再报 `PicklingError`。
3.  **计算优势**: 在 CPU 变换逻辑复杂（如高分辨率图像增强）的场景下，即使有并行预加载，OPT 缓存依然能通过消除重复计算带来客观的性能提升。

## Install
```bash
pip install OPT4TorchDataset
```
> 目前，我们尚未将wheel包推送至pypi，所以该方法无法使用。
> 你可以先前往我们的Github Actions页面，获取自动构建的whl包手动安装。

## Quick Start

### Method 1: API

```python
from OPT4TorchDataSet.cachelib import generate_precomputed_file, OPTCacheDecorator
from torch.utils.data import DataLoader

# Step 1: 离线生成预计算文件（一次性）
generate_precomputed_file(
    dataset_size=10000,
    total_iterations=100000,
    persist_path="precomputed/my_experiment.safetensors",
    random_seed=0,
    replacement=True,
    maxsize=3000
)

# Step 2: 运行时创建缓存装饰器
decorator = OPTCacheDecorator(
    precomputed_path="precomputed/my_experiment.safetensors",
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
    --output precomputed/my_experiment.safetensors \
    --seed 0
```

- `--dataset-size`: **必需**。数据集大小
- `--total-iter`: **必需**。预计算的总访问次数
- `--output`: **必需**。保存预计算结果的文件路径（.safetensors格式）
- `--seed`: 随机种子，确保结果可重现（可选）
- `--no-replacement`: 禁用有放回采样（可选）

## 开发者指南

### 本地开发使用

如果你没有安装包，而是直接使用源码，可以通过以下方式启动CLI：

```bash
# 在项目根目录下
python -m src.OPT4TorchDataSet.cli \
    --total-iter 100000 \
    --output ./precomputed/imagenet_opt.safetensors \
    --seed 42
```

### 环境配置

#### 兼容性矩阵

| 操作系统     | CUDA版本 | GPU型号       | SM架构 |
| ------------ | -------- | ------------- | ------ |
| Ubuntu 24.04 | 12.8.2   | H800          | sm90   |
| Windows 11   | 12.9.1   | NVIDIA 4060Ti | sm89   |
| Windows 11   | 13.0.2   | NVIDIA 4060Ti | sm89   |

#### 安装步骤

**创建 Conda 环境**
```bash
uv venv --python 3.14
.venv\Scripts\activate.ps1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
uv pip install -U "triton-windows" # Windows 系统可选
```

**选择 GPU 设备**
```bash
export CUDA_VISIBLE_DEVICES=0
```

**设置 Hugging Face 镜像**（提升下载速度）
```bash
$env:HF_ENDPOINT = "https://hf-mirror.com" # 在 Windows 上
```

### 构建 Python wheel 包

```bash
uv pip install build
uv python -m build
```

## 实验结果
所有实验结果均以 JSON 文件形式保存在各个实验目录下的 `results/` 子目录中。

| Model                        | FIFO Time(s) | LFU Time(s) | LRU Time(s) | OPT Time(s) | RR Time(s) | none Time(s) | warmUp Time(s) |
| ---------------------------- | ------------ | ----------- | ----------- | ----------- | ---------- | ------------ | -------------- |
| convnextv2_base              | 179.1515     | 167.4298    | 149.3036    | 136.9226    | 163.4252   | 195.6518     | 159.9803       |
| davit_base                   | 118.902      | 118.8557    | 711.6681    | 82.9771     | 132.2939   | 120.6464     | 137.6575       |
| mobilenetv3_small_100        | 97.6476      | 95.3437     | 82.9322     | 47.3384     | 69.0704    | 103.9736     | 112.7548       |
| mobilenetv5_base             | 168.7884     | 166.6462    | 171.9088    | 150.2256    | 172.5975   | 189.045      | 230.02         |
| resnet50                     | 65.3436      | 71.7038     | 69.5199     | 42.7169     | 69.1713    | 84.1985      | 73.032         |
| swin_base_patch4_window7_224 | 126.8982     | 137.4744    | 155.4214    | 85.9536     | 114.0388   | 125.5925     | 140.2842       |
| swinv2_cr_base_224           | 126.8298     | 156.9438    | 181.8895    | 104.4104    | 182.2423   | 191.7603     | 158.8843       |
| vit_base_patch16_224         | 115.6438     | 98.5015     | 105.8998    | 67.547      | 116.7467   | 127.2664     | 108.5507       |
| vit_base_patch16_dinov3      | 94.6338      | 97.7933     | 124.6119    | 96.1758     | 91.3743    | 160.4503     | 135.4348       |

```bash
Batch Size: 16 | Num Workers: 0 | AMP Enabled: TRUE | Epochs: 5 | Cache Size Ratio: 0.3
```

# OPT4TorchDataset
Plug-and-Play Optimal Page Replacement Algorithm (OPT) for Torch Dataset

## What is OPT?

**OPT (Optimal Page Replacement Algorithm)** 是理论上最优的页面替换算法，也被称为 Bélády's algorithm。它通过预知未来的访问模式来决定当前应该替换缓存中的哪个数据项，从而达到最小的缓存未命中率。


- OPT需要知道未来的访问序列
- 在理论上提供最佳的缓存命中率
- 总是替换"在未来最晚被访问"的数据项

在深度学习训练中，数据访问模式通常是可预测的（由采样器决定），这使得OPT算法能够发挥最大效用：
- 提升数据加载速度
- 提高缓存命中率，比LRU、LFU等传统算法更高效

## Why OPT NOT work?
思路：建立简单的访问模型（Zipf/Markov），分析在何种分布下 OPT 明显优于 LRU/LFU，给出上界/下界或渐近分析。
创新点：提供理论或半理论解释，减少“纯工程”质疑。
验证：对 synthetic traces（Zipf α 不同, Markov 转移）跑仿真并绘制空间/命中率曲线。
MVP：一页数学推导 + 一套仿真实验图（heatmap）。
产出：补充材料中的理论段落和 synthetic results。

## Install
```bash
pip install OPT4TorchDataset
```
> 目前，我们尚未将wheel包推送至pypi，所以该方法无法使用。

## Quick Start

### Method 1: API

```python
from OPT4TorchDataSet.cachelib import generate_precomputed_file

generate_precomputed_file(
    dataset_size=10000,        # 数据集大小
    total_iterations=100000,   # 总迭代次数
    persist_path="precomputed/my_experiment.pkl", # 预计算结果保存路径
    random_seed=0,             # 随机种子
    replacement=True           # 有放回采样
)
```

### Method 2: CLI

```bash
opt4torch-precompute \
    --total-iter 100000 \
    --output precomputed/my_experiment.pkl \
    --seed 42
```

- `--total-iter`: **必需**。预计算的总访问次数，通常等于你训练中的总采样次数
- `--output`: **必需**。保存预计算结果的文件路径（.pkl格式）
- `--sampler`: 采样器类型，格式为`module:callable`（默认：`torch.utils.data:RandomSampler`）
- `--seed`: 随机种子，确保结果可重现（可选）


### 在PyTorch数据集中使用
```python
import torch
import torch.utils.data as data
from OPT4TorchDataSet.cachelib import (
    generate_precomputed_file,
    OPTCacheDecorator
)

class CustomDataset(data.Dataset):
    def __init__(self, data_source, cache_size=None, precomputed_path=None):
        self.data_source = data_source
        
        # 如果提供了预计算文件，设置缓存
        if precomputed_path and cache_size:
            self.setup_cache(precomputed_path, cache_size)

    def setup_cache(self, precomputed_path, cache_size):
        """设置OPT缓存"""
        # 检查预计算文件是否存在，不存在则自动生成
        if not os.path.exists(precomputed_path):
            generate_precomputed_file(
                dataset_size=len(self.data_source),
                total_iterations=len(self.data_source) * 3,  # 假设训练3个epoch
                persist_path=precomputed_path,
                random_seed=42
            )
        
        self.cache_decorator = OPTCacheDecorator(
            precomputed_path=precomputed_path,
            maxsize=cache_size,
            prediction_window=10000,  # 根据需要调整
            total_iter=len(self.data_source) * 3  # 应与预计算时的total_iterations一致
        )
        self._cached_getitem = self.cache_decorator(self._load_data)

    def _load_data(self, obj, index):
        """实际的数据加载方法（会被缓存）"""
        # 在这里实现实际的数据加载逻辑
        return self.data_source[index]

    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, index):
        if hasattr(self, '_cached_getitem'):
            return self._cached_getitem(self, index)
        else:
            return self._load_data(self, index)

# 使用示例
if __name__ == "__main__":
    from torch.utils.data import DataLoader, RandomSampler

    # 创建数据集
    data_source = list(range(1000))  # 示例数据
    dataset = CustomDataset(
        data_source,
        cache_size=300,  # 缓存30%的数据
        precomputed_path="./precomputed/opt_cache.pkl"
    )
    
    # 创建采样器
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=len(dataset) * 3,
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,  # 使用sampler时必须为False
        num_workers=4,
        pin_memory=True,
        sampler=sampler
    )

    # 使用数据加载器
    for batch_idx, data in enumerate(dataloader):
        if batch_idx >= 10:  # 测试10个批次
            break
        print(f"批次 {batch_idx}: {data.shape}")
```

**重要提醒**：
- 训练时使用的采样器配置必须与预计算时完全一致
- 包括相同的随机种子、相同的采样器类型和参数
- 总访问次数也必须匹配，否则会抛出异常

## 本地开发使用

如果你没有安装包，而是直接使用源码，可以通过以下方式启动CLI：

```bash
# 在项目根目录下
python -m src.OPT4TorchDataSet.cli \
    --total-iter 100000 \
    --output ./precomputed/imagenet_opt.pkl \
    --seed 42
```


## Environment (for development only)
```bash
# tested ubuntu 24.04 cuda 12.8 h800 sm90
# tested windows11 cuda 12.9.1 NVIDIA 4060Ti sm89
apt update
apt upgrade
apt install build-essential

conda create -n opt4 python=3.13
conda activate opt4
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt

# install triton for windows(optional)
pip install -U "triton-windows<3.5"

# login swanlab (optional)
swanlab login
# then following https://docs.swanlab.cn/guide_cloud/general/quick-start.html

# choice device (optional)
export CUDA_VISIBLE_DEVICES=2

# set mirror (optional)
export HF_ENDPOINT=https://hf-mirror.com
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## Dataset
```bash
hf download --repo-type dataset ILSVRC/imagenet-1k --cache-dir ./data/imagenet-1k --token {your_token_here}
hf download --repo-type dataset timm/mini-imagenet --cache-dir ./data/mini-imagenet
```

## Experiment timm/mini-imagenet
dataset: mini-imagenet
device: NVIDIA H800 sm90 CUDA 12.9  
system: ubuntu 24.04  
python: 3.13.5  
cachetools: 6.2.0  
OPT: 1.0.0  
torch: 2.8.0 compiled True  
cython: OFF  
seed: 0  
DataIter summary: 977 (50000)  
Epoch: 10  
num_workers: 16  
batch_size: 512  

### Training Speed - Method (one device)

Cache Size: 25000 (50% dataset) (? GB RAM)

| model                        | BaseLine | OPT ON | LRU ON | LFU ON | FIFO ON | RR ON |
| ---------------------------- | -------- | ------ | ------ | ------ | ------- | ----- |
| resnet50                     | 5:21±    | 4:36±  | 4:44±  | 4:44±  | 4:41±   | 4:47± |
| efficientnet_b0              | 5:34±    | 4:46±  | 4:49±  | 4:50±  | 4:59±   | 4:52± |
| mobilenetv4_conv_small       | 5:07±    | 3:12±  | 3:16±  | 3:14±  | 3:16±   | 3:15± |
| convnext_base                | 6:54±    | 6:10±  | 6:14±  | 6:23±  | 6:18±   | 6:10± |
| deit3_small_patch16_224      | 4:33±    | 3:17±  | 3:15±  | 3:16±  | 3:13±   | 3:16± |
| vit_small_patch8_224         | 6:36±    | 6:58±  | 6:33±  | 6:35±  | 6:39±   | 6:41± |
| swin_tiny_patch4_window7_224 | 4:09±    | 4:08±  | 3:59±  | 4:03±  | 3:59±   | 6:21± |


log:https://swanlab.cn/@Sail2Dream/opt4/overview

## Experiment CIFAR-10

## Experiment MLP

### Hit rate - Method

### Hit rate - Cache Size

## build up whl
```bash
pip install build
python -m build
pip install dist/opt4torchdataset-1.0.0-cp313-cp313-linux_x86_64.whl --force-reinstall
```
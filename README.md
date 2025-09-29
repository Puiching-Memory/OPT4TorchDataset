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

## Usage

#### 1. 离线预计算CLI
```bash
# 使用内置CLI预计算OPT访问索引
opt4 \
    --total-iter 100000 \
    --output ./precomputed/imagenet_opt.pkl \
    --seed 42 \
    --num-workers 8
```

#### 2. 在PyTorch数据集中使用
```python
import torch
import torchvision
import torch.utils.data as data
from OPT4TorchDataSet.cachelib import (
    load_precomputed_opt_indices,
    make_opt_cache,
    OPTCacheDecorator
)

class Imagenet1K(data.Dataset):
    def __init__(self, root_dir, split, cache_size=None, precomputed_path=None):
        self.root_dir = root_dir
        self.split = split
        self.idx_list = os.listdir(os.path.join(root_dir, split))
        
        # 如果提供了预计算文件，设置缓存
        if precomputed_path and cache_size:
            self.setup_cache(precomputed_path, cache_size)

    def setup_cache(self, precomputed_path, cache_size):
        """设置OPT缓存"""
        precomputed = load_precomputed_opt_indices(precomputed_path)
        self.cache_decorator = OPTCacheDecorator(
            precomputed_path=precomputed_path,
            maxsize=cache_size,
            prediction_window=len(precomputed.future_index),
            total_iter=len(precomputed.future_index)
        )
        self._cached_getitem = self.cache_decorator(self._load_data)

    def _load_data(self, obj, index):
        """实际的数据加载方法（会被缓存）"""
        image = torchvision.io.decode_image(
            os.path.join(self.root_dir, self.split, self.idx_list[index]),
            mode=torchvision.io.ImageReadMode.RGB
        )
        label = torch.tensor(int(self.idx_list[index].split("-")[0]))
        return image, label

    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, index):
        if hasattr(self, '_cached_getitem'):
            return self._cached_getitem(self, index)
        else:
            return self._load_data(self, index)

# 使用示例
if __name__ == "__main__":
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset = Imagenet1K(
        r".cache/imagenet-1k-jpeg-256",
        "train",
        cache_size=int(0.3 * 50000),  # 缓存30%的数据集
        precomputed_path="./precomputed/imagenet_opt.pkl"
    )
    
    # 使用相同的采样器配置
    weights = [1.0] * len(dataset)  # 或使用你的权重分布
    sampler = WeightedRandomSampler(
        weights, 
        num_samples=len(dataset) * 3,
        replacement=True,
        generator=torch.Generator().manual_seed(42)
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,  # 使用sampler时必须为False
        num_workers=4,
        pin_memory=True,
        sampler=sampler
    )

    for batch_idx, (image, label) in enumerate(dataloader):
        if batch_idx >= 10:  # 测试10个批次
            break
        print(f"批次 {batch_idx}: {image.shape}, {label.shape}")
```

### 方法二：本地开发使用

如果你没有安装包，而是直接使用源码，可以通过以下方式启动CLI：

```bash
# 在项目根目录下
python -m src.OPT4TorchDataSet.cli \
    --total-iter 100000 \
    --output ./precomputed/imagenet_opt.pkl \
    --seed 42
```

**替代方案 - 直接使用Python代码**:

如果不想安装torch或遇到CLI问题，可以直接在Python中预计算：

```python
import sys
from pathlib import Path
import random
import pickle

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def simple_precompute(total_iter, output_path, seed=42):
    """简单的预计算函数，不依赖torch"""
    random.seed(seed)
    
    # 生成随机访问序列 (模拟RandomSampler)
    future_index = [random.randint(0, total_iter-1) for _ in range(total_iter)]
    
    # 构建未来映射
    future_map = {}
    for pos, key in enumerate(future_index):
        if key not in future_map:
            future_map[key] = []
        future_map[key].append(pos)
    
    # 保存到文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'future_index': future_index,
            'future_map': future_map
        }, f)
    
    print(f"预计算完成: {len(future_index)} 个访问, {len(future_map)} 个唯一键")

# 使用示例
if __name__ == "__main__":
    simple_precompute(10000, "local_cache.pkl", seed=42)
```

#### 在代码中使用本地模块
```python
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from OPT4TorchDataSet.cachelib import precompute_opt_indices, OPTCacheDecorator
import torch
from torch.utils.data import WeightedRandomSampler

# 预计算示例
def precompute_example():
    sampler = WeightedRandomSampler
    generator = torch.Generator().manual_seed(42)
    
    precomputed = precompute_opt_indices(
        sampler=sampler,
        generator=generator,
        total_iter=10000,
        persist_path="local_cache.pkl"
    )
    print("预计算完成")

# 使用缓存装饰器
def use_cache_decorator():
    cache = OPTCacheDecorator(
        precomputed_path="local_cache.pkl",
        maxsize=100,
        prediction_window=500,
        total_iter=10000
    )
    
    def expensive_operation(obj, index):
        # 模拟耗时操作
        print(f"执行昂贵操作: {index}")
        return f"result_{index}"
    
    cached_func = cache(expensive_operation)
    
    # 测试
    for i in range(20):
        result = cached_func(None, i % 10)
        if i % 5 == 0:
            stats = cache.stats()
            print(f"命中率: {stats['hit_rate']:.2%}")

if __name__ == "__main__":
    precompute_example()
    use_cache_decorator()
```

### CLI参数说明

#### 安装后使用：
```bash
opt4 --help
```

#### 本地开发使用：
```bash
python -m src.OPT4TorchDataSet.cli --help
# 或
python src/OPT4TorchDataSet/cli.py --help
```

#### 参数详解：

- `--total-iter`: **必需**。预计算的总访问次数，通常等于你训练中的总采样次数
- `--output`: **必需**。保存预计算结果的文件路径（.pkl格式）
- `--sampler`: 采样器类型，格式为`module:callable`（默认：`torch.utils.data:RandomSampler`）
- `--seed`: 随机种子，确保结果可重现（可选）
- `--overwrite`: 允许覆盖已存在的输出文件

#### 使用示例：

**基础用法**：
```bash
# 安装后使用
opt4 --total-iter 50000 --output cache.pkl --seed 42

# 本地使用 (需要torch)
python -m src.OPT4TorchDataSet.cli --total-iter 50000 --output cache.pkl --seed 42
```

**自定义采样器**：
```bash
# 使用WeightedRandomSampler
opt4 \
    --total-iter 100000 \
    --output weighted_cache.pkl \
    --seed 42 \
    --sampler torch.utils.data:WeightedRandomSampler

# 使用自定义采样器
opt4 \
    --total-iter 100000 \
    --output custom_cache.pkl \
    --sampler mypackage.samplers:CustomSampler
```

### 加载和使用预计算结果

预计算完成后，在训练代码中加载和使用：

```python
from OPT4TorchDataSet.cachelib import (
    load_precomputed_opt_indices,
    make_opt_cache,
)

# 加载预计算结果
precomputed = load_precomputed_opt_indices("./precomputed/imagenet_opt.pkl")

# 创建OPT缓存
cache = make_opt_cache(
    total_iter=len(precomputed.future_index),
    maxsize=int(0.3 * dataset_size),  # 缓存30%的数据集大小
    prediction_window=len(precomputed.future_index),
    precomputed=precomputed,
)

# 使用缓存（详见上面的完整示例）
```

**重要提醒**：
- 训练时使用的采样器配置必须与预计算时完全一致
- 包括相同的随机种子、相同的采样器类型和参数
- 总访问次数也必须匹配，否则会抛出异常

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
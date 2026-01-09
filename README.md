# OPT4TorchDataset
Plug-and-Play Optimal Page Replacement Algorithm (OPT) for Torch Dataset

[English](README.md) | [中文](README_zh.md)

![Experiment Overview A](media/Experiment_Overview_A.svg)
![Experiment Overview B](media/Experiment_Overview_B.svg)


## What is OPT?

**OPT (Optimal Page Replacement Algorithm)** is the **theoretically optimal** page replacement algorithm, also known as Bélády's algorithm.
It requires **knowing the future** access pattern to decide which data item in the cache should be replaced, thereby achieving the **maximum** cache hit rate.

- OPT needs to know the future access sequence
- Theoretically best cache hit rate
- Always replaces the data item that will be accessed "furthest in the future"

In deep learning training, data access patterns are usually **predictable** (determined by the sampler), which allows us to apply the OPT algorithm to deep learning training:
- Improve data loading speed
- Increase cache hit rate, more efficient than cache algorithms like LRU, LFU, etc.

![flow.png](media/伪随机数缓存的一种方法.drawio.svg)

## Multi-process Loading and Distributed Support

**The project now fully supports multi-process environments (`num_workers > 0`):**

1.  **Shared OPT Cache**: The `SharedOPTCacheDecorator` implements efficient cross-process shared caching. It utilizes shared memory technology to ensure all data loading processes access the same cache pool, maximizing memory utilization and significantly reducing computation overhead for sample generation.
2.  **Picklable Caches**: All traditional caches (LRU, LFU, FIFO, RR) are now picklable via the `CachetoolsDecorator`. They can run correctly under the `spawn` start method (common on Windows) without `PicklingError`.
3.  **Computational Advantage**: In scenarios with complex CPU transformations (e.g., high-resolution image augmentation), OPT caching still provides substantial performance gains by eliminating redundant computations, even when parallel prefetching is enabled.

## Install
```bash
pip install OPT4TorchDataset
```
> Currently, we have not pushed the wheel package to pypi, so this method cannot be used.
> You can go to our Github Actions page to get the automatically built whl package for manual installation.

## Quick Start

### Method 1: API

```python
from OPT4TorchDataSet.cachelib import generate_precomputed_file, OPTCacheDecorator
from torch.utils.data import DataLoader

# Step 1: Offline generation of precomputed file (one-time)
generate_precomputed_file(
    dataset_size=10000,
    total_iterations=100000,
    persist_path="precomputed/my_experiment.safetensors",
    random_seed=0,
    replacement=True,
    maxsize=3000
)

# Step 2: Create cache decorator at runtime
decorator = OPTCacheDecorator(
    precomputed_path="precomputed/my_experiment.safetensors",
    maxsize=3000, # Must be consistent with maxsize during precomputation
    total_iter=100000
)

# Step 3: Apply to dataset
dataset = MyDataset()
dataset.__getitem__ = decorator(dataset.__getitem__)

# Use DataLoader
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

- `--dataset-size`: **Required**. Dataset size
- `--total-iter`: **Required**. Total number of accesses for precomputation
- `--output`: **Required**. File path to save precomputed results (.safetensors format)
- `--seed`: Random seed to ensure reproducible results (optional)
- `--no-replacement`: Disable replacement sampling (optional)

## Developer Guide

### Local Development Usage

If you have not installed the package but are using the source code directly, you can start the CLI in the following way:

```bash
# In the project root directory
python -m src.OPT4TorchDataSet.cli \
    --total-iter 100000 \
    --output ./precomputed/imagenet_opt.safetensors \
    --seed 42
```

### Environment Configuration

#### Compatibility Matrix

| OS           | CUDA Version | GPU Model     | SM Architecture |
| ------------ | ------------ | ------------- | --------------- |
| Ubuntu 24.04 | 12.8.2       | H800          | sm90            |
| Windows 11   | 12.9.1       | NVIDIA 4060Ti | sm89            |
| Windows 11   | 13.0.2       | NVIDIA 4060Ti | sm89            |

#### Installation Steps

**Create Conda Environment**
```bash
uv venv --python 3.14
.venv\Scripts\activate.ps1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
uv pip install -U "triton-windows" # Optional for Windows
```

**Select GPU Device**
```bash
export CUDA_VISIBLE_DEVICES=0
```

**Set Hugging Face Mirror** (improve download speed)
```bash
$env:HF_ENDPOINT = "https://hf-mirror.com" # on Windows
```

### Build Python wheel package

```bash
uv pip install build
uv python -m build
```

## Experiment
All experiment results are saved as JSON files in the `results/` subdirectory of each experiment.

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

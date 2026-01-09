# 缓存内存占用实验 (RAM Usage Experiment)

## 简介
该实验旨在测量不同缓存策略在运行时对 RAM 的占用情况。特别是对于大规模数据集，缓存条目本身可能会消耗大量内存。该实验可以帮助用户在“访问速度”与“内存开销”之间找到平衡点。

## CLI 使用教程
```bash
python experiment.py --dataset-size 10000 --epochs 10 --batch-size 32
```

**参数说明：**
- `--dataset-size`: 模拟数据集的大小（默认值：`10000`）。
- `--epochs`: 数据访问循环次数（默认值：`10`）。
- `--batch-size`: 批处理大小（默认值：`32`）。

## 实验结果格式
实验结果保存于 `results/results.json`。

**JSON 结构：**
```json
[
    {
        "name": "OPT",
        "cache_size": 1000,
        "ram_usage_mb": 250.50,
        "peak_ram_usage_mb": 310.20,
        "entry_count": 1000
    },
    ...
]
```

*注：`ram_usage` 和 `peak_ram_usage` 单位均为 MB。*

# 缓存命中率实验 (Cache Hit Rate Experiment)

## 简介
该实验旨在评估和比较不同缓存替换策略（LRU, LFU, FIFO, RR 和 OPT）在模拟数据集访问场景下的性能表现。通过测量各种缓存大小配置下的命中率和未命中次数，量化 OPT 算法在减少 I/O 压力方面的理论上限。

## CLI 使用教程
你可以通过命令行灵活配置实验参数。使用以下命令运行实验：

```bash
python experiment.py --dataset-size 10000 --epochs 10 --batch-size 32
```

**参数说明：**
- `--dataset-size`: 模拟数据集的大小（默认值：`10000`）。
- `--epochs`: 模拟训练的轮数，用于产生重复的访问序列（默认值：`10`）。
- `--batch-size`: 数据加载的批处理大小（默认值：`32`）。
- `--help`: 查看完整的帮助信息。

## 实验结果格式
实验结果将保存为 `results/results.json`。

**JSON 结构：**
```json
[
    {
        "name": "OPT",
        "cache_size": 1000,
        "hit_rate": 0.45,
        "miss_count": 5500,
        "total_accesses": 10000
    },
    ...
]
```

*注：`name` 代表策略名称，`cache_size` 为实际缓存条目数，`hit_rate` 为 0-1 之间的命中率。*

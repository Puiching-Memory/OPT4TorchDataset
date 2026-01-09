# 访问分布与性能收益对比 (When OPT is Better)

## 简介
该实验研究数据访问分布（通过 Zipf 分布的 $\alpha$ 参数控制）对不同缓存算法效果的影响。通过调整 $\alpha$（从均匀分布到极度倾斜的分布），观察 OPT 何时能产生最大的领先优势。

## CLI 使用教程
```bash
python experiment.py --dataset-size 10000 --epochs 10 --batch-size 32
```

**参数说明：**
- `--dataset-size`: 模拟数据集大小（默认：`10000`）。
- `--epochs`: 数据循环轮数（默认：`10`）。
- `--batch-size`: 访问批次大小（默认：`32`）。

## 实验结果格式
该实验会针对多个 $\alpha$ 值进行循环，并将结果聚合。
实验结果保存于 `results/all_results.json`。

**JSON 结构：**
```json
[
    {
        "zipf_alpha": 0.0,
        "name": "OPT",
        "cache_size": 500,
        "hit_rate": 0.05,
        "miss_count": 9500
    },
    ...
]
```

*注：`zipf_alpha` 为 0.0 时代表均匀随机分布。*

# 真实模型训练速度实验 (Training Speed Experiment)

## 简介
该实验使用 `timm` 库提供的真实计算机视觉模型（如 ResNet, ViT, Swin Transformer 等）来测量在不同缓存策略下的实际端到端训练速度。这反映了 OPT 在实际深度学习工作流中的性能收益。

## CLI 使用教程
```bash
python experiment.py --epochs 5 --batch-size 16 --num-workers 0 --use-amp --cache-types "none,LRU,OPT"
```

**参数说明：**
- `--epochs`: 每个模型测试的轮数（默认：`5`）。
- `--batch-size`: 训练批次大小（默认：`16`）。
- `--num-workers`: 数据加载线程数（默认：`0`）。对于 `OPT` 策略，设置大于 0 会自动使用共享内存加速。
- `--use-amp`: 是否开启自动混合精度训练（默认：`True`）。
- `--cache-types`: 参与对比的缓存方法，逗号分隔（默认：`none,LRU,LFU,FIFO,RR,OPT`）。

## 实验结果格式
实验结果保存于 `results/results.json`。

**JSON 结构：**
```json
[
    {
        "model_name": "resnet50",
        "cache_type": "OPT",
        "cache_size_ratio": 0.5,
        "training_time": 42.71,
        "batch_size": 16,
        ...
    },
    ...
]
```

*注：表格显示不同策略完成指定 Epoch 的总耗时。*

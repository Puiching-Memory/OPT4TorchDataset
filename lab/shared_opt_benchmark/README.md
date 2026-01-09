# 共享内存缓存性能测试 (Shared Memory Cache Benchmark)

## 简介
该实验对比了三种模式下的数据加载性能：
1. **No Cache**: 无缓存基准。
2. **Local Cache (Py)**: 每个 Worker 拥有独立的 Python 对象缓存（数据重复占用内存）。
3. **Shared Cache (C++)**: 跨 Worker 共享的 C++ 后端缓存（数据在进程间共享，内存利用率最高）。

## CLI 使用教程
该实验整合了性能对比与功能校验。使用 `python experiment.py [COMMAND] --help` 查看各子命令详情。

### 1. 性能对比测试 (Performance Benchmark)
对比无缓存、模拟单进程本地缓存以及跨进程共享缓存的吞吐率。
```bash
python experiment.py perf --dataset-size 1000 --cache-ratio 0.2 --total-iters 5000 --workers 4
```
- `perf` 子命令。
- `--cache-ratio`: 缓存占数据集的比例（默认：`0.2`）。
- `--total-iters`: 总迭代次数（默认：`5000`）。
- `--workers`: DataLoader 的工作进程数（默认：`4`）。
- `--sim-time`: 模拟单次 IO/计算的时间（秒）（默认：`0.005`）。

### 2. 功能一致性校验 (Functional Validation)
验证共享缓存在多进程环境下是否能维持正确的命中率（不关心时间效率）。
```bash
python experiment.py validate --workers 4 --iterations 2000 --cache-ratio 0.2
```
- `validate` 子命令。

## 实验结果格式
- `perf` 结果保存于 `results/perf_results.json`。
- `validate` 结果保存于 `results/validation_results.json`。

**JSON 结构 (perf_compare.json)：**
```json
[
    {
        "method": "Shared Cache (C++)",
        "throughput": 150.25,
        "hit_rate": 20.00
    },
    ...
]
```

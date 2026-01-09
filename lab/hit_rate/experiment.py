#!/usr/bin/env python3
"""
缓存命中率实验

该脚本用于比较不同缓存替换策略（LRU、LFU、FIFO、RR和OPT）在模拟数据集上的命中率表现。
通过测量各种缓存大小配置下的命中率和未命中次数，评估各策略的有效性。
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import torch
import typer
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler
from cachetools import cached, LRUCache, LFUCache, FIFOCache, RRCache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lib.hit_rate_dataset import HitRateDataset
from OPT4TorchDataSet.cachelib import (
    OPTCacheDecorator,
    CachetoolsDecorator,
    generate_precomputed_file,
)

# Setup logging
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(OUTPUT_DIR / "experiment.log", rotation="10 MB")


class CacheExperiment:
    def __init__(
        self,
        caches: Optional[List[Tuple[str, float, Any]]] = None,
        output_dir: Union[str, Path] = "results",
        batch_size: int = 32,
        num_workers: int = 0,
        dataset: Optional[torch.utils.data.Dataset] = None,
        epochs: int = 1,
    ):
        """
        初始化缓存实验

        Args:
            caches: 缓存配置列表，每项包含(名称, 缓存大小比例, 缓存实例)
            output_dir: 结果输出目录
            batch_size: 批处理大小
            num_workers: 数据加载器工作进程数
            dataset: 实验数据集
            epochs: 训练轮数
        """
        self.caches = caches
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.epochs = epochs

        self.output_dir.mkdir(exist_ok=True)

    def _run_single_experiment(self, cache) -> float:
        """
        运行单个缓存实验

        Args:
            cache: 缓存实例

        Returns:
            float: 未命中次数
        """

        # 创建新的数据集实例，确保每次实验都从干净状态开始
        dataset = HitRateDataset(len(self.dataset.dataset))
        dataset.setCache(cache)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=RandomSampler(
                dataset,
                replacement=True,
                num_samples=len(dataset) * self.epochs,  # 足够大的轮数确保采样充足
                generator=dataset.getGenerator(),
            ),
        )

        for batch_data in dataloader:
            pass

        return dataset.getMissCount()

    def run(self):
        """运行所有缓存实验并保存结果"""

        logger.info("Starting Cache Performance Experiments")
        results = []

        for name, cache_size, cache in self.caches:
            # 对于OPT缓存，需要重置状态
            if isinstance(cache, OPTCacheDecorator):
                cache.reset()

            miss_count = self._run_single_experiment(cache)
            total_accesses = len(self.dataset) * self.epochs
            hit_rate = (
                total_accesses - miss_count
            ) / total_accesses  # 命中率是 0-1 之间的小数

            results.append(
                {
                    "name": name,
                    "cache_size": cache_size,
                    "hit_rate": hit_rate,
                    "miss_count": miss_count,
                    "total_accesses": total_accesses,
                }
            )

            logger.info(f"Cache: {name} hit rate: {hit_rate:.2%}")

        # 保存结果到JSON文件
        with open(self.output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"Results saved to {self.output_dir / 'results.json'}")


app = typer.Typer()


@app.command()
def main(
    dataset_size: int = typer.Option(10000, help="Maximum dataset size"),
    epochs: int = typer.Option(10, help="Number of epochs to run"),
    batch_size: int = typer.Option(32, help="Batch size for experiments"),
):
    """
    Run the cache hit rate experiment with various strategies and sizes.
    """
    total_iter = dataset_size * epochs

    cache_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cache_types = ["LRU", "LFU", "FIFO", "RR"]
    caches: List[Tuple[str, float, object]] = []

    caches.append(("Warmup", 0.0, CachetoolsDecorator(LRUCache(maxsize=dataset_size))))

    # 为每种缓存类型和缓存大小生成配置
    for cache_type in cache_types:
        for size in cache_sizes:
            cache_size = int(size * dataset_size)
            if cache_type == "LRU":
                item = CachetoolsDecorator(LRUCache(maxsize=cache_size))
            elif cache_type == "LFU":
                item = CachetoolsDecorator(LFUCache(maxsize=cache_size))
            elif cache_type == "FIFO":
                item = CachetoolsDecorator(FIFOCache(maxsize=cache_size))
            elif cache_type == "RR":
                item = CachetoolsDecorator(RRCache(maxsize=cache_size))
            caches.append((cache_type, size, item))

    # 添加OPT缓存配置 - 为每个cache_size生成独立的预计算文件
    precomputed_dir = PROJECT_ROOT / "precomputed"
    precomputed_dir.mkdir(parents=True, exist_ok=True)

    for size in cache_sizes:
        cache_size = int(size * dataset_size)
        if cache_size > 0:  # 避免缓存大小为0的情况
            # 为每个cache_size创建独立的预计算文件
            precomputed_path = (
                precomputed_dir / f"hit_rate_experiment_opt_{cache_size}.safetensors"
            )

            # 检查预计算文件是否存在，不存在则生成
            if not precomputed_path.exists():
                logger.info(
                    f"生成预计算文件（maxsize={cache_size}）: {precomputed_path}"
                )

                # 使用新的API生成预计算文件，包含maxsize参数
                generate_precomputed_file(
                    dataset_size=dataset_size,
                    total_iterations=total_iter,
                    persist_path=precomputed_path,
                    random_seed=0,  # 与 HitRateDataset 使用相同的种子
                    replacement=True,
                    maxsize=cache_size,
                )

                logger.info(f"预计算文件生成完成: {precomputed_path}")

            opt_decorator = OPTCacheDecorator(
                precomputed_path=precomputed_path,
                maxsize=cache_size,
                total_iter=total_iter,
            )
            caches.append(("OPT", size, opt_decorator))

    experiment = CacheExperiment(
        caches=caches,
        output_dir=OUTPUT_DIR,
        batch_size=batch_size,
        dataset=HitRateDataset(dataset_size),
        num_workers=0,
        epochs=epochs,
    )
    experiment.run()


if __name__ == "__main__":
    app()

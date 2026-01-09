#!/usr/bin/env python3
"""
Cache Performance Experiment - Track RAM Usage

This script focuses on recording RAM usage of different cache algorithms.
"""

import json
import sys
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from copy import deepcopy

import torch
import psutil
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
from OPT4TorchDataSet.cachelib import OPTCacheDecorator, generate_precomputed_file

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
        self.caches = caches
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.epochs = epochs

        self.output_dir.mkdir(exist_ok=True)

    def _run_single_experiment(self, cache) -> Dict:
        """Run single experiment focused on RAM usage"""

        # 强制进行垃圾回收以获得更准确的基线内存使用量
        gc.collect()

        # 获取初始内存使用量
        initial_ram = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # 创建新的数据集实例，确保每次实验都从干净状态开始
        dataset = deepcopy(self.dataset)
        # 应用缓存
        dataset.setCache(cache)

        # 获取数据集加载后的内存使用量
        dataset_loaded_ram = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=RandomSampler(
                dataset,
                replacement=True,
                num_samples=len(dataset) * self.epochs,
                generator=dataset.getGenerator(),
            ),
        )

        # 获取数据加载器创建后的内存使用量
        dataloader_created_ram = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # 迭代数据加载器
        for batch_data in dataloader:
            pass

        # 获取迭代完成后的峰值内存使用量
        peak_ram = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # 获取缓存统计信息
        entry_count = 0
        if hasattr(dataset, "_getitem_impl"):
            if hasattr(dataset._getitem_impl, "__wrapped__"):
                cache = getattr(dataset._getitem_impl, "__wrapped__", None)
                if (
                    cache
                    and hasattr(cache, "cache")
                    and hasattr(cache.cache, "__dict__")
                ):
                    entry_count = (
                        len(cache.cache) if hasattr(cache.cache, "__len__") else 0
                    )
                elif (
                    cache and hasattr(cache, "__dict__") and "_cache" in cache.__dict__
                ):
                    entry_count = len(cache._cache)

        # 清理
        del dataloader
        del dataset
        gc.collect()

        final_ram = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        return {
            "initial_ram": initial_ram,
            "dataset_loaded_ram": dataset_loaded_ram,
            "dataloader_created_ram": dataloader_created_ram,
            "peak_ram": peak_ram,
            "final_ram": final_ram,
            "entry_count": entry_count,
        }

    def run(self):
        """Run experiments using internal configuration"""

        logger.info("Starting Cache RAM Usage Experiments")
        results = []

        for name, cache_size, cache in self.caches:
            metrics = self._run_single_experiment(cache)

            # 计算各种RAM使用量指标
            ram_usage = metrics["final_ram"] - metrics["initial_ram"]
            peak_ram_usage = metrics["peak_ram"] - metrics["initial_ram"]
            entry_count = metrics["entry_count"]

            results.append(
                {
                    "name": name,
                    "cache_size": cache_size,
                    "ram_usage_mb": ram_usage,
                    "peak_ram_usage_mb": peak_ram_usage,
                    "entry_count": entry_count,
                }
            )

            logger.info(
                f"Cache: {name} RAM usage: {ram_usage:.2f} MB, Peak: {peak_ram_usage:.2f} MB, Entries: {entry_count}"
            )

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
    Run the RAM usage experiment for different cache strategies.
    """
    total_iter = dataset_size * epochs

    cache_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cache_types = ["LRU", "LFU", "FIFO", "RR"]
    caches: List[Tuple[str, float, object]] = []

    caches.append(("Warmup", 0.0, cached(LRUCache(maxsize=dataset_size))))

    # 为每种缓存类型和缓存大小生成配置
    for cache_type in cache_types:
        for size in cache_sizes:
            cache_size = int(size * dataset_size)
            if cache_type == "LRU":
                caches.append((cache_type, size, cached(LRUCache(maxsize=cache_size))))
            elif cache_type == "LFU":
                caches.append((cache_type, size, cached(LFUCache(maxsize=cache_size))))
            elif cache_type == "FIFO":
                caches.append((cache_type, size, cached(FIFOCache(maxsize=cache_size))))
            elif cache_type == "RR":
                caches.append((cache_type, size, cached(RRCache(maxsize=cache_size))))

    precomputed_dir = PROJECT_ROOT / "precomputed"
    precomputed_dir.mkdir(parents=True, exist_ok=True)

    # 添加OPT缓存配置 - 为每个cache_size生成独立的预计算文件
    for size in cache_sizes:
        cache_size = int(size * dataset_size)
        if cache_size > 0:
            precomputed_path = (
                precomputed_dir / f"ram_usage_opt_precomputed_{cache_size}.safetensors"
            )

            if not precomputed_path.exists():
                logger.info(
                    f"生成预计算文件（maxsize={cache_size}）: {precomputed_path}"
                )
                generate_precomputed_file(
                    dataset_size=dataset_size,
                    total_iterations=total_iter,
                    persist_path=precomputed_path,
                    random_seed=0,
                    replacement=True,
                    maxsize=cache_size,
                )
                logger.info("预计算文件生成完成")

            caches.append(
                (
                    "OPT",
                    size,
                    OPTCacheDecorator(
                        precomputed_path=precomputed_path,
                        maxsize=cache_size,
                        total_iter=total_iter,
                    ),
                )
            )

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

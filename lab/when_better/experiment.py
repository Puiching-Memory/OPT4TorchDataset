#!/usr/bin/env python3
"""
Cache Performance Experiment - Simplified Implementation

This script focuses on recording training speed and final time cost.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from copy import deepcopy

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

from lib.zipf_dataset import ZipfDataset
from OPT4TorchDataSet.cachelib import OPTCacheDecorator, generate_precomputed_file

# Setup logging
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(OUTPUT_DIR / "experiment.log", rotation="10 MB")


def save_results_to_json(results: List[Dict], output_path: Path):
    """
    将实验结果保存到JSON文件中

    Args:
        results: 包含实验结果的字典列表
        output_path: JSON文件输出路径
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


class CacheExperiment:
    def __init__(
        self,
        caches: Optional[List[Tuple[str, float, Any]]] = None,
        output_dir: Union[str, Path] = "results",
        batch_size: int = 32,
        num_workers: int = 0,
        dataset: Optional[torch.utils.data.Dataset] = None,
        epochs: int = 1,
        zipf_alpha: float = 1.0,
    ):
        self.caches = caches
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.epochs = epochs
        self.zipf_alpha = zipf_alpha

        self.output_dir.mkdir(exist_ok=True)

    def _run_single_experiment(self, cache, access_sequence: torch.Tensor) -> float:
        """Run single experiment focused on speed and time cost"""

        # 创建新的数据集实例，确保每次实验都从干净状态开始
        dataset = deepcopy(self.dataset)
        # 应用缓存
        dataset.setCache(cache)

        # 使用 Subset 来强制执行预定义的访问序列
        from torch.utils.data import Subset
        subset = Subset(dataset, access_sequence.tolist())

        dataloader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            # 使用 SequentialSampler 确保按照 Subset 的索引顺序访问
        )

        for batch_data in dataloader:
            pass

        return dataset.getMissCount()

    def run(self, access_sequence: torch.Tensor):
        """Run experiments using internal configuration"""

        logger.info(
            f"Starting Cache Performance Experiments with Zipf alpha={self.zipf_alpha}"
        )
        results = []

        for name, cache_size, cache in self.caches:
            miss_count = self._run_single_experiment(cache, access_sequence)
            total_accesses = len(access_sequence)
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
                    "zipf_alpha": self.zipf_alpha,
                }
            )

            logger.info(f"Cache: {name} hit rate: {hit_rate:.2%}")

        return results


app = typer.Typer()


@app.command()
def main(
    dataset_size: int = typer.Option(10000, help="Maximum dataset size"),
    epochs: int = typer.Option(10, help="Number of epochs to run"),
    batch_size: int = typer.Option(32, help="Batch size for experiments"),
):
    """
    Run the 'when better' experiment for Zipf distributions.
    """
    total_iter = dataset_size * epochs

    # 定义要测试的Zipf参数
    zipf_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5]

    # 定义缓存大小和类型
    cache_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cache_types = ["LRU", "LFU", "FIFO", "RR"]

    # 收集所有实验结果
    all_results = []

    for alpha in zipf_alphas:
        caches: List[Tuple[str, float, object]] = []
        
        # 为当前 alpha 创建数据集并生成固定的访问序列
        dataset = ZipfDataset(dataset_size, alpha=alpha, seed=0)
        future_index = torch.tensor(dataset.generate_access_sequence(total_iter, seed=0), dtype=torch.int64)

        caches.append(("Warmup", 0.0, cached(LRUCache(maxsize=dataset_size))))

        # 为每种缓存类型和缓存大小生成配置
        for cache_type in cache_types:
            for size in cache_sizes:
                cache_size = int(size * dataset_size)
                if cache_type == "LRU":
                    caches.append(
                        (cache_type, size, cached(LRUCache(maxsize=cache_size)))
                    )
                elif cache_type == "LFU":
                    caches.append(
                        (cache_type, size, cached(LFUCache(maxsize=cache_size)))
                    )
                elif cache_type == "FIFO":
                    caches.append(
                        (cache_type, size, cached(FIFOCache(maxsize=cache_size)))
                    )
                elif cache_type == "RR":
                    caches.append(
                        (cache_type, size, cached(RRCache(maxsize=cache_size)))
                    )

        precomputed_dir = PROJECT_ROOT / "precomputed"
        precomputed_dir.mkdir(parents=True, exist_ok=True)

        # 添加OPT缓存配置 - 为每个cache_size生成独立的预计算文件
        for size in cache_sizes:
            cache_size = int(size * dataset_size)
            if cache_size > 0:
                precomputed_path = (
                    precomputed_dir
                    / f"zipf_v2_opt_alpha_{alpha:.1f}_{cache_size}.safetensors"
                )

                # 如果 alpha 设置改变或序列变化，理想情况下应重新生成
                # 为简单起见，这里假设 alpha 精度为 .1 足以区分
                if not precomputed_path.exists():
                    logger.info(
                        f"生成预计算文件（alpha={alpha}, maxsize={cache_size}）: {precomputed_path}"
                    )
                    generate_precomputed_file(
                        dataset_size=dataset_size,
                        total_iterations=total_iter,
                        persist_path=precomputed_path,
                        random_seed=0,
                        replacement=True,
                        maxsize=cache_size,
                        distribution_seq=future_index,
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
            dataset=dataset,
            num_workers=0,
            epochs=epochs,
            zipf_alpha=alpha,
        )
        # 收集实验结果
        all_results.extend(experiment.run(future_index))

    # 保存所有结果到一个JSON文件
    save_results_to_json(all_results, OUTPUT_DIR / "all_results.json")
    logger.info(f"All results saved to {OUTPUT_DIR / 'all_results.json'}")


if __name__ == "__main__":
    app()

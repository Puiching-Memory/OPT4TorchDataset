#!/usr/bin/env python3
"""
Cache Performance Experiment - Simplified Implementation

This script focuses on recording training speed and final time cost.
"""

import csv
import sys
import torch
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler
import os
from typing import List, Dict, Tuple
from pathlib import Path
from copy import deepcopy
from cachetools import cached, LRUCache, LFUCache, FIFOCache, RRCache

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

from lib.zipf_dataset import ZipfDataset
from OPT4TorchDataSet.cachelib import OPTCacheDecorator, generate_precomputed_file

def save_results_to_csv(results: List[Dict], output_path: Path):
    """
    将实验结果保存到CSV文件中，便于在Excel中绘制图表
    
    Args:
        results: 包含实验结果的字典列表
        output_path: CSV文件输出路径
    """
    # 重新组织数据结构以方便Excel绘图
    cache_types = sorted(list(set(result["name"] for result in results)))
    cache_sizes = sorted(list(set(result["cache_size"] for result in results)))
    zipf_alphas = sorted(list(set(result["zipf_alpha"] for result in results)))
    
    # 创建一个二维表，行是缓存大小，列是缓存类型
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 为每个Zipf参数写入结果
        for alpha in zipf_alphas:
            writer.writerow([f"Zipf Alpha = {alpha}"])
            # 写入表头
            header = ["Cache Size"] + cache_types
            writer.writerow(header)
            
            # 写入命中率数据
            for size in cache_sizes:
                row = [size]
                for cache_type in cache_types:
                    # 查找对应的命中率
                    hit_rate = next((r["hit_rate"] for r in results 
                                   if r["name"] == cache_type and r["cache_size"] == size and r["zipf_alpha"] == alpha), None)
                    row.append(f"{hit_rate:.4f}" if hit_rate is not None else "")
                writer.writerow(row)
            
            # 添加一行空行作为分隔
            writer.writerow([])
            
            # 写入miss count数据
            writer.writerow(["Miss Count"] + cache_types)
            for size in cache_sizes:
                row = [size]
                for cache_type in cache_types:
                    # 查找对应的未命中次数
                    miss_count = next((r["miss_count"] for r in results 
                                     if r["name"] == cache_type and r["cache_size"] == size and r["zipf_alpha"] == alpha), None)
                    row.append(miss_count if miss_count is not None else "")
                writer.writerow(row)
            
            # 添加一行空行作为分隔
            writer.writerow([])
            
            # 写入total accesses数据
            writer.writerow(["Total Accesses"] + [""] * (len(cache_types) - 1) + [next(r["total_accesses"] for r in results if r["zipf_alpha"] == alpha)])
            writer.writerow([])  # 额外空行分隔不同alpha值的结果


class CacheExperiment:
    def __init__(self,
                 caches: List[Tuple[str, object]] = None,
                 output_dir: str = "results",
                 batch_size: int = 32,
                 num_workers: int = 0,
                 dataset: torch.utils.data.Dataset = None,
                 epochs: int = 1,
                 zipf_alpha: float = 1.0
                 ):
        self.caches = caches
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.epochs = epochs
        self.zipf_alpha = zipf_alpha

        self.output_dir.mkdir(exist_ok=True)

    def _run_single_experiment(self) -> float:
        """Run single experiment focused on speed and time cost"""
        
        dataset = deepcopy(self.dataset)
        
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                sampler=RandomSampler(dataset,
                                                      replacement=True,
                                                      num_samples=len(dataset) * self.epochs, # 足够大的轮数确保采样充足
                                                      generator=dataset.getGenerator()
                                                      )
                                )
        
        for batch_data in dataloader:
            pass

        return dataset.getMissCount()
    
    def run(self):
        """Run experiments using internal configuration"""
        
        logger.info(f"Starting Cache Performance Experiments with Zipf alpha={self.zipf_alpha}")
        results = []
        
        for name, cache_size, cache in self.caches:
            self.dataset.setCache(cache)
            miss_count = self._run_single_experiment()
            total_accesses = len(self.dataset) * self.epochs
            hit_rate = (total_accesses - miss_count) / total_accesses  # 命中率是 0-1 之间的小数
            
            results.append({
                "name": name,
                "cache_size": cache_size,
                "hit_rate": hit_rate,
                "miss_count": miss_count,
                "total_accesses": total_accesses,
                "zipf_alpha": self.zipf_alpha
            })
            
            self.dataset.resetMissCount()
            logger.info(f"Cache: {name} hit rate: {hit_rate:.2%}")

        return results

if __name__ == "__main__":
    MAX_DATASET_SIZE = 10000
    epochs = 10
    batch_size = 32
    total_iter = MAX_DATASET_SIZE * epochs

    # 定义要测试的Zipf参数
    zipf_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5]
    
    # 定义缓存大小和类型
    cache_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cache_types = ["LRU", "LFU", "FIFO", "RR"]
    
    # 收集所有实验结果
    all_results = []
    
    for alpha in zipf_alphas:
        caches: List[Tuple[str, object]] = []
        
        caches.append(("Warmup", 0.0, cached(LRUCache(maxsize=MAX_DATASET_SIZE))))

        # 为每种缓存类型和缓存大小生成配置
        for cache_type in cache_types:
            for size in cache_sizes:
                cache_size = int(size * MAX_DATASET_SIZE)
                if cache_type == "LRU":
                    caches.append((cache_type, size, cached(LRUCache(maxsize=cache_size))))
                elif cache_type == "LFU":
                    caches.append((cache_type, size, cached(LFUCache(maxsize=cache_size))))
                elif cache_type == "FIFO":
                    caches.append((cache_type, size, cached(FIFOCache(maxsize=cache_size))))
                elif cache_type == "RR":
                    caches.append((cache_type, size, cached(RRCache(maxsize=cache_size))))
        
        precomputed_dir = Path(ROOT) / "precomputed"
        precomputed_dir.mkdir(parents=True, exist_ok=True)
        precomputed_path = precomputed_dir / "zipf_opt_precomputed.pkl"

        if not precomputed_path.exists():
            logger.info(f"预计算文件不存在，正在生成: {precomputed_path}")
            generate_precomputed_file(
                dataset_size=MAX_DATASET_SIZE,
                total_iterations=total_iter,
                persist_path=precomputed_path,
                random_seed=0,
                replacement=True,
            )
            logger.info("预计算文件生成完成")

        # 添加OPT缓存配置
        for size in cache_sizes:
            cache_size = int(size * MAX_DATASET_SIZE)
            if cache_size > 0:
                caches.append((
                    "OPT",
                    size,
                    OPTCacheDecorator(
                        precomputed_path=precomputed_path,
                        maxsize=cache_size,
                        total_iter=total_iter,
                        seed=0,
                    ),
                ))

        experiment = CacheExperiment(caches=caches,
                                     batch_size=batch_size,
                                     dataset=ZipfDataset(MAX_DATASET_SIZE, alpha=alpha, seed=0),
                                     num_workers=0,
                                     epochs=epochs,
                                     zipf_alpha=alpha
                                     )
        # 收集实验结果
        all_results.extend(experiment.run())
    
    # 创建输出目录
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # 保存所有结果到一个CSV文件
    save_results_to_csv(all_results, output_dir / "all_results.csv")
    logger.info(f"All results saved to {output_dir / 'all_results.csv'}")
#!/usr/bin/env python3
"""
Cache Performance Experiment - Track RAM Usage

This script focuses on recording RAM usage of different cache algorithms.
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
import psutil
import gc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

from lib.hit_rate_dataset import HitRateDataset
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
    
    # 创建一个二维表，行是缓存大小，列是缓存类型
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        header = ["Cache Size"] + cache_types
        writer.writerow(header)
        
        # 写入RAM使用量数据 (MB)
        for size in cache_sizes:
            row = [size]
            for cache_type in cache_types:
                # 查找对应的RAM使用量
                ram_usage = next((r["ram_usage_mb"] for r in results 
                                if r["name"] == cache_type and r["cache_size"] == size), None)
                row.append(f"{ram_usage:.2f}" if ram_usage is not None else "")
            writer.writerow(row)
        
        # 添加一行空行作为分隔
        writer.writerow([])
        
        # 写入缓存大小数据
        writer.writerow(["Cache Entry Count"] + cache_types)
        for size in cache_sizes:
            row = [size]
            for cache_type in cache_types:
                # 查找对应的缓存条目数
                entry_count = next((r["entry_count"] for r in results 
                                  if r["name"] == cache_type and r["cache_size"] == size), None)
                row.append(entry_count if entry_count is not None else "")
            writer.writerow(row)
        
        # 添加一行空行作为分隔
        writer.writerow([])
        
        # 写入峰值RAM使用量数据
        writer.writerow(["Peak RAM Usage (MB)"] + cache_types)
        for size in cache_sizes:
            row = [size]
            for cache_type in cache_types:
                # 查找对应的峰值RAM使用量
                peak_ram = next((r["peak_ram_usage_mb"] for r in results 
                               if r["name"] == cache_type and r["cache_size"] == size), None)
                row.append(f"{peak_ram:.2f}" if peak_ram is not None else "")
            writer.writerow(row)

class CacheExperiment:
    def __init__(self,
                 caches: List[Tuple[str, object]] = None,
                 output_dir: str = "results",
                 batch_size: int = 32,
                 num_workers: int = 0,
                 dataset: torch.utils.data.Dataset = None,
                 epochs: int = 1,
                 ):
        self.caches = caches
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.epochs = epochs

        self.output_dir.mkdir(exist_ok=True)

    def _run_single_experiment(self) -> Dict:
        """Run single experiment focused on RAM usage"""
        
        # 强制进行垃圾回收以获得更准确的基线内存使用量
        gc.collect()
        
        # 获取初始内存使用量
        initial_ram = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        dataset = deepcopy(self.dataset)
        
        # 获取数据集加载后的内存使用量
        dataset_loaded_ram = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                sampler=RandomSampler(dataset,
                                                      replacement=True,
                                                      num_samples=len(dataset) * self.epochs,
                                                      generator=dataset.getGenerator()
                                                      )
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
        if hasattr(dataset, '_getitem_impl'):
            if hasattr(dataset._getitem_impl, '__wrapped__'):
                cache = getattr(dataset._getitem_impl, '__wrapped__', None)
                if cache and hasattr(cache, 'cache') and hasattr(cache.cache, '__dict__'):
                    entry_count = len(cache.cache) if hasattr(cache.cache, '__len__') else 0
                elif cache and hasattr(cache, '__dict__') and '_cache' in cache.__dict__:
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
            "entry_count": entry_count
        }
    
    def run(self):
        """Run experiments using internal configuration"""
        
        logger.info("Starting Cache RAM Usage Experiments")
        results = []
        
        for name, cache_size, cache in self.caches:
            self.dataset.setCache(cache)
            metrics = self._run_single_experiment()
            
            # 计算各种RAM使用量指标
            ram_usage = metrics["final_ram"] - metrics["initial_ram"]
            peak_ram_usage = metrics["peak_ram"] - metrics["initial_ram"]
            entry_count = metrics["entry_count"]
            
            results.append({
                "name": name,
                "cache_size": cache_size,
                "ram_usage_mb": ram_usage,
                "peak_ram_usage_mb": peak_ram_usage,
                "entry_count": entry_count
            })
            
            self.dataset.resetMissCount()
            logger.info(f"Cache: {name} RAM usage: {ram_usage:.2f} MB, Peak: {peak_ram_usage:.2f} MB, Entries: {entry_count}")

        # 保存结果到CSV文件
        save_results_to_csv(results, self.output_dir / "results.csv")
        logger.info(f"Results saved to {self.output_dir / 'results.csv'}")

if __name__ == "__main__":
    MAX_DATASET_SIZE = 10000
    epochs = 10
    batch_size = 32
    total_iter = MAX_DATASET_SIZE * epochs

    cache_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cache_types = ["LRU", "LFU", "FIFO", "RR"]
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

    # 添加OPT缓存配置 - 为每个cache_size生成独立的预计算文件
    for size in cache_sizes:
        cache_size = int(size * MAX_DATASET_SIZE)
        if cache_size > 0:
            precomputed_path = precomputed_dir / f"ram_usage_opt_precomputed_{cache_size}.pkl"
            
            if not precomputed_path.exists():
                logger.info(f"生成预计算文件（maxsize={cache_size}）: {precomputed_path}")
                generate_precomputed_file(
                    dataset_size=MAX_DATASET_SIZE,
                    total_iterations=total_iter,
                    persist_path=precomputed_path,
                    random_seed=0,
                    replacement=True,
                    maxsize=cache_size
                )
                logger.info("预计算文件生成完成")

            caches.append((
                "OPT",
                size,
                OPTCacheDecorator(
                    precomputed_path=precomputed_path,
                    maxsize=cache_size,
                    total_iter=total_iter
                ),
            ))

    experiment = CacheExperiment(caches=caches,
                                 batch_size=batch_size,
                                 dataset=HitRateDataset(MAX_DATASET_SIZE),
                                 num_workers=0,
                                 epochs=epochs,
                                 )
    experiment.run()
#!/usr/bin/env python3
"""
Cache Performance Experiment - Simplified Implementation

This script focuses on recording training speed and final time cost.
"""

import csv
import sys
import time
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

from lib.hit_rate_dataset import HitRateDataset
from OPT4TorchDataSet.cachelib import OPTCacheDecorator
import tempfile

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
        
        # 写入命中率数据
        for size in cache_sizes:
            row = [size]
            for cache_type in cache_types:
                # 查找对应的命中率
                hit_rate = next((r["hit_rate"] for r in results 
                               if r["name"] == cache_type and r["cache_size"] == size), None)
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
                                 if r["name"] == cache_type and r["cache_size"] == size), None)
                row.append(miss_count if miss_count is not None else "")
            writer.writerow(row)
        
        # 添加一行空行作为分隔
        writer.writerow([])

        # 写入耗时数据
        writer.writerow(["Time Cost (s)"] + cache_types)
        for size in cache_sizes:
            row = [size]
            for cache_type in cache_types:
                # 查找对应的耗时
                time_cost = next((r["time_cost"] for r in results 
                                if r["name"] == cache_type and r["cache_size"] == size), None)
                row.append(f"{time_cost:.4f}" if time_cost is not None else "")
            writer.writerow(row)
        
        # 添加一行空行作为分隔
        writer.writerow([])

        # 写入total accesses数据（所有策略和缓存大小的总访问次数都相同）
        writer.writerow(["Total Accesses"] + [""] * (len(cache_types) - 1) + [results[0]["total_accesses"]])
    

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

    def _run_single_experiment(self) -> Tuple[float, float]:
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
        
        start_time = time.perf_counter()
        for batch_data in dataloader:
            pass
        end_time = time.perf_counter()
        
        time_cost = end_time - start_time
        miss_count = dataset.getMissCount()
        
        return miss_count, time_cost
    
    def run(self):
        """Run experiments using internal configuration"""
        
        logger.info("Starting Cache Performance Experiments")
        results = []
        
        for name, cache_size, cache in self.caches:
            self.dataset.setCache(cache)
            miss_count, time_cost = self._run_single_experiment()
            total_accesses = len(self.dataset) * self.epochs
            hit_rate = (total_accesses - miss_count) / total_accesses  # 命中率是 0-1 之间的小数
            
            results.append({
                "name": name,
                "cache_size": cache_size,
                "hit_rate": hit_rate,
                "miss_count": miss_count,
                "total_accesses": total_accesses,
                "time_cost": time_cost
            })
            
            self.dataset.resetMissCount()
            logger.info(f"{name} cache_size: {cache_size}, hit rate: {hit_rate:.2%}, time cost: {time_cost:.4f}s")

        # 保存结果到CSV文件
        save_results_to_csv(results, self.output_dir / "results.csv")
        logger.info(f"Results saved to {self.output_dir / 'results.csv'}")


if __name__ == "__main__":
    MAX_DATASET_SIZE = 10000
    epochs = 10
    batch_size = 32
    total_iter = MAX_DATASET_SIZE * epochs

    cache_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    prediction_windows = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # 窗口大小比例

    caches: List[Tuple[str, object]] = []

    # 创建临时文件用于存储预计算数据
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        opt_generator = torch.Generator()
        opt_generator.manual_seed(0)
        
        # 生成预计算数据
        from OPT4TorchDataSet.cachelib import generate_precomputed_file
        generate_precomputed_file(
            dataset_size=MAX_DATASET_SIZE,
            total_iterations=total_iter,
            persist_path=tmp_path,
            random_seed=0,
            replacement=True
        )

        # 添加WarmUp缓存（使用最小的窗口大小）
        caches.append(
            ("WarmUp", 0.1, OPTCacheDecorator(
                precomputed_path=tmp_path,
                maxsize=int(0.1 * MAX_DATASET_SIZE),
                prediction_window=int(total_iter * prediction_windows[0]),
                total_iter=total_iter,
            ))
        )

        # 循环创建不同缓存大小和窗口大小的组合
        for size in cache_sizes:
            for window in prediction_windows:
                caches.append((
                    f"OPT-{window}", 
                    size, 
                    OPTCacheDecorator(
                        precomputed_path=tmp_path,
                        maxsize=int(size * MAX_DATASET_SIZE),
                        prediction_window=int(total_iter * window),
                        total_iter=total_iter,
                    )
                ))

        experiment = CacheExperiment(caches=caches,
                                     batch_size=batch_size,
                                     dataset=HitRateDataset(MAX_DATASET_SIZE),
                                     num_workers=0,
                                     epochs=epochs,
                                     )
        experiment.run()
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
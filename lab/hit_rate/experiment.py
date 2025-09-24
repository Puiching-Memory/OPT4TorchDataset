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

from lib.hit_rate_dataset import HitRateDataset
from OPT4TorchDataSet.cachelib import make_opt_cache

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
        
        logger.info("Starting Cache Performance Experiments")
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
                "total_accesses": total_accesses
            })
            
            self.dataset.resetMissCount()
            logger.info(f"Cache: {name} hit rate: {hit_rate:.2%}")

        # 保存结果到CSV文件
        save_results_to_csv(results, self.output_dir / "results.csv")
        logger.info(f"Results saved to {self.output_dir / 'results.csv'}")

if __name__ == "__main__":
    MAX_DATASET_SIZE = 10000
    # 注意: 对于 OPT 策略, 我们需要预先创建对应 future 序列 (total_iter = len(dataset)*epochs)
    total_iter = MAX_DATASET_SIZE * 10  # epochs=10, 与下方 experiment 初始化保持一致

    # 占位: 其他策略仍使用 cachetools.cached 包装器
    CACHE_CONFIGS: List[Tuple[str, object]] = [
        ("LRU", 0.1, cached(LRUCache(maxsize=int(0.1 * MAX_DATASET_SIZE)))),
        ("LRU", 0.2, cached(LRUCache(maxsize=int(0.2 * MAX_DATASET_SIZE)))),
        ("LRU", 0.3, cached(LRUCache(maxsize=int(0.3 * MAX_DATASET_SIZE)))),
        ("LRU", 0.4, cached(LRUCache(maxsize=int(0.4 * MAX_DATASET_SIZE)))),
        ("LRU", 0.5, cached(LRUCache(maxsize=int(0.5 * MAX_DATASET_SIZE)))),
        ("LRU", 0.6, cached(LRUCache(maxsize=int(0.6 * MAX_DATASET_SIZE)))),
        ("LRU", 0.7, cached(LRUCache(maxsize=int(0.7 * MAX_DATASET_SIZE)))),
        ("LRU", 0.8, cached(LRUCache(maxsize=int(0.8 * MAX_DATASET_SIZE)))),
        ("LRU", 0.9, cached(LRUCache(maxsize=int(0.9 * MAX_DATASET_SIZE)))),
        ("LFU", 0.1, cached(LFUCache(maxsize=int(0.1 * MAX_DATASET_SIZE)))),
        ("LFU", 0.2, cached(LFUCache(maxsize=int(0.2 * MAX_DATASET_SIZE)))),
        ("LFU", 0.3, cached(LFUCache(maxsize=int(0.3 * MAX_DATASET_SIZE)))),
        ("LFU", 0.4, cached(LFUCache(maxsize=int(0.4 * MAX_DATASET_SIZE)))),
        ("LFU", 0.5, cached(LFUCache(maxsize=int(0.5 * MAX_DATASET_SIZE)))),
        ("LFU", 0.6, cached(LFUCache(maxsize=int(0.6 * MAX_DATASET_SIZE)))),
        ("LFU", 0.7, cached(LFUCache(maxsize=int(0.7 * MAX_DATASET_SIZE)))),
        ("LFU", 0.8, cached(LFUCache(maxsize=int(0.8 * MAX_DATASET_SIZE)))),
        ("LFU", 0.9, cached(LFUCache(maxsize=int(0.9 * MAX_DATASET_SIZE)))),
        ("FIFO", 0.1, cached(FIFOCache(maxsize=int(0.1 * MAX_DATASET_SIZE)))),
        ("FIFO", 0.2, cached(FIFOCache(maxsize=int(0.2 * MAX_DATASET_SIZE)))),
        ("FIFO", 0.3, cached(FIFOCache(maxsize=int(0.3 * MAX_DATASET_SIZE)))),
        ("FIFO", 0.4, cached(FIFOCache(maxsize=int(0.4 * MAX_DATASET_SIZE)))),
        ("FIFO", 0.5, cached(FIFOCache(maxsize=int(0.5 * MAX_DATASET_SIZE)))),
        ("FIFO", 0.6, cached(FIFOCache(maxsize=int(0.6 * MAX_DATASET_SIZE)))),
        ("FIFO", 0.7, cached(FIFOCache(maxsize=int(0.7 * MAX_DATASET_SIZE)))),
        ("FIFO", 0.8, cached(FIFOCache(maxsize=int(0.8 * MAX_DATASET_SIZE)))),
        ("FIFO", 0.9, cached(FIFOCache(maxsize=int(0.9 * MAX_DATASET_SIZE)))), 
        ("RR", 0.1, cached(RRCache(maxsize=int(0.1 * MAX_DATASET_SIZE)))),
        ("RR", 0.2, cached(RRCache(maxsize=int(0.2 * MAX_DATASET_SIZE)))),
        ("RR", 0.3, cached(RRCache(maxsize=int(0.3 * MAX_DATASET_SIZE)))),
        ("RR", 0.4, cached(RRCache(maxsize=int(0.4 * MAX_DATASET_SIZE)))),
        ("RR", 0.5, cached(RRCache(maxsize=int(0.5 * MAX_DATASET_SIZE)))),
        ("RR", 0.6, cached(RRCache(maxsize=int(0.6 * MAX_DATASET_SIZE)))),
        ("RR", 0.7, cached(RRCache(maxsize=int(0.7 * MAX_DATASET_SIZE)))),
        ("RR", 0.8, cached(RRCache(maxsize=int(0.8 * MAX_DATASET_SIZE)))),
        ("RR", 0.9, cached(RRCache(maxsize=int(0.9 * MAX_DATASET_SIZE)))), 
        # OPT 新接口: 使用 make_opt_cache 工厂。为公平比较, 对每种缓存大小建立独立 context。
        ("OPT", 0.1, make_opt_cache(
            sampler=RandomSampler,
            generator=torch.Generator().manual_seed(0) or torch.Generator(),  # manual_seed 返回 None, 用 or 链接保留对象
            total_iter=total_iter,
            maxsize=int(0.1 * MAX_DATASET_SIZE),
            assert_alignment=False,
            name="OPT-0.1"
        )),
        ("OPT", 0.2, make_opt_cache(
            sampler=RandomSampler,
            generator=torch.Generator().manual_seed(0) or torch.Generator(),
            total_iter=total_iter,
            maxsize=int(0.2 * MAX_DATASET_SIZE),
            name="OPT-0.2"
        )),
        ("OPT", 0.3, make_opt_cache(
            sampler=RandomSampler,
            generator=torch.Generator().manual_seed(0) or torch.Generator(),
            total_iter=total_iter,
            maxsize=int(0.3 * MAX_DATASET_SIZE),
            name="OPT-0.3"
        )),
        ("OPT", 0.4, make_opt_cache(
            sampler=RandomSampler,
            generator=torch.Generator().manual_seed(0) or torch.Generator(),
            total_iter=total_iter,
            maxsize=int(0.4 * MAX_DATASET_SIZE),
            name="OPT-0.4"
        )),
        ("OPT", 0.5, make_opt_cache(
            sampler=RandomSampler,
            generator=torch.Generator().manual_seed(0) or torch.Generator(),
            total_iter=total_iter,
            maxsize=int(0.5 * MAX_DATASET_SIZE),
            name="OPT-0.5"
        )),
        ("OPT", 0.6, make_opt_cache(
            sampler=RandomSampler,
            generator=torch.Generator().manual_seed(0) or torch.Generator(),
            total_iter=total_iter,
            maxsize=int(0.6 * MAX_DATASET_SIZE),
            name="OPT-0.6"
        )),
        ("OPT", 0.7, make_opt_cache(
            sampler=RandomSampler,
            generator=torch.Generator().manual_seed(0) or torch.Generator(),
            total_iter=total_iter,
            maxsize=int(0.7 * MAX_DATASET_SIZE),
            name="OPT-0.7"
        )),
        ("OPT", 0.8, make_opt_cache(
            sampler=RandomSampler,
            generator=torch.Generator().manual_seed(0) or torch.Generator(),
            total_iter=total_iter,
            maxsize=int(0.8 * MAX_DATASET_SIZE),
            name="OPT-0.8"
        )),
        ("OPT", 0.9, make_opt_cache(
            sampler=RandomSampler,
            generator=torch.Generator().manual_seed(0) or torch.Generator(),
            total_iter=total_iter,
            maxsize=int(0.9 * MAX_DATASET_SIZE),
            name="OPT-0.9"
        )),
    ]
    experiment = CacheExperiment(caches=CACHE_CONFIGS,
                                 batch_size=32,
                                 dataset=HitRateDataset(MAX_DATASET_SIZE),
                                 num_workers=0,
                                 epochs=10,
                                 )
    experiment.run()
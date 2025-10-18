#!/usr/bin/env python3
"""
缓存命中率实验

该脚本用于比较不同缓存替换策略（LRU、LFU、FIFO、RR和OPT）在模拟数据集上的命中率表现。
通过测量各种缓存大小配置下的命中率和未命中次数，评估各策略的有效性。
"""

import csv
import sys
import torch
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler
import os
from typing import List, Dict, Tuple
from pathlib import Path
from cachetools import cached, LRUCache, LFUCache, FIFOCache, RRCache

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
        """运行所有缓存实验并保存结果"""
        
        logger.info("Starting Cache Performance Experiments")
        results = []
        
        for name, cache_size, cache in self.caches:
            # 对于OPT缓存，需要重置状态
            if isinstance(cache, OPTCacheDecorator):
                cache.reset()
            
            miss_count = self._run_single_experiment(cache)
            total_accesses = len(self.dataset) * self.epochs
            hit_rate = (total_accesses - miss_count) / total_accesses  # 命中率是 0-1 之间的小数
            
            results.append({
                "name": name,
                "cache_size": cache_size,
                "hit_rate": hit_rate,
                "miss_count": miss_count,
                "total_accesses": total_accesses
            })
            
            logger.info(f"Cache: {name} hit rate: {hit_rate:.2%}")

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
    
    # 添加OPT缓存配置 - 为每个cache_size生成独立的预计算文件
    precomputed_dir = os.path.join(ROOT, 'precomputed')
    os.makedirs(precomputed_dir, exist_ok=True)
    
    for size in cache_sizes:
        cache_size = int(size * MAX_DATASET_SIZE)
        if cache_size > 0:  # 避免缓存大小为0的情况
            # 为每个cache_size创建独立的预计算文件
            precomputed_path = os.path.join(precomputed_dir, f'hit_rate_experiment_opt_{cache_size}.pkl')
            
            # 检查预计算文件是否存在，不存在则生成
            if not os.path.exists(precomputed_path):
                logger.info(f"生成预计算文件（maxsize={cache_size}）: {precomputed_path}")
                
                # 使用新的API生成预计算文件，包含maxsize参数
                generate_precomputed_file(
                    dataset_size=MAX_DATASET_SIZE,
                    total_iterations=total_iter,
                    persist_path=precomputed_path,
                    random_seed=0,  # 与 HitRateDataset 使用相同的种子
                    replacement=True,
                    maxsize=cache_size
                )
                
                logger.info(f"预计算文件生成完成: {precomputed_path}")
            
            opt_decorator = OPTCacheDecorator(
                precomputed_path=precomputed_path,
                maxsize=cache_size,
                total_iter=total_iter
            )
            caches.append(("OPT", size, opt_decorator))

    experiment = CacheExperiment(caches=caches,
                                 batch_size=batch_size,
                                 dataset=HitRateDataset(MAX_DATASET_SIZE),
                                 num_workers=0,
                                 epochs=epochs,
                                 )
    experiment.run()
#!/usr/bin/env python3
"""
Cache Performance Experiment - Simplified Implementation

This script focuses on recording training speed and final time cost.
"""

import json
import sys
import torch
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler
import os
from typing import List, Dict
from pathlib import Path
from cachetools import cached, LRUCache, LFUCache, FIFOCache, RRCache

# 添加项目根目录和src目录到Python路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

from lib.hit_rate_dataset import HitRateDataset


class ExperimentConfig:
    """Central configuration for all experiments"""
    output_dir: str = "results"
    batch_size: int = 32
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = HitRateDataset()
    
    # 缓存类型配置字典
    cache_configs: Dict[str, object] = {
        "None": None,
        "LRU-1000": cached(LRUCache(maxsize=1000)),
        "LFU-1000": cached(LFUCache(maxsize=1000)),
        "FIFO-1000": cached(FIFOCache(maxsize=1000)),
        "RR-1000": cached(RRCache(maxsize=1000))
    }


class CacheExperiment:
    
    def __init__(self,
                 cache_configs: Dict[str, object] = {"None": None},
                 output_dir: str = "results",
                 batch_size: int = 32,
                 num_workers: int = 0,
                 dataset: HitRateDataset = HitRateDataset(),
                 device: str = "cpu",
                 ):
        self.cache_configs = cache_configs
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.device = device

        self.output_dir.mkdir(exist_ok=True)

    def _run_single_experiment(self) -> float:
        """Run single experiment focused on speed and time cost"""
        
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                sampler=RandomSampler(self.dataset,
                                                      replacement=True,
                                                      num_samples=len(self.dataset),
                                                      generator=self.dataset.getGenerator()
                                                      )
                                )
        
        for batch_data in dataloader:
            pass

        return self.dataset.getMissCount()
    
    def run(self):
        """Run experiments using internal configuration"""
        
        logger.info("Starting Cache Performance Experiments")
        results = []
        
        for name, cache in self.cache_configs.items():
            if cache is not None:
                self.dataset.setCache(cache)

            result = self._run_single_experiment()
            miss_count = self.dataset.getMissCount()
            hit_rate = (len(self.dataset) - miss_count) / len(self.dataset)
            
            results.append({
                "cache_type": name,
                "hit_rate": hit_rate,
                "miss_count": miss_count
            })
            
            self.dataset.resetMissCount()
            logger.info(f"Cache: {name}, hit rate: {hit_rate:.2%}")

        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Results saved to {self.output_dir / 'results.json'}")


if __name__ == "__main__":
    CONFIG = ExperimentConfig()
    experiment = CacheExperiment(cache_configs=CONFIG.cache_configs,
                                 output_dir=CONFIG.output_dir,
                                 batch_size=CONFIG.batch_size,
                                 num_workers=CONFIG.num_workers,
                                 dataset=CONFIG.dataset,
                                 device=CONFIG.device
                                 )
    experiment.run()
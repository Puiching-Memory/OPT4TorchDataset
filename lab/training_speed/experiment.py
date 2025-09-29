#!/usr/bin/env python3
"""
Cache Performance Experiment - Simplified Implementation

This script focuses on recording training speed and final time cost.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List
import torch
import timm
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import os

# 添加项目根目录和src目录到Python路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

from lib.mini_imagenet_dataset import MiniImageNetDataset


class ExperimentConfig:
    """Central configuration for all experiments"""

    output_dir: str = "results"
    models: List[str] = ["resnet50",
                         "efficientnet_b0",
                         "mobilenetv4_conv_small",
                         ]
    
    # Training parameters
    batch_size: int = 128
    num_workers: int = 8
    enable_amp: bool = True

    # Dataset
    dataset = MiniImageNetDataset(split='train')

class CacheExperiment:
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _run_single_experiment(self, model_name: str) -> float:
        """Run single experiment focused on speed and time cost"""
        
        dataset = CONFIG.dataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model(model_name, pretrained=False, num_classes=100)
        model = model.to(device)
        
        generator = torch.Generator()
        generator.manual_seed(42)
        sampler = RandomSampler(dataset, 
                              replacement=False, 
                              generator=generator)
        dataloader = DataLoader(dataset, 
                              batch_size=CONFIG.batch_size, 
                              sampler=sampler, 
                              num_workers=CONFIG.num_workers)
        
        model.train()
        scaler = torch.amp.grad_scaler.GradScaler()
        optimizer = torch.optim.AdamW(model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Train for one epoch and measure timing
        start_time = time.perf_counter()
        
        # Add progress bar for training
        pbar = tqdm(dataloader, desc=f"Training {model_name}")
        
        for batch_data in pbar:
            images, labels = batch_data
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=CONFIG.enable_amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        total_training_time = time.perf_counter() - start_time
        logger.info(f"Total time: {total_training_time:.4f}s")
        
        return total_training_time
    
    def run(self):
        """Run experiments using internal configuration"""
        
        logger.info("Starting Cache Performance Experiments")
        results = []
        
        # Use direct configuration
        models = CONFIG.models
    
        for model in models:
            logger.info("="*50)
            logger.info(f"Model: {model}")
            result = self._run_single_experiment(model)
            results.append(
                {
                    "model_name": model,
                    "training_time": result,
                }
                )

        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {self.output_dir / 'results.json'}")


if __name__ == "__main__":
    CONFIG = ExperimentConfig()
    experiment = CacheExperiment(output_dir=CONFIG.output_dir)
    experiment.run()
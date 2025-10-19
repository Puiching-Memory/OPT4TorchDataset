#!/usr/bin/env python3

import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import math
import torch
import timm
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import os
import csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

# from lib.mini_imagenet_dataset import MiniImageNetDataset
from lib.ramRGB_dataset import RandomRGBDataset
from OPT4TorchDataSet.cachelib import OPTCacheDecorator, generate_precomputed_file
from OPT4TorchDataSet.logger import ExperimentLogger

from cachetools import cached, LRUCache, LFUCache, FIFOCache, RRCache

class ExperimentConfig:
    output_dir: str = "results"
    models: List[str] = [
        "resnet50",
        "vit_base_patch16_224",
        "swin_base_patch4_window7_224",
        "mobilenetv3_small_100",
        "vit_base_patch16_dinov3",
        "convnextv2_base",
        "swinv2_cr_base_224",
        "davit_base",
        "mobilenetv5_base"
    ]
    
    batch_size: int = 16
    num_workers: int = 0  # 改为 0 以避免 Windows 多进程 pickle 问题
    enable_amp: bool = True
    epochs: int = 5
    pin_memory: bool = True

    # dataset_class = MiniImageNetDataset
    # dataset_params: Dict[str, Any] = {"split": "train"}
    dataset_class = RandomRGBDataset
    dataset_params: Dict[str, Any] = {"data_dir": "../../data/random_rgb_dataset"}

    cache_types: List[str] = ["warmUp", "OPT", "none", "LRU", "LFU", "FIFO", "RR"]
    # cache_types: List[str] = ["OPT"]
    cache_size_ratio: float = 0.3
    
    # Logging configuration
    log_backends: List[str] = ["swanlab"]  # 支持的日志后端：swanlab, tensorboard
    log_dir: str = "runs/experiment"  # TensorBoard 日志目录
    swanlab_project: str = "opt4"
    swanlab_workspace: str = "Sail2Dream"


def save_results_to_csv(results: List[Dict], output_path: Path):
    """
    Save experiment results to CSV file
    """
    models = sorted(list(set(result["model_name"] for result in results)))
    cache_types = sorted(list(set(result["cache_type"] for result in results)))
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        header = ["Model"] + [f"{cache_type} Time(s)" for cache_type in cache_types]
        writer.writerow(header)
        
        for model in models:
            # 时间行
            time_row = [f"{model} Time"]
            for cache_type in cache_types:
                training_time = next((r["training_time"] for r in results 
                                    if r["model_name"] == model and r["cache_type"] == cache_type), None)
                time_row.append(f"{training_time:.4f}" if training_time is not None else "")
            writer.writerow(time_row)
        
        writer.writerow([])
        
        writer.writerow(["Batch Size"] + [""] * (len(cache_types) - 1) + [results[0]["batch_size"]])
        writer.writerow(["Num Workers"] + [""] * (len(cache_types) - 1) + [results[0]["num_workers"]])
        writer.writerow(["AMP Enabled"] + [""] * (len(cache_types) - 1) + [results[0]["enable_amp"]])
        writer.writerow(["Epochs"] + [""] * (len(cache_types) - 1) + [results[0]["epochs"]])
        writer.writerow(["Cache Size Ratio"] + [""] * (len(cache_types) - 1) + [results[0]["cache_size_ratio"]])


class CacheExperiment:
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize experiment logger
        self.exp_logger = ExperimentLogger(backends=config.log_backends)
        logger_config = {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "enable_amp": config.enable_amp,
            "epochs": config.epochs,
            "cache_size_ratio": config.cache_size_ratio,
            "dataset_class": config.dataset_class.__name__,
            "cache_types": config.cache_types,
            "project": config.swanlab_project,
            "workspace": config.swanlab_workspace,
            "log_dir": config.log_dir,
        }
        self.exp_logger.init(logger_config)
        
        # Generate OPT cache precomputed file if needed
        if "OPT" in self.config.cache_types:
            self._prepare_opt_cache()

    def _prepare_opt_cache(self):
        """Prepare OPT cache precomputed file"""
        dataset = self.config.dataset_class(**self.config.dataset_params)
        dataset_size = len(dataset)
        total_iterations = dataset_size * self.config.epochs
        cache_size = int(dataset_size * self.config.cache_size_ratio)
        
        # 使用统一的预计算目录
        precomputed_dir = Path(__file__).parent.parent.parent / "precomputed"
        precomputed_dir.mkdir(parents=True, exist_ok=True)
        self.precomputed_path = precomputed_dir / "opt_precomputed_training_speed.pkl"
        
        # 如果文件不存在则生成
        if not self.precomputed_path.exists():
            generate_precomputed_file(
                dataset_size=dataset_size,
                total_iterations=total_iterations,
                persist_path=self.precomputed_path,
                random_seed=42,
                replacement=True,
                maxsize=cache_size
            )
        
        logger.info(f"Using OPT precomputed file: {self.precomputed_path}")

    def _setup_dataset(self, cache_type: str):
        """Setup dataset with specified cache type"""
        dataset = self.config.dataset_class(**self.config.dataset_params)
        dataset_size = len(dataset)
        cache_size = int(dataset_size * self.config.cache_size_ratio)
        
        if cache_type == "none":
            # No cache - do nothing
            pass
        elif cache_type in ["LRU", "LFU", "FIFO", "RR"]:
            cache_classes = {
                "LRU": LRUCache,
                "LFU": LFUCache,
                "FIFO": FIFOCache,
                "RR": RRCache
            }
            
            cache = cache_classes[cache_type](maxsize=cache_size)
            # 使用 setCache 方法应用缓存装饰器
            dataset.setCache(cached(cache))
        elif cache_type == "OPT":
            total_iterations = dataset_size * self.config.epochs
            
            opt_cache = OPTCacheDecorator(
                precomputed_path=self.precomputed_path,
                maxsize=cache_size,
                total_iter=total_iterations
            )
            
            # 使用 setCache 方法应用 OPT 缓存装饰器
            dataset.setCache(opt_cache)
            
        return dataset

    def _run_single_experiment(self, model_name: str, cache_type: str) -> dict:
        """Run single experiment focused on speed and time cost"""
        
        dataset = self._setup_dataset(cache_type)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model(model_name, pretrained=False, num_classes=100)
        model = model.to(device)

        dataset_size = len(dataset)
        total_iterations = dataset_size * self.config.epochs
        batches_per_epoch = max(1, math.ceil(dataset_size / self.config.batch_size))
        total_batches = math.ceil(total_iterations / self.config.batch_size)

        generator = torch.Generator()
        generator.manual_seed(42)
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=total_iterations,
            generator=generator,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
            pin_memory=self.config.pin_memory,
        )
        
        model.train()
        scaler = torch.amp.grad_scaler.GradScaler()
        optimizer = torch.optim.AdamW(model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Train for one epoch and measure timing
        start_time = time.perf_counter()
        
        # Add progress bar for training
        pbar = tqdm(
            dataloader,
            total=total_batches,
            desc=f"Training {model_name} with {cache_type} cache",
        )
        
        for step, batch_data in enumerate(pbar, start=1):
            images, labels = batch_data
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.config.enable_amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            current_epoch = 1 + (step - 1) // batches_per_epoch
            next_epoch_boundary = step % batches_per_epoch == 0 or step == total_batches
            if next_epoch_boundary:
                pbar.set_postfix(epoch=f"{current_epoch}/{self.config.epochs}", loss=float(loss.item()))
                
                # Log metrics to experiment logger (only numeric values for SwanLab compatibility)
                self.exp_logger.log({
                    "loss": float(loss.item()),
                    "epoch": current_epoch,
                })
        
        total_training_time = time.perf_counter() - start_time
        logger.info(f"Total time: {total_training_time:.4f}s")
        
        # Log final metrics to experiment logger (only numeric values for SwanLab compatibility)
        self.exp_logger.log({
            "training_time": total_training_time,
        })
        
        # Collect results
        result = {
            "model_name": model_name,
            "cache_type": cache_type,
            "training_time": total_training_time,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "enable_amp": self.config.enable_amp,
            "dataset_class": self.config.dataset_class.__name__,
            "cache_size_ratio": self.config.cache_size_ratio,
            "epochs": self.config.epochs,
        }
        
        return result
    
    def run(self):
        """Run experiments using internal configuration"""
        
        logger.info("Starting Cache Performance Experiments")
        results = []
        
        models = self.config.models
        cache_types = self.config.cache_types
    
        try:
            for model in models:
                for cache_type in cache_types:
                    logger.info("="*50)
                    logger.info(f"Model: {model}, Cache: {cache_type}")
                    result = self._run_single_experiment(model, cache_type)
                    results.append(result)

            save_results_to_csv(results, self.output_dir / "results.csv")
            logger.info(f"Results saved to {self.output_dir / 'results.csv'}")
        finally:
            # Finish experiment logger
            self.exp_logger.finish()


if __name__ == "__main__":
    config = ExperimentConfig()
    experiment = CacheExperiment(config)
    experiment.run()
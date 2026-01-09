#!/usr/bin/env python3

import sys
import time
import json
import math
from pathlib import Path
from typing import List, Dict, Any

import torch
import timm
import typer
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from cachetools import LRUCache, LFUCache, FIFOCache, RRCache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lib.ramRGB_dataset import RandomRGBDataset
from OPT4TorchDataSet.cachelib import (
    OPTCacheDecorator,
    SharedOPTCacheDecorator,
    CachetoolsDecorator,
    generate_precomputed_file,
)

# Setup logging
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(OUTPUT_DIR / "experiment.log", rotation="10 MB")


class ExperimentConfig:
    output_dir: Path = OUTPUT_DIR
    models: List[str] = [
        "resnet18",
        "resnet50",
        "mobilenetv3_small_100",
        "vit_tiny_patch16_224",
    ]

    batch_size: int = 32
    num_workers: int = 4
    enable_amp: bool = True
    epochs: int = 3  # 增加模型和比例后，调低 Epoch 以保持总时长可控
    warmup_epochs: int = 1  # 预热轮次
    pin_memory: bool = True

    # dataset_class = MiniImageNetDataset
    # dataset_params: Dict[str, Any] = {"split": "train"}
    dataset_class = RandomRGBDataset
    dataset_params: Dict[str, Any] = {"data_dir": str(PROJECT_ROOT / "data" / "random_rgb_dataset")}

    cache_types: List[str] = ["none", "LRU", "LFU", "FIFO", "RR", "OPT"]
    # cache_types: List[str] = ["OPT"]
    cache_size_ratios: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Logging configuration
    swanlab_workspace: str = "Sail2Dream"
    log_dir: str = str(OUTPUT_DIR / "tensorboard")  # TensorBoard 日志目录


class CacheExperiment:

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(exist_ok=True)

    def _prepare_opt_cache(self, cache_size_ratio: float):
        """Prepare OPT cache precomputed file"""
        dataset = self.config.dataset_class(**self.config.dataset_params)
        dataset_size = len(dataset)
        total_iterations = dataset_size * self.config.epochs
        cache_size = int(dataset_size * cache_size_ratio)

        # 使用统一的预计算目录
        precomputed_dir = PROJECT_ROOT / "precomputed"
        precomputed_dir.mkdir(parents=True, exist_ok=True)
        self.precomputed_path = (
            precomputed_dir
            / f"opt_precomputed_training_speed_{cache_size_ratio}.safetensors"
        )

        # 如果文件不存在则生成
        if not self.precomputed_path.exists():
            logger.info(f"Generating precomputed file for ratio {cache_size_ratio}...")
            generate_precomputed_file(
                dataset_size=dataset_size,
                total_iterations=total_iterations,
                persist_path=self.precomputed_path,
                random_seed=42,
                replacement=True,
                maxsize=cache_size,
            )

        logger.info(f"Using OPT precomputed file: {self.precomputed_path}")
        return self.precomputed_path

    def _setup_dataset(self, cache_type: str, cache_size_ratio: float):
        """Setup dataset with specified cache type"""
        dataset = self.config.dataset_class(**self.config.dataset_params)
        dataset_size = len(dataset)
        cache_size = int(dataset_size * cache_size_ratio)

        if cache_type == "none":
            # No cache - do nothing
            pass
        elif cache_type in ["LRU", "LFU", "FIFO", "RR"]:
            cache_classes = {
                "LRU": LRUCache,
                "LFU": LFUCache,
                "FIFO": FIFOCache,
                "RR": RRCache,
            }

            cache = cache_classes[cache_type](maxsize=cache_size)
            # 使用 CachetoolsDecorator 包装以支持 Windows 多进程 (Picklable)
            dataset.setCache(CachetoolsDecorator(cache))
        elif cache_type == "OPT":
            total_iterations = dataset_size * self.config.epochs

            if self.config.num_workers > 0:
                # 获取样本形状和类型以初始化共享内存
                sample_img, _ = dataset[0]
                opt_cache = SharedOPTCacheDecorator(
                    precomputed_path=self.precomputed_path,
                    maxsize=cache_size,
                    dataset_size=dataset_size,
                    item_shape=sample_img.shape,
                    item_dtype=sample_img.dtype,
                )
            else:
                opt_cache = OPTCacheDecorator(
                    precomputed_path=self.precomputed_path,
                    maxsize=cache_size,
                    total_iter=total_iterations,
                )

            # 使用 setCache 方法应用 OPT 缓存装饰器
            dataset.setCache(opt_cache)

        return dataset

    def _run_single_experiment(
        self, model_name: str, cache_type: str, cache_size_ratio: float
    ) -> dict:
        """Run single experiment focused on speed and time cost"""

        dataset = self._setup_dataset(cache_type, cache_size_ratio)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model(model_name, pretrained=False, num_classes=100)
        model = model.to(device)

        dataset_size = len(dataset)
        warmup_iterations = dataset_size * self.config.warmup_epochs
        timed_iterations = dataset_size * self.config.epochs
        
        total_iterations = warmup_iterations + timed_iterations
        batches_per_epoch = max(1, math.ceil(dataset_size / self.config.batch_size))
        
        warmup_batches = warmup_iterations // self.config.batch_size
        total_batches = total_iterations // self.config.batch_size
        timed_batches = total_batches - warmup_batches

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

        # Add progress bar for training
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[postfix]}"),
        ) as progress:
            task = progress.add_task(
                f"Training {model_name} (Cache: {cache_type}, Ratio: {cache_size_ratio})",
                total=total_batches,
                postfix="",
            )

            # Warm-up phase
            if self.config.warmup_epochs > 0:
                progress.update(task, description=f"Warmup {model_name}")
                data_iter = iter(dataloader)
                for _ in range(warmup_batches):
                    batch_data = next(data_iter)
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
                    
                    progress.update(task, advance=1, postfix="[Warmup]")
            else:
                data_iter = iter(dataloader)

            # Timed phase
            progress.update(task, description=f"Training {model_name}")
            start_time = time.perf_counter()
            
            for step in range(1, timed_batches + 1):
                batch_data = next(data_iter)
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
                progress.update(
                    task,
                    advance=1,
                    postfix=f" [bold blue]Epoch:[/] {current_epoch}/{self.config.epochs} [bold red]Loss:[/] {loss.item():.4f}",
                )

            total_training_time = time.perf_counter() - start_time
            logger.info(f"Total time (timed): {total_training_time:.4f}s")

        # Collect results
        result = {
            "model_name": model_name,
            "cache_type": cache_type,
            "cache_size_ratio": cache_size_ratio,
            "training_time": total_training_time,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "enable_amp": self.config.enable_amp,
            "dataset_class": self.config.dataset_class.__name__,
            "epochs": self.config.epochs,
        }

        return result

    def run(self):
        """Run experiments using internal configuration"""

        logger.info("Starting Cache Performance Experiments")
        results = []

        models = self.config.models
        cache_types = self.config.cache_types
        cache_size_ratios = self.config.cache_size_ratios

        try:
            for cache_size_ratio in cache_size_ratios:
                # Prepare OPT cache for current ratio
                if "OPT" in cache_types:
                    self.precomputed_path = self._prepare_opt_cache(cache_size_ratio)

                for model in models:
                    for cache_type in cache_types:
                        logger.info("=" * 50)
                        logger.info(
                            f"Model: {model}, Cache: {cache_type}, Ratio: {cache_size_ratio}"
                        )
                        result = self._run_single_experiment(
                            model, cache_type, cache_size_ratio
                        )
                        results.append(result)

            if results:
                results_path = self.output_dir / "results.json"
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                logger.info(f"Results saved to {results_path}")
            else:
                logger.warning("No results to save.")
        finally:
            # Finish experiment logger
            # self.exp_logger.finish()
            pass


app = typer.Typer()


@app.command()
def main(
    epochs: int = typer.Option(5, help="Number of epochs to train"),
    warmup_epochs: int = typer.Option(1, help="Number of warmup epochs"),
    batch_size: int = typer.Option(16, help="Batch size for training"),
    num_workers: int = typer.Option(0, help="Number of workers for dataloader"),
    use_amp: bool = typer.Option(True, help="Enable automatic mixed precision"),
    cache_types: str = typer.Option("none,LRU,LFU,FIFO,RR,OPT", help="Comma-separated list of cache types"),
):
    """
    Run the training speed experiment comparing different cache strategies.
    """
    config = ExperimentConfig()
    config.epochs = epochs
    config.warmup_epochs = warmup_epochs
    config.batch_size = batch_size
    config.num_workers = num_workers
    config.enable_amp = use_amp
    config.cache_types = cache_types.split(",")

    experiment = CacheExperiment(config)
    experiment.run()


if __name__ == "__main__":
    app()

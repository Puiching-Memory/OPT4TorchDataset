"""Benchmark and Validation: Shared Memory OPT Cache.

This script combines performance comparison (Throughput) and functional validation (Hit Rate)
for different cache implementations:
1. No Cache (Baseline) - Raw IO performance.
2. Local Python Cache (OPTCacheDecorator) - Memory redundant but simple.
3. Shared C++ Cache (SharedOPTCacheDecorator) - High performance, cross-process shared.
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple

import torch
import typer
from torch.utils.data import DataLoader, Dataset
from torch import multiprocessing as mp
from loguru import logger

# Ensure repository root and src are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from OPT4TorchDataSet.cachelib import (
    generate_precomputed_file,
    OPTCacheDecorator,
    SharedOPTCacheDecorator,
)

# Setup logging
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(OUTPUT_DIR / "experiment.log", rotation="10 MB")


class HeavyDataset(Dataset):
    """A dataset that simulates heavy computation or IO."""
    def __init__(
        self,
        size: int,
        compute_time: float = 0.002,
        cache_obj: Optional[Callable] = None,
        shape: Tuple[int, ...] = (3, 64, 64)
    ):
        self.size = size
        self.compute_time = compute_time
        self.cache_obj = cache_obj
        # Generate dummy data
        self.data = torch.randn(size, *shape)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.cache_obj:
            # Wrap the fetch call inside the cache decorator
            return self.cache_obj(self._fetch)(index)
        return self._fetch(index)

    def _fetch(self, index):
        # Simulate heavy IO/Computation
        if self.compute_time > 0:
            time.sleep(self.compute_time)
        return self.data[index]


def run_test(
    label: str,
    dataset: Dataset,
    num_workers: int,
    total_iters: int,
    seed: int,
    batch_size: int = 16,
    stats_func: Optional[Callable] = None,
):
    sampler = torch.utils.data.RandomSampler(
        list(range(len(dataset))),
        replacement=True,
        num_samples=total_iters,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
    )

    logger.info(f"Running [{label}] with {num_workers} workers...")
    start = time.time()
    count = 0
    try:
        for _ in loader:
            count += 1
    except Exception as e:
        logger.error(f"Test [{label}] failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0

    end = time.time()
    duration = end - start
    throughput = total_iters / duration

    hit_rate = 0.0
    if stats_func:
        stats = stats_func()
        hit_rate = stats.get("hit_rate", 0.0) * 100

    logger.info(f"  Result: {throughput:.2f} items/s, Hit Rate: {hit_rate:.2f}%")
    return throughput, hit_rate


app = typer.Typer(help="Shared OPT Cache benchmark and validation tool.")


@app.command()
def perf(
    dataset_size: int = typer.Option(1000, help="Dataset size"),
    cache_ratio: float = typer.Option(0.2, help="Cache ratio (0.0 - 1.0)"),
    total_iters: int = typer.Option(5000, help="Total iterations to simulate"),
    workers: int = typer.Option(4, help="Number of DataLoader workers"),
    sim_time: float = typer.Option(0.005, help="Simulated IO/Compute time per item (seconds)"),
):
    """
    Performance Comparison: Compare No-Cache vs Local-Python-Cache vs Shared-C++-Cache.
    """
    mp.freeze_support()
    maxsize = int(dataset_size * cache_ratio)
    seed = 123
    precomputed_path = OUTPUT_DIR / "perf_bench.safetensors"
    item_shape = (3, 64, 64)

    logger.info("=== Performance Benchmark Setup ===")
    logger.info(f"Dataset: {dataset_size}, Cache: {maxsize} ({cache_ratio*100}%)")
    logger.info(f"Workers: {workers}, Sim Time: {sim_time*1000}ms/item")

    # 1. 预计算路径
    generate_precomputed_file(
        dataset_size=dataset_size,
        total_iterations=total_iters,
        persist_path=precomputed_path,
        random_seed=seed,
        maxsize=maxsize,
    )

    results_data = []

    # Scenario A: No Cache
    ds_none = HeavyDataset(dataset_size, sim_time, shape=item_shape)
    t, h = run_test("No Cache", ds_none, workers, total_iters, seed)
    results_data.append({"method": "No Cache", "throughput": t, "hit_rate": h})

    # Scenario B: Local Python Cache
    local_cache = OPTCacheDecorator(precomputed_path, maxsize, total_iters)
    ds_local = HeavyDataset(dataset_size, sim_time, cache_obj=local_cache, shape=item_shape)
    t, h = run_test("Local Cache (Py)", ds_local, workers, total_iters, seed, stats_func=local_cache.stats)
    results_data.append({"method": "Local Cache (Py)", "throughput": t, "hit_rate": h})

    # Scenario C: Shared C++ Cache
    shared_cache = SharedOPTCacheDecorator(
        precomputed_path=precomputed_path,
        maxsize=maxsize,
        dataset_size=dataset_size,
        item_shape=item_shape,
    )
    ds_shared = HeavyDataset(dataset_size, sim_time, cache_obj=shared_cache, shape=item_shape)
    t, h = run_test("Shared Cache (C++)", ds_shared, workers, total_iters, seed, stats_func=shared_cache.stats)
    results_data.append({"method": "Shared Cache (C++)", "throughput": t, "hit_rate": h})

    # Print summary table
    logger.info("\n" + "=" * 55)
    logger.info(f"{'Method':<25} | {'Throughput':<15} | {'Hit Rate':<10}")
    logger.info("-" * 55)
    for res in results_data:
        logger.info(f"{res['method']:<25} | {res['throughput']:>10.2f} it/s | {res['hit_rate']:>8.2f}%")
    logger.info("=" * 55)

    # Save
    with open(OUTPUT_DIR / "perf_results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
    logger.info(f"Results saved to {OUTPUT_DIR / 'perf_results.json'}")

    if precomputed_path.exists():
        precomputed_path.unlink()


@app.command()
def validate(
    workers: int = typer.Option(4, help="Number of worker processes"),
    iterations: int = typer.Option(2000, help="Total iterations to run"),
    cache_ratio: float = typer.Option(0.2, help="Cache size ratio"),
):
    """
    Functional Validation: Verify SharedOPTCacheDecorator consistency across multiple workers.
    """
    mp.freeze_support()
    dataset_size = 500
    maxsize = int(dataset_size * cache_ratio)
    precomputed_path = OUTPUT_DIR / "bench_shared_validate.safetensors"
    seed = 42

    logger.info("=== Functional Validation ===")
    generate_precomputed_file(
        dataset_size=dataset_size,
        total_iterations=iterations,
        persist_path=precomputed_path,
        random_seed=seed,
        maxsize=maxsize,
    )

    opt_cache = SharedOPTCacheDecorator(
        precomputed_path=precomputed_path,
        maxsize=maxsize,
        dataset_size=dataset_size,
        item_shape=(3, 32, 32),
    )

    dataset = HeavyDataset(dataset_size, compute_time=0, cache_obj=opt_cache, shape=(3, 32, 32))
    
    t, h = run_test("Shared Validation", dataset, workers, iterations, seed, batch_size=1, stats_func=opt_cache.stats)

    stats = opt_cache.stats()
    logger.info(f"Final Stats: Hits={stats['hits']}, Misses={stats['miss']}, Hit Rate={h:.2f}%")
    
    # Save
    results_path = OUTPUT_DIR / "validation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    
    if precomputed_path.exists():
        precomputed_path.unlink()
    
    logger.info("Shared OPT Cache validation successful! ✅")


if __name__ == "__main__":
    app()

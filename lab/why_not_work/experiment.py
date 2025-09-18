from __future__ import annotations
import os
# no timing needed
import csv
from typing import List, Sequence, Callable, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from tqdm import tqdm
from cachetools import FIFOCache, LFUCache, LRUCache, RRCache
from multiprocessing import Pool
import sys
sys.path.append(os.path.abspath("./src/OPT4TorchDataSet"))
from cachelib import OPTCache, OPTInit

# Use Times New Roman as the global font (with common fallbacks)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

# force output directory to be next to this script (lab/why_opt_not_work/out)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'out')
N = 20000
T = 100000
ALPHAS = [0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
SEEDS = [0, 1, 2]
CACHE_PERCENTS = [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

def percents_to_sizes(percs: List[float], N: int) -> List[int]:
    return [max(1, min(N, int(round(p / 100.0 * N)))) for p in percs]

def zipf_trace(N: int, T: int, alpha: float, seed: int | None = None) -> List[int]:
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, N + 1, dtype=np.float64)
    probs = 1.0 / (ranks ** alpha)
    probs /= probs.sum()
    return rng.choice(np.arange(N, dtype=np.int64), size=T, p=probs).tolist()

def simulate_with_cache(trace: Sequence[int], C: int, cache_cls: Callable[..., Any]) -> float:
    if C <= 0:
        return 0.0
    cache = cache_cls(maxsize=C)
    hits = 0
    for x in trace:
        if x in cache:
            hits += 1
        else:
            cache[x] = True
    return hits / len(trace)

def simulate_opt(trace: Sequence[int], C: int) -> float:
    if C <= 0:
        return 0.0
    OPTInit(lambda seq, replacement=True, num_samples=None, generator=None: list(trace), None, len(trace))

    @OPTCache(cache_max=C)
    def _access(dummy, idx):
        return object()

    seen = set()
    hits = 0
    for x in trace:
        res = _access(None, x)
        if res in seen:
            hits += 1
        else:
            seen.add(res)
    return hits / len(trace)

POLICIES = {
    'OPT': simulate_opt,
    'LRU': None,
    'LFU': None,
    'FIFO': None,
    'RR': None,
}


def simulate_lru(trace: Sequence[int], C: int) -> float:
    return simulate_with_cache(trace, C, LRUCache)


def simulate_lfu(trace: Sequence[int], C: int) -> float:
    return simulate_with_cache(trace, C, LFUCache)


def simulate_fifo(trace: Sequence[int], C: int) -> float:
    return simulate_with_cache(trace, C, FIFOCache)


def simulate_rr(trace: Sequence[int], C: int) -> float:
    return simulate_with_cache(trace, C, RRCache)


# fill in policy functions (avoids lambdas so they are picklable)
POLICIES['LRU'] = simulate_lru
POLICIES['LFU'] = simulate_lfu
POLICIES['FIFO'] = simulate_fifo
POLICIES['RR'] = simulate_rr


def _worker(args):
    """Worker for a single (si, ai, ci) task. Returns (si, ai, ci, results_tuple)."""
    si, ai, ci, seed, alpha, C, N_local, T_local = args
    trace = zipf_trace(N_local, T_local, alpha, seed)
    res = tuple(POLICIES[name](trace, C) for name in POLICIES.keys())
    return si, ai, ci, res

def run_grid(out_dir: str, N: int, T: int, alphas: List[float], cache_sizes: List[int], seeds: List[int], processes: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    A = len(alphas)
    Cn = len(cache_sizes)
    P = len(POLICIES)
    S = len(seeds)
    raw = np.zeros((S, A, Cn, P), dtype=np.float32)

    # prepare tasks
    tasks = []
    for si, seed in enumerate(seeds):
        for ai, alpha in enumerate(alphas):
            for ci, C in enumerate(cache_sizes):
                tasks.append((si, ai, ci, seed, alpha, C, N, T))

    total = len(tasks)
    processes = processes or max(1, os.cpu_count() or 1)
    with Pool(processes) as pool:
        with tqdm(total=total, desc='grid', unit='task') as pbar:
            for si, ai, ci, res in pool.imap_unordered(_worker, tasks):
                raw[si, ai, ci, :] = res
                pbar.update(1)

    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    print(f"Done grid; Output directory: {out_dir}")
    return mean, std

def plot_results(mean: np.ndarray, std: np.ndarray, alphas: List[float], cache_sizes: List[int], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    names = list(POLICIES.keys())
    percents = [c / N * 100.0 for c in cache_sizes]

    # heatmap OPT - LRU
    diff = mean[:, :, names.index('OPT')] - mean[:, :, names.index('LRU')]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(diff, aspect='auto', origin='lower', cmap='RdBu')
    fig.colorbar(im, ax=ax, label='OPT - LRU')
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([str(a) for a in alphas])
    ax.set_xticks(range(len(percents)))
    ax.set_xticklabels([f"{p:.1f}%" for p in percents], rotation=45, ha='right')
    ax.set_xlabel(f'Cache (% of N={N})')
    ax.set_ylabel('alpha')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'heatmap.png'), dpi=300)

    # per-alpha plots
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for ai, a in enumerate(alphas):
        fig, ax = plt.subplots(figsize=(8, 4))
        # simple line + scatter; use solid lines for all
        for pi, name in enumerate(names):
            m = mean[ai, :, pi]
            # choose color: highlight OPT and LFU, others black (LRU/FIFO/RR same color)
            if name in ('OPT', 'LFU'):
                color = colors[pi % len(colors)]
            else:
                color = 'black'
            ax.plot(percents, m, label=name, color=color, linestyle='-', alpha=0.9, linewidth=1.5)
            ax.scatter(percents, m, color=color, edgecolors='white', s=40, zorder=3)
        ax.set_xscale('log')
        ax.set_xticks(percents)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:.1f}%"))
        ax.set_xlabel(f'Cache (% of N={N})')
        ax.set_ylabel('hit rate')
        ax.set_title(f'alpha={a}')
        # custom legend: highlight OPT and LFU, group others
        opt_color = colors[names.index('OPT') % len(colors)] if 'OPT' in names else colors[0]
        lfu_color = colors[names.index('LFU') % len(colors)] if 'LFU' in names else colors[1 % len(colors)]
        others_color = 'black'
        h_opt = Line2D([0], [0], color=opt_color, lw=1.5)
        h_lfu = Line2D([0], [0], color=lfu_color, lw=1.5)
        h_others = Line2D([0], [0], color=others_color, lw=1.5)
        legend = ax.legend([h_opt, h_lfu, h_others], ['OPT', 'LFU', 'Others (LRU/FIFO/RR)'], title='Policies', framealpha=0.9)
        # small caption below legend explaining style
        legend.get_frame().set_edgecolor('gray')
        ax.grid(True, ls='--', alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'hitrate_alpha_{a}.png'), dpi=300)

    print(f"Plots saved to {out_dir}")

def summarize_results(mean: np.ndarray, std: np.ndarray, alphas: List[float], cache_sizes: List[int], out_dir: str):
    names = list(POLICIES.keys())
    csv_path = os.path.join(out_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['alpha', 'cache', 'policy', 'mean', 'std'])
        for ai, a in enumerate(alphas):
            for ci, c in enumerate(cache_sizes):
                for pi, name in enumerate(names):
                    w.writerow([a, c, name, float(mean[ai, ci, pi]), float(std[ai, ci, pi])])
    print(f"Summary saved to {csv_path}")

def main():
    cache_sizes = percents_to_sizes(CACHE_PERCENTS, N)
    mean, std = run_grid(OUT_DIR, N, T, ALPHAS, cache_sizes, SEEDS)
    summarize_results(mean, std, ALPHAS, cache_sizes, OUT_DIR)
    plot_results(mean, std, ALPHAS, cache_sizes, OUT_DIR)

if __name__ == '__main__':
    main()

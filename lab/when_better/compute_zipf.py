from __future__ import annotations
import os
import sys
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'out')
os.makedirs(OUT_DIR, exist_ok=True)

split = sys.argv[1] if len(sys.argv) > 1 else 'train'

# Try to load from local datasets cache (arrow files) to avoid network calls.
from datasets import Dataset
import glob

cache_base = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'mini-imagenet', 'timm___mini-imagenet', 'default', '0.0.0', 'bd8779f9d33c061ea6e75fdd3bce4e43dd679060')
print(f"Looking for local arrow files under: {cache_base}")
pattern = os.path.join(cache_base, f"mini-imagenet-{split}-*.arrow")
files = sorted(glob.glob(pattern))
if files:
    print(f"Found {len(files)} arrow files; loading locally without network...")
    ds = Dataset.from_file(files[0])
    # if multiple files, concatenate
    if len(files) > 1:
        rest = [Dataset.from_file(p) for p in files[1:]]
        ds = Dataset.from_list(list(ds) + [i for r in rest for i in list(r)])
    # extract labels robustly
    labels = []
    if 'label' in ds.column_names:
        for x in ds['label']:
            # label may be int or mapping; handle common cases
            if isinstance(x, (int, float)):
                labels.append(int(x))
            elif isinstance(x, dict) and 'label' in x:
                labels.append(int(x['label']))
            else:
                try:
                    labels.append(int(x))
                except Exception:
                    # fallback: skip
                    continue
else:
    # fallback: try normal load_dataset (may require network)
    from datasets import load_dataset
    print(f"No local arrows found, falling back to load_dataset (may access network)")
    ds = load_dataset('timm/mini-imagenet', cache_dir=os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'mini-imagenet'), split=split)
    labels = [int(x) for x in ds['label']]
print(f"Loaded {len(labels)} samples; unique labels: {len(set(labels))}")

cnt = Counter(labels)
# sort by descending frequency
counts = np.array(sorted(cnt.values(), reverse=True), dtype=np.float64)
ranks = np.arange(1, len(counts) + 1, dtype=np.float64)
probs = counts / counts.sum()

# fit a power-law: probs ~ k * rank^{-s} => log(probs) = log(k) - s*log(rank)
log_r = np.log(ranks)
log_p = np.log(probs)

slope, intercept = np.polyfit(log_r, log_p, 1)
s_est = -slope

# R^2 for the log-log fit
pred = intercept + slope * log_r
ss_res = np.sum((log_p - pred) ** 2)
ss_tot = np.sum((log_p - log_p.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

print(f"Estimated Zipf exponent s = {s_est:.4f}")
print(f"Fit intercept (log k) = {intercept:.4f}, R^2 = {r2:.4f}")

# Save rank/prob table
out_csv = os.path.join(OUT_DIR, f"zipf_{split}_ranks.csv")
with open(out_csv, 'w') as f:
    f.write('rank,count,prob\n')
    for r, c, p in zip(ranks, counts, probs):
        f.write(f"{int(r)},{int(c)},{p:.12f}\n")
print(f"Saved rank/prob CSV to {out_csv}")

# plot log-log
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(ranks, probs, s=10, c='tab:blue', label='data')
ax.plot(ranks, np.exp(intercept) * ranks ** (-s_est), color='red', lw=1.5, label=f'fit s={s_est:.3f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('rank (log)')
ax.set_ylabel('probability (log)')
ax.set_title(f'Zipf plot: {split} (s={s_est:.3f}, R^2={r2:.3f})')
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend()
fig.tight_layout()
plot_path = os.path.join(OUT_DIR, f'zipf_{split}.png')
fig.savefig(plot_path, dpi=200)
print(f"Saved plot to {plot_path}")

# print top-10
print('\nTop-10 labels by count:')
for i, (lab, c) in enumerate(cnt.most_common(10), 1):
    print(f"{i}. label={lab}: count={c}")

print('\nDone')

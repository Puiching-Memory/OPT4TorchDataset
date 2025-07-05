import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import time
import sys
from functools import lru_cache
from cachetools import cached, LRUCache, FIFOCache, LFUCache, RRCache
import random

sys.path.append(os.path.abspath("./"))


class testDataset(data.Dataset):
    def __init__(self):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        self.idx_list = np.arange(0, 10000)
        self.cache_miss = 0

    def __len__(self):
        return len(self.idx_list)

    # @lru_cache(maxsize=int(10000 * 1.0))
    @cached(cache=LRUCache(maxsize=int(10000 * 0.1)))
    def __getitem__(self, index):
        self.cache_miss += 1
        return index


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import RandomSampler
    from matplotlib import pyplot as plt
    import tqdm

    gen = torch.Generator()
    gen.manual_seed(0)

    dataset = testDataset()
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=RandomSampler(dataset, generator=gen),
        num_workers=0,
    )

    for epoch in tqdm.tqdm(range(100), leave=True, position=0):
        pbar = tqdm.tqdm(dataloader, leave=False, position=1)
        # out_book = []
        for batch_idx, (output) in enumerate(dataloader):
            # print(output)
            # out_book.append(output.item())
            pbar.update()
        # print(out_book)

    print(f"cache_miss: {dataset.cache_miss}")
    print(f"total: {len(dataset)*100}")
    print(f"hit: {(1-(dataset.cache_miss/(len(dataset)*100)))*100}%")

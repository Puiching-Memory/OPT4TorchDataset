import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import time
import sys
from functools import lru_cache
from cachetools import cached, LRUCache, FIFOCache, LFUCache, RRCache

sys.path.append(os.path.abspath("./"))
#from lib.datasets._optcache import OptCache
import random
from collections import OrderedDict
from tqdm import tqdm
import bisect
import copy

sys.path.append(os.path.abspath("./"))

class testDataset(data.Dataset):
    def __init__(self):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        self.idx_list = np.arange(0, 10000)
        self.cache_miss = 0

        self.current = 0
        self.cache_max = int(10000 * 1)
        self.cache = dict()

        gen = torch.Generator()
        gen.manual_seed(0)
        samper = RandomSampler(
            [None] * 10000, replacement=True, num_samples=10000 * 100, generator=gen
        )
        self.future_index = []
        for i in samper:
            self.future_index.append(i)
        #self.future_index = np.array(self.future_index)

    def __len__(self):
        return len(self.idx_list)

    # @lru_cache(maxsize=int(10000 * 0.5))
    # @cached(cache=OptCache(maxsize=int(10000 * 0.1)))
    # @opt_cache(max_size=int(10000 * 0.1),future_calls=)
    def __getitem__(self, index):
        self.current += 1
        if index in self.cache:
            return self.cache[index]
        
        # porcess data here
        result = index
        self.cache_miss += 1

        if len(self.cache) < self.cache_max:
            self.cache[index] = result
        else:
            max_distance = ["key",0]
            for k in self.cache.keys():
                try:
                    distance = self.future_index.index(k, self.current, self.current + int(10000 * 100 * 0.0005)) - self.current
                except ValueError:
                    max_distance = [k,float("inf")]
                    break
                else:  
                    if distance > max_distance[1]:
                        max_distance = [k,distance]
                
            self.cache.pop(max_distance[0])
            self.cache[index] = result
        
        return result


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import RandomSampler
    from collections import Counter
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    gen = torch.Generator()
    gen.manual_seed(0)

    dataset = testDataset()
    dataloader = DataLoader(
        dataset=dataset,
        sampler=RandomSampler(
            dataset, replacement=True, num_samples=10000 * 100, generator=gen
        ),
        batch_size=1,
        num_workers=0,
    )

    out_book = []
    pbar = tqdm(dataloader)
    for batch_idx, (output) in enumerate(dataloader):
        out_book.append(output.item())
        pbar.update()

    # out_book = Counter(out_book)
    print(out_book[0:10])
    # plt.bar(out_book.keys(), out_book.values())
    # plt.xlabel('Numbers')
    # plt.ylabel('Frequency')
    # plt.title('Frequency of Numbers in List')
    # plt.show()

    print(f"cache_miss: {dataset.cache_miss}")
    print(f"total: {len(dataset)*100}")
    print(f"hit: {(1-(dataset.cache_miss/(len(dataset)*100)))*100}%")

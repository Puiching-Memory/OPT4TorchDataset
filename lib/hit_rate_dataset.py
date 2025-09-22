import types
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import os
import sys
from cachetools import cached, LRUCache, LFUCache, FIFOCache, RRCache


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

class HitRateDataset(Dataset):
    def __init__(self):
        self.miss = 0
        self.dataset = range(10000)
        self.data_generator = torch.Generator()
        self.data_generator.manual_seed(0)
        self._getitem_impl = types.MethodType(self._raw_getitem, self)

    def __len__(self):
        return len(self.dataset)
    
    def _raw_getitem(self, self_obj, idx):
        self_obj.miss += 1
        return idx

    def __getitem__(self, idx):
        return self._getitem_impl(idx)

    def getMissCount(self):
        return self.miss

    def getGenerator(self):
        return self.data_generator
    
    def setCache(self, cacheMethod):
        wrapped = cacheMethod(self._raw_getitem)
        self._getitem_impl = types.MethodType(wrapped, self)

    def resetMissCount(self):
        self.miss = 0

if __name__ == "__main__":
    train_dataset = HitRateDataset()
    train_dataset.setCache(cached(LFUCache(maxsize=int(10000 * 0.5))))

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=32, 
                                  shuffle=False, 
                                  num_workers=0,
                                  pin_memory=True,
                                  sampler=RandomSampler(train_dataset,
                                                        replacement=True,
                                                        num_samples=len(train_dataset),
                                                        generator=train_dataset.getGenerator()
                                                        )
                                  )

    for idx in train_dataloader:
        pass

    print("="*80)
    print(f"cache miss count: {train_dataset.getMissCount()} miss rate: {train_dataset.getMissCount()/ len(train_dataset)}")
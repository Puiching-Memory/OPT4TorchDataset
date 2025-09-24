import types
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch

class HitRateDataset(Dataset):
    def __init__(self, size=10000):
        self.miss = 0
        self.dataset = list(range(size))
        self._getitem_impl = types.MethodType(self._raw_getitem, self)
        self._generator = torch.Generator()
        self._generator.manual_seed(0)

    def __len__(self):
        return len(self.dataset)
    
    def _raw_getitem(self, self_obj, idx):
        self_obj.miss += 1
        return idx

    def __getitem__(self, idx):
        return self._getitem_impl(idx)

    def getMissCount(self):
        return self.miss
    
    def resetMissCount(self):
        self.miss = 0

    def setCache(self, cacheMethod):
        wrapped = cacheMethod(self._raw_getitem)
        self._getitem_impl = types.MethodType(wrapped, self)

    def getGenerator(self):
        return self._generator


if __name__ == "__main__":
    from cachetools import cached, LRUCache, LFUCache, FIFOCache, RRCache
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
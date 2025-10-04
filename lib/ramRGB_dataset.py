import types
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
import os
import torchvision
from torchvision.transforms import v2

class RandomRGBDataset(Dataset):
    def __init__(self, path):
        self.dataset = []
        for i in os.listdir(path):
            self.dataset.append(os.path.join(path, i))
            
        self.miss = 0
        self._getitem_impl = types.MethodType(self._raw_getitem, self)
        self._generator = torch.Generator()
        self._generator.manual_seed(0)

        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataset)
    
    def _raw_getitem(self, self_obj, idx):
        self_obj.miss += 1
        image = torchvision.io.read_image(self.dataset[idx])
        image = self.transforms(image)
        return image

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
    train_dataset = RandomRGBDataset("./data/random_rgb_dataset")
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

    for image in train_dataloader:
        print(image)

    print("="*80)
    print(f"cache miss count: {train_dataset.getMissCount()} miss rate: {train_dataset.getMissCount()/ len(train_dataset)}")
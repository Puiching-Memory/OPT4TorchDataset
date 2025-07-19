import torch.utils.data as data
import os
import sys
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("src/OPT4TorchDataSet"))

from torchvision.transforms import v2
import torch
import torchvision

from cachelib import OPTCache,OPTInit
from cachetools import cached, LRUCache, LFUCache, FIFOCache, RRCache

class Imagenet1K(data.Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split
        self.idx_list = os.listdir(os.path.join(root_dir, split))
        self.data_generator = torch.Generator()
        self.data_generator.manual_seed(0)
        OPTInit(data.RandomSampler,self.data_generator,self.__len__())
        print(len(self.idx_list))
    def get_generator(self):
        return self.data_generator

    def __len__(self):
        return len(self.idx_list)

    @OPTCache(cache_max=128116)
    #@cached(cache=LRUCache(maxsize=128116))
    #@cached(cache=LFUCache(maxsize=128116))
    #@cached(cache=FIFOCache(maxsize=128116))
    #@cached(cache=RRCache(maxsize=128116))
    def __getitem__(self, index):
        image = torchvision.io.decode_image(os.path.join(self.root_dir, self.split,  self.idx_list[index]),
                                            mode=torchvision.io.ImageReadMode.RGB)
        label = torch.tensor(int(self.idx_list[index].split("-")[0]))
        return image,label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = Imagenet1K(
        r".cache/imagenet-1k-jpeg-256",
        "train"
    )
    dataloader = DataLoader(dataset=dataset,
                            batch_size=2,
                            shuffle=False, # shuffle must be False
                            num_workers=0,
                            pin_memory=True,
                            sampler=data.RandomSampler(dataset,
                                                       replacement=True,
                                                       num_samples=len(dataset) * 3,
                                                       generator=dataset.get_generator()
                                                       ),
                            )

    for batch_idx, (image, label) in enumerate(dataloader):
        break
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from loguru import logger


class RandomRGBDataset(ImageFolder):
    """Random RGB dataset using torchvision ImageFolder"""
    
    def __init__(self, data_dir='./data/random_rgb_dataset'):
        # ImageFolder expects root directory with class subdirectories
        super().__init__(root=data_dir)
        
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.miss = 0
        self._getitem_impl = self._raw_getitem
        self._generator = torch.Generator()
        self._generator.manual_seed(0)
    
    def _raw_getitem(self, idx):
        self.miss += 1
        # ImageFolder's __getitem__ returns (PIL.Image, class_index)
        image, label = super(RandomRGBDataset, self).__getitem__(idx)
        image = self.transforms(image)
        return image, label
    
    def __getitem__(self, idx):
        return self._getitem_impl(idx)
    
    def getMissCount(self):
        return self.miss
    
    def resetMissCount(self):
        self.miss = 0
    
    def setCache(self, cacheMethod):
        wrapped = cacheMethod(self._raw_getitem)
        self._getitem_impl = wrapped
    
    def getGenerator(self):
        return self._generator
    


if __name__ == "__main__":
    from cachetools import cached, LRUCache, LFUCache, FIFOCache, RRCache
    from torch.utils.data import RandomSampler
    
    logger.info("Testing RandomRGBDataset")
    logger.info("-" * 30)
    
    train_dataset = RandomRGBDataset(
        data_dir='./data/random_rgb_dataset'
    )
    
    logger.info(f"Train size: {len(train_dataset)}")
    
    image, label = train_dataset[0]
    logger.info(f"Image shape: {image.shape}")
    logger.info(f"Label: {label}")
    
    logger.info("Applying LFU cache...")
    train_dataset.setCache(cached(LFUCache(maxsize=int(len(train_dataset) * 0.5))))
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True,
        sampler=RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=len(train_dataset),
            generator=train_dataset.getGenerator()
        )
    )
    
    logger.info("Iterating through dataset...")
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        total_samples += images.shape[0]
        if batch_idx == 0:
            logger.info(f"Batch shape: {images.shape}")
            logger.info(f"Labels shape: {labels.shape}")
            logger.info(f"Sample labels: {labels[:5]}")
    
    miss_count = train_dataset.getMissCount()
    miss_rate = miss_count / total_samples
    logger.info(f"Total samples accessed: {total_samples}")
    logger.info(f"Cache miss count: {miss_count}")
    logger.info(f"Cache miss rate: {miss_rate:.2%}")
    logger.info(f"Cache hit rate: {(1 - miss_rate):.2%}")
    
    logger.info("All tests passed!")
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from loguru import logger


class RandomRGBDataset(ImageFolder):
    """Random RGB dataset using torchvision ImageFolder"""

    def __init__(self, data_dir="./data/random_rgb_dataset"):
        # ImageFolder expects root directory with class subdirectories
        super().__init__(root=data_dir)

        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                # 添加一些计算密集型的 CPU 变换，以凸显缓存优势
                v2.RandomRotation(degrees=45),
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.miss = 0
        self.cache_decorator = None
        self._wrapped_getitem = None
        self._generator = torch.Generator()
        self._generator.manual_seed(0)

    def _raw_load(self, idx):
        """Only load and transform the image, increment miss count"""
        self.miss += 1
        path, _ = self.samples[idx]
        image = self.loader(path)
        image = self.transforms(image)
        return image

    def __getitem__(self, idx):
        # ImageFolder's samples is a list of (path, class_index)
        _, label = self.samples[idx]
        
        if self.cache_decorator is not None:
            if self._wrapped_getitem is None:
                # Lazy wrap to avoid pickling issues with bound methods on Windows
                self._wrapped_getitem = self.cache_decorator(self._raw_load)
            image = self._wrapped_getitem(idx)
        else:
            image = self._raw_load(idx)
            
        return image, label

    def getMissCount(self):
        return self.miss

    def resetMissCount(self):
        self.miss = 0

    def setCache(self, cacheDecorator):
        self.cache_decorator = cacheDecorator
        self._wrapped_getitem = None

    def getGenerator(self):
        return self._generator


if __name__ == "__main__":
    from cachetools import cached, LFUCache
    from torch.utils.data import RandomSampler

    logger.info("Testing RandomRGBDataset")
    logger.info("-" * 30)

    train_dataset = RandomRGBDataset(data_dir="./data/random_rgb_dataset")

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
            generator=train_dataset.getGenerator(),
        ),
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

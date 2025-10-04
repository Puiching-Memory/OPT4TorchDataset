import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Image
from torchvision.transforms import v2
import torch.utils.data as data
import os
import sys
from cachetools import cached, LRUCache, LFUCache, FIFOCache, RRCache

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

class MiniImageNetDataset(Dataset):
    def __init__(self, split='train'):
        # 加载Hugging Face数据集
        self.dataset = load_dataset('timm/mini-imagenet',split=split)
        self.dataset = self.dataset.cast_column("image", Image(mode="RGB"))
        
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
    
        image = item['image']
        label = item['label']

        image = self.transforms(image)

        return image, label
    


if __name__ == "__main__":
    train_dataset = MiniImageNetDataset(split='train')
    val_dataset = MiniImageNetDataset(split='validation')
    test_dataset = MiniImageNetDataset(split='test')

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    for images, labels in train_dataloader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break
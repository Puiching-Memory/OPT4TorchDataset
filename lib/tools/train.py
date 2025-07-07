from datasets import load_dataset
import os
import timm
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import torchmetrics
import torchmetrics.classification
import warnings
from src.optlibA import optA
from multiprocess import set_start_method

# 屏蔽 EXIF 警告
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", message="Truncated File Read", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", message="Metadata Warning, tag 274 had too many entries", category=UserWarning, module="PIL.TiffImagePlugin")

transforms = v2.Compose([v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Lambda(lambda x: x[:3]),  # 移除 alpha 通道
                            v2.Resize((224,224)),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])

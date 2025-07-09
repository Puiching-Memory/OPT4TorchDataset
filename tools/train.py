from datasets import load_dataset
import os
import sys
sys.path.append(os.path.abspath("./"))
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
from lib.dataset import imagenet_1k
from torch.utils.data import DataLoader

# 屏蔽 EXIF 警告
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", message="Truncated File Read", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", message="Metadata Warning, tag 274 had too many entries", category=UserWarning, module="PIL.TiffImagePlugin")

# check Device
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

transforms = v2.Compose([v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            #v2.Resize((256,256)),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])

dataset = imagenet_1k.Imagenet1K(
    r".cache/imagenet-1k-jpeg-256",
    "train"
)
dataloader = DataLoader(dataset=dataset,
                        batch_size=8,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True,
                        )

model = timm.create_model('resnetv2_50',
                        pretrained=True,
                        cache_dir="./.cache/",
                        num_classes=1000,
                        )
model.train()
model = torch.compile(model)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = torch.amp.GradScaler()
loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

progress = tqdm(total=len(dataloader), desc="Training Progress",leave=True)

for batch_idx, (image, label) in enumerate(dataloader):
    image = transforms(image).to(device)
    label = label.to(device)


    optimizer.zero_grad()
    with torch.autocast(device_type="cuda"):
        output = model(image)
        loss_value = loss(output, label)

    scaler.scale(loss_value).backward()
    scaler.step(optimizer)
    scaler.update()

    progress.update()
    progress.set_description(f"Loss: {loss_value.item():.4f}")

from datasets import load_dataset
import os
import sys
sys.path.append(os.path.abspath("./"))
import timm
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.utils.data as data
from tqdm import tqdm
from torch.nn.parallel import DataParallel as DP
import torchmetrics
import torchmetrics.classification
import warnings
from lib.dataset import imagenet_1k
from torch.utils.data import DataLoader


# 屏蔽 EXIF 警告
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", message="Truncated File Read", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", message="Metadata Warning, tag 274 had too many entries", category=UserWarning, module="PIL.TiffImagePlugin")

# print(timm.list_models())

# check Device
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

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
                        batch_size=1024,
                        shuffle=False, # shuffle must be False
                        num_workers=4,
                        pin_memory=True,
                        sampler=data.RandomSampler(dataset,
                                                    replacement=True,
                                                    num_samples=len(dataset) * 3, # * batch size
                                                    generator=dataset.get_generator()
                                                    ),
                        )

model = timm.create_model('resnet50',
                        pretrained=True,
                        cache_dir="./.cache/",
                        num_classes=1000,
                        ).to(device)
model.train()
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = torch.amp.GradScaler()
loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

metricACC = torchmetrics.classification.Accuracy(task="multiclass",num_classes=1000).to(device)
metricAuroc = torchmetrics.classification.AUROC(task="multiclass",num_classes=1000).to(device)
metricAP = torchmetrics.classification.AveragePrecision(task="multiclass",num_classes=1000).to(device)

progress = tqdm(total=len(dataloader), desc="Training Progress",leave=True)

for batch_idx, (image, label) in enumerate(dataloader):
    image = transforms(image).to(device)
    label = label.to(device)

    optimizer.zero_grad()
    with torch.autocast(device_type="cuda"):
        output = model(image)
        loss_value = loss(output, label)
        acc = metricACC(output, label)
        auroc = metricAuroc(output, label)
        ap = metricAP(output, label)

    scaler.scale(loss_value).backward()
    scaler.step(optimizer)
    scaler.update()

    progress.update()
    progress.set_description(f"Loss: {loss_value.item():.4f} ACC: {acc.item():.4f} AUROC: {auroc.item():.4f} AP: {ap.item():.4f}")
    # break

progress.close()
print("Finished training, Starting validation")

# val

dataset = imagenet_1k.Imagenet1K(
    r".cache/imagenet-1k-jpeg-256",
    "validation"
)
dataloader = DataLoader(dataset=dataset,
                        batch_size=1024,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True,
                        )

model.eval()

metricACC = torchmetrics.classification.Accuracy(task="multiclass",num_classes=1000).to(device)
metricAuroc = torchmetrics.classification.AUROC(task="multiclass",num_classes=1000).to(device)
metricAP = torchmetrics.classification.AveragePrecision(task="multiclass",num_classes=1000).to(device)

progress = tqdm(total=len(dataloader), desc="Val Progress",leave=True)

for batch_idx, (image, label) in enumerate(dataloader):
    image = transforms(image).to(device)
    label = label.to(device)

    with torch.autocast(device_type="cuda"):
        output = model(image)
        acc = metricACC(output, label)
        auroc = metricAuroc(output, label)
        ap = metricAP(output, label)

    progress.update()
    progress.set_description(f"ACC: {acc.item():.4f} AUROC: {auroc.item():.4f} AP: {ap.item():.4f}")

progress.close()

acc = metricACC.compute()
auroc = metricAuroc.compute()
ap = metricAP.compute()
print(f"ACC: {acc:.4f} AUROC: {auroc:.4f} AP: {ap:.4f}")



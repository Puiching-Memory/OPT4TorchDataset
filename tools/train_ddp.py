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

world_size = torch.cuda.device_count()
local_rank = int(os.environ["LOCAL_RANK"])
init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
device = torch.device(f"cuda:{os.environ["LOCAL_RANK"]}" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')
torch.cuda.set_device(local_rank)

# 屏蔽 EXIF 警告
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", message="Truncated File Read", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", message="Metadata Warning, tag 274 had too many entries", category=UserWarning, module="PIL.TiffImagePlugin")

# 自定义 collate_fn 将 PIL 图像转换为张量
def custom_collate_fn(batch):
    transforms = v2.Compose([v2.ToImage(),
                             v2.ToDtype(torch.float32, scale=True),
                             v2.Lambda(lambda x: x[:3]),  # 移除 alpha 通道
                             v2.Resize((224,224)),
                             v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             ])
    images = [transforms(item['image']) for item in batch]
    labels = [item['label'] for item in batch]
    return {'image': torch.stack(images), 'label': torch.tensor(labels)}

def train():
    #print(timm.list_models())
    model = timm.create_model('resnetv2_50',
                            pretrained=True,
                            cache_dir="./.cache/",
                            num_classes=1000,
                            )
    model.train()
    model = torch.compile(model)
    model.to(device)
    model = DDP(model,device_ids=[local_rank])
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    dataset = load_dataset("./imagenet-1k.py",
                        trust_remote_code=True,
                        cache_dir="./.cache/",
                        num_proc=os.cpu_count()
                        )
    dataset.with_format("torch", device=device)
    dataloader = DataLoader(dataset["train"],
                            batch_size=128,
                            num_workers=8,
                            collate_fn=custom_collate_fn,
                            shuffle=False,
                            sampler=DistributedSampler(dataset["train"], shuffle=True),
                            pin_memory=True,
                            drop_last=True,
                            persistent_workers=True,
                            )

    dataloader = optA(dataloader)

    scaler = torch.amp.GradScaler()
    metric = torchmetrics.classification.Accuracy(task="multiclass",num_classes=1000).to(device)
    progress = tqdm(total=len(dataloader), desc="Training Progress",leave=True)
    epoch = 5

    for e in range(epoch):
        dataloader.sampler.set_epoch(e)
        for i in dataloader:
            opt.zero_grad()
            image = i["image"].to(device)
            label = i["label"].to(device)
            with torch.autocast(device_type="cuda"):
                output = model(image)
                loss_value = loss(output, label)
                acc = metric(output, label)

            torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.AVG)
            if local_rank == 0:
                progress.set_description(f"Epoch: {e+1} Loss: {loss_value.item():.4f} | Acc: {acc.item():.4f}")
                progress.update()
            
            scaler.scale(loss_value).backward()
            scaler.step(opt)
            scaler.update()

        if local_rank == 0:
            #acc = metric.compute()
            #print(f"Accuracy on all data: {acc}")
            progress.reset()
        
    if local_rank == 0:
        progress.close()

    destroy_process_group()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    train()
    
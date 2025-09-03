import os
import sys
import timm
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.utils.data as data
from tqdm import tqdm
import torchmetrics
import torchmetrics.classification
import warnings
import swanlab
import datetime
from typing import cast

sys.path.append(os.path.abspath("./"))
# from lib.dataset import imagenet_1k
from lib.dataset import mini_imagenet_dataloader

if __name__ == "__main__":
    print(timm.list_models())

    # experiment / run configuration (used for naming and reproducibility)
    dataset_name = "mini-imagenet"
    model_name = "deit3_small_patch16_224"
    batch_size = 512
    lr = 1e-3
    epoch_size = 10
    num_workers = 16
    lr_str = f"{lr:.0e}"
    experiment_name = f"{dataset_name}_{model_name}_bs{batch_size}_lr{lr_str}_ep{epoch_size}"

    run = swanlab.init(project="opt4", experiment_name=experiment_name)
    
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    dataset = mini_imagenet_dataloader.MiniImageNetDataset(split='train')

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False, # shuffle must be False
                            num_workers=num_workers,
                            pin_memory=True,
                            sampler=data.RandomSampler(dataset,
                                                        replacement=True,
                                                        num_samples=len(dataset) * epoch_size,
                                                        generator=dataset.get_generator()
                                                        ),
                            )

    model = timm.create_model(model_name,
                            pretrained=True,
                            cache_dir="./models/",
                            num_classes=100,
                            ).to(device)
    model.train()

    # FIXME: memory leak. H800 CUDA:12.8 torch:2.7.1 ubuntu:24.04 Driver:535.161.08
    try:
        compiled = torch.compile(model)
        # help static type checkers: treat compiled object as Module for downstream calls
        model = cast(torch.nn.Module, compiled)
    except Exception as e:
        print("torch.compile error -> Skip\n", e)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    loss = torch.nn.CrossEntropyLoss()

    metricACC = torchmetrics.classification.Accuracy(task="multiclass",num_classes=100).to(device)
    metricAuroc = torchmetrics.classification.AUROC(task="multiclass",num_classes=100).to(device)
    metricAP = torchmetrics.classification.AveragePrecision(task="multiclass",num_classes=100).to(device)

    progress = tqdm(total=len(dataloader), desc="Training Progress",leave=True)

    for batch_idx, (image, label) in enumerate(dataloader):
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

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
        swanlab.log({"loss": loss_value.item(), "ACC": acc.item(), "AUROC": auroc.item(), "AP": ap.item()})
        # break

    progress.close()
    print("Finished training, Starting validation")

    # val

    dataset = mini_imagenet_dataloader.MiniImageNetDataset(split='validation')
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True,
                            )

    model.eval()

    metricACC = torchmetrics.classification.Accuracy(task="multiclass",num_classes=100).to(device)
    metricAuroc = torchmetrics.classification.AUROC(task="multiclass",num_classes=100).to(device)
    metricAP = torchmetrics.classification.AveragePrecision(task="multiclass",num_classes=100).to(device)

    progress = tqdm(total=len(dataloader), desc="Val Progress",leave=True)

    with torch.inference_mode():
        for batch_idx, (image, label) in enumerate(dataloader):
            image = image.to(device)
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
    swanlab.log({"val_ACC": acc.item(), "val_AUROC": auroc.item(), "val_AP": ap.item()})



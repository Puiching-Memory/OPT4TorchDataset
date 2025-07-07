# convert huggingface dataset to raw images
# 171GB -> train:199GB + 
from datasets import load_dataset
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

output_dir = "./.cache/imagenet-1k-avif-256"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir,"train"), exist_ok=True)
os.makedirs(os.path.join(output_dir,"validation"), exist_ok=True)
os.makedirs(os.path.join(output_dir,"test"), exist_ok=True)

dataset = load_dataset("./lib/dataset/imagenet_1k_datasets.py",
                    trust_remote_code=True,
                    cache_dir="./.cache/",
                    num_proc=os.cpu_count(),
                    )
dataset.with_format("torch")

def process_item(d, split):
    image = d["image"]
    label = d["label"]
    image = image.resize((256, 256))
    UID = uuid.uuid4()
    file_name = f"{str(label)}-{str(UID)}.avif"
    print(os.path.join(output_dir, split ,file_name))
    image.save(os.path.join(output_dir, split ,file_name), format="AVIF", lossless=True, quality=100, speed=0)
    return file_name

for split in dataset.keys():  # 遍历所有切片
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)  # 为每个切片创建目录
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_item, d, split) for d in dataset[split]]
        
        for future in as_completed(futures):
            file_name = future.result()
# convert huggingface dataset to raw images
# 171GB -> train:199GB + 
from datasets import load_dataset
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm

output_dir = "./.cache/imagenet-1k-jpeg-256"
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

print(f"total number of train: {len(dataset["train"])}")
print(f"total number of validation: {len(dataset["validation"])}")
print(f"total number of test: {len(dataset["test"])}")

progress = tqdm(desc="Converting Progress",leave=True)

def process_item(index, d, split):
    image = d["image"]
    label = d["label"]
    if image.mode != 'RGB': image = image.convert('RGB')
    image = image.resize((256, 256))
    file_name = f"{label}-{split}-{index}.jpeg"
    if os.path.exists(os.path.join(output_dir, split ,file_name)):return
    image.save(os.path.join(output_dir, split ,file_name), format="JPEG", quality=95, optimize=True)

    progress.n = index
    progress.set_description(f"Converting {os.path.join(output_dir, split ,file_name)}")
    progress.update()

for split in dataset.keys():  # 遍历所有切片
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)  # 为每个切片创建目录
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_item, index, d, split) for index,d in enumerate(dataset[split])]
        
        for future in as_completed(futures):
            future.result()
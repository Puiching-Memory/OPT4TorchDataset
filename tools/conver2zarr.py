# convert huggingface dataset to raw images
# 171GB -> train:199GB + 
from datasets import load_dataset
import os
import zarr
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor


dataset = load_dataset("./lib/dataset/imagenet_1k_datasets.py",
                    trust_remote_code=True,
                    cache_dir="./.cache/",
                    num_proc=os.cpu_count(),
                    )
dataset.with_format("torch")

root = zarr.group("./.cache/imagenet_1k_256.zarr",overwrite=True,zarr_format=3)
train_group = root.create_group(name="train")
train_array = train_group.create_array(name="images",
                                       shape=(len(dataset["train"]), 256, 256, 3),
                                       shards=(1, 256, 256, 3),
                                       chunks=(1, 256, 256, 3),
                                       dtype=np.int8,
                                       compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
                                       )
train_label = train_group.create_array(name="labels",
                                       shape=(len(dataset["train"]),),
                                       shards=(1,),
                                       chunks=(1,),
                                       dtype=np.int64,
                                       compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
                                       )

validation_group = root.create_group(name="validation")
validation_array = validation_group.create_array(name="images",
                                                shape=(len(dataset["validation"]), 256, 256, 3),
                                                shards=(1, 256, 256, 3),
                                                chunks=(1, 256, 256, 3),
                                                dtype=np.int8,
                                                compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
                                                 )
validation_label = validation_group.create_array(name="labels",
                                       shape=(len(dataset["validation"]),),
                                       shards=(1,),
                                       chunks=(1,),
                                       dtype=np.int64,
                                       compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
                                       )

test_group = root.create_group(name="test")
test_array = test_group.create_array(name="images", 
                                    shape=(len(dataset["test"]), 256, 256, 3),
                                    shards=(1, 256, 256, 3),
                                    chunks=(1, 256, 256, 3),
                                    dtype=np.int8,
                                    compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
                                     )
test_label = test_group.create_array(name="labels",
                                       shape=(len(dataset["test"]),),
                                       shards=(1,),
                                       chunks=(1,),
                                       dtype=np.int64,
                                       compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
                                       )


progress = tqdm(total=len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"]), desc="Processing images")
def process_item(index, d, split):
    image = d["image"]
    label = d["label"]
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((256, 256))
    image = np.array(image,dtype=np.int8)

    if split == "train":
        train_array[index] = image
        train_label[index] = label
    elif split == "validation":
        validation_array[index] = image
        validation_label[index] = label
    elif split == "test":
        test_label[index] = label
        test_array[index] = image

    progress.n = index
    progress.refresh()

for split in dataset.keys():  # 遍历所有切片
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_item,index, d, split) for index,d in enumerate(dataset[split])]
        
        for future in as_completed(futures):
            file_name = future.result()
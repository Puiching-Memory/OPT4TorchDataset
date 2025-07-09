import nvidia.dali.plugin.pytorch.experimental.proxy as dali_proxy
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
import torchvision.datasets
import nvidia.dali.plugin.pytorch as nvidia
import numpy as np
import time

@pipeline_def
def rn50_train_pipe():
    rng = fn.random.coin_flip(probability=0.5)
    filepaths = fn.external_source(name="images", no_copy=True)
    jpegs = fn.io.file.read(filepaths)
    images = fn.decoders.image_random_crop(
        jpegs,
        device="mixed",
        output_type=types.RGB,
        random_aspect_ratio=[0.75, 4.0 / 3.0],
        random_area=[0.08, 1.0],
    )
    images = fn.resize(
        images,
        size=[256, 256],
        interp_type=types.INTERP_LINEAR,
        antialias=False,
    )
    output = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(256, 256),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=rng,
    )
    return output

def read_filepath(path):
    return np.frombuffer(path.encode(), dtype=np.int8)

nworkers = 8
pipe = rn50_train_pipe(
    batch_size=16, num_threads=8, device_id=0,
    prefetch_queue_depth=2*nworkers)

# The scope makes sure the server starts and stops at enter/exit
with dali_proxy.DALIServer(pipe) as dali_server:
    # DALI proxy instance can be used as a transform callable
    dataset = torchvision.datasets.ImageFolder(
        ".cache/imagenet-1k-jpeg-256", transform=dali_server.proxy, loader=read_filepath)

    # Same interface as torch DataLoader, but takes a dali_server as first argument
    loader = dali_proxy.DataLoader(
        dali_server,
        dataset,
        batch_size=16,
        num_workers=nworkers,
        drop_last=True,
    )

    for data, target in loader:
        starts = time.time()
        # consume it
        print(data.shape)
        print(target)

        print(time.time() - starts)
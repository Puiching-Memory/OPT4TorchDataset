# OPT4TorchDataset
Plug-and-Play Optimal Page Replacement Algorithm (OPT) for Torch Dataset

## There's no free lunch
this method requires a substantial amount of additional RAM. 
If you're using a personal computer with 16GB or less memory, this may offer limited benefit to you.

## install
```bash
pip install OPT4TorchDataset
```

## usage
```python
import torch,torchvision
import torch.utils.data as data
from OPT4TorchDataSet.cachelib import OPTCache,OPTInit

class Imagenet1K(data.Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split
        self.idx_list = os.listdir(os.path.join(root_dir, split))

        # must Init first!
        self.data_generator = torch.Generator()
        self.data_generator.manual_seed(0)
        OPTInit(data.RandomSampler,self.data_generator,self.__len__())

    # Provide a convenient access method for DataLoader
    def get_generator(self):
        return self.data_generator

    def __len__(self):
        return len(self.idx_list)
    
    # use cache here!
    @OPTCache() 
    def __getitem__(self, index):
        image = torchvision.io.decode_image(os.path.join(self.root_dir, self.split,  self.idx_list[index]),
                                            mode=torchvision.io.ImageReadMode.RGB)
        label = torch.tensor(int(self.idx_list[index].split("-")[0]))
        return image,label

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = Imagenet1K(
        r".cache/imagenet-1k-jpeg-256",
        "train"
    )
    dataloader = DataLoader(dataset=dataset,
                            batch_size=2,
                            shuffle=False, # shuffle must be False
                            num_workers=0,
                            pin_memory=True,
                            sampler=data.RandomSampler(dataset,
                                                       replacement=True,
                                                       num_samples=len(dataset) * 3,
                                                       generator=dataset.get_generator()
                                                       ),
                            )

    for batch_idx, (image, label) in enumerate(dataloader):
        break
```

## dev env
```bash
# using ubuntu 24.04 cuda 12.8 h800 sm90
apt update
apt upgrade
apt install build-essential
```

```bash
conda create -n opt4 python=3.13
conda activate opt4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install timm huggingface-hub datasets[vision] torchmetrics pillow cachetools swanlab
pip install zarr nvidia-dali-cuda120 # optional
```

## dataset
```bash
huggingface-cli download --repo-type dataset --resume-download ILSVRC/imagenet-1k --local-dir ./imagenet-1k --token {your_token_here}
```

## experiment
dataset: imagenet-1k
device: NVIDIA H800 CUDA 12.8
system: ubuntu 24.04

python:3.13
cachetools: 6.1.0
OPT: 1.0.0
torch: 2.7.1 compiled False

Cache Size: 12811 (1% dataset) (? GB RAM) 12
DataIter summary: 3754 (1281167)
Epoch: 3
num_workers: 16
batch_size: 512

## Training Speed (one device)

| model    | BaseLine | OPT ON | LRU ON | LFU ON | FIFO ON | RR ON | log |
| -------- | -------- | ------ | ------ | ------ | ------- | ----- | --- |
| resnet50 | 50:14    | 43:31  | 44:57  | 44:55  | 44:30   | 44:41 |     |

## Training Speed (multi devices DDP)


## Hit rate
| BaseLine | OPT ON | LRU ON |
| -------- | ------ | ------ |
| 0%       |

## build up whl
```bash
pip install build
python -m build
pip install dist/opt4torchdataset-1.0.0-cp313-cp313-linux_x86_64.whl
```
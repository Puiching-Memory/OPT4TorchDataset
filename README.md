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
# tested ubuntu 24.04 cuda 12.8 h800 sm90
# tested windows11 cuda 12.9.1 NVIDIA 4060Ti sm89
apt update
apt upgrade
apt install build-essential

conda create -n opt4 python=3.13
conda activate opt4
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt

# install triton for windows(optional)
pip install -U "triton-windows<3.5"

# login swanlab (optional)
swanlab login
# then following https://docs.swanlab.cn/guide_cloud/general/quick-start.html

# choice device (optional)
export CUDA_VISIBLE_DEVICES=2

# set mirror (optional)
export HF_ENDPOINT=https://hf-mirror.com
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## dataset
```bash
hf download --repo-type dataset ILSVRC/imagenet-1k --cache-dir ./data/imagenet-1k --token {your_token_here}
hf download --repo-type dataset timm/mini-imagenet --cache-dir ./data/mini-imagenet
```

## experiment timm/mini-imagenet
dataset: mini-imagenet
device: NVIDIA H800 sm90 CUDA 12.9
system: ubuntu 24.04
python:3.13
cachetools: 6.2.0
OPT: 1.0.0
torch: 2.8.0 compiled True
Cache Size: 25000 (50% dataset) (? GB RAM)
DataIter summary: 293 (50000)
Epoch: 3
num_workers: 16
batch_size: 512

### Training Speed (one device)

| model    | BaseLine | OPT ON | LRU ON | LFU ON | FIFO ON | RR ON | log                                          |
| -------- | -------- | ------ | ------ | ------ | ------- | ----- | -------------------------------------------- |
| resnet50 | 2:27     | 1:41   | 1:54   | 1:49   | 1:51    | 1:52  | https://swanlab.cn/@Sail2Dream/opt4/overview |

### Training Speed (multi devices DDP)

### Hit rate

### eval

## build up whl
```bash
pip install build
python -m build
pip install dist/opt4torchdataset-1.0.0-cp313-cp313-linux_x86_64.whl --force-reinstall
```
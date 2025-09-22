# OPT4TorchDataset
Plug-and-Play Optimal Page Replacement Algorithm (OPT) for Torch Dataset

## What is OPT?

## Why OPT work?

## Install
```bash
pip install OPT4TorchDataset
```

## Usage
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

## Environment (for development only)
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

## Dataset
```bash
hf download --repo-type dataset ILSVRC/imagenet-1k --cache-dir ./data/imagenet-1k --token {your_token_here}
hf download --repo-type dataset timm/mini-imagenet --cache-dir ./data/mini-imagenet
```

## Experiment timm/mini-imagenet
dataset: mini-imagenet
device: NVIDIA H800 sm90 CUDA 12.9
system: ubuntu 24.04
python: 3.13.5
cachetools: 6.2.0
OPT: 1.0.0
torch: 2.8.0 compiled True
cython: OFF
seed: 0
DataIter summary: 977 (50000)
Epoch: 10
num_workers: 16
batch_size: 512

### Training Speed - Method (one device)

Cache Size: 25000 (50% dataset) (? GB RAM)

| model                        | BaseLine | OPT ON | LRU ON | LFU ON | FIFO ON | RR ON |
| ---------------------------- | -------- | ------ | ------ | ------ | ------- | ----- |
| resnet50                     | 5:21±    | 4:36±  | 4:44±  | 4:44±  | 4:41±   | 4:47± |
| efficientnet_b0              | 5:34±    | 4:46±  | 4:49±  | 4:50±  | 4:59±   | 4:52± |
| mobilenetv4_conv_small       | 5:07±    | 3:12±  | 3:16±  | 3:14±  | 3:16±   | 3:15± |
| convnext_base                | 6:54±    | 6:10±  | 6:14±  | 6:23±  | 6:18±   | 6:10± |
| deit3_small_patch16_224      | 4:33±    | 3:17±  | 3:15±  | 3:16±  | 3:13±   | 3:16± |
| vit_small_patch8_224         | 6:36±    | 6:58±  | 6:33±  | 6:35±  | 6:39±   | 6:41± |
| swin_tiny_patch4_window7_224 | 4:09±    | 4:08±  | 3:59±  | 4:03±  | 3:59±   | 6:21± |


log:https://swanlab.cn/@Sail2Dream/opt4/overview

## Experiment CIFAR-10

## Experiment MLP

### Training Speed - Method (multi devices DDP)
TODO

### Hit rate - Method

### Hit rate - Cache Size

## Why OPT NOT work?
思路：建立简单的访问模型（Zipf/Markov），分析在何种分布下 OPT 明显优于 LRU/LFU，给出上界/下界或渐近分析。
创新点：提供理论或半理论解释，减少“纯工程”质疑。
验证：对 synthetic traces（Zipf α 不同, Markov 转移）跑仿真并绘制空间/命中率曲线。
MVP：一页数学推导 + 一套仿真实验图（heatmap）。
产出：补充材料中的理论段落和 synthetic results。


## build up whl
```bash
pip install build
python -m build
pip install dist/opt4torchdataset-1.0.0-cp313-cp313-linux_x86_64.whl --force-reinstall
```
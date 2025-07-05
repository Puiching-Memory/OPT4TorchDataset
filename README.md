# OPT4TorchDataset
Plug-and-Play Optimal Page Replacement Algorithm (OPT) for Torch Dataset

## env
conda create -n opt4 python=3.13
conda activate opt4
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install timm
pip install huggingface-hub

## fix up opencv

## dataset
huggingface-cli download --repo-type dataset --resume-download ILSVRC/imagenet-1k --local-dir ./imagenet-1k

## exp
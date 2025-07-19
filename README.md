# OPT4TorchDataset
Plug-and-Play Optimal Page Replacement Algorithm (OPT) for Torch Dataset

## env
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
pip install timm huggingface-hub datasets[vision] torchmetrics pillow==11.3.0
pip install swanlab  # optional
pip install zarr nvidia-dali-cuda120 # optional
```

## dataset
```bash
huggingface-cli download --repo-type dataset --resume-download ILSVRC/imagenet-1k --local-dir ./imagenet-1k --token {your_token_here}
```

## exp

## build up whl
```bash
pip install build
python -m build
```
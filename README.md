# OPT4TorchDataset
Plug-and-Play Optimal Page Replacement Algorithm (OPT) for Torch Dataset

## env
```bash
conda create -n opt4 python=3.13
conda activate opt4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install timm huggingface-hub datasets
```

## dataset
```bash
huggingface-cli download --repo-type dataset --resume-download ILSVRC/imagenet-1k --local-dir ./imagenet-1k --token {your_token_here}
```

## exp
import torch.utils.data as data
import os
from torchvision.transforms import v2
import torch
import torchvision

class Imagenet1K(data.Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split
        self.idx_list = os.listdir(os.path.join(root_dir, split))

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        #print(index)
        image = torchvision.io.decode_image(os.path.join(self.root_dir, self.split,  self.idx_list[index]),
                                            mode=torchvision.io.ImageReadMode.RGB)
        label = torch.tensor(int(self.idx_list[index].split("-")[0]))
        #print(image,label)
        return image,label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = Imagenet1K(
        r".cache/imagenet-1k-jpeg-256",
        "train"
    )
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True,num_workers=0,pin_memory=True)

    for batch_idx, (image, label) in enumerate(dataloader):
        break
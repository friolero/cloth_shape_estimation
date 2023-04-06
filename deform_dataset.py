import glob
import os
import pickle as pkl

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeformDataset(Dataset):
    def __init__(self, data_dir, mode):

        assert os.path.isdir(
            f"{data_dir}/{mode}"
        ), "Data folder does not exist."
        self.pkl_files = glob.glob(f"{data_dir}/{mode}/*_offset.pkl")
        self.rgb_files = [
            fn.replace("_offset.pkl", "_rgb.png") for fn in self.pkl_files
        ]
        self.normals_files = [
            fn.replace("_offset.pkl", "_normals.png") for fn in self.pkl_files
        ]
        self.img2tensor = transforms.Compose(
            [
                Image.open,
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        with open(self.pkl_files[idx], "rb") as fp:
            data = pkl.load(fp)
            scale = torch.FloatTensor([data["scale"]])
            offsets = data["offsets"]
            depths = data["depth"].permute([2, 0, 1]).repeat([3, 1, 1])
        rgb = self.img2tensor(self.rgb_files[idx])
        normals = self.img2tensor(self.normals_files[idx])
        normals = normals * 2.0 - 1.0
        return rgb, depths, normals, scale, offsets


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    batch_size = 8
    data_dir = "/home/zyuwei/Projects/cloth_shape_estimation/data"
    train_ds = DeformDataset(data_dir=data_dir, mode="train")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for idx, batch in enumerate(train_dl):
        batch = [data.to(device) for data in batch]
        rgb, depths, normals, scale, offsets = batch
        print(
            rgb.shape, depths.shape, normals.shape, scale.shape, offsets.shape
        )
        if idx >= 10:
            break

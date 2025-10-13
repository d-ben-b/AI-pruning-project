import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as F
import re

IMG_SIZE = 1024


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split("(\d+)", s)
    ]


class UAVdataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None):
        self.img_paths = sorted(
            glob(os.path.join(img_dir, "*.png")), key=natural_sort_key
        )
        self.mask_paths = (
            sorted(glob(os.path.join(mask_dir, "*.png")), key=natural_sort_key)
            if mask_dir
            else None
        )
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]) if self.mask_paths else None

        if self.transform:
            img = self.transform(img)
            if mask:
                mask = F.resize(
                    mask,
                    [IMG_SIZE, IMG_SIZE],
                    interpolation=F.InterpolationMode.NEAREST,
                )

        if mask:
            mask = np.array(mask, dtype=np.int64)
            return img, torch.tensor(mask)
        else:
            return img


def get_datasets(root_dir="./data"):
    training_img_dir = os.path.join(root_dir, "train/imgs")
    mask_img_dir = os.path.join(root_dir, "train/masks")
    testing_img_dir = os.path.join(root_dir, "test/imgs")

    # train_transform = T.Compose(
    #     [
    #         T.Resize((256, 256)),
    #         T.RandomHorizontalFlip(p=0.5),
    #         T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    #         T.RandomRotation(10),
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    # test_transform = T.Compose(
    #     [
    #         T.Resize((256, 256)),
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    train_transform = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            # T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = UAVdataset(training_img_dir, mask_img_dir, train_transform)
    test_dataset = UAVdataset(testing_img_dir, None, test_transform)

    return train_dataset, test_dataset

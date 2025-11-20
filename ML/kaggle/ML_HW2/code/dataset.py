import os
import segmentation_models_pytorch as smp
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as F
import re
import torchvision.transforms.functional as F
import random

IMG_SIZE = 512


class JointTransform:
    def __init__(self, img_size=512, hflip=True, rotation=True, color_jitter=True):
        self.img_size = img_size
        self.hflip = hflip
        self.rotation = rotation
        self.color_jitter = color_jitter

        if color_jitter:
            self.jitter = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            )

    def __call__(self, img, mask):
        # 1️⃣ Random Horizontal Flip
        if self.hflip and random.random() < 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)

        # 2️⃣ Random Rotation
        if self.rotation:
            angle = random.uniform(-10, 10)
            img = F.rotate(img, angle, interpolation=F.InterpolationMode.BILINEAR)
            mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)

        # 3️⃣ Color Jitter (only image)
        if self.color_jitter:
            img = self.jitter(img)

        # 4️⃣ Resize both to same size
        img = F.resize(img, [self.img_size, self.img_size])
        mask = F.resize(
            mask,
            [self.img_size, self.img_size],
            interpolation=F.InterpolationMode.NEAREST,
        )

        # 5️⃣ To Tensor
        img = F.to_tensor(img)
        mask = torch.tensor(np.array(mask, dtype=np.int64))

        # 6️⃣ Normalize
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img, mask


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
            if mask is not None:
                img, mask = self.transform(img, mask)
            else:
                img = self.transform(img)

        if mask is not None:
            return img, mask
        else:
            return img


def get_datasets(root_dir="./data"):
    training_img_dir = os.path.join(root_dir, "train/imgs")
    mask_img_dir = os.path.join(root_dir, "train/masks")
    testing_img_dir = os.path.join(root_dir, "test/imgs")

    joint_transform = JointTransform(
        img_size=512, hflip=True, rotation=True, color_jitter=True
    )

    # 測試集只做 resize + toTensor，不做增強
    test_transform = T.Compose(
        [
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = UAVdataset(training_img_dir, mask_img_dir, joint_transform)
    test_dataset = UAVdataset(testing_img_dir, None, test_transform)
    return train_dataset, test_dataset

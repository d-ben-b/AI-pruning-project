# code/dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegDataset(Dataset):
    """
    UAV Semantic Segmentation Dataset
    - image: RGB image
    - mask: grayscale mask (0–15)
    """

    def __init__(self, img_dir, mask_dir=None, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.fnames = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(
            cv2.imread(os.path.join(self.img_dir, fname)), cv2.COLOR_BGR2RGB
        )
        mask = None
        if self.mask_dir:
            mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            if mask is not None:
                aug = self.transform(image=img, mask=mask)
                img, mask = aug["image"], aug["mask"]
            else:
                img = self.transform(image=img)["image"]
        return (img, mask.long()) if mask is not None else (img, fname)


# ---------------------------------------------------------------
# Transform utilities
# ---------------------------------------------------------------
def get_transforms(phase="train"):
    if phase == "train":
        return A.Compose(
            [
                A.Resize(576, 576),
                A.RandomCrop(512, 512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.ColorJitter(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(512, 512),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

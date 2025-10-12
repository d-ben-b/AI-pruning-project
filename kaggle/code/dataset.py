import os, cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None, subset_files=None):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        fnames = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        if subset_files is not None:
            sset = set(subset_files)
            fnames = [f for f in fnames if f in sset]
        self.fnames = fnames
        self.tf = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        fname = self.fnames[i]
        img = cv2.cvtColor(
            cv2.imread(os.path.join(self.img_dir, fname)), cv2.COLOR_BGR2RGB
        )
        if self.mask_dir is None:
            if self.tf:
                img = self.tf(image=img)["image"]
            return img, fname
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        if self.tf:
            out = self.tf(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return img, mask.long()


def get_transforms(
    mode="train", train_resize=(576, 576), train_crop=(512, 512), val_size=(512, 512)
):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(*train_resize),
                A.RandomCrop(*train_crop),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=20, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, p=0.3),
                A.RandomBrightnessContrast(0.3, 0.3, p=0.4),
                A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.4),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.RandomRain(p=0.1),
                A.RandomFog(p=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(*val_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

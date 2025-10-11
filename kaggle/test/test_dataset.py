# test/test_dataset.py
import torch
from kaggle.code.dataset import SegDataset, get_transforms
import cv2
import numpy as np
import os


def test_dataset_sample():
    img_dir = "./kaggle/data/train/imgs"
    mask_dir = "./kaggle/data/train/masks"

    assert os.path.exists(img_dir), "❌ train/imgs 資料夾不存在"
    assert os.path.exists(mask_dir), "❌ train/masks 資料夾不存在"

    dataset = SegDataset(img_dir, mask_dir, transform=get_transforms("train"))
    img, mask = dataset[0]

    print("✅ Dataset 測試成功")
    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)
    print("Unique labels in mask:", torch.unique(mask))
    unique_labels = set()
    for f in os.listdir(mask_dir):
        m = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
        unique_labels.update(np.unique(m).tolist())
    print(sorted(unique_labels))


if __name__ == "__main__":
    print("Testing SegDataset...")
    test_dataset_sample()

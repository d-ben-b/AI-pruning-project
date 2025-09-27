from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch


def build_loaders(data_path, img_size=224, bs=128, workers=8):
    # TODO: val dataloader，train dataloader (subset 選項)
    # Data augmentation and normalization for training
    training_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # === Dataset ===

    pass

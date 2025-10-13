# code/train_utils.py
import torch
import numpy as np
from tqdm import tqdm


def calculate_iou(pred, mask, num_classes=16):
    pred = pred.argmax(1)
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (mask == c)).float().sum()
        union = ((pred == c) | (mask == c)).float().sum()
        if union == 0:
            continue
        ious.append((inter / union).item())
    return np.mean(ious) if ious else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_iou = 0, 0
    for imgs, masks in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_iou += calculate_iou(out, masks)
    return total_loss / len(loader), total_iou / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou = 0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validating"):
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            loss = criterion(out, masks)
            total_loss += loss.item()
            total_iou += calculate_iou(out, masks)
    return total_loss / len(loader), total_iou / len(loader)

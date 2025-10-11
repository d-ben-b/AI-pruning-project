# code/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        intersection = (pred * target_onehot).sum()
        union = pred.sum() + target_onehot.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


def combined_loss(pred, target):
    ce = nn.CrossEntropyLoss()(pred, target)
    dice = DiceLoss()(pred, target)
    focal = FocalLoss()(pred, target)
    return 0.4 * ce + 0.4 * dice + 0.2 * focal

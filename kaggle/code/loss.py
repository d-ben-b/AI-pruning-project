import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_1h = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1)
        inter = (probs * target_1h).sum()
        union = probs.sum() + target_1h.sum()
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.a = alpha
        self.g = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        return (self.a * (1 - pt) ** self.g * ce).mean()


_ce = nn.CrossEntropyLoss()
_dice = DiceLoss()
_focal = FocalLoss()


def combined_loss(logits, target, w_ce=0.5, w_dice=0.3, w_focal=0.2):
    return (
        w_ce * _ce(logits, target)
        + w_dice * _dice(logits, target)
        + w_focal * _focal(logits, target)
    )

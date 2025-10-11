# test/test_loss.py
import torch
from kaggle.code.loss import DiceLoss, FocalLoss, combined_loss


def test_loss_functions():
    pred = torch.randn(2, 16, 256, 256)
    target = torch.randint(0, 16, (2, 256, 256))

    dice = DiceLoss()(pred, target)
    focal = FocalLoss()(pred, target)
    combo = combined_loss(pred, target)

    print(
        f"✅ DiceLoss: {dice.item():.4f}, FocalLoss: {focal.item():.4f}, Combined: {combo.item():.4f}"
    )


if __name__ == "__main__":
    test_loss_functions()

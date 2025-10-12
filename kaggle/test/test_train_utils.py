# test/test_train_utils.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from kaggle.scr.train_utils import train_one_epoch, validate, calculate_iou
from kaggle.scr.loss import combined_loss
from kaggle.scr.model import build_model


def test_training_utils():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("DeepLabV3Plus", "resnet50", num_classes=16).to(device)

    # dummy dataset (2 images, 3x512x512)
    imgs = torch.randn(2, 3, 512, 512)
    masks = torch.randint(0, 16, (2, 512, 512))
    loader = DataLoader(TensorDataset(imgs, masks), batch_size=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_loss, train_iou = train_one_epoch(
        model, loader, optimizer, combined_loss, device
    )
    val_loss, val_iou = validate(model, loader, combined_loss, device)

    print("✅ Train Utils 測試成功")
    print(f"Train Loss={train_loss:.4f}, IoU={train_iou:.4f}")
    print(f"Val Loss={val_loss:.4f}, IoU={val_iou:.4f}")


if __name__ == "__main__":
    test_training_utils()

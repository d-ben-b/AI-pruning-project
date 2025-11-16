import os
import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from termcolor import colored
import segmentation_models_pytorch as smp
from tqdm import tqdm
from kaggle.ML_HW2.code.dataset import get_datasets

# ==========================================================
# 設定
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100  # ✅ 延長訓練週期
BATCH_SIZE = 8
LR_ENCODER = 1e-5
LR_DECODER = 1e-4
NUM_CLASSES = 16
SAVE_PATH = "./checkpoints"
os.makedirs(SAVE_PATH, exist_ok=True)

# ==========================================================
# Dataset & DataLoader
# ==========================================================
train_ds, _ = get_datasets("./data")
total_size = len(train_ds)
val_size = int(0.2 * total_size)
train_size = total_size - val_size

train_subset, val_subset = random_split(train_ds, [train_size, val_size])
train_loader = DataLoader(
    train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==========================================================
# 模型定義 (U-Net++)
# ==========================================================
model = smp.UnetPlusPlus(
    encoder_name="resnet101",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)

# ==========================================================
# 損失函數 (Hybrid Loss)
# ==========================================================
ce_loss = nn.CrossEntropyLoss()
dice_loss = smp.losses.DiceLoss(mode="multiclass")


def hybrid_loss(pred, target):
    """CrossEntropy + Dice"""
    return 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)


# ==========================================================
# 分組學習率 (encoder / decoder)
# ==========================================================
optimizer = optim.Adam(
    [
        {"params": model.encoder.parameters(), "lr": LR_ENCODER},
        {"params": model.decoder.parameters(), "lr": LR_DECODER},
        {"params": model.segmentation_head.parameters(), "lr": LR_DECODER},
    ]
)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)


# ==========================================================
# mIoU 計算
# ==========================================================
def compute_iou(pred, mask, num_classes=NUM_CLASSES):
    pred = torch.argmax(pred, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = mask == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)


# ==========================================================
# 訓練主迴圈
# ==========================================================
best_miou = 0
history = {"train_loss": [], "val_miou": [], "lr": []}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for imgs, masks in tqdm(
        train_loader, desc=f"Train {epoch+1}/{EPOCHS}", colour="green", leave=False
    ):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = hybrid_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # 驗證階段
    model.eval()
    with torch.no_grad():
        miou_scores = []
        for imgs, masks in tqdm(
            val_loader, desc=f"Val {epoch+1}/{EPOCHS}", colour="magenta", leave=False
        ):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            miou = compute_iou(outputs.cpu(), masks.cpu(), NUM_CLASSES)
            miou_scores.append(miou)
        mean_miou = np.nanmean(miou_scores)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(
        colored(
            f"[Epoch {epoch+1:03d}] LR={current_lr:.2e} | Loss={avg_loss:.4f} | Val mIoU={mean_miou:.4f}",
            "cyan",
        )
    )

    history["train_loss"].append(avg_loss)
    history["val_miou"].append(mean_miou)
    history["lr"].append(current_lr)

    # ✅ 儲存最佳模型
    if mean_miou > best_miou:
        best_miou = mean_miou
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, "best_unetpp.pth"))
        print(colored(f"✅ Saved new best model (mIoU={best_miou:.4f})", "green"))

print(colored(f"Training complete! Best mIoU={best_miou:.4f}", "yellow"))

with open(os.path.join(SAVE_PATH, "training_log_unetpp.json"), "w") as f:
    json.dump(history, f)
print(f"✅ Saved training log to {os.path.join(SAVE_PATH, 'training_log_unetpp.json')}")

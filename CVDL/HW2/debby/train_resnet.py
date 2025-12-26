import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# 引用模組
from model import get_modified_resnet18
from utils import set_seed

# 設定隨機種子
set_seed(9966)

if __name__ == "__main__":
    # --- 參數設定 ---
    BATCH_SIZE = 128
    NUM_EPOCHS = 40  # 建議至少 40-50 以達到穩定準確度
    LEARNING_RATE = 0.01
    DATA_PATH = "./data/cifar10"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 準備資料與下載  ---
    # Data Augmentation (提升準確度)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 下載 CIFAR-10
    full_train_dataset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=val_transform)

    # 分割訓練集與驗證集 (80% / 20%) 
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train)) # 20% validation
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    
    # 使用 Subset 建立訓練與驗證集
    train_dataset = Subset(full_train_dataset, train_idx)
    # 驗證集需使用不含 Augmentation 的 transform
    val_dataset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=False, transform=val_transform)
    val_dataset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 載入模型  
    model = get_modified_resnet18().to(device)

    # --- 設定 Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    # 搭配 LR Scheduler 以獲得更好效果
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- 紀錄歷史數據
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    best_val_acc = 0.0

    print("Start Training ResNet18 on CIFAR-10...")

    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix(loss=loss.item(), acc=100.*train_correct/total)

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # 紀錄歷史
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_correct / total)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_correct / val_total)
        
        print(f"Epoch {epoch+1}: Val Acc: {100.*val_correct/val_total:.2f}%")

        # --- 儲存最佳模型 (Highest Validation Accuracy)
        if (val_correct / val_total) > best_val_acc:
            best_val_acc = val_correct / val_total
            torch.save(model.state_dict(), "resnet18_best.pth")
            print(f" -> Best Model Saved! Accuracy: {100.*best_val_acc:.2f}%")
        
        scheduler.step()

    # --- 繪製並儲存 Acc/Loss 圖表
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Training Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("resnet_acc_loss.png")
    plt.show()
    print("Training Complete. Graphs saved as 'resnet_acc_loss.png'.")
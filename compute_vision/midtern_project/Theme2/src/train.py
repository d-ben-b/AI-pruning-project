import os
import shutil
import random
from colorama import Fore, Style, init
init(autoreset=True)
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

SEED = 43
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


def evaluate_test(model):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = correct / total
    print(f"Test Accuracy = {acc:.4f}")
    return acc

def plot_curve(train, val, acc, save):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(train, label="Train Loss")
    plt.plot(val, label="Val Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(acc, label="Val Acc")
    plt.legend()

    plt.savefig(save, dpi=200)
    plt.show()

def plot_cm(model, save):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for images, label in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, pred = outputs.max(1)
            preds.extend(pred.cpu().numpy())
            labels.extend(label.numpy())

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    plt.figure(figsize=(7,7))
    disp.plot(cmap="Blues", values_format="d")
    plt.savefig(save, dpi=200)
    plt.show()

data_root = "/home/ben/project/compute_vision/midtern_project/Theme2/data"

train_dir = f"{data_root}/train"
val_dir = f"{data_root}/validation"
test_dir = f"{data_root}/test"

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
print("Classes:", class_names)

num_classes = 7 

# --- ResNet18 ---
model_resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_resnet18.fc = nn.Linear(model_resnet18.fc.in_features, num_classes)

# --- EfficientNet-B0 ---
model_effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model_effnet.classifier[1] = nn.Linear(model_effnet.classifier[1].in_features, num_classes)

# --- Simple CNN ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model_cnn = SimpleCNN()


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=1e-3,
    patience=5,
    save_prefix="theme2_model"
):

    # -------- 基礎設置 --------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    os.makedirs("/home/ben/project/compute_vision/midtern_project/Theme2/out", exist_ok=True)
    log_path = "/home/ben/project/compute_vision/midtern_project/Theme2/out/theme2_training.log"

    # 紀錄
    best_val_loss = float("inf")
    no_improve = 0

    train_losses, val_losses, val_accs = [], [], []

    # -------- Logging Start --------
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n==== Theme2 Training Start {datetime.now()} ====\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Patience: {patience}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write("\n")

    # -------- Training Loop --------
    for epoch in range(epochs):

        # ==================================================
        # (1) TRAINING
        # ==================================================
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", colour="green")

        for images, labels in train_pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ==================================================
        # (2) VALIDATION
        # ==================================================
        model.eval()
        total = 0
        correct = 0
        val_loss_sum = 0.0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", colour="cyan")

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss_sum / len(val_loader)
        val_losses.append(avg_val_loss)

        val_acc = correct / total
        val_accs.append(val_acc)

        scheduler.step(avg_val_loss)

        # ==================================================
        # (3) 彩色輸出
        # ==================================================
        msg = (
            f"{Fore.YELLOW}Epoch {epoch+1}/{epochs} | "
            f"{Fore.GREEN}Train Loss: {avg_train_loss:.4f} | "
            f"{Fore.CYAN}Val Loss: {avg_val_loss:.4f} | "
            f"{Fore.MAGENTA}Val Acc: {val_acc:.4f}{Style.RESET_ALL}"
        )

        print(msg)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        # ==================================================
        # (4) Save Best Model
        # ==================================================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0

            save_path = f"/home/ben/project/compute_vision/midtern_project/Theme2/out/{save_prefix}.pth"
            torch.save(model.state_dict(), save_path)

            print(Fore.GREEN + f"[Save] Best model saved: {save_path}")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[Save] Best model saved: {save_path}\n")

        else:
            no_improve += 1
            print(Fore.RED + f"[No Improve] {no_improve}/{patience}")

            if no_improve >= patience:
                print(Fore.RED + "[Early Stop] Stopping training.")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write("[Early Stop]\n")
                break

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("==== Training End ====\n\n")

    return train_losses, val_losses, val_accs

models = {
    "resnet18_t2": model_resnet18.to(DEVICE),
    "effnet_b0_t2": model_effnet.to(DEVICE),
    "simplecnn_t2": model_cnn.to(DEVICE)
}

results = {}

for name, model in models.items():
    print("\n===== Training", name, "=====")
    train_losses, val_losses, val_accs = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE,
        save_prefix=name
    )
    results[name] = {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs
    }

for name, r in results.items():
    print("\nTesting", name)
    evaluate_test(r["model"])


for name, r in results.items():
    plot_curve(
        r["train_losses"], r["val_losses"], r["val_accs"],
        f"/home/ben/project/compute_vision/midtern_project/Theme2/out/{name}_curve.png"
    )
for name, r in results.items():
    plot_cm(r["model"], f"/home/ben/project/compute_vision/midtern_project/Theme2/out/{name}_cm.png")
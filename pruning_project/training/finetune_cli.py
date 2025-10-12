import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pruning.nm_pruner import NMPruner
from training.trainer import finetune
import os


# ======= TinyMLP 測試模型 =======
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _cli():
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    out_dir = "out/finetune_test"
    os.makedirs(out_dir, exist_ok=True)

    # ======= 模型 & pruner =======
    model = TinyMLP().to(device)
    pruner = NMPruner(model, N=2, M=4)
    pruner.compute_masks()
    pruner.apply_once()

    # ======= 假資料 =======
    X_train = torch.randn(512, 128)
    y_train = torch.randint(0, 10, (512,))
    X_val = torch.randn(128, 128)
    y_val = torch.randint(0, 10, (128,))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # ======= 執行 finetune =======
    finetune(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=5,
        lr=1e-3,
        out_dir=out_dir,
        pruner=pruner,
    )


if __name__ == "__main__":
    _cli()

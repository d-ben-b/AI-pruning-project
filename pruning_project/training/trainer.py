import torch
import torch.nn.functional as F


def finetune(model, train_loader, val_loader, device, epochs, lr, pruner):
    # TODO:
    # 1. optimizer & scaler
    # 2. 訓練 loop (每個 batch 更新後套用 pruner.apply_once)
    # 3. 每 epoch evaluate
    # 4. 回傳最佳結果
    pass

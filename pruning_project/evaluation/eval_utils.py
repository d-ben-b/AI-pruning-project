from calendar import c
import torch
import torch.nn as nn
from tqdm import tqdm


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct_1 = 0
    total_correct_5 = 0
    total_samples = 0

    progress_bar = tqdm(loader, desc="🔍 Evaluating", leave=False)

    for images, targets in progress_bar:
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        # Top-1 / Top-5 accuracy
        maxk = min(5, outputs.size(1))
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))

        total_correct_1 += correct[:, :1].sum().item()
        total_correct_5 += correct[:, :maxk].sum().item()

        avg_loss = total_loss / total_samples
        top1_acc = 100.0 * total_correct_1 / total_samples
        top5_acc = 100.0 * total_correct_5 / total_samples

    return {
        "loss": avg_loss,
        "top1": top1_acc,
        "top5": top5_acc,
    }

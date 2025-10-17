import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from termcolor import colored
from evaluation.eval_utils import evaluate


# TODO: need to add patient(earily stop)
def finetune(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    out_dir,
    pruner=None,
    arch=None,
    target=None,
):
    """
    Fine-tune model with optional N:M pruning constraint.
    Includes:
      - AMP (torch.amp)
      - Cosine LR scheduler
      - tqdm progress bar (blue)
      - Per-epoch validation + summary
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_top1 = 0.0
    best_epoch = -1

    print(colored("🚀 Starting finetuning...", "cyan"))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        # === 藍色 tqdm bar ===
        pbar = tqdm(
            train_loader,
            desc=colored(f"[Epoch {epoch+1}/{epochs}] Training", "blue"),
            leave=False,
            dynamic_ncols=True,
        )

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if pruner is not None:
                pruner.apply_once(verbose=False)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # 更新 tqdm 狀態
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        scheduler.step()
        avg_loss = total_loss / total_samples

        # === 驗證 ===
        val_metrics = evaluate(model, val_loader, device)

        top1 = val_metrics.get("top1", 0)
        top5 = val_metrics.get("top5", 0)

        if top1 > best_top1:
            best_top1 = top1
            best_epoch = epoch + 1
            filename = (
                f"{arch}_{target.replace('.', '_')}_best.pth"
                if arch and target
                else "best_model.pth"
            )
            save_path = os.path.join(out_dir, filename)
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model to {save_path}")

        print(
            colored(
                f"\n[Epoch {epoch+1:02d}/{epochs}] "
                f"Loss={avg_loss:.4f} | Top1={top1:.2f}% | Top5={top5:.2f}% | "
                f"LR={scheduler.get_last_lr()[0]:.2e}",
                "cyan",
            )
        )

    print(
        colored(
            f"\n✅ Finetuning done! Best Top-1: {best_top1:.2f}% (at epoch {best_epoch})",
            "green",
        )
    )

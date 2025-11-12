import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from termcolor import colored
from evaluation.eval_utils import evaluate


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
    patience=5,  # ✅ 新增：early stop 等待次數
    min_delta=1e-5,  # ✅ 新增：最小改善幅度 (以 top1 為準)
):
    """
    Fine-tune model with optional N:M pruning constraint.
    Includes:
      - AMP (torch.amp)
      - Cosine LR scheduler
      - tqdm progress bar (blue)
      - Per-epoch validation + summary
      - Early stopping (patience & min_delta)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_top1 = 0.0
    best_epoch = -1
    patience_counter = 0

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

        improved = top1 - best_top1 > min_delta

        if improved:
            best_top1 = top1
            best_epoch = epoch + 1
            patience_counter = 0

            if arch and target:
                # 支援多層剪枝，例如 ["mlp.fc1", "mlp.fc2"]
                target_str = (
                    "_".join([t.replace(".", "_") for t in target])
                    if isinstance(target, list)
                    else target.replace(".", "_")
                )
                filename = f"{arch}_{target_str}_best.pth"
            else:
                filename = "best_model.pth"
            save_path = os.path.join(out_dir, filename)
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model to {save_path}")
        else:
            patience_counter += 1
        print(
            colored(
                f"\n[Epoch {epoch+1:02d}/{epochs}] "
                f"Loss={avg_loss:.4f} | Top1={top1:.2f}% | Top5={top5:.2f}% | "
                f"LR={scheduler.get_last_lr()[0]:.2e}",
                "cyan",
            )
        )
        # === Early stop 條件 ===
        if patience_counter >= patience:
            print(
                colored(
                    f"⏹️ Early stopping triggered at epoch {epoch+1}. "
                    f"Best Top-1 = {best_top1:.2f}% (at epoch {best_epoch})",
                    "yellow",
                )
            )
            break

    print(
        colored(
            f"\n✅ Finetuning done! Best Top-1: {best_top1:.2f}% (at epoch {best_epoch})",
            "green",
        )
    )

    return best_epoch

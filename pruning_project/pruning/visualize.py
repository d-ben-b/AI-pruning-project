import os
import torch
import matplotlib.pyplot as plt


def visualize_weights_before_pruning(model, save_root="out/mask_vis/before_pruning"):
    """
    可視化剪枝前的權重分佈。

    Args:
        model: 未剪枝的模型
        save_root: 圖片輸出根目錄
    """
    os.makedirs(save_root, exist_ok=True)
    print("\n🎨 Visualizing weights before pruning...")

    for name, param in model.named_parameters():
        if "weight" not in name or param.ndim < 2:
            continue
        W = param.detach().cpu()
        plt.figure(figsize=(4, 4))
        plt.imshow(W, cmap="viridis", interpolation="nearest")
        plt.title(f"{name}\nWeights before pruning")
        plt.tight_layout()
        save_path = os.path.join(
            save_root, f"{name.replace('.', '_')}_weight_before.png"
        )
        plt.savefig(save_path)
        plt.close()
        print(f"💾 Saved weight visualization to {save_path}")

    print("\n✅ Visualization complete.")


def visualize_masks_and_weights(pruner, model, save_root="out/mask_vis"):
    """
    可視化 N:M 剪枝後的 mask 與實際權重分佈。

    Args:
        pruner: NMPruner 實例 (包含 pruner.masks)
        model: 被剪枝後的模型
        save_root: 圖片輸出根目錄
    """
    # === 建立資料夾 ===
    mask_dir = os.path.join(save_root, "masks")
    weight_dir = os.path.join(save_root, "after_pruning")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    print("\n🎨 Visualizing masks and pruned weights...")

    # === 可視化 mask ===
    for name, mask in pruner.masks.items():
        if mask.ndim < 2:
            print(f"Skipping {name} (1D, likely bias)")
            continue
        plt.figure(figsize=(4, 4))
        plt.imshow(mask.cpu(), cmap="Greys", interpolation="nearest")
        plt.title(f"{name}\nMask (N:M={pruner.N}:{pruner.M})")
        plt.tight_layout()
        save_path = os.path.join(mask_dir, f"{name.replace('.', '_')}_mask.png")
        plt.savefig(save_path)
        plt.close()
        print(f"💾 Saved mask visualization to {save_path}")

    # === 可視化剪枝後權重 ===
    for name, param in model.named_parameters():
        if "weight" not in name or param.ndim < 2:
            continue
        W = param.detach().cpu()
        plt.figure(figsize=(4, 4))
        plt.imshow(W != 0, cmap="Greys", interpolation="nearest")
        plt.title(f"{name}\nNonzero={(W != 0).sum().item()}/{W.numel()}")
        plt.tight_layout()
        save_path = os.path.join(weight_dir, f"{name.replace('.', '_')}_weight.png")
        plt.savefig(save_path)
        plt.close()
        print(f"💾 Saved pruned weight visualization to {save_path}")

    print("\n✅ Visualization complete.")

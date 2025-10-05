import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pruning.nm_pruner import NMPruner
from pruning.visualize import (
    visualize_masks_and_weights,
    visualize_weights_before_pruning,
)

save_dir = "out/mask_vis"
os.makedirs(save_dir, exist_ok=True)


# ======= TinyMLP 測試模型 =======
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ======= 假命名封裝器 =======
class FakeDeiTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].mlp = model  # 讓 fc1, fc2 變成 blocks.0.mlp.fc1 等命名

    def forward(self, x):
        return self.blocks[0].mlp(x)


def _cli():
    model = TinyMLP()
    wrapped = FakeDeiTWrapper(model)
    visualize_weights_before_pruning(
        wrapped, save_root=os.path.join(save_dir, "before_pruning")
    )
    pruner = NMPruner(wrapped, N=2, M=4, targets=["mlp.fc1"])
    pruner.compute_masks()

    # ====== Apply mask ======
    print("\n✂️ Before pruning:")
    w = wrapped.blocks[0].mlp.fc1.weight
    print("Nonzero before:", torch.count_nonzero(w).item())

    pruner.apply_once()

    print("\n✅ After pruning:")
    print("Nonzero after:", torch.count_nonzero(w).item())
    print(f"Sparsity = {1 - torch.count_nonzero(w).item() / w.numel():.2%}")
    visualize_masks_and_weights(pruner, wrapped)

    # ====== 測試梯度 hook ======
    pruner.attach_gradient_hooks()

    # 準備一個假資料 forward + backward
    x = torch.randn(4, 8, requires_grad=False)
    y = wrapped(x).sum()
    y.backward()

    # 驗證梯度是否被遮蔽
    print("\n🎯 Gradient check after backward:")
    for name, param in wrapped.named_parameters():
        if name in pruner.masks:
            grad = param.grad
            mask = pruner.masks[name]
            masked_grad = grad * (mask == 0)
            zeroed = (masked_grad == 0).all().item()
            print(
                f"  - {name}: grad masked correctly = {zeroed}, "
                f"nonzero_grad={(grad != 0).sum().item()}, "
                f"mask_zero={(mask == 0).sum().item()}"
            )
    # ====== 稀疏度與 N:M 規範檢查 ======
    print("\n📊 Checking density and N:M compliance...")
    overall_density, layer_stats = pruner.density_stats()
    print(f"🔢 Overall density = {overall_density:.4f}")

    for stat in layer_stats:
        print(
            f"  - {stat['layer']}: {stat['density']*100:.2f}% "
            f"({stat['nonzero']}/{stat['total']})"
        )

    nm_ok = pruner.nm_compliance()
    print("\n🧩 N:M compliance check:")
    for name, ok in nm_ok.items():
        print(f"  - {name}: {'✅ OK' if ok else '❌ FAIL'}")


if __name__ == "__main__":
    _cli()

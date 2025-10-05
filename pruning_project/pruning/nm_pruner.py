import torch
import torch.nn as nn
import torch.nn.functional as F


class NMPruner:

    def __init__(
        self,
        model: nn.Module,
        N: int,
        M: int,
        targets=[
            "attn.qkv",
            "attn.proj",
            "mlp.fc1",
            "mlp.fc2",
        ],
    ):
        """
        Args:
            model: 要被剪枝的模型 (通常是 DeiT / ViT)
            N: 保留的權重數 (ex. 2)
            M: group 大小 (ex. 4)
        """
        self.model = model
        self.N = N
        self.M = M
        self.masks = {}
        self.prune_targets = targets

    def compute_masks(self):
        """
        計算 attn.qkv, attn.proj, mlp.fc1, mlp.fc2 的 mask。
        每 M 個權重保留 N 個最重要 (|w| 最大)。
        """
        print(f"🔧 Computing {self.N}:{self.M} masks...")
        for name, param in self.model.named_parameters():
            if any(k in name for k in self.prune_targets):
                W = param.data
                mask = self._nm_mask(W, self.N, self.M)
                self.masks[name] = mask.to(W.device)
                print(
                    f"  - {name}: {mask.sum().item():.0f}/{mask.numel()} weights kept"
                )

    def apply_once(self):
        """
        將已經算好的 mask 套用到模型的權重上（in-place）
        讓被剪掉的權重歸零。
        """
        print("✂️  Applying masks to weights...")
        for name, param in self.model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                param.data.mul_(mask)  # in-place mask
                kept = mask.sum().item()
                total = mask.numel()
                print(f"  - {name}: kept {kept}/{total} ({kept/total:.2%})")
        print("✅  All masks applied.\n")

    def attach_gradient_hooks(self):
        """
        註冊 gradient hook，確保被剪掉的權重在反向傳播時梯度永遠為 0。

        原理：
        - PyTorch 支援對 tensor 註冊 hook 函式（在 backward() 時自動呼叫）
        - 我們利用這機制，將梯度乘上對應的 mask。
          若 mask[i,j] = 0 → 對應位置的梯度會被清零，該權重不再更新。
          若 mask[i,j] = 1 → 保留原梯度，正常更新。

        實作流程：
        1. 遍歷所有 model 參數。
        2. 若該層有對應的 mask，就註冊 hook。
        3. hook 函式會在 backward() 時自動執行： grad = grad * mask
        4. 這樣可確保被剪掉的權重不會「長回來」。
        """
        print("📌 Attaching gradient hooks...")
        for name, param in self.model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]

                def hook_fn(grad, mask=mask):
                    # 遮蔽梯度：只有 mask=1 的權重才允許更新
                    return grad * mask

                param.register_hook(hook_fn)
                print(f"  - Hook attached to {name}")

    def density_stats(self):
        """
        計算整體與各層的稀疏度統計。
        Returns:
            overall_density (float): 全模型保留權重比例
            layer_stats (list of dict): 每層統計資料
        """
        layer_stats = []
        total_nonzero, total_params = 0, 0

        for name, mask in self.masks.items():
            nonzero = mask.sum().item()
            total = mask.numel()
            density = nonzero / total
            total_nonzero += nonzero
            total_params += total

            layer_stats.append(
                {
                    "layer": name,
                    "nonzero": int(nonzero),
                    "total": int(total),
                    "density": round(density, 4),
                }
            )

        overall_density = total_nonzero / total_params if total_params > 0 else 0
        return overall_density, layer_stats

    def nm_compliance(self):
        """
        檢查每個 mask 是否符合 N:M 條件。
        Returns:
            compliance (dict): {layer_name: True/False}
        """
        compliance = {}

        for name, mask in self.masks.items():
            flat = mask.view(-1)
            total_groups = flat.numel() // self.M
            flat = flat[: total_groups * self.M]  # 修剪多餘部分
            groups = flat.view(-1, self.M)
            nonzero_per_group = groups.sum(dim=1)

            valid = torch.all(nonzero_per_group == self.N)
            compliance[name] = valid.item()

        return compliance

    def _nm_mask(self, W, N, M):
        """
        產生 N:M 結構化 mask。
        以權重絕對值排序，每 M 個中保留 N 個最大者。
        """
        # 展平成 1D
        flat = W.view(-1)

        # 補齊長度，確保可整除 M
        total_elems = flat.numel()
        pad_len = (M - (total_elems % M)) % M
        if pad_len > 0:
            flat = torch.cat([flat, flat.new_zeros(pad_len)])

        # 分組
        groups = flat.view(-1, M)

        # 每組選出 N 個最大值
        _, idx = torch.topk(groups.abs(), N, dim=1, largest=True)
        mask = torch.zeros_like(groups)
        mask.scatter_(1, idx, 1.0)

        # 還原形狀
        mask = mask.view(-1)[:total_elems].view_as(W)
        return mask

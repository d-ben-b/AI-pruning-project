import torch
import torch.nn as nn
import torch.nn.functional as F


class NMPruner:
    def __init__(self, model: nn.Module, N: int, M: int):
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

    def compute_masks(self):
        # TODO: 計算每層 mask (attn.proj / attn.qkv / mlp.fc1 / mlp.fc2)
        pass

    def apply_once(self):
        # TODO: 把 mask 乘到 weight 上
        pass

    def attach_gradient_hooks(self):
        # TODO: 註冊 hook，讓被 mask 掉的權重梯度永遠是 0
        pass

    def density_stats(self):
        # TODO: 回傳 overall density 與每層統計
        pass

    def nm_compliance(self):
        # TODO: 檢查每 group 是否符合 N:M
        pass

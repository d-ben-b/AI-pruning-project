import time
import torch


def linear_flops(out_f, in_f):
    return out_f * in_f


def flops_dense_and_nm(model, pruner):
    # TODO: 計算 dense FLOPs 與 N:M FLOPs
    pass


@torch.inference_mode()
def measure_latency(model, device, bs=32, warmup=30, iters=100):
    # TODO: 使用隨機輸入測試平均 latency
    pass

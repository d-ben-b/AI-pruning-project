import time
import torch


def linear_flops(out_f, in_f):
    return out_f * in_f


def flops_dense_and_nm(model, pruner):
    """
    比較 Dense vs N:M 模型的 FLOPs。
    假設只考慮 Linear (fc) 層。
    """
    dense_flops, sparse_flops = 0, 0

    for name, param in model.named_parameters():
        if "weight" in name and param.ndim == 2:  # Linear 層
            out_f, in_f = param.shape
            dense_flops += linear_flops(out_f, in_f)

            # 找對應 mask
            mask = pruner.masks.get(name)
            if mask is not None:
                density = mask.sum().item() / mask.numel()
            else:
                density = 1.0
            sparse_flops += dense_flops * density

    return dense_flops, sparse_flops


@torch.inference_mode()
def measure_latency(model, device, bs=32, warmup=30, iters=100):
    """
    用隨機輸入測試模型平均 latency (毫秒)
    """
    model.eval().to(device)
    x = torch.randn(bs, 3, 224, 224, device=device)

    # 預熱 (避免第一次 CUDA 會慢)
    for _ in range(warmup):
        _ = model(x)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = model(x)
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000 / iters
    return avg_ms


def count_nonzero_params(model):
    total, nonzero = 0, 0
    for p in model.parameters():
        total += p.numel()
        nonzero += torch.count_nonzero(p).item()
    return nonzero, total, nonzero / total

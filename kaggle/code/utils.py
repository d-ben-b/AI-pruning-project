import os, random, numpy as np, torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_iou(logits, target, num_classes=16):
    with torch.no_grad():
        pred = logits.argmax(1)
        ious = []
        for c in range(num_classes):
            p = pred == c
            g = target == c
            inter = (p & g).sum().item()
            uni = (p | g).sum().item()
            ious.append((inter + 1e-6) / (uni + 1e-6))
        return float(np.mean(ious))


def rle_encode(mask_np):
    # mask_np: {0,1} np.ndarray (H,W), Fortran order
    import numpy as np

    pixels = mask_np.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[:-1:2]
    return " ".join(map(str, runs))

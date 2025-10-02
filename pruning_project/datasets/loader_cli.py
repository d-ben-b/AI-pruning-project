# loader_cli.py
from config import get_config
from datasets.loaders import build_loaders
import torch


def main():
    args = get_config()
    train_loader, val_loader = build_loaders(args)

    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")

    # 抽一個 batch 出來看看
    images, labels = next(iter(train_loader))
    print(f"One train batch images: {images.shape}, labels: {labels.shape}")

    images, labels = next(iter(val_loader))
    print(f"One val batch images: {images.shape}, labels: {labels.shape}")

    # 檢查 GPU 是否可用
    if torch.cuda.is_available():
        device = torch.device(args.device)
        images = images.to(device)
        labels = labels.to(device)
        print(f"Moved one batch to {device}: {images.shape}, {labels.shape}")


if __name__ == "__main__":
    main()

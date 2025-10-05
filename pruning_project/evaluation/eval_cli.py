import timm
import torch
from config import get_config
from datasets.loaders import build_loaders
from evaluation.eval_utils import evaluate


def main():
    args = get_config()
    print(f"🧠 Loading model: {args.arch}")
    model = timm.create_model(
        args.arch, pretrained=not args.no_pretrained, num_classes=1000
    ).to(args.device)

    _, val_loader = build_loaders(args)
    print("🚀 Start evaluation...")
    metrics = evaluate(model, val_loader, device=args.device)

    print(f"\n✅ Done! Results:")
    print(f"  Loss:  {metrics['loss']:.4f}")
    print(f"  Top-1: {metrics['top1']:.2f}%")
    print(f"  Top-5: {metrics['top5']:.2f}%")


if __name__ == "__main__":
    main()

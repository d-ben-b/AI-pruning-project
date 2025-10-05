from config import get_config
from pruning.nm_pruner import NMPruner
from training.trainer import finetune
from training.rewind import save_rewind_point, load_rewind_point
from evaluation.eval_utils import evaluate
from log.logger import log_experiment, log_per_layer, plot_results
from datasets.loaders import build_loaders
import timm
import os
import torch
from pathlib import Path


def main():
    exp_csv = Path(args.out) / "experiments.csv"
    per_csv = Path(args.out) / "per_layer_stats.csv"
    args = get_config()
    # 1. 建立模型 (用 timm)
    if args.arch == "all":
        archs = [
            "deit_tiny_patch16_224",
            "deit_small_patch16_224",
            "deit_base_patch16_224",
            "deit_tiny_distilled_patch16_224",
            "deit_small_distilled_patch16_224",
            "deit_base_distilled_patch16_224 ",
        ]
    else:
        archs = [args.arch]
    for arch in archs:
        model = timm.create_model(arch, pretrained=args.pretrained, num_classes=1000)
        model.to(args.device)
        print(f"Model {arch} created.")

        # Save rewind point
        rewind_path = os.path.join(args.out, f"rewind_{args.rewind_tag}.pth")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # dummy optimizer
        save_rewind_point(model, optimizer, rewind_path)
        load_rewind_point(model, optimizer, rewind_path, map_location=args.device)

        # Build dataloaders
        train_loader, val_loader = build_loaders(args)

        # Baseline evaluation
        baseline_metrics = evaluate(model, val_loader, device=args.device)
        print(f"Baseline Top-1: {baseline_metrics['top1']:.2f}%")
        print(f"Baseline Top-5: {baseline_metrics['top5']:.2f}%")
        # baseline_acc = baseline_metrics["top1"]

        # Initialize pruner and prune
        pruner = NMPruner(model, N=args.n, M=args.m)
        pruner.compute_masks()
        pruner.apply_once()
        pruner.attach_gradient_hooks()
        overall_density, layer_stats = pruner.density_stats()
        print(f"Overall density after pruning: {overall_density:.4f}")
        log_per_layer(layer_stats, args.out)

        # Check N:M compliance
        compliance = pruner.nm_compliance()
        print("N:M compliance per layer:", compliance)

        # Rewind to the specified point
        load_rewind_point(model, tag=args.rewind_tag)
        print(f"Model rewound to {args.rewind_tag} state.")

        # Finetune
        finetune(
            model,
            train_loader,
            val_loader,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            device=args.device,
            out_dir=args.out,
        )

        # Final evaluation
        final_metrics = evaluate(model, val_loader, device=args.device)
        print(f"Final Top-1 after finetuning: {final_metrics['top1']:.2f}%")

        # Log experiment results
        log_per_layer(per_csv, layer_stats)
        log_experiment(
            exp_csv,
            {
                "arch": arch,
                "baseline_top1": baseline_metrics["top1"],
                "final_top1": final_metrics["top1"],
                "density": overall_density,
                "n": args.n,
                "m": args.m,
                "epochs": args.finetune_epochs,
                "lr": args.finetune_lr,
            },
        )
        plot_results(args.out)


if __name__ == "__main__":
    main()

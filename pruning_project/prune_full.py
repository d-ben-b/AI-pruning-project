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
    args = get_config()
    exp_csv = Path(args.out) / "experiments.csv"
    per_csv = Path(args.out) / "per_layer_stats.csv"

    # 1. 建立模型 (用 timm)
    # === 模型選擇 ===
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

    # === 要分別測試剪的四種 Layer ===
    prune_targets = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]

    for arch in archs:
        print(f"\n🚀 Model {arch} start pruning experiments...")
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

        # === baseline 儲存 ===
        base_ckpt = Path(args.out) / f"{arch}_baseline.pth"
        torch.save(model.state_dict(), base_ckpt)
        print(f"💾 Saved baseline model to {base_ckpt}")

        # Initialize pruner and prune

        print(f"\n============================")
        print(f"✂️ Pruning target: {prune_targets}")
        print(f"============================")

        model.load_state_dict(torch.load(base_ckpt, map_location=args.device))

        pruner = NMPruner(model, N=args.n, M=args.m, targets=prune_targets)
        pruner.compute_masks()
        pruner.apply_once()
        pruner.attach_gradient_hooks()

        # Check density
        overall_density, layer_stats = pruner.density_stats()
        print(f"Overall density after pruning: {overall_density:.4f}")
        log_per_layer(
            layer_stats, args.out, arch=arch, n=args.n, m=args.m, tag=prune_targets
        )

        # Check N:M compliance
        compliance = pruner.nm_compliance()
        print("N:M compliance per layer:", compliance)

        # Save pruned model
        pruned_ckpt = Path(args.out) / f"{arch}_fc1_fc2_pruned.pth"
        torch.save(model.state_dict(), pruned_ckpt)
        print(f"💾 Saved pruned model to {pruned_ckpt}")

        # Rewind to the specified point
        load_rewind_point(model, optimizer, rewind_path, map_location=args.device)
        print(f"Model rewound to {args.rewind_tag} state.")

        # Finetune
        current_epoch = finetune(
            model,
            train_loader,
            val_loader,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            device=args.device,
            out_dir=args.out,
            pruner=pruner,
            arch=arch,
            target=prune_targets,
        )

        # Final evaluation
        final_metrics = evaluate(model, val_loader, device=args.device)
        print(f"Final Top-1 after finetuning: {final_metrics['top1']:.2f}%")

        # Log experiment results
        log_per_layer(
            layer_stats, args.out, arch=arch, n=args.n, m=args.m, tag=prune_targets
        )
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
                "early_stop_epoch": current_epoch,
                "lr": args.finetune_lr,
                "tag": prune_targets,
            },
        )
        plot_results(args.out)
        print(f"\n🎉 Completed all layer pruning tests for {arch}.")


if __name__ == "__main__":
    main()

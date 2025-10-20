# pruning_project/test_nm_compliance.py
import torch
from pruning.nm_pruner import NMPruner
import timm
import argparse


def check_nm_compliance(model, N=2, M=4):
    """
    檢查模型是否符合 N:M 結構化稀疏。
    """
    pruner = NMPruner(model, N=N, M=M)
    compliance = {}

    for name, param in model.named_parameters():
        if any(k in name for k in ["mlp.fc2"]):
            W = param.data
            flat = W.view(-1)
            total_groups = flat.numel() // M
            flat = flat[: total_groups * M]
            groups = flat.view(-1, M)
            nonzero_per_group = (groups != 0).sum(dim=1)
            valid = torch.all(nonzero_per_group <= N)
            compliance[name] = valid.item()
            print(
                f"{name:35s} | "
                f"Groups={total_groups:6d} | "
                f"OK={valid.item():9d} | "
                f"Density={W.nonzero().size(0)/W.numel():.2f}"
            )

    all_ok = all(compliance.values())
    print(
        "\n✅ Model is N:M compliant!" if all_ok else "\n❌ Non-compliant layers found!"
    )
    return compliance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check N:M compliance of a pruned model"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--arch", type=str, default="deit_tiny_patch16_224")
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--M", type=int, default=4)
    args = parser.parse_args()

    print(f"🔍 Loading model {args.arch} from {args.model_path} ...")
    model = timm.create_model(args.arch, pretrained=False, num_classes=1000)
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu", weights_only=True)
    )

    check_nm_compliance(model, N=args.N, M=args.M)

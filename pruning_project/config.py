import argparse


def get_config():
    parser = argparse.ArgumentParser(description="Configuration for pruning project")
    parser.add_argument("--arch", default="all", type=str, help="Model architecture")
    parser.add_argument(
        "--prune_target",
        nargs="+",  # 允許輸入多個參數，並自動存成 List
        default=["all"],  # 預設值改為 List
        help="Layer to prune",
    )
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Do not use pretrained weights"
    )
    parser.add_argument("--data_path", type=str, default="/homes/nfs/Parker/ImageNet/")
    parser.add_argument("--out", type=str, default="out/deit_nm")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--rewind_tag", type=str, default="init")
    parser.add_argument(
        "--cache_mode", default="part", choices=["part", "full"], help="Cache mode"
    )

    args = parser.parse_args()

    args.pretrained = not args.no_pretrained
    return args

import argparse


def get_congig():
    parser = argparse.ArgumentParser(description="Configuration for pruning project")
    parser.add_argument(
        "--arch", default="deit_tiny_patch16_224", type=str, help="Model architecture"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="如果加上這個 flag，則不載入 pretrained 權重",
    )
    parser.add_argument("--data_path", type=str, default="", required=True)
    parser.add_argument("--out", type=str, default="out/deit_nm")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--finetune_epochs", type=int, default=50)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--rewind_tag", type=str, default="init")
    args = parser.parse_args()
    args.pretrained = not args.no_pretrained
    return args

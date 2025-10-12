import os, argparse, hashlib, subprocess, shlex
import numpy as np, pandas as pd, torch, cv2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.datasets import SegDataset, get_transforms
from src.models import build_model
from src.utils import rle_encode


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="kaggle/data")
    ap.add_argument("--model_name", type=str, default="Unet++")
    ap.add_argument("--encoder", type=str, default="efficientnet-b5")
    ap.add_argument("--num_classes", type=int, default=16)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--orig_h", type=int, default=1024)
    ap.add_argument("--orig_w", type=int, default=1024)
    ap.add_argument("--hflip_tta", action="store_true")
    ap.add_argument(
        "--ms_tta",
        type=float,
        nargs="*",
        default=[],
        help="e.g. --ms_tta 0.75 1.0 1.25",
    )
    ap.add_argument("--out_csv", type=str, default="kaggle/outputs/submission.csv")
    ap.add_argument("--kaggle_submit", action="store_true")
    ap.add_argument(
        "--kaggle_comp", type=str, default="2025-ncku-ee-ml-16-classes-segmentation"
    )
    return ap.parse_args()


def infer_logits(model, imgs, scales, device):
    # multi-scale + flip TTA → average probs
    import torch.nn.functional as F

    if not scales:
        scales = [1.0]
    agg = None
    with torch.no_grad():
        for s in scales:
            x = (
                imgs
                if s == 1.0
                else F.interpolate(
                    imgs, scale_factor=s, mode="bilinear", align_corners=False
                )
            )
            logit = model(x)
            logit = F.interpolate(
                logit, size=imgs.shape[-2:], mode="bilinear", align_corners=False
            )
            if agg is None:
                agg = torch.softmax(logit, dim=1)
            else:
                agg += torch.softmax(logit, dim=1)
    return agg / len(scales)


def main():
    args = parse_args()
    device = (
        args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    )

    test_img_dir = os.path.join(args.data_root, "test", "imgs")
    assert os.path.exists(test_img_dir)

    tf = get_transforms("val", val_size=(512, 512))
    ds = SegDataset(test_img_dir, mask_dir=None, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(
        args.model_name, args.encoder, args.num_classes, pretrained=False
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    H, W = args.orig_h, args.orig_w
    records = []
    for imgs, names in tqdm(loader, desc="Predicting"):
        imgs = imgs.to(device)
        probs = infer_logits(model, imgs, args.ms_tta or [1.0], device)
        if args.hflip_tta:
            probs_f = infer_logits(
                model, torch.flip(imgs, dims=[3]), args.ms_tta or [1.0], device
            )
            probs = (probs + torch.flip(probs_f, dims=[3])) / 2
        preds = probs.argmax(1).cpu().numpy().astype(np.uint8)

        for pred, fname in zip(preds, names):
            pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
            row = {"img": fname}
            for c in range(args.num_classes):
                m = (pred == c).astype(np.uint8)
                row[f"class_{c}"] = "none" if m.sum() == 0 else rle_encode(m)
            records.append(row)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(args.out_csv, index=False)
    md5 = hashlib.md5(open(args.out_csv, "rb").read()).hexdigest()[:12]
    print(f"Saved: {args.out_csv} (rows={len(df)}) md5={md5}")

    # quick sanity: non-empty counts
    non_empty = (df.drop(columns=["img"]) != "none").sum()
    print("非空RLE數量 (1000張):\n", non_empty.sort_values())

    if args.kaggle_submit:
        msg = f"{args.model_name} {args.encoder} | 1024 NN | F-order RLE | md5={md5}"
        cmd = f'kaggle competitions submit -c {args.kaggle_comp} -f "{args.out_csv}" -m "{msg}"'
        print("Submitting:", cmd)
        subprocess.run(shlex.split(cmd), check=True)


if __name__ == "__main__":
    main()

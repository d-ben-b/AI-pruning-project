# kaggle/train.py
import os, json, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# ===== project imports =====
from kaggle.code.dataset import SegDataset, get_transforms
from kaggle.code.model import build_model
from kaggle.code.loss import combined_loss
from kaggle.code.utils import set_seed, calculate_iou


# -------------------------
# argparse
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="kaggle/data")
    ap.add_argument(
        "--model_name", type=str, default="Unet++"
    )  # "DeepLabV3Plus" | "Unet++" | "FPN"
    ap.add_argument(
        "--encoder", type=str, default="efficientnet-b5"
    )  # e.g. "resnet101", "efficientnet-b4/b5"
    ap.add_argument("--num_classes", type=int, default=16)

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train_resize", type=int, nargs=2, default=[800, 800])
    ap.add_argument("--train_crop", type=int, nargs=2, default=[768, 768])
    ap.add_argument("--val_size", type=int, nargs=2, default=[768, 768])

    ap.add_argument("--save_dir", type=str, default="kaggle")
    ap.add_argument("--patience", type=int, default=10)

    ap.add_argument(
        "--eval_1024",
        action="store_true",
        help="validation 時把 pred/gt 回縮到 1024 再算 mIoU（模擬 Kaggle 評分尺寸）",
    )

    # 續訓相關
    ap.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="要續訓的 checkpoint 路徑；可為 weights-only .pth 或 *_full.pth",
    )
    ap.add_argument(
        "--resume_strict",
        action="store_true",
        help="載入 checkpoint 時使用 strict=True（預設 False：僅載入匹配/同 shape 權重）",
    )
    return ap.parse_args()


# -------------------------
# helpers
# -------------------------
def _safe_get_transforms(mode, train_resize, train_crop, val_size):
    """兼容兩種介面：get_transforms(mode, ...) 或 get_transforms(mode)"""
    try:
        return get_transforms(mode, train_resize, train_crop, val_size)
    except TypeError:
        return get_transforms(mode)


def _strip_module_keys(state_dict):
    # 把 DataParallel 存的 'module.' 拿掉
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _load_as_much_as_possible(model, state_dict, strict=False):
    """非嚴格載入：僅把名稱存在且 shape 相同的權重覆蓋，其他跳過。"""
    if strict:
        model.load_state_dict(state_dict, strict=True)
        return
    cur = model.state_dict()
    matched = {
        k: v for k, v in state_dict.items() if k in cur and cur[k].shape == v.shape
    }
    missing = [k for k in cur.keys() if k not in matched]
    extra = [k for k in state_dict.keys() if k not in cur]
    print(
        f"↪️ matched={len(matched)}  missing_in_ckpt={len(missing)}  extra_in_ckpt={len(extra)}"
    )
    cur.update(matched)
    model.load_state_dict(cur, strict=False)


def miou_on_loader_upsampled(model, loader, device, n_classes=16, H=1024, W=1024):
    import cv2

    inter = np.zeros(n_classes, dtype=np.float64)
    uni = np.zeros(n_classes, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            pred = model(imgs).argmax(1).cpu().numpy().astype(np.uint8)
            gt = masks.numpy().astype(np.uint8)
            for p, g in zip(pred, gt):
                p = cv2.resize(p, (W, H), interpolation=cv2.INTER_NEAREST)
                g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
                for c in range(n_classes):
                    pi, gi = (p == c), (g == c)
                    inter[c] += np.logical_and(pi, gi).sum()
                    uni[c] += np.logical_or(pi, gi).sum()
    iou = inter / (uni + 1e-6)
    return float(np.nanmean(iou))


# -------------------------
# main
# -------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = (
        args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"
    )
    print(f"[Device] {device} | CUDA available: {torch.cuda.is_available()}")

    img_dir = os.path.join(args.data_root, "train", "imgs")
    mask_dir = os.path.join(args.data_root, "train", "masks")
    assert os.path.exists(img_dir), f"Missing: {img_dir}"
    assert os.path.exists(mask_dir), f"Missing: {mask_dir}"

    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    tr_files, val_files = train_test_split(
        all_files, test_size=0.2, random_state=args.seed
    )

    train_tf = _safe_get_transforms(
        "train", tuple(args.train_resize), tuple(args.train_crop), tuple(args.val_size)
    )
    val_tf = _safe_get_transforms(
        "val", tuple(args.train_resize), tuple(args.train_crop), tuple(args.val_size)
    )

    tr_ds = SegDataset(img_dir, mask_dir, transform=train_tf, subset_files=tr_files)
    va_ds = SegDataset(img_dir, mask_dir, transform=val_tf, subset_files=val_files)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # model / optim / sched / amp
    model = build_model(
        args.model_name, args.encoder, args.num_classes, pretrained=True
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(enabled=("cuda" in device and args.use_amp))

    # 續訓
    start_epoch = 1
    best_iou = -1.0
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"🔄 Resuming from: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device)

        # (A) 完整 checkpoint：含 optimizer/scheduler/scaler/epoch 等
        if isinstance(ckpt, dict) and ("model" in ckpt or "state_dict" in ckpt):
            state = ckpt.get("model", ckpt.get("state_dict"))
            state = _strip_module_keys(state)
            _load_as_much_as_possible(
                model, state, strict=getattr(args, "resume_strict", False)
            )

            # 盡力恢復 opt/sch/scaler
            try:
                if "optimizer" in ckpt and ckpt["optimizer"]:
                    opt.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print("⚠️ optimizer state 載入失敗，將重新初始化：", e)
            try:
                if "scheduler" in ckpt and ckpt["scheduler"]:
                    sch.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print("⚠️ scheduler state 載入失敗，將重新初始化：", e)
            try:
                if args.use_amp and "scaler" in ckpt and ckpt["scaler"] is not None:
                    scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print("⚠️ scaler state 載入失敗（可忽略）：", e)

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_iou = float(ckpt.get("best_iou", -1.0))
            print(
                f"✔️  已載入完整 checkpoint：從 epoch {start_epoch} 繼續（best_iou={best_iou:.4f}）"
            )

        # (B) 權重-only：一般 .pth
        else:
            if not isinstance(ckpt, dict):
                raise RuntimeError(
                    "未知的 checkpoint 格式（期望為 state_dict 或包含 'model' 的完整 ckpt）"
                )
            state = _strip_module_keys(ckpt)
            _load_as_much_as_possible(
                model, state, strict=getattr(args, "resume_strict", False)
            )
            print("✔️  已載入權重；optimizer/scheduler 將從頭開始。")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.save_dir, f"{args.model_name.replace('+','plus')}_{args.encoder}_best.pth"
    )
    json.dump(
        vars(args), open(os.path.join(args.save_dir, "train_cfg.json"), "w"), indent=2
    )

    patience = 0
    for epoch in range(start_epoch, args.epochs + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        tr_iou = 0.0
        for x, y in tqdm(tr_loader, desc=f"Train {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss = combined_loss(out, y)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                out = model(x)
                loss = combined_loss(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tr_loss += loss.item()
            tr_iou += calculate_iou(out, y, args.num_classes)
        tr_loss /= len(tr_loader)
        tr_iou /= len(tr_loader)

        # ---- val ----
        model.eval()
        va_loss = 0.0
        va_iou = 0.0
        with torch.no_grad():
            for x, y in tqdm(va_loader, desc="Validating"):
                x, y = x.to(device), y.to(device)
                out = model(x)
                va_loss += combined_loss(out, y).item()
                va_iou += calculate_iou(out, y, args.num_classes)
        va_loss /= len(va_loader)
        va_iou /= len(va_loader)

        sch.step()

        if args.eval_1024:
            miou1024 = miou_on_loader_upsampled(
                model, va_loader, device, args.num_classes, 1024, 1024
            )
            print(f"[Epoch {epoch}] mIoU@1024={miou1024:.4f} (Kaggle 規格)")

        # ---- save best ----
        if va_iou > best_iou:
            best_iou = va_iou
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ New best IoU={best_iou:.4f} → saved {ckpt_path}")

            # 另外存完整 ckpt，之後可無縫續訓
            full_ckpt_path = ckpt_path.replace("_best.pth", "_best_full.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sch.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "best_iou": best_iou,
                    "args": vars(args),
                },
                full_ckpt_path,
            )
            print(f"🧷 Saved full checkpoint -> {full_ckpt_path}")
        else:
            patience += 1

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"Train Loss={tr_loss:.4f} IoU={tr_iou:.4f} | "
            f"Val Loss={va_loss:.4f} IoU={va_iou:.4f}"
        )

        if patience >= args.patience:
            print("Early stopping.")
            break


if __name__ == "__main__":
    main()

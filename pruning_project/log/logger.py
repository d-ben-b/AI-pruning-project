import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def log_experiment(exp_csv, row_dict):
    """
    將一筆整體實驗結果 append 到 experiments.csv

    Args:
        exp_csv (str or Path): 輸出 CSV 路徑
        row_dict (dict): 實驗結果，例如：
            {
                "arch": "deit_tiny_patch16_224",
                "baseline_top1": 72.1,
                "final_top1": 73.5,
                "density": 0.50,
                "n": 2,
                "m": 4,
                "epochs": 50,
                "lr": 1e-5
            }
    """
    exp_csv = Path(exp_csv)
    exp_csv.parent.mkdir(parents=True, exist_ok=True)

    file_exists = exp_csv.exists()
    fieldnames = [
        "arch",
        "baseline_top1",
        "final_top1",
        "density",
        "n",
        "m",
        "epochs",
        "lr",
    ]

    with open(exp_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

    print(f"💾 Logged experiment results to {exp_csv}")


def log_per_layer(layer_stats, out_dir, arch="unknown", n=None, m=None, tag=None):
    """
    記錄每層的剪枝結果到 CSV。
    Args:
        layer_stats (list[dict]): 每層的統計資訊
            e.g., [{"layer": "mlp.fc1.weight", "density": 0.5, "nonzero": 32, "total": 64}, ...]
        out_dir (str or Path): 輸出資料夾
        arch (str): 模型名稱
        n, m (int): N:M 結構參數
        tag (str): 剪枝層名稱或實驗標籤
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_layer_stats.csv"

    fieldnames = ["arch", "layer", "density", "nonzero", "total", "n", "m", "tag"]
    new_rows = []

    for stat in layer_stats:
        new_rows.append(
            {
                "arch": arch,
                "layer": stat["layer"],
                "density": f"{stat['density']:.4f}",
                "nonzero": stat["nonzero"],
                "total": stat["total"],
                "n": n,
                "m": m,
                "tag": tag,
            }
        )

    # 若檔案不存在，先寫 header
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"💾 Logged per-layer stats to {csv_path}")


def plot_results(out_dir):
    """
    從 experiments.csv 讀取結果並繪圖。
    輸出折線圖 (density vs acc) / (N:M 結構 vs acc)
    """
    out_dir = Path(out_dir)
    csv_path = out_dir / "experiments.csv"
    if not csv_path.exists():
        print("⚠️ No experiments.csv found, skip plotting.")
        return

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))

    # Top-1 accuracy 改善 vs density
    plt.plot(df["density"], df["final_top1"], "o-", label="Final Top-1")
    plt.plot(df["density"], df["baseline_top1"], "x--", label="Baseline Top-1")

    plt.xlabel("Density (1 - Sparsity)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Accuracy vs Density across Experiments")
    plt.legend()
    plt.grid(True)

    save_path = out_dir / "accuracy_vs_density.png"
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Saved plot to {save_path}")

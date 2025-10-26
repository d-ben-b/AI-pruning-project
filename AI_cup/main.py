# ============================================================
# main.py — 玉山 AI CUP 2025: PU Learning 智能修正版 (v4)
# 支援帳戶雙向特徵 (from + to)，解決全零特徵問題
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier
from pulearn import ElkanotoPuClassifier

# ============================================================
# 1. 讀取資料
# ============================================================
DATA_DIR = "./data"
alert_path = os.path.join(DATA_DIR, "acct_alert.csv")
trans_path = os.path.join(DATA_DIR, "acct_transaction.csv")
predict_path = os.path.join(DATA_DIR, "acct_predict.csv")

df_pos = pd.read_csv(alert_path)
df_unlabeled = pd.read_csv(predict_path)
df_trans = pd.read_csv(trans_path)

# 統一欄位名稱小寫
for df in [df_pos, df_unlabeled, df_trans]:
    df.columns = [c.strip().lower() for c in df.columns]

df_pos["label"] = 1
df_unlabeled["label"] = 0
df_accounts = pd.concat([df_pos, df_unlabeled], ignore_index=True)

# ============================================================
# 2. 交易特徵 — 同時計算 from 與 to 特徵
# ============================================================
print("📊 Building bidirectional transaction features...")


df_trans["txn_time_min"] = (
    pd.to_datetime(df_trans["txn_time"], format="%H:%M:%S").dt.hour * 60
    + pd.to_datetime(df_trans["txn_time"], format="%H:%M:%S").dt.minute
)
df_trans["hour"] = pd.to_datetime(df_trans["txn_time"], format="%H:%M:%S").dt.hour

# ---- From 帳戶 ----
agg_from = (
    df_trans.groupby("from_acct")
    .agg(
        txn_amt_mean=("txn_amt", "mean"),
        txn_amt_sum=("txn_amt", "sum"),
        txn_amt_std=("txn_amt", "std"),
        txn_date_nunique=("txn_date", "nunique"),
        cross_bank_ratio=("to_acct_type", lambda x: np.mean(x == "02")),
        night_ratio=("hour", lambda x: np.mean(x.between(0, 6))),
        self_ratio=("is_self_txn", lambda x: np.mean(x == "Y")),
    )
    .reset_index()
    .rename(columns={"from_acct": "acct"})
)

# ---- To 帳戶 ----
agg_to = (
    df_trans.groupby("to_acct")
    .agg(
        recv_amt_mean=("txn_amt", "mean"),
        recv_amt_sum=("txn_amt", "sum"),
        recv_amt_std=("txn_amt", "std"),
        recv_from_nunique=("from_acct", "nunique"),
    )
    .reset_index()
    .rename(columns={"to_acct": "acct"})
)

# ---- 合併 ----
df_feat = pd.merge(agg_from, agg_to, on="acct", how="outer").fillna(0)

# ============================================================
# 3. 合併帳戶標籤
# ============================================================
df_all = pd.merge(df_accounts, df_feat, on="acct", how="left").fillna(0)

X = df_all.drop(columns=["acct", "label"])
y = df_all["label"]

# ============================================================
# 4. PU 結構資料分割
# ============================================================
df_pos_train, df_pos_val = train_test_split(
    df_all[df_all["label"] == 1], test_size=0.2, random_state=42
)
df_unlabeled_train, df_unlabeled_val = train_test_split(
    df_all[df_all["label"] == 0], test_size=0.2, random_state=42
)

df_train = pd.concat([df_pos_train, df_unlabeled_train], ignore_index=True)
df_val = pd.concat([df_pos_val, df_unlabeled_val], ignore_index=True)

X_train = np.array(df_train.drop(columns=["acct", "label"]))
y_train = np.array(df_train["label"])
X_val = np.array(df_val.drop(columns=["acct", "label"]))
y_val = np.array(df_val["label"])

print(f"🧩 Train size: {len(X_train)}, Positives: {df_pos_train.shape[0]}")
print(f"🧩 Val size:   {len(X_val)}, Positives: {df_pos_val.shape[0]}")

# ============================================================
# 5. PU Learning 模型
# ============================================================
base_estimator = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=5,
    min_gain_to_split=0.0,
    n_jobs=-1,
    random_state=42,
)

pu_clf = ElkanotoPuClassifier(
    estimator=base_estimator, hold_out_ratio=0.1, random_state=42
)

print("🚀 Training PU model...")
pu_clf.fit(X_train, y_train)

# ============================================================
# 6. 閾值最佳化
# ============================================================
val_pred = pu_clf.predict_proba(X_val)[:, 1]
best_t, best_f1 = 0, 0
for t in np.linspace(0.01, 0.5, 100):
    f1 = f1_score(y_val, (val_pred > t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"✅ Best threshold = {best_t:.3f}, F1 = {best_f1:.4f}")
print(classification_report(y_val, (val_pred > best_t).astype(int)))

# ============================================================
# 7. 產生 submission
# ============================================================
pred_df = df_all[df_all["label"] == 0].copy()
pred_df["pred_score"] = pu_clf.predict_proba(pred_df.drop(columns=["acct", "label"]))[
    :, 1
]
pred_df["sar_flag"] = (pred_df["pred_score"] > best_t).astype(int)

submission = pred_df[["acct", "sar_flag"]]
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv 已生成，可上傳 T-Brain！")

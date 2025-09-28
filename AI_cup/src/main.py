import pandas as pd
import os

# 讀取交易資料

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 取得 src/ 路徑
DATA_DIR = os.path.join(BASE_DIR, "../data")  # 指向 data/ 資料夾
df_txn = pd.read_csv(os.path.join(DATA_DIR, "acct_transaction.csv"))
df_alert = pd.read_csv(os.path.join(DATA_DIR, "acct_alert.csv"))


acct_features = (
    df_txn.groupby("from_acct")
    .agg(
        txn_count=("txn_amt", "count"),  # 交易筆數
        txn_sum=("txn_amt", "sum"),  # 總交易金額
        txn_mean=("txn_amt", "mean"),  # 平均交易金額
        txn_max=("txn_amt", "max"),  # 最大金額
        to_acct_unique=("to_acct", "nunique"),  # 收款帳戶數
        channel_unique=("channel_type", "nunique"),  # 使用通路數
    )
    .reset_index()
)


df_alert = df_alert.rename(columns={"acct": "from_acct"})

train_data = acct_features.merge(df_alert, on="from_acct", how="left")

# train_data["label"] = train_data["label"].fillna(0).astype(int)

print(train_data.head())

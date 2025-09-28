import pandas as pd

# 建立小範例 DataFrame
df_txn = pd.DataFrame(
    {
        "from_acct": ["A", "A", "B", "C"],
        "to_acct": ["X", "Y", "Z", "X"],
        "txn_amt": [100, 200, 300, 150],
    }
)

df_alert = pd.DataFrame({"acct": ["A", "C"], "event_date": [10, 15], "label": [1, 1]})

# Step 1: groupby，彙總交易
acct_features = (
    df_txn.groupby("from_acct")
    .agg(
        txn_count=("txn_amt", "count"),
        txn_sum=("txn_amt", "sum"),
        txn_mean=("txn_amt", "mean"),
    )
    .reset_index()
)

# Step 2: 將 acct_alert 的 acct 改名成 from_acct
df_alert = df_alert.rename(columns={"acct": "from_acct"})

# Step 3: merge
train_data = acct_features.merge(df_alert, on="from_acct", how="left")

# Step 4: 填補 NaN
train_data["label"] = train_data["event_date"].notna().astype(int)

print(train_data)

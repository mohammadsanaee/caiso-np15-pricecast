from __future__ import annotations

import pandas as pd
from pathlib import Path
import numpy as np


def mae(y, yhat): 
    return float(np.mean(np.abs(y - yhat)))

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def main():
    feat_path = Path("data/clean/features_np15.parquet")
    df = pd.read_parquet(feat_path).sort_values("ts").reset_index(drop=True)

    # Baseline: predict y(t) = lag_24
    df["yhat_naive24"] = df["lag_24"]
    df["yhat_naive168"] = df["lag_168"]

    # Use a simple time-based split: last 20% as test
    split = int(len(df) * 0.8)
    test = df.iloc[split:].copy()

    y = test["y"].to_numpy()
    m1 = test["yhat_naive24"].to_numpy()
    m2 = test["yhat_naive168"].to_numpy()

    print("Test period:", test["ts"].min(), "→", test["ts"].max())
    print("Naive-24  MAE:", mae(y, m1), "RMSE:", rmse(y, m1))
    print("Naive-168 MAE:", mae(y, m2), "RMSE:", rmse(y, m2))


if __name__ == "__main__":
    main()
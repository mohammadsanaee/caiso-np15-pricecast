from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import joblib


FEATURE_PATH = Path("data/clean/features_np15.parquet")
ARTIFACT_DIR = Path("artifacts")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    df = pd.read_parquet(FEATURE_PATH).sort_values("ts").reset_index(drop=True)

    target = "y"
    feature_cols = [c for c in df.columns if c not in {"ts", "node", "market", target}]

    # ---- Walk-forward setup ----
    # Train on first 60% then predict next 10%, then roll forward by 10%
    n = len(df)
    train_start = 0
    train_end = int(n * 0.6)
    step = int(n * 0.1)

    params = {
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    fold_metrics = []
    all_preds = []

    fold = 0
    while train_end + step <= n:
        fold += 1
        train = df.iloc[train_start:train_end]
        test = df.iloc[train_end:train_end + step]

        X_train, y_train = train[feature_cols], train[target]
        X_test, y_test = test[feature_cols], test[target]

        # Baseline: naive24
        yhat_naive24 = test["lag_24"].to_numpy()

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)

        m = {
            "fold": fold,
            "train_start": str(train["ts"].min()),
            "train_end": str(train["ts"].max()),
            "test_start": str(test["ts"].min()),
            "test_end": str(test["ts"].max()),
            "rmse": rmse(y_test, yhat),
            "mae": float(mean_absolute_error(y_test, yhat)),
            "baseline_naive24_rmse": rmse(y_test, yhat_naive24),
            "baseline_naive24_mae": float(mean_absolute_error(y_test, yhat_naive24)),
        }
        fold_metrics.append(m)

        pred_df = pd.DataFrame({
            "ts": test["ts"].values,
            "y": y_test.values,
            "yhat": yhat,
            "yhat_naive24": yhat_naive24,
            "fold": fold,
        })
        all_preds.append(pred_df)

        # roll forward: expand training window
        train_end += step

    metrics_df = pd.DataFrame(fold_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True)

    print("\n=== WALK-FORWARD SUMMARY ===")
    print("Folds:", len(metrics_df))
    print("Model RMSE (mean):", metrics_df["rmse"].mean())
    print("Baseline RMSE (mean):", metrics_df["baseline_naive24_rmse"].mean())
    print("Model MAE  (mean):", metrics_df["mae"].mean())
    print("Baseline MAE (mean):", metrics_df["baseline_naive24_mae"].mean())

    # ---- Train final model on ALL data (deployable artifact) ----
    final_model = LGBMRegressor(**params)
    final_model.fit(df[feature_cols], df[target])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, ARTIFACT_DIR / "model.pkl")
    (ARTIFACT_DIR / "feature_columns.txt").write_text("\n".join(feature_cols))

    metadata = {
        "node": str(df["node"].iloc[0]) if "node" in df.columns else "TH_NP15_GEN-APND",
        "market": str(df["market"].iloc[0]) if "market" in df.columns else "DAM",
        "trained_rows": int(len(df)),
        "train_start": str(df["ts"].min()),
        "train_end": str(df["ts"].max()),
        "params": params,
        "walkforward_mean_rmse": float(metrics_df["rmse"].mean()),
        "walkforward_mean_mae": float(metrics_df["mae"].mean()),
        "walkforward_mean_baseline_rmse": float(metrics_df["baseline_naive24_rmse"].mean()),
        "walkforward_mean_baseline_mae": float(metrics_df["baseline_naive24_mae"].mean()),
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Save metrics + preds for plots/report
    metrics_df.to_csv(ARTIFACT_DIR / "walkforward_metrics.csv", index=False)
    preds_df.to_csv(ARTIFACT_DIR / "walkforward_predictions.csv", index=False)

    print("\nSaved deployable artifacts to ./artifacts/")
    print("- artifacts/model.pkl")
    print("- artifacts/feature_columns.txt")
    print("- artifacts/metadata.json")
    print("- artifacts/walkforward_metrics.csv")
    print("- artifacts/walkforward_predictions.csv")


if __name__ == "__main__":
    main()
    
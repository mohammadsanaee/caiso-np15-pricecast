from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

import mlflow
import mlflow.sklearn


FEATURE_PATH = Path("data/clean/features_np15.parquet")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    df = pd.read_parquet(FEATURE_PATH).sort_values("ts").reset_index(drop=True)

    target = "y"
    feature_cols = [c for c in df.columns if c not in {"ts", "node", "market", target}]

    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    X_train, y_train = train[feature_cols], train[target]
    X_test, y_test = test[feature_cols], test[target]

    # Baseline: naive 24 (already in features)
    yhat_naive24 = test["lag_24"].to_numpy()
    baseline_rmse = rmse(y_test, yhat_naive24)
    baseline_mae = float(mean_absolute_error(y_test, yhat_naive24))

    params = {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    model = LGBMRegressor(**params)

    mlflow.set_experiment("caiso_np15_dam_lmp")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("features_count", len(feature_cols))
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("test_rows", len(test))
        mlflow.log_param("test_start", str(test["ts"].min()))
        mlflow.log_param("test_end", str(test["ts"].max()))

        model.fit(X_train, y_train)

        yhat = model.predict(X_test)

        metrics = {
            "rmse": rmse(y_test, yhat),
            "mae": float(mean_absolute_error(y_test, yhat)),
            "baseline_naive24_rmse": baseline_rmse,
            "baseline_naive24_mae": baseline_mae,
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, artifact_path="model")
        # Save feature list as an artifact (critical for reproducible inference)
        feat_list_path = Path("data/clean/feature_columns.txt")
        feat_list_path.write_text("\n".join(feature_cols))
        mlflow.log_artifact(str(feat_list_path))

        print("Test period:", test["ts"].min(), "to", test["ts"].max())
        print("Baseline naive24  RMSE:", baseline_rmse, "MAE:", baseline_mae)
        print("LightGBM         RMSE:", metrics["rmse"], "MAE:", metrics["mae"])
        print("Saved feature list to", feat_list_path)


if __name__ == "__main__":
    main()
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


RAW_GLOB = "data/clean/lmp_dam_*.parquet"          # output from downloader
FEAT_PATH = Path("data/clean/features_np15.parquet")  # output from build_features script


def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    # -------------------------
    # 1) Load clean LMP series
    # -------------------------
    # Pick the latest clean parquet file
    clean_files = sorted(Path("data/clean").glob("lmp_dam_*.parquet"))
    if not clean_files:
        raise FileNotFoundError("No clean parquet found. Run downloader first.")
    clean_path = clean_files[-1]
    df = pd.read_parquet(clean_path).copy()

    df["ts"] = pd.to_datetime(df["interval_start_utc"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    print("\n=== CLEAN DATASET ===")
    print("File:", clean_path)
    print("Rows:", len(df))
    print("Time range:", df["ts"].min(), "→", df["ts"].max())
    print("Unique nodes:", df["node"].nunique())
    print("Unique markets:", df["market"].nunique(), "| examples:", df["market"].astype(str).unique()[:10])
    print(df.head(5))

    # Basic stats
    y = df["lmp_usd_per_mwh"]
    print("\nLMP stats:")
    print(y.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
    print("Negative price share:", float((y < 0).mean()))

    # Hourly continuity check (important for lag features)
    # We don't require perfect continuity, but we should know the gaps.
    diffs = df["ts"].diff().dropna()
    gap_hours = diffs.dt.total_seconds() / 3600.0
    gaps = (gap_hours > 1.01).sum()
    print("\nTime gaps (>1 hour):", int(gaps))
    if gaps:
        print("Largest gap (hours):", float(gap_hours.max()))

    # -------------------------
    # 2) Time series plot
    # -------------------------
    plt.figure()
    plt.plot(df["ts"], y)
    plt.title("NP15 DAM LMP over time")
    plt.xlabel("Time (UTC)")
    plt.ylabel("LMP (USD/MWh)")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # 3) Distribution plot
    # -------------------------
    plt.figure()
    plt.hist(y.dropna(), bins=60)
    plt.title("Distribution of NP15 DAM LMP")
    plt.xlabel("LMP (USD/MWh)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # 4) Seasonality checks
    # -------------------------
    tmp = df.copy()
    tmp["hour"] = tmp["ts"].dt.hour
    tmp["dow"] = tmp["ts"].dt.dayofweek

    hourly = tmp.groupby("hour")["lmp_usd_per_mwh"].median()
    plt.figure()
    plt.plot(hourly.index, hourly.values)
    plt.title("Median LMP by hour of day (UTC)")
    plt.xlabel("Hour")
    plt.ylabel("Median LMP (USD/MWh)")
    plt.tight_layout()
    plt.show()

    dow = tmp.groupby("dow")["lmp_usd_per_mwh"].median()
    plt.figure()
    plt.plot(dow.index, dow.values)
    plt.title("Median LMP by day of week (0=Mon)")
    plt.xlabel("Day of week")
    plt.ylabel("Median LMP (USD/MWh)")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # 5) If features exist, inspect them + baseline check
    # -------------------------
    if FEAT_PATH.exists():
        feat = pd.read_parquet(FEAT_PATH).sort_values("ts").reset_index(drop=True)
        print("\n=== FEATURES DATASET ===")
        print("File:", FEAT_PATH)
        print("Rows:", len(feat))
        print("Time range:", feat["ts"].min(), "→", feat["ts"].max())
        print("Columns:", list(feat.columns))
        print(feat.head(5))

        # Quick correlation sanity (just a few key lags)
        corr_cols = ["y", "lag_1", "lag_24", "lag_168", "roll_mean_24", "roll_mean_168"]
        corr_cols = [c for c in corr_cols if c in feat.columns]
        print("\nCorrelations (sanity):")
        print(feat[corr_cols].corr(numeric_only=True))

        # Baseline quality on the same split as earlier (last 20% test)
        split = int(len(feat) * 0.8)
        test = feat.iloc[split:].copy()
        y_true = test["y"].to_numpy()
        yhat_24 = test["lag_24"].to_numpy()
        yhat_168 = test["lag_168"].to_numpy()

        print("\nBaseline on feature table (last 20%):")
        print("Naive-24  RMSE:", rmse(y_true, yhat_24))
        print("Naive-168 RMSE:", rmse(y_true, yhat_168))

        # Plot a small window of true vs naive-24
        w = min(24 * 7, len(test))  # 1 week
        view = test.iloc[:w]
        plt.figure()
        plt.plot(view["ts"], view["y"], label="Actual")
        plt.plot(view["ts"], view["lag_24"], label="Naive-24")
        plt.title("Actual vs Naive-24 (first week of test)")
        plt.xlabel("Time (UTC)")
        plt.ylabel("LMP (USD/MWh)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        print("\n(features_np15.parquet not found yet — run build_features_np15.py first)")

    print("\nEDA done.")


if __name__ == "__main__":
    main()
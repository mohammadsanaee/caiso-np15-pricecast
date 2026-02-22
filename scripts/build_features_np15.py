from __future__ import annotations

from pathlib import Path
import glob
import pandas as pd


NODE = "TH_NP15_GEN-APND"
MARKET = "DAM"


def load_latest_clean_parquet() -> pd.DataFrame:
    files = sorted(glob.glob("data/clean/lmp_dam_*.parquet"))
    if not files:
        raise FileNotFoundError("No clean parquet found under data/clean/. Run the downloader first.")
    return pd.read_parquet(files[-1])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[(df["node"] == NODE) & (df["market"].astype(str).str.upper() == MARKET)].copy()
    df = df.sort_values("interval_start_utc").reset_index(drop=True)

    # Ensure hourly index continuity is not required, but helpful to know
    df["ts"] = pd.to_datetime(df["interval_start_utc"], utc=True)

    # Calendar features
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek  # 0=Mon
    df["month"] = df["ts"].dt.month

    # Target
    df["y"] = df["lmp_usd_per_mwh"]

    # Lag features (hours)
    for lag in [1, 24, 48, 72, 168]:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # Rolling features (use past only)
    df["roll_mean_24"] = df["y"].shift(1).rolling(24).mean()
    df["roll_std_24"] = df["y"].shift(1).rolling(24).std()
    df["roll_mean_168"] = df["y"].shift(1).rolling(168).mean()
    df["roll_std_168"] = df["y"].shift(1).rolling(168).std()

    # Drop rows where features aren't available yet
    feature_cols = [
        "hour","dow","month",
        "lag_1","lag_24","lag_48","lag_72","lag_168",
        "roll_mean_24","roll_std_24","roll_mean_168","roll_std_168",
    ]
    out = df[["ts", "node", "market", "y"] + feature_cols].dropna().reset_index(drop=True)
    return out


def main():
    df = load_latest_clean_parquet()
    feat = build_features(df)

    Path("data/clean").mkdir(parents=True, exist_ok=True)
    out_path = Path("data/clean/features_np15.parquet")
    feat.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print("Rows:", len(feat), "| Columns:", list(feat.columns))
    print(feat.head(5))


if __name__ == "__main__":
    main()
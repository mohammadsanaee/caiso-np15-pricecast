from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ARTIFACT_DIR = Path("artifacts")
NODE = "TH_NP15_GEN-APND"
MARKET = "DAM"


def load_latest_clean() -> pd.DataFrame:
    files = sorted(Path("data/clean").glob("lmp_dam_*.parquet"))
    if not files:
        raise FileNotFoundError("No clean parquet found in data/clean/. Run downloader first.")

    df = pd.read_parquet(files[-1]).copy()
    df["ts"] = pd.to_datetime(df["interval_start_utc"], utc=True)
    df = df[(df["node"] == NODE) & (df["market"].astype(str).str.upper() == MARKET)].copy()
    df = df.sort_values("ts").reset_index(drop=True)

    # Hourly grid within available range
    df = df.set_index("ts").sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full_idx)

    # Refill static columns + gap-fill prices
    df["node"] = NODE
    df["market"] = MARKET
    df["lmp_usd_per_mwh"] = df["lmp_usd_per_mwh"].ffill().bfill()

    df = df.reset_index().rename(columns={"index": "ts"})
    return df


def build_feature_row_from_series(
    s: pd.Series, ts_future_utc: pd.Timestamp
) -> dict:
    """
    Build features for ts_future_utc using a price series s indexed by UTC timestamps.
    This works for recursive forecasting because s can include predicted values
    for earlier future hours.
    """
    ts_future_utc = pd.Timestamp(ts_future_utc)
    if ts_future_utc.tzinfo is None:
        ts_future_utc = ts_future_utc.tz_localize("UTC")
    else:
        ts_future_utc = ts_future_utc.tz_convert("UTC")

    end_ts = ts_future_utc - pd.Timedelta(hours=1)

    # Need enough history for 168h window ending at t-1
    need_start = end_ts - pd.Timedelta(hours=167)
    if s.index.min() > need_start or end_ts not in s.index:
        raise ValueError(
            f"Not enough history to build features for {ts_future_utc}. "
            f"Need 168h ending at {end_ts}. "
            f"Series range is {s.index.min()} → {s.index.max()}."
        )

    def y_at(ts: pd.Timestamp) -> float:
        ts = pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).tz_localize("UTC")
        return float(s.loc[ts])

    # Lags
    lag_1 = y_at(ts_future_utc - pd.Timedelta(hours=1))
    lag_24 = y_at(ts_future_utc - pd.Timedelta(hours=24))
    lag_48 = y_at(ts_future_utc - pd.Timedelta(hours=48))
    lag_72 = y_at(ts_future_utc - pd.Timedelta(hours=72))
    lag_168 = y_at(ts_future_utc - pd.Timedelta(hours=168))

    window_24 = pd.date_range(end=end_ts, periods=24, freq="h", tz="UTC")
    window_168 = pd.date_range(end=end_ts, periods=168, freq="h", tz="UTC")

    vals_24 = s.loc[window_24].to_numpy(dtype=float)
    vals_168 = s.loc[window_168].to_numpy(dtype=float)

    feats = {
        "hour": int(ts_future_utc.hour),
        "dow": int(ts_future_utc.dayofweek),
        "month": int(ts_future_utc.month),
        "lag_1": lag_1,
        "lag_24": lag_24,
        "lag_48": lag_48,
        "lag_72": lag_72,
        "lag_168": lag_168,
        "roll_mean_24": float(np.mean(vals_24)),
        "roll_std_24": float(np.std(vals_24, ddof=1)),
        "roll_mean_168": float(np.mean(vals_168)),
        "roll_std_168": float(np.std(vals_168, ddof=1)),
    }
    return feats


def main():
    model_path = ARTIFACT_DIR / "model.pkl"
    feat_path = ARTIFACT_DIR / "feature_columns.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Run walkforward_train_np15.py first.")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}. Run walkforward_train_np15.py first.")

    model = joblib.load(model_path)
    feature_cols = feat_path.read_text().splitlines()

    hist = load_latest_clean()

    # Build a working series we can extend with predictions
    hist = hist.sort_values("ts")
    price_series = pd.Series(hist["lmp_usd_per_mwh"].to_numpy(), index=pd.DatetimeIndex(hist["ts"]))

    last_ts = price_series.index.max()
    future_hours = list(pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC"))

    out = []

    # Recursive rollout: predict one hour, append prediction, move to next hour
    for ts_future in future_hours:
        feats = build_feature_row_from_series(price_series, ts_future)
        X = pd.DataFrame([feats])[feature_cols]
        pred = float(model.predict(X)[0])

        # Append prediction so subsequent steps can use lag_1 / rolling windows
        price_series.loc[ts_future] = pred

        out.append(
            {
                "ts_utc": ts_future.isoformat(),
                "node": NODE,
                "market": MARKET,
                "lmp_pred_usd_per_mwh": pred,
            }
        )

    out_path = ARTIFACT_DIR / "next24_forecast.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"Last observed hour: {last_ts}")
    print(f"Saved forecast: {out_path}")
    print("First 3 rows:")
    print(json.dumps(out[:3], indent=2))


if __name__ == "__main__":
    main()
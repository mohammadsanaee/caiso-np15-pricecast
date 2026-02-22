from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


ARTIFACT_DIR = Path("artifacts")
NODE = "TH_NP15_GEN-APND"
MARKET = "DAM"

MODEL_PATH = ARTIFACT_DIR / "model.pkl"
FEAT_PATH = ARTIFACT_DIR / "feature_columns.txt"
META_PATH = ARTIFACT_DIR / "metadata.json"

DATA_DIR = Path("data/clean")


app = FastAPI(title="CAISO NP15 DAM LMP Forecast API", version="0.1.0")


def load_latest_clean() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("lmp_dam_*.parquet"))
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

    df["node"] = NODE
    df["market"] = MARKET
    df["lmp_usd_per_mwh"] = df["lmp_usd_per_mwh"].ffill().bfill()

    df = df.reset_index().rename(columns={"index": "ts"})
    return df


def build_feature_row_from_series(s: pd.Series, ts_future_utc: pd.Timestamp) -> dict:
    ts_future_utc = pd.Timestamp(ts_future_utc)
    if ts_future_utc.tzinfo is None:
        ts_future_utc = ts_future_utc.tz_localize("UTC")
    else:
        ts_future_utc = ts_future_utc.tz_convert("UTC")

    end_ts = ts_future_utc - pd.Timedelta(hours=1)
    need_start = end_ts - pd.Timedelta(hours=167)

    if s.index.min() > need_start or end_ts not in s.index:
        raise ValueError(
            f"Not enough history to build features for {ts_future_utc}. "
            f"Need 168h ending at {end_ts}. "
            f"Series range is {s.index.min()} → {s.index.max()}."
        )

    def y_at(ts: pd.Timestamp) -> float:
        ts = pd.Timestamp(ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return float(s.loc[ts])

    lag_1 = y_at(ts_future_utc - pd.Timedelta(hours=1))
    lag_24 = y_at(ts_future_utc - pd.Timedelta(hours=24))
    lag_48 = y_at(ts_future_utc - pd.Timedelta(hours=48))
    lag_72 = y_at(ts_future_utc - pd.Timedelta(hours=72))
    lag_168 = y_at(ts_future_utc - pd.Timedelta(hours=168))

    window_24 = pd.date_range(end=end_ts, periods=24, freq="h", tz="UTC")
    window_168 = pd.date_range(end=end_ts, periods=168, freq="h", tz="UTC")

    vals_24 = s.loc[window_24].to_numpy(dtype=float)
    vals_168 = s.loc[window_168].to_numpy(dtype=float)

    return {
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


def forecast_next_hours(model, feature_cols, hours: int = 24):
    hist = load_latest_clean().sort_values("ts")
    s = pd.Series(hist["lmp_usd_per_mwh"].to_numpy(), index=pd.DatetimeIndex(hist["ts"]))

    last_ts = s.index.max()
    future_hours = list(pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=hours, freq="h", tz="UTC"))

    out = []
    for ts_future in future_hours:
        feats = build_feature_row_from_series(s, ts_future)
        X = pd.DataFrame([feats])[feature_cols]
        pred = float(model.predict(X)[0])
        s.loc[ts_future] = pred  # recursive
        out.append(
            {
                "ts_utc": ts_future.isoformat(),
                "node": NODE,
                "market": MARKET,
                "lmp_pred_usd_per_mwh": pred,
            }
        )
    return out


class ForecastRequest(BaseModel):
    hours: int = 24


@app.on_event("startup")
def _startup():
    if not MODEL_PATH.exists() or not FEAT_PATH.exists():
        raise RuntimeError("Missing artifacts. Run training first to create artifacts/model.pkl and feature_columns.txt")

    app.state.model = joblib.load(MODEL_PATH)
    app.state.feature_cols = FEAT_PATH.read_text().splitlines()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {"node": NODE, "market": MARKET, "note": "metadata.json not found"}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    if req.hours < 1 or req.hours > 168:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 168")

    try:
        preds = forecast_next_hours(app.state.model, app.state.feature_cols, hours=req.hours)
        return {"node": NODE, "market": MARKET, "hours": req.hours, "forecast": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
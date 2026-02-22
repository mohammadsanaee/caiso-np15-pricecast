from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx
import pytest
from pathlib import Path
import pandas as pd
import numpy as np


BASE_URL = "http://127.0.0.1:8000"

def _ensure_test_parquet():
    """
    CI doesn't have real data/clean files (we don't commit them).
    Create a small synthetic parquet file that matches the schema the API expects.
    Must contain at least 168+ hours of history.
    """
    out_dir = Path("data/clean")
    out_dir.mkdir(parents=True, exist_ok=True)

    # If a parquet already exists, do nothing
    if list(out_dir.glob("lmp_dam_*.parquet")):
        return

    end = pd.Timestamp("2026-02-22T03:00:00Z")
    n_hours = 24 * 14  # 336 hours (2 weeks) -> enough for 168h lags + rolls
    ts = pd.date_range(end=end, periods=n_hours, freq="h", tz="UTC")

    # Simple synthetic price pattern
    rng = np.random.default_rng(42)
    base = 30 + 5 * np.sin(np.arange(n_hours) * 2 * np.pi / 24)  # daily seasonality
    noise = rng.normal(0, 1.0, size=n_hours)
    lmp = base + noise

    df = pd.DataFrame(
        {
            "interval_start_utc": ts.astype("datetime64[ns, UTC]"),
            "node": "TH_NP15_GEN-APND",
            "market": "DAM",
            "lmp_usd_per_mwh": lmp.astype(float),
        }
    )

    out_path = out_dir / "lmp_dam_ci_test.parquet"
    df.to_parquet(out_path, index=False)

def _wait_for_server(timeout_s: int = 20) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("API did not become ready in time")


@pytest.fixture(scope="session")
def api_server():
    _ensure_test_parquet()
    """
    Start uvicorn in a subprocess for tests, then tear down.
    """
    env = os.environ.copy()
    # Ensure we use the current python environment
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    try:
        _wait_for_server(timeout_s=25)
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_health(api_server):
    r = httpx.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_forecast_24(api_server):
    r = httpx.post(f"{BASE_URL}/forecast", json={"hours": 24})
    assert r.status_code == 200
    body = r.json()
    assert body["node"] == "TH_NP15_GEN-APND"
    assert body["market"] == "DAM"
    assert body["hours"] == 24
    assert isinstance(body["forecast"], list)
    assert len(body["forecast"]) == 24

    first = body["forecast"][0]
    assert "ts_utc" in first
    assert "lmp_pred_usd_per_mwh" in first


def test_forecast_bounds(api_server):
    r = httpx.post(f"{BASE_URL}/forecast", json={"hours": 0})
    assert r.status_code == 400
    r = httpx.post(f"{BASE_URL}/forecast", json={"hours": 999})
    assert r.status_code == 400
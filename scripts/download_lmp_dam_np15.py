from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


OASIS_SINGLEZIP = "https://oasis.caiso.com/oasisapi/SingleZip"


@dataclass(frozen=True)
class OasisQuery:
    queryname: str = "PRC_LMP"
    market_run_id: str = "DAM"
    node: str = "TH_NP15_GEN-APND"
    version: str = "1"
    resultformat: str = "6"  # CSV format (otherwise default is XML)
    # startdatetime / enddatetime will be set per request


def to_oasis_dt(dt_utc: datetime) -> str:
    """
    Format per OASIS examples: YYYYMMDDTHH:MM-0000 (UTC offset).
    """
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware UTC datetime")
    dt_utc = dt_utc.astimezone(timezone.utc)
    return dt_utc.strftime("%Y%m%dT%H:%M-0000")


def fetch_singlezip_csv(params: dict) -> pd.DataFrame:
    """
    Downloads the zip from OASIS and returns the first CSV as a DataFrame.
    """
    r = requests.get(OASIS_SINGLEZIP, params=params, timeout=60)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]

    if not csv_names:
        raise RuntimeError(f"No CSV found in zip. Files: {z.namelist()[:10]}")

    # Usually one CSV per request for PRC_LMP when using node=
    with z.open(csv_names[0]) as f:
        df = pd.read_csv(f)

    return df


def normalize_lmp_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.upper(): c for c in df.columns}

    def pick(*candidates: str) -> str:
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        raise KeyError(f"Missing expected column. Tried {candidates}. Have {list(df.columns)[:30]}")

    node_col = pick("NODE", "APNODE", "PNODE")

    # timestamp (you DO have INTERVALSTARTTIME_GMT)
    t_col = pick("INTERVALSTARTTIME_GMT", "INTERVAL_START_GMT")
    ts = pd.to_datetime(df[t_col], utc=True)

    # In your file, the numeric value is in MW; LMP_TYPE tells what it is
    value_col = pick("MW")
    lmp_type_col = pick("LMP_TYPE")

    # Keep total LMP only (not ENERGY/CONG/LOSS components)
    dff = df[df[lmp_type_col].astype(str).str.upper().eq("LMP")].copy()

    out = pd.DataFrame(
        {
            "interval_start_utc": ts.loc[dff.index],
            "node": dff[node_col].astype(str),
            "market": dff["MARKET_RUN_ID"].astype(str),
            "lmp_usd_per_mwh": pd.to_numeric(dff[value_col], errors="coerce"),
        }
    ).dropna(subset=["lmp_usd_per_mwh"])

    out = out.sort_values(["node", "interval_start_utc"]).reset_index(drop=True)
    return out


def main(
    start_utc: datetime,
    end_utc: datetime,
    out_dir: Path = Path("data"),
) -> None:
    out_raw = out_dir / "raw"
    out_clean = out_dir / "clean"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_clean.mkdir(parents=True, exist_ok=True)

    q = OasisQuery()
    params = {
        "queryname": q.queryname,
        "market_run_id": q.market_run_id,
        "node": q.node,                 # use node OR grp_type (recommended)
        "version": q.version,
        "resultformat": q.resultformat, # CSV format
        "startdatetime": to_oasis_dt(start_utc),
        "enddatetime": to_oasis_dt(end_utc),
    }

    df_raw = fetch_singlezip_csv(params)
    raw_path = out_raw / f"PRC_LMP_DAM_{q.node}_{start_utc:%Y%m%dT%H%M}_to_{end_utc:%Y%m%dT%H%M}.csv"
    df_raw.to_csv(raw_path, index=False)

    df = normalize_lmp_df(df_raw)
    clean_path = out_clean / f"lmp_dam_{q.node}_{start_utc:%Y%m%d}_to_{end_utc:%Y%m%d}.parquet"
    df.to_parquet(clean_path, index=False)

    print(f"Saved raw:   {raw_path}")
    print(f"Saved clean: {clean_path}")
    print(df.head(5))


if __name__ == "__main__":
    # MVP choice: pull last 30 days in UTC.
    # (We can refine to CAISO "operating day" boundaries later.)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30)
    main(start, end)
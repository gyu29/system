"""Sharadar (Nasdaq Data Link) bulk export client.

Tables used:
  TICKERS  security master (permaticker = stable ID, delisted names included)
  SEP      daily prices. open/high/low/close/volume are split-adjusted, closeadj is
           split+dividend adjusted, closeunadj is the raw traded price.
  SF1      fundamentals. AR* dimensions are as-first-reported; datekey is the SEC filing date.
  DAILY    daily marketcap/EV computed from the latest filing known on that date.
  ACTIONS  splits, dividends, delistings, ticker changes.
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path

import polars as pl
import requests

BASE_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
API_KEY_ENV = "NASDAQ_DATA_LINK_API_KEY"
TABLES = ("TICKERS", "SEP", "SF1", "DAILY", "ACTIONS")

DATE_COLS = {
    "TICKERS": ("firstadded", "firstpricedate", "lastpricedate", "firstquarter", "lastquarter", "lastupdated"),
    "SEP": ("date", "lastupdated"),
    "SF1": ("calendardate", "datekey", "reportperiod", "lastupdated"),
    "DAILY": ("date", "lastupdated"),
    "ACTIONS": ("date",),
}
STRING_COLS = {
    "TICKERS": (
        "table", "ticker", "name", "exchange", "isdelisted", "category", "cusips", "siccode",
        "sicsector", "sicindustry", "famasector", "famaindustry", "sector", "industry",
        "scalemarketcap", "scalerevenue", "relatedtickers", "currency", "location",
        "secfilings", "companysite",
    ),
    "SEP": ("ticker",),
    "SF1": ("ticker", "dimension", "fiscalperiod"),
    "DAILY": ("ticker",),
    "ACTIONS": ("action", "ticker", "name", "contraticker", "contraname"),
}
INT_COLS = {"permaticker"}


def api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"set {API_KEY_ENV} to your Nasdaq Data Link API key")
    return key


def normalize(df: pl.DataFrame, table: str) -> pl.DataFrame:
    """Cast an all-string vendor frame to proper types. Unknown columns must be numeric, or this raises."""
    dates, strings = set(DATE_COLS[table]), set(STRING_COLS[table])
    exprs = []
    for c in df.columns:
        col = pl.col(c).str.strip_chars()
        if c in dates:
            exprs.append(col.str.to_date("%Y-%m-%d", strict=True).alias(c))
        elif c in strings:
            exprs.append(pl.when(col == "").then(None).otherwise(col).alias(c))
        elif c in INT_COLS:
            exprs.append(col.cast(pl.Int64, strict=True).alias(c))
        else:
            exprs.append(pl.when(col == "").then(None).otherwise(col).cast(pl.Float64, strict=True).alias(c))
    return df.select(exprs)


def read_csv(source: bytes | Path, table: str) -> pl.DataFrame:
    df = pl.read_csv(source, infer_schema=False)
    return normalize(df, table)


def load_export(path: Path, table: str) -> pl.DataFrame:
    """Load a bulk export that was downloaded by hand (.zip or .csv)."""
    path = Path(path)
    if path.suffix == ".zip":
        return read_csv(_csv_from_zip(path.read_bytes()), table)
    return read_csv(path, table)


def _csv_from_zip(blob: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in export, found {names}")
        return zf.read(names[0])


def export_link(table: str, key: str, session: requests.Session, poll_seconds: float = 15, timeout: float = 1800) -> str:
    """Ask Nasdaq Data Link for a bulk export and wait until the file is ready."""
    url = f"{BASE_URL}/{table}.json"
    deadline = time.monotonic() + timeout
    while True:
        r = session.get(url, params={"qopts.export": "true", "api_key": key}, timeout=60)
        r.raise_for_status()
        file = r.json()["datatable_bulk_download"]["file"]
        if file["status"].lower() == "fresh" and file.get("link"):
            return file["link"]
        if time.monotonic() > deadline:
            raise TimeoutError(f"{table} export still {file['status']!r} after {timeout}s")
        time.sleep(poll_seconds)


def download_table(table: str, key: str | None = None, session: requests.Session | None = None) -> pl.DataFrame:
    if table not in TABLES:
        raise KeyError(f"unsupported Sharadar table {table!r}")
    key = key or api_key()
    session = session or requests.Session()
    link = export_link(table, key, session)
    r = session.get(link, timeout=1800)
    r.raise_for_status()
    return read_csv(_csv_from_zip(r.content), table)

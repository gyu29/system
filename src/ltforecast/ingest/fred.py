"""FRED / ALFRED client producing point-in-time macro observations."""

from __future__ import annotations

import datetime as dt
import os

import polars as pl
import requests

from ltforecast.config import MacroSeries

URL = "https://api.stlouisfed.org/fred/series/observations"
API_KEY_ENV = "FRED_API_KEY"

# An initial-release date this far after the observation means ALFRED has no real vintage for
# it (its history starts later), so the release date is unknown and we fall back to the lag rule.
MAX_PLAUSIBLE_RELEASE_DAYS = 120


def api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"set {API_KEY_ENV} to a free FRED API key")
    return key


def parse_observations(payload: dict, series: MacroSeries) -> pl.DataFrame:
    """FRED JSON -> macro_pit rows: series, observation_date, available_date, value, first_release."""
    obs = pl.DataFrame(
        payload["observations"],
        schema={"realtime_start": pl.String, "realtime_end": pl.String, "date": pl.String, "value": pl.String},
    )
    lag = pl.duration(days=series.release_lag_days)
    df = obs.filter(pl.col("value") != ".").select(
        pl.lit(series.id).alias("series"),
        pl.col("date").str.to_date().alias("observation_date"),
        pl.col("realtime_start").str.to_date().alias("release_date"),
        pl.col("value").cast(pl.Float64),
    )
    if series.vintage:
        has_vintage = (pl.col("release_date") - pl.col("observation_date")).dt.total_days() <= MAX_PLAUSIBLE_RELEASE_DAYS
        df = df.with_columns(
            pl.when(has_vintage)
            .then(pl.col("release_date"))
            .otherwise(pl.col("observation_date") + lag)
            .alias("available_date"),
            has_vintage.alias("first_release"),
        )
    else:
        # Market series are never revised, so the latest value is the value that was known.
        df = df.with_columns(
            (pl.col("observation_date") + lag).alias("available_date"),
            pl.lit(True).alias("first_release"),
        )
    return df.select("series", "observation_date", "available_date", "value", "first_release").sort("observation_date")


def fetch_series(
    series: MacroSeries,
    start: dt.date,
    key: str | None = None,
    session: requests.Session | None = None,
) -> pl.DataFrame:
    key = key or api_key()
    session = session or requests.Session()
    params = {
        "series_id": series.id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start.isoformat(),
    }
    if series.vintage:
        # output_type=4: initial release only, realtime_start = the day it was first published
        params |= {"output_type": 4, "realtime_start": "1776-07-04", "realtime_end": "9999-12-31"}
    r = session.get(URL, params=params, timeout=60)
    r.raise_for_status()
    return parse_observations(r.json(), series)

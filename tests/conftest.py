"""Synthetic Sharadar/FRED-shaped data covering the cases the PIT layer must get right.

AAA  healthy; 2:1 split on 2020-03-02 (adjusted close 4 -> 6, unadjusted 8 -> 6); dividend 2019-06-14
BBB  unadjusted price $3: fails the price filter
CCC  bankrupt, last price 2020-06-30
DDD  acquired 2020-09-15 (generic 'delisted' + specific 'acquisitionby' actions)
EEE  an ETF: not common stock
FFF  $200M market cap: fails the size filter
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest
import yaml

from ltforecast.config import DEFAULT_CONFIG, load_config

START, END = dt.date(2019, 1, 1), dt.date(2020, 12, 31)
SPLIT_DATE = dt.date(2020, 3, 2)


def business_days(start=START, end=END) -> pl.Series:
    days = pl.date_range(start, end, "1d", eager=True)
    return days.filter(days.dt.weekday() <= 5).alias("date")


TICKERS = [
    # ticker, permaticker, category, isdelisted, lastpricedate, mcap ($M), close
    ("AAA", 101, "Domestic Common Stock", "N", END, 1000.0, None),
    ("BBB", 102, "Domestic Common Stock", "N", END, 500.0, 3.0),
    ("CCC", 103, "Domestic Common Stock", "Y", dt.date(2020, 6, 30), 400.0, 20.0),
    ("DDD", 104, "Domestic Common Stock Primary Class", "Y", dt.date(2020, 9, 15), 800.0, 30.0),
    ("EEE", 105, "ETF", "N", END, 5000.0, 100.0),
    ("FFF", 106, "Domestic Common Stock", "N", END, 200.0, 25.0),
]


@pytest.fixture
def raw_tickers() -> pl.DataFrame:
    rows = []
    for t, pid, cat, delisted, last, _, _ in TICKERS:
        rows.append(
            {
                "table": "SEP", "permaticker": pid, "ticker": t, "name": f"{t} Inc", "exchange": "NYSE",
                "isdelisted": delisted, "category": cat, "cusips": None, "siccode": "3571",
                "sicsector": None, "sicindustry": None, "famasector": None, "famaindustry": "Computers",
                "sector": "Technology", "industry": "Hardware", "scalemarketcap": None, "scalerevenue": None,
                "relatedtickers": None, "currency": "USD", "location": None,
                "secfilings": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={pid:010d}",
                "companysite": None, "firstadded": START, "firstpricedate": START, "lastpricedate": last,
                "firstquarter": START, "lastquarter": last, "lastupdated": END,
            }
        )
    # The same company also appears in the fundamentals table list; only table=SEP rows are used.
    rows.append({**rows[0], "table": "SF1"})
    return pl.DataFrame(rows)


@pytest.fixture
def raw_sep() -> pl.DataFrame:
    days = business_days()
    frames = []
    for t, _, _, _, last, _, close in TICKERS:
        d = days.filter(days <= last).to_frame()
        if t == "AAA":
            pre = pl.col("date") < SPLIT_DATE
            d = d.with_columns(
                pl.when(pre).then(4.0).otherwise(6.0).alias("close"),
                pl.when(pre).then(8.0).otherwise(6.0).alias("closeunadj"),
            )
        else:
            d = d.with_columns(pl.lit(close).alias("close"), pl.lit(close).alias("closeunadj"))
        frames.append(
            d.with_columns(
                pl.lit(t).alias("ticker"),
                pl.col("close").alias("open"),
                pl.col("close").alias("high"),
                pl.col("close").alias("low"),
                pl.lit(1_000_000.0).alias("volume"),
                pl.col("close").alias("closeadj"),
                pl.lit(END).alias("lastupdated"),
            )
        )
    return pl.concat(frames).select(
        "ticker", "date", "open", "high", "low", "close", "volume", "closeadj", "closeunadj", "lastupdated"
    )


@pytest.fixture
def raw_actions() -> pl.DataFrame:
    rows = [
        (dt.date(2019, 6, 14), "dividend", "AAA", 0.5, None),
        (SPLIT_DATE, "split", "AAA", 2.0, None),
        (dt.date(2020, 6, 30), "bankruptcyliquidation", "CCC", None, None),
        (dt.date(2020, 9, 15), "acquisitionby", "DDD", 35.0, "ZZZ"),
        (dt.date(2020, 9, 15), "delisted", "DDD", None, None),
    ]
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "action": pl.String, "ticker": pl.String, "value": pl.Float64, "contraticker": pl.String},
        orient="row",
    ).with_columns(pl.col("ticker").alias("name"), pl.lit(None, pl.String).alias("contraname"))


@pytest.fixture
def raw_daily(raw_sep) -> pl.DataFrame:
    mcaps = pl.DataFrame({"ticker": [t[0] for t in TICKERS], "marketcap": [t[5] for t in TICKERS]})
    return raw_sep.select("ticker", "date").join(mcaps, on="ticker").with_columns(pl.lit(END).alias("lastupdated"))


@pytest.fixture
def raw_sf1() -> pl.DataFrame:
    d = dt.date
    rows = [
        # ticker, dimension, reportperiod, datekey, revenue, sharesbas
        ("AAA", "ARQ", d(2019, 3, 31), d(2019, 5, 1), 100.0, 250.0),
        ("AAA", "ARQ", d(2019, 3, 31), d(2019, 6, 3), 110.0, 250.0),  # amendment: must be ignored
        ("AAA", "MRQ", d(2019, 3, 31), d(2019, 6, 3), 110.0, 250.0),  # restated dimension: excluded
        ("AAA", "ARQ", d(2019, 6, 30), d(2019, 8, 2), 120.0, None),   # filed on a Friday
        ("CCC", "ARQ", d(2019, 12, 31), d(2020, 3, 10), 50.0, 20.0),
    ]
    return pl.DataFrame(
        rows,
        schema={"ticker": pl.String, "dimension": pl.String, "reportperiod": pl.Date, "datekey": pl.Date,
                "revenue": pl.Float64, "sharesbas": pl.Float64},
        orient="row",
    ).with_columns(
        pl.col("reportperiod").alias("calendardate"),
        pl.lit(END).alias("lastupdated"),
        pl.lit("2019-Q1").alias("fiscalperiod"),
    )


def cpi_payload(start=dt.date(2017, 1, 1), end=dt.date(2020, 12, 1), stale_before=dt.date(2018, 1, 1)) -> dict:
    """CPI rising 0.2/month, first released ~45 days after the observation date.
    Observations before `stale_before` carry a 2018-06-01 'first vintage' (ALFRED gap)."""
    obs, level = [], 250.0
    month = start
    while month <= end:
        release = month + dt.timedelta(days=45) if month >= stale_before else dt.date(2018, 6, 1)
        obs.append({"realtime_start": release.isoformat(), "realtime_end": "9999-12-31",
                    "date": month.isoformat(), "value": f"{level:.1f}"})
        level += 0.2
        month = (month.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return {"observations": obs}


@pytest.fixture
def cfg(tmp_path: Path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["data_dir"] = str(tmp_path / "data")
    raw["dates"]["history_start"] = START
    raw["universe"]["cpi_base_date"] = dt.date(2020, 12, 31)
    path = tmp_path / "config" / "universe.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


@pytest.fixture
def macro(cfg) -> pl.DataFrame:
    from ltforecast.ingest.fred import parse_observations

    cpi = next(s for s in cfg.macro if s.id == "CPIAUCSL")
    return parse_observations(cpi_payload(), cpi)

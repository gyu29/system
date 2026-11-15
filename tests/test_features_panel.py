"""Point-in-time behaviour of features/panel.py on hand-built filings (plan 1.3).

The trading calendar is every weekday of 2018-2020. Filings are usable from the next trading day.
Useful weekdays: 2019-08-02 and 2019-08-30 are Fridays (08-30 is also a month end), 2019-07-31 is a
Wednesday month end.
"""

import calendar as pycal
import datetime as dt
import random
import sys

import polars as pl
import pytest
from conftest import business_days
from feature_helpers import (
    d,
    make_benchmark,
    make_ctx,
    make_filings,
    make_fundamentals,
    make_macro,
    make_marketcap,
    make_prices,
    make_securities,
    make_universe,
    period_ends,
)

from ltforecast.features import panel
from ltforecast.features.panel import (
    SCHEMAS,
    context_from_frames,
    fundamentals_asof,
    make_context,
    market_cap_at,
    month_end_prices,
    months_back,
    sector_map,
)
from ltforecast.pit.asof import LookaheadError, assert_no_lookahead
from ltforecast.store import Store
from ltforecast.universe import month_end_dates

CAL = business_days(d(2018, 1, 1), d(2020, 12, 31))


def asof(rows, entries, dimension="ARQ", metrics=("revenue",), lags=(0,), **ctx_kw) -> list[dict]:
    """fundamentals_asof on a context holding only these filings; rows sorted by security, date."""
    ctx = make_ctx(fundamentals=rows, entries=entries, calendar=CAL, **ctx_kw)
    return fundamentals_asof(ctx, dimension, list(metrics), lags).to_dicts()


OLD = (1, "ARQ", d(2019, 3, 31), d(2019, 5, 1), {"revenue": 100.0})


@pytest.mark.parametrize(
    ("filed", "still_old", "first_new"),
    [
        (d(2019, 7, 31), d(2019, 7, 31), d(2019, 8, 1)),   # filed on the prediction date itself
        (d(2019, 8, 30), d(2019, 8, 30), d(2019, 9, 2)),   # filed on a friday month end -> monday
        (d(2019, 8, 2), d(2019, 8, 2), d(2019, 8, 5)),     # friday -> monday
        (d(2019, 8, 3), d(2019, 8, 2), d(2019, 8, 5)),     # saturday -> monday
        (d(2019, 8, 6), d(2019, 8, 6), d(2019, 8, 7)),     # tuesday -> wednesday
    ],
)
def test_filing_is_visible_only_from_the_next_trading_day(filed, still_old, first_new):
    new = (1, "ARQ", d(2019, 6, 30), filed, {"revenue": 120.0})
    before, after = asof([OLD, new], [(1, still_old), (1, first_new)])
    assert before == {"security_id": 1, "date": still_old, "fiscal_period_end": d(2019, 3, 31),
                      "src_date": d(2019, 5, 1), "revenue": 100.0}
    assert after == {"security_id": 1, "date": first_new, "fiscal_period_end": d(2019, 6, 30),
                     "src_date": filed, "revenue": 120.0}


def test_fundamental_lag_is_counted_in_trading_days():
    new = (1, "ARQ", d(2019, 6, 30), d(2019, 8, 2), {"revenue": 120.0})  # a friday
    days = [(1, d(2019, 8, 2)), (1, d(2019, 8, 5)), (1, d(2019, 8, 6))]
    assert [r["revenue"] for r in asof([OLD, new], days, fundamental_lag=0)] == [120.0, 120.0, 120.0]
    assert [r["revenue"] for r in asof([OLD, new], days, fundamental_lag=1)] == [100.0, 120.0, 120.0]
    assert [r["revenue"] for r in asof([OLD, new], days, fundamental_lag=2)] == [100.0, 100.0, 120.0]


def test_nothing_before_the_first_filing_and_nothing_for_other_securities():
    out = asof([OLD], [(1, d(2019, 4, 30)), (1, d(2019, 5, 1)), (1, d(2019, 5, 2)), (2, d(2019, 5, 31))])
    assert [(r["security_id"], r["revenue"], r["src_date"]) for r in out] == [
        (1, None, None), (1, None, None), (1, 100.0, d(2019, 5, 1)), (2, None, None),
    ]


def test_metric_missing_from_the_latest_filing_is_null_not_carried_forward():
    rows = [
        (1, "ARQ", d(2019, 9, 30), d(2019, 11, 8), {"revenue": 100.0, "debt": 50.0}),
        (1, "ARQ", d(2019, 12, 31), d(2020, 2, 14), {"revenue": 120.0}),  # no debt in this filing
    ]
    (row,) = asof(rows, [(1, d(2020, 2, 28))], metrics=("revenue", "debt"), lags=(0, 1))
    assert row == {"security_id": 1, "date": d(2020, 2, 28), "fiscal_period_end": d(2019, 12, 31),
                   "src_date": d(2020, 2, 14), "revenue": 120.0, "debt": None, "revenue_lag1": 100.0,
                   "debt_lag1": 50.0}


def test_amendment_that_adds_a_metric_is_not_backfilled_into_the_original_filing():
    rows = [
        (1, "ARQ", d(2019, 9, 30), d(2019, 11, 8), {"revenue": 100.0, "netinc": 7.0}),
        (1, "ARQ", d(2019, 12, 31), d(2020, 2, 14), {"revenue": 120.0}),   # original 10-K: no netinc
        (1, "ARQ", d(2019, 12, 31), d(2020, 3, 20), {"netinc": 9.0}),      # amendment supplies it
        (1, "ARQ", d(2020, 3, 31), d(2020, 5, 8), {"revenue": 130.0, "netinc": 11.0}),
    ]
    months = [d(2020, 1, 31), d(2020, 2, 28), d(2020, 3, 31), d(2020, 4, 30), d(2020, 5, 29)]
    out = asof(rows, [(1, m) for m in months], metrics=("revenue", "netinc"), lags=(0, 1))
    got = [(r["fiscal_period_end"], r["src_date"], r["revenue"], r["netinc"], r["revenue_lag1"], r["netinc_lag1"])
           for r in out]
    q3, q4, q1 = d(2019, 9, 30), d(2019, 12, 31), d(2020, 3, 31)
    assert got == [
        (q3, d(2019, 11, 8), 100.0, 7.0, None, None),
        (q4, d(2020, 2, 14), 120.0, None, 100.0, 7.0),
        (q4, d(2020, 2, 14), 120.0, None, 100.0, 7.0),   # after the amendment: still the original filing
        (q4, d(2020, 2, 14), 120.0, None, 100.0, 7.0),
        (q1, d(2020, 5, 8), 130.0, 11.0, 120.0, 9.0),    # the amended value is known by the next filing
    ]
    # the filing is dated by its first metric, whichever metrics are asked for: asking for netinc
    # alone must not make Q3's value linger until the amendment
    alone = asof(rows, [(1, m) for m in months], metrics=("netinc",))
    assert [(r["fiscal_period_end"], r["netinc"]) for r in alone] == [
        (q3, 7.0), (q4, None), (q4, None), (q4, None), (q1, 11.0)]


def test_late_filing_of_an_older_period_never_replaces_a_newer_one():
    rows = [
        (1, "ARQ", d(2018, 12, 31), d(2019, 2, 15), {"revenue": 100.0}),
        (1, "ARQ", d(2019, 6, 30), d(2019, 8, 9), {"revenue": 110.0}),    # Q2, on time
        (1, "ARQ", d(2019, 3, 31), d(2019, 9, 16), {"revenue": 105.0}),   # Q1, filed after Q2
        (1, "ARQ", d(2019, 9, 30), d(2019, 11, 8), {"revenue": 115.0}),
    ]
    months = [d(2019, 7, 31), d(2019, 8, 30), d(2019, 9, 30), d(2019, 11, 29)]
    out = asof(rows, [(1, m) for m in months], lags=(0, 1, 2))
    got = [(r["fiscal_period_end"], r["src_date"], r["revenue"], r["revenue_lag1"], r["revenue_lag2"]) for r in out]
    assert got == [
        (d(2018, 12, 31), d(2019, 2, 15), 100.0, None, None),
        # Q1 is not filed yet: one quarter back is unknown, and Q4 is correctly two quarters back
        (d(2019, 6, 30), d(2019, 8, 9), 110.0, None, 100.0),
        # Q1 arrives on 09-16: Q2 stays the latest period, as the market saw it on its filing date
        (d(2019, 6, 30), d(2019, 8, 9), 110.0, None, 100.0),
        (d(2019, 9, 30), d(2019, 11, 8), 115.0, 110.0, 105.0),
    ]
    assert all(r["src_date"] < r["date"] for r in out)


def test_periods_filed_on_the_same_day_resolve_to_the_newest_period():
    rows = [
        (1, "ARQ", d(2019, 3, 31), d(2019, 8, 9), {"revenue": 105.0}),
        (1, "ARQ", d(2019, 6, 30), d(2019, 8, 9), {"revenue": 110.0}),
    ]
    (row,) = asof(rows, [(1, d(2019, 8, 30))], lags=(0, 1))
    assert (row["fiscal_period_end"], row["revenue"], row["revenue_lag1"]) == (d(2019, 6, 30), 110.0, 105.0)


def test_skipped_quarter_nulls_only_the_lags_that_point_at_it():
    gap = [d(2018, 9, 30), d(2018, 12, 31), d(2019, 6, 30), d(2019, 9, 30), d(2019, 12, 31), d(2020, 3, 31)]
    full = period_ends(d(2020, 3, 31), 6)
    assert full == [d(2018, 12, 31), d(2019, 3, 31), d(2019, 6, 30), d(2019, 9, 30), d(2019, 12, 31), d(2020, 3, 31)]
    rows = [
        *make_filings(1, "ARQ", gap, lag_days=40, assets=[100, 110, 130, 140, 150, 160]),   # no 2019-03-31
        *make_filings(2, "ARQ", full, lag_days=40, assets=[10, 20, 30, 40, 50, 60]),
    ]
    entries = [(s, m) for s in (1, 2) for m in (d(2020, 2, 28), d(2020, 5, 29))]
    out = asof(rows, entries, metrics=("assets",), lags=(0, 1, 3, 4))
    got = {(r["security_id"], r["date"]): (r["assets"], r["assets_lag1"], r["assets_lag3"], r["assets_lag4"])
           for r in out}
    assert got == {
        # lag4 points at 2019-03-31, which was never filed: null, not the older 110
        (1, d(2020, 5, 29)): (160.0, 150.0, 130.0, None),
        # lag3 points at the missing 2019-03-31 (null); lag4 points at 2018-12-31, which exists
        (1, d(2020, 2, 28)): (150.0, 140.0, None, 110.0),
        (2, d(2020, 5, 29)): (60.0, 50.0, 30.0, 20.0),
        (2, d(2020, 2, 28)): (50.0, 40.0, 20.0, 10.0),
    }


def test_annual_lags_tolerate_52_week_years_but_not_a_fiscal_year_change():
    rows = [
        *make_filings(1, "ARY", [d(2017, 12, 30), d(2018, 12, 29), d(2019, 12, 28)], lag_days=60, eps=[1, 2, 3]),
        # fiscal year end moved from december to september: the "prior year" is only nine months back
        *make_filings(2, "ARY", [d(2017, 12, 31), d(2018, 12, 31), d(2019, 9, 30)], lag_days=60, eps=[1, 2, 3]),
    ]
    out = asof(rows, [(1, d(2020, 3, 31)), (2, d(2020, 3, 31))], dimension="ARY", metrics=("eps",), lags=(0, 1, 2))
    assert [(r["eps"], r["eps_lag1"], r["eps_lag2"]) for r in out] == [(3.0, 2.0, 1.0), (3.0, None, None)]


def test_dimensions_do_not_mix_and_art_lags_are_quarters():
    ends = period_ends(d(2019, 12, 31), 5)
    rows = [
        *make_filings(1, "ARQ", ends, lag_days=40, revenue=[10, 11, 12, 13, 14]),
        *make_filings(1, "ART", ends, lag_days=40, revenue=[40, 42, 44, 46, 50]),
    ]
    entry = [(1, d(2020, 2, 28))]
    (q,) = asof(rows, entry, dimension="ARQ", lags=(0, 4))
    (ttm,) = asof(rows, entry, dimension="ART", lags=(0, 4))
    assert (q["revenue"], q["revenue_lag4"]) == (14.0, 10.0)
    assert (ttm["revenue"], ttm["revenue_lag4"]) == (50.0, 40.0)   # one year earlier = four reports back
    (none,) = asof(rows, entry, dimension="ARY")
    assert none["revenue"] is None and none["fiscal_period_end"] is None


def test_stale_filing_is_null():
    filed = d(2018, 5, 10)
    assert filed + dt.timedelta(days=550) == d(2019, 11, 11)
    rows = [(1, "ARQ", d(2018, 3, 31), filed, {"revenue": 100.0})]
    fresh, stale = asof(rows, [(1, d(2019, 11, 11)), (1, d(2019, 11, 12))])
    assert (fresh["revenue"], fresh["src_date"], fresh["fiscal_period_end"]) == (100.0, filed, d(2018, 3, 31))
    assert stale == {"security_id": 1, "date": d(2019, 11, 12), "fiscal_period_end": None, "src_date": None,
                     "revenue": None}
    # 2018-08-17 is 99 days after the filing, 2018-08-20 is 102
    short = asof(rows, [(1, d(2018, 8, 17)), (1, d(2018, 8, 20))], max_fundamental_age_days=100)
    assert [r["revenue"] for r in short] == [100.0, None]


def test_bypassing_usable_from_raises_lookahead(monkeypatch):
    rows = [(1, "ARQ", d(2019, 12, 31), d(2020, 2, 14), {"revenue": 120.0})]
    ctx = make_ctx(fundamentals=rows, entries=[(1, d(2020, 1, 31)), (1, d(2020, 2, 28))], calendar=CAL)
    ok = fundamentals_asof(ctx, "ARQ", ["revenue"])
    assert ok["revenue"].to_list() == [None, 120.0]
    assert_no_lookahead(ok, "date", "src_date")

    # the classic bug: a fiscal period treated as known on the day it ends
    def period_end_as_usable_from(df, calendar, lag, col="available_date"):
        return df.with_columns(pl.col("fiscal_period_end").alias("usable_from"))

    monkeypatch.setattr(panel, "add_usable_from", period_end_as_usable_from)
    with pytest.raises(LookaheadError):
        fundamentals_asof(ctx, "ARQ", ["revenue"])


def test_output_shape_and_arguments():
    rows = [OLD, (1, "ARQ", d(2019, 6, 30), d(2019, 8, 2), {"revenue": 120.0, "assets": 5.0})]
    ctx = make_ctx(fundamentals=rows, entries=[(2, d(2019, 9, 30)), (1, d(2019, 9, 30)), (1, d(2019, 5, 31))],
                   calendar=CAL)
    out = fundamentals_asof(ctx, "ARQ", ["revenue", "assets", "revenue", "never_reported"], lags=(1, 0))
    assert out.columns == ["security_id", "date", "fiscal_period_end", "src_date", "revenue", "assets",
                           "never_reported", "revenue_lag1", "assets_lag1", "never_reported_lag1"]
    assert out.select("security_id", "date").rows() == [(1, d(2019, 5, 31)), (1, d(2019, 9, 30)), (2, d(2019, 9, 30))]
    assert out.schema["never_reported"] == pl.Float64 and out["never_reported"].null_count() == 3
    assert out.schema["src_date"] == pl.Date and out.schema["revenue_lag1"] == pl.Float64
    assert out["revenue_lag1"].to_list() == [None, 100.0, None]
    # lag 0 only on request
    assert fundamentals_asof(ctx, "ARQ", ["revenue"], lags=(1,)).columns == [
        "security_id", "date", "fiscal_period_end", "src_date", "revenue_lag1"]
    for bad in (dict(dimension="MRQ", metrics=["revenue"]), dict(dimension="ARQ", metrics=[]),
                dict(dimension="ARQ", metrics=["revenue"], lags=(-1,))):
        with pytest.raises(ValueError):
            fundamentals_asof(ctx, **bad)


def test_no_fundamentals_at_all_gives_null_rows():
    ctx = make_ctx(entries=[(1, d(2019, 9, 30))], calendar=CAL)
    out = fundamentals_asof(ctx, "ART", ["revenue"], lags=(0, 4))
    assert out.to_dicts() == [{"security_id": 1, "date": d(2019, 9, 30), "fiscal_period_end": None,
                               "src_date": None, "revenue": None, "revenue_lag4": None}]


# ---- an independent, slow statement of the same rules, checked on random filings -------------------

def _next_trading_day(day: dt.date) -> dt.date:
    day += dt.timedelta(days=1)
    while day.weekday() >= 5:
        day += dt.timedelta(days=1)
    return day


def _months_before(day: dt.date, months: int) -> dt.date:
    year, month = divmod(day.year * 12 + day.month - 1 - months, 12)
    return dt.date(year, month + 1, min(day.day, pycal.monthrange(year, month + 1)[1]))


def _reference(rows, sid, t, lags, metrics, max_age=550):
    filed, cell = {}, {}
    for s, _, period, available, values in rows:
        if s != sid:
            continue
        filed[period] = min(filed.get(period, available), available)
        for m, v in values.items():
            cell[period, m] = (available, v)
    names = [m if k == 0 else f"{m}_lag{k}" for k in lags for m in metrics]
    nothing = {"fiscal_period_end": None, "src_date": None, **dict.fromkeys(names)}
    visible = [p for p, a in filed.items() if _next_trading_day(a) <= t]
    if not visible:
        return nothing
    latest = max(visible)
    src = filed[latest]
    if (t - src).days > max_age:
        return nothing
    out = {"fiscal_period_end": latest, "src_date": src, **dict.fromkeys(names)}
    for k in lags:
        # the lag is looked up by date: the period nearest to k quarters back, within 45 days
        target = _months_before(latest, 3 * k)
        near = [p for p in filed if abs((p - target).days) <= 45 and (k == 0 or p < latest)]
        if not near:
            continue
        period = min(near, key=lambda p: abs((p - target).days))
        for m in metrics:
            available, value = cell.get((period, m), (None, None))
            if available is not None and available <= src:
                out[m if k == 0 else f"{m}_lag{k}"] = value
    return out


def test_matches_a_brute_force_reference_on_random_late_and_amended_filings():
    rng = random.Random(7)
    rows = []
    for sid in range(1, 9):
        for period in period_ends(d(2020, 9, 30), 11):
            if rng.random() < 0.12:      # a quarter that is never filed
                continue
            # up to 200 days late: some periods arrive after their successor, some go stale
            filed = period + dt.timedelta(days=rng.randint(20, 200))
            values = {"a": float(rng.randint(1, 99))}
            if rng.random() < 0.8:
                values["b"] = float(rng.randint(1, 99))
            elif rng.random() < 0.7:     # b only arrives in an amendment
                rows.append((sid, "ARQ", period, filed + dt.timedelta(days=rng.randint(1, 120)),
                             {"b": float(rng.randint(1, 99))}))
            rows.append((sid, "ARQ", period, filed, values))
    months = [m for m in month_end_dates(CAL).to_list() if m >= d(2018, 6, 1)]
    entries = [(sid, m) for sid in range(1, 9) for m in months]
    lags, metrics = (0, 1, 2, 4), ("a", "b")
    out = asof(rows, entries, metrics=metrics, lags=lags, max_fundamental_age_days=150)
    only_b = asof(rows, entries, metrics=("b",), lags=lags, max_fundamental_age_days=150)
    assert len(out) == len(only_b) == len(entries)
    n_values = n_stale = 0
    for row, row_b in zip(out, only_b):
        want = _reference(rows, row["security_id"], row["date"], lags, metrics, max_age=150)
        assert {k: row[k] for k in want} == want, (row["security_id"], row["date"])
        assert all(row_b[k] == want[k] for k in row_b if k not in ("security_id", "date")), row_b
        n_values += sum(row[k] is not None for k in ("a", "b", "a_lag4", "b_lag1"))
        n_stale += row["src_date"] is None and row["date"] > d(2019, 6, 30)
    assert n_values > 200 and n_stale >= 10   # the comparison is not vacuous


def test_fundamentals_asof_is_fast_enough_for_the_real_panel():
    # 300 securities x 60 quarters x 6 metrics, 54,000 entries: well under a second when vectorized
    ends = pl.Series(period_ends(d(2020, 6, 30), 60))
    base = pl.DataFrame({"security_id": pl.int_range(300, eager=True)}).join(ends.to_frame("fiscal_period_end"), how="cross")
    cal = business_days(d(2005, 1, 1), d(2020, 12, 31))
    fund = pl.concat([
        base.with_columns(
            pl.lit("ARQ").alias("dimension"), pl.lit(m).alias("metric"),
            (pl.col("fiscal_period_end") + pl.duration(days=45)).alias("available_date"),
            (pl.col("security_id") * 1.0).alias("value"),
        )
        for m in ("a", "b", "c", "d", "e", "f")
    ]).select(list(SCHEMAS["fundamentals"]))
    months = month_end_dates(cal).filter(month_end_dates(cal) >= d(2006, 1, 1))
    entries = pl.DataFrame({"security_id": pl.int_range(300, eager=True)}).join(months.to_frame("date"), how="cross")
    ctx = context_from_frames(entries=entries, fundamentals=fund, calendar=cal)
    out = fundamentals_asof(ctx, "ARQ", ["a", "b", "c"], lags=(0, 4, 8))
    assert out.height == 300 * 180 and out["a"].null_count() == 0
    assert (out["a"] == out["security_id"]).all()
    # the first period is 2005-09-30; the ninth (2007-09-30, filed 11-14) is first seen at the end of
    # november 2007, so the 22 month ends from january 2006 to october 2007 have no period 8 back
    assert out["c_lag8"].null_count() == 300 * 22 and out["c_lag4"].null_count() == 300 * 10


# ---- prices, market cap, sectors ---------------------------------------------------------------

@pytest.fixture
def price_ctx():
    cal = business_days(d(2019, 1, 1), d(2020, 12, 31))
    prices = make_prices(
        {1: {d(2019, 1, 1): 10.0, d(2020, 3, 2): 12.0}, 2: 20.0, 3: 5.0},
        cal,
        splits={1: {d(2020, 3, 2): 2.0}},
        last={3: d(2019, 6, 14)},   # stops trading mid-june
    )
    mcap = make_marketcap(prices, {1: {d(2019, 1, 1): 1.0e9, d(2020, 3, 30): 1.5e9}, 2: 2.0e9, 3: 3.0e8})
    return make_ctx(prices=prices, marketcap=mcap, securities=make_securities({1: ("Tech", "Software"), 2: "Energy"}))


def test_helpers_build_core_shaped_prices(price_ctx):
    p = price_ctx.prices
    assert p.schema == pl.Schema(SCHEMAS["prices"])
    one = p.filter(pl.col("security_id") == 1).sort("date")
    split = one.filter(pl.col("date").is_in([d(2020, 2, 28), d(2020, 3, 2)]))
    assert split["close_unadj"].to_list() == [20.0, 12.0]       # 2:1 split: unadjusted halves, adjusted does not
    assert split["adj_close"].to_list() == [10.0, 12.0] and split["split_factor"].to_list() == [1.0, 2.0]
    assert split["ret"].to_list() == pytest.approx([0.0, 0.2])
    assert one["ret"][0] is None
    assert p.filter(pl.col("security_id") == 3)["date"].max() == d(2019, 6, 14)


def test_default_entries_are_every_month_end_with_a_price(price_ctx):
    e = price_ctx.entries
    assert e.filter(pl.col("security_id") == 1).height == 24
    assert e.filter(pl.col("security_id") == 3)["date"].to_list() == [
        d(2019, 1, 31), d(2019, 2, 28), d(2019, 3, 29), d(2019, 4, 30), d(2019, 5, 31)]
    assert e.rows() == sorted(e.rows())
    assert price_ctx.calendar.to_list() == business_days(d(2019, 1, 1), d(2020, 12, 31)).to_list()


def test_month_end_prices(price_ctx):
    me = month_end_prices(price_ctx)
    assert me.columns == ["security_id", "date", "adj_close", "close", "close_unadj"]
    assert set(me["date"].to_list()) == set(month_end_dates(price_ctx.calendar).to_list())
    one = dict(me.filter(pl.col("security_id") == 1).select("date", "adj_close").iter_rows())
    assert one[d(2020, 2, 28)] == 10.0 and one[d(2020, 3, 31)] == 12.0
    # the stock that stopped on june 14 has no june row: nothing is carried forward
    assert me.filter(pl.col("security_id") == 3)["date"].max() == d(2019, 5, 31)


def test_months_back_uses_month_end_trading_days(price_ctx):
    back = {k: dict(months_back(price_ctx, k).iter_rows()) for k in (0, 1, 6, 12)}
    assert back[0][d(2020, 5, 29)] == d(2020, 5, 29)
    assert back[1][d(2020, 3, 31)] == d(2020, 2, 28)     # feb 29 2020 is a saturday
    assert back[6][d(2020, 5, 29)] == d(2019, 11, 29)    # nov 30 2019 is a saturday
    assert back[12][d(2020, 5, 29)] == d(2019, 5, 31)    # 261 weekdays earlier, not 252
    assert back[12][d(2019, 12, 31)] is None             # before the calendar
    assert back[12][d(2020, 1, 31)] == d(2019, 1, 31)
    assert len(back[12]) == 24 and months_back(price_ctx, 12).columns == ["date", "date_k"]
    with pytest.raises(ValueError):
        months_back(price_ctx, -1)


def test_market_cap_at_uses_only_values_on_or_before_t():
    cal = business_days(d(2020, 1, 1), d(2020, 4, 30))
    prices = make_prices({1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0}, cal)
    mcap = pl.DataFrame(
        [
            (1, d(2020, 3, 30), 100.0), (1, d(2020, 3, 31), 110.0), (1, d(2020, 4, 1), 999.0),  # value at t
            (2, d(2020, 3, 26), 200.0), (2, d(2020, 4, 1), 999.0),      # 5 days old at t: used
            (3, d(2020, 3, 23), 300.0), (3, d(2020, 4, 1), 999.0),      # 8 days old: too old
            (4, d(2020, 3, 27), 400.0), (4, d(2020, 3, 31), 0.0),       # a zero is not a market cap
        ],
        schema=SCHEMAS["marketcap"], orient="row",
    )
    ctx = make_ctx(prices=prices, marketcap=mcap, entries=[(s, d(2020, 3, 31)) for s in (1, 2, 3, 4, 5)])
    out = market_cap_at(ctx)
    assert out.columns == ["security_id", "date", "market_cap", "market_cap_date"]
    assert out.select("security_id", "market_cap", "market_cap_date").rows() == [
        (1, 110.0, d(2020, 3, 31)),
        (2, 200.0, d(2020, 3, 26)),
        (3, None, None),
        (4, 400.0, d(2020, 3, 27)),
        (5, None, None),
    ]


def test_sector_map(price_ctx):
    assert sector_map(price_ctx).rows() == [(1, "Tech", "Software"), (2, "Energy", None)]


def test_context_from_frames_derives_entries_and_calendar(price_ctx):
    universe = make_universe(price_ctx.prices, exclude=[2, (1, d(2019, 1, 31))])
    ctx = context_from_frames(prices=price_ctx.prices, universe=universe)
    assert 2 not in ctx.entries["security_id"].to_list()
    assert ctx.entries.filter(pl.col("security_id") == 1)["date"].min() == d(2019, 2, 28)
    assert ctx.entries.columns == ["security_id", "date"]
    assert ctx.fundamental_lag == 1 and ctx.max_fundamental_age_days == 550
    assert ctx.fundamentals.schema == pl.Schema(SCHEMAS["fundamentals"]) and ctx.macro.is_empty()
    # entries without a universe: the universe is exactly the entries
    only = make_ctx(entries=[(7, d(2019, 3, 29))], calendar=price_ctx.calendar)
    assert only.universe.select("date", "security_id", "in_universe").rows() == [(d(2019, 3, 29), 7, True)]
    with pytest.raises(ValueError):
        context_from_frames(prices=price_ctx.prices)             # no output grid
    with pytest.raises(ValueError):
        context_from_frames(entries=only.entries)                # no calendar


def test_make_context_reads_the_store(tmp_path, cfg, price_ctx):
    store = Store(tmp_path / "data")
    fund = make_fundamentals([(1, "ARQ", d(2019, 12, 31), d(2020, 2, 14), {"revenue": 120.0, "assets": 9.0})])
    universe = make_universe(price_ctx.prices, exclude=[3])
    tables = {
        "prices_daily": price_ctx.prices, "universe": universe, "fundamentals_pit": fund,
        "marketcap_daily": price_ctx.marketcap, "securities": price_ctx.securities,
        "macro_pit": make_macro({"DGS10": {d(2020, 1, 2): 1.5}}),
        "benchmark_daily": make_benchmark(price_ctx.calendar, {d(2019, 1, 2): 0.1}),
    }
    for name, df in tables.items():
        store.write(name, df)
    ctx = make_context(store, cfg)
    assert ctx.fundamental_lag == cfg.pit.fundamental_lag_trading_days == 1
    assert ctx.entries.rows() == sorted(universe.filter(pl.col("security_id") != 3).select("security_id", "date").rows())
    assert ctx.calendar.to_list() == price_ctx.calendar.to_list()
    assert ctx.prices.height == price_ctx.prices.height
    row = fundamentals_asof(ctx, "ARQ", ["revenue"]).filter(
        (pl.col("security_id") == 1) & (pl.col("date") == d(2020, 2, 28))).row(0, named=True)
    assert (row["revenue"], row["src_date"]) == (120.0, d(2020, 2, 14))
    assert ctx.macro.select("series", "available_date", "value").rows() == [("DGS10", d(2020, 1, 3), 1.5)]
    assert ctx.benchmark["bench_level"][:3].to_list() == pytest.approx([1.0, 1.1, 1.1])
    assert make_context(store, cfg, metrics=["assets"]).fundamentals["metric"].to_list() == ["assets"]


def test_result_does_not_depend_on_the_row_order_a_join_returns(monkeypatch):
    # polars gives no row-order guarantee for a join without maintain_order; the supersede step is
    # positional, so it must not rely on the order a join happens to return
    rows = [
        (sid, "ARQ", end, end + dt.timedelta(days=40), {"revenue": float(sid * 100 + i)})
        for sid in (1, 2, 3)
        for i, end in enumerate(period_ends(d(2020, 6, 30), 10))
    ]
    entries = [(sid, m) for sid in (1, 2, 3) for m in month_end_dates(CAL).to_list() if m >= d(2019, 1, 1)]
    want = asof(rows, entries, lags=(0, 4))

    real_join = pl.DataFrame.join

    def shuffling_join(self, *args, **kwargs):
        out = real_join(self, *args, **kwargs)
        caller = sys._getframe(1).f_code.co_name
        if caller == "fundamentals_asof" and kwargs.get("maintain_order") in (None, "none"):
            out = out.sample(fraction=1.0, shuffle=True, seed=3)
        return out

    monkeypatch.setattr(pl.DataFrame, "join", shuffling_join)
    assert asof(rows, entries, lags=(0, 4)) == want

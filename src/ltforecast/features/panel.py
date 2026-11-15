"""Shared point-in-time inputs for the feature families (plan 1.3 and 3.2).

A `FeatureContext` holds the core tables plus the output grid (`entries`: every in-universe
(security, month end)). The helpers below answer "what was known at the close of t" once, so the
family modules never touch dates themselves:

- fundamentals: the latest fiscal period whose filing was usable at t (`fundamentals_asof`)
- prices / market cap: rows dated <= t (`month_end_prices`, `months_back`, `market_cap_at`)
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from ltforecast.pit.asof import add_usable_from, asof_join, assert_no_lookahead, trading_calendar
from ltforecast.universe import month_end_dates

if TYPE_CHECKING:
    from ltforecast.config import Config
    from ltforecast.store import Store

# Months between consecutive fiscal periods of an SF1 dimension. ART is a trailing-12-month figure
# that is reported every quarter, so "one year earlier" is lag 4 for ARQ and ART and lag 1 for ARY.
PERIOD_MONTHS = {"ARQ": 3, "ART": 3, "ARY": 12}
# Half a quarter: a lagged period this close to its expected date cannot be a neighbouring quarter.
LAG_TOLERANCE_DAYS = 45
MCAP_TOLERANCE = "7d"
_FAR = dt.date(9999, 12, 31)

_F, _D, _S, _I = pl.Float64, pl.Date, pl.String, pl.Int64
SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "prices": {
        "security_id": _I, "date": _D, "open": _F, "high": _F, "low": _F, "close": _F, "close_unadj": _F,
        "adj_close": _F, "volume": _F, "dividend": _F, "split_factor": _F, "ret": _F,
    },
    "fundamentals": {
        "security_id": _I, "dimension": _S, "metric": _S, "fiscal_period_end": _D, "available_date": _D,
        "value": _F,
    },
    "marketcap": {"security_id": _I, "date": _D, "market_cap": _F},
    "securities": {
        "security_id": _I, "ticker": _S, "name": _S, "cik": _I, "category": _S, "exchange": _S, "sector": _S,
        "industry": _S, "siccode": _S, "famaindustry": _S, "first_price_date": _D, "last_price_date": _D,
        "delisted_date": _D, "delist_reason": _S, "ticker_history": _S,
    },
    "macro": {
        "series": _S, "observation_date": _D, "available_date": _D, "value": _F, "first_release": pl.Boolean,
    },
    "benchmark": {"date": _D, "bench_ret": _F, "n_members": _I, "bench_level": _F},
    "universe": {
        "date": _D, "security_id": _I, "close_unadj": _F, "market_cap": _F, "adv": _F, "mcap_threshold": _F,
        "is_common": pl.Boolean, "in_universe": pl.Boolean,
    },
}


@dataclass(frozen=True)
class FeatureContext:
    entries: pl.DataFrame       # security_id, date: every in-universe (stock, month end), sorted that way
    prices: pl.DataFrame        # prices_daily
    fundamentals: pl.DataFrame  # fundamentals_pit
    marketcap: pl.DataFrame     # marketcap_daily
    securities: pl.DataFrame
    macro: pl.DataFrame         # macro_pit
    benchmark: pl.DataFrame     # benchmark_daily
    universe: pl.DataFrame
    calendar: pl.Series         # sorted trading days
    fundamental_lag: int        # trading days between a filing and its first use
    max_fundamental_age_days: int = 550  # a filing older than this (calendar days) at t is stale -> null


def context_from_frames(
    *,
    prices: pl.DataFrame | None = None,
    universe: pl.DataFrame | None = None,
    entries: pl.DataFrame | None = None,
    fundamentals: pl.DataFrame | None = None,
    marketcap: pl.DataFrame | None = None,
    securities: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    benchmark: pl.DataFrame | None = None,
    calendar: pl.Series | None = None,
    fundamental_lag: int = 1,
    max_fundamental_age_days: int = 550,
) -> FeatureContext:
    """A context from in-memory frames. A frame left out is empty with the core-table schema.

    `entries` defaults to the `in_universe` rows of `universe` (and `universe` to the entries, all
    members); `calendar` defaults to the distinct price dates.
    """
    if entries is None:
        if universe is None:
            raise ValueError("pass `universe` or `entries`: one of them must define the output grid")
        entries = universe.filter(pl.col("in_universe"))
    entries = entries.select("security_id", "date").unique().sort("security_id", "date")
    if universe is None:
        members = entries.select("date", "security_id", pl.lit(True).alias("in_universe"))
        universe = pl.concat([pl.DataFrame(schema=SCHEMAS["universe"]), members], how="diagonal")
    frames = {
        "prices": prices, "fundamentals": fundamentals, "marketcap": marketcap, "securities": securities,
        "macro": macro, "benchmark": benchmark, "universe": universe,
    }
    frames = {k: pl.DataFrame(schema=SCHEMAS[k]) if v is None else v for k, v in frames.items()}
    if calendar is None:
        if frames["prices"].is_empty():
            raise ValueError("pass `calendar` (sorted trading days) when there are no prices to derive it from")
        calendar = trading_calendar(frames["prices"])
    return FeatureContext(
        entries=entries,
        calendar=calendar.sort().alias("date"),
        fundamental_lag=fundamental_lag,
        max_fundamental_age_days=max_fundamental_age_days,
        **frames,
    )


def make_context(store: Store, cfg: Config, *, metrics: Sequence[str] | None = None) -> FeatureContext:
    """The context over the stored core tables.

    `metrics` keeps only those SF1 indicators in memory (fundamentals_pit holds every indicator in
    long form, which is the largest table by far); None loads them all.
    """
    fund = store.scan("fundamentals_pit")
    if metrics is not None:
        fund = fund.filter(pl.col("metric").is_in(list(metrics)))
    return context_from_frames(
        prices=store.scan("prices_daily").collect(),
        universe=store.scan("universe").collect(),
        fundamentals=fund.collect(),
        marketcap=store.scan("marketcap_daily").collect(),
        securities=store.scan("securities").collect(),
        macro=store.scan("macro_pit").collect(),
        benchmark=store.scan("benchmark_daily").collect(),
        fundamental_lag=cfg.pit.fundamental_lag_trading_days,
    )


def fundamentals_asof(
    ctx: FeatureContext, dimension: str, metrics: Sequence[str], lags: Sequence[int] = (0,)
) -> pl.DataFrame:
    """For each entry, the most recent fiscal period of `dimension` whose filing was usable at t.

    Columns: security_id, date, fiscal_period_end, src_date, then `<metric>` (lag 0, if requested) and
    `<metric>_lag<k>` for each k in `lags`. One row per entry, sorted by security_id, date.

    Conventions:
    - The filing of a fiscal period is dated by the earliest `available_date` of ANY metric of that
      (security, dimension, period); it is usable from `fundamental_lag` trading days later. The latest
      period at t is the one with the greatest fiscal_period_end among the filings usable at t, so a
      late filing of an older period never replaces a newer one. `src_date` is that filing's date.
    - Each row is the snapshot the market had when that filing arrived: a value (of the latest period or
      of a lagged one) is used only if its own available_date is <= src_date. A metric the filing did
      not carry is null (no carry-forward from older filings), and stays null if an amendment supplies
      it later; the amended value is picked up, as a lag, once the next period's filing arrives.
    - `<metric>_lag<k>` is the value for the fiscal period ending k periods (k quarters for ARQ and
      ART, k years for ARY) before the latest one, looked up by date: the security's period whose end
      is nearest to `fiscal_period_end - k*months`, within LAG_TOLERANCE_DAYS, and strictly earlier
      than the latest period. A lag is null only when the period it points at is missing or was not
      yet known at the filing; a skipped period nulls the lags that point at it and no other.
    - A filing more than `ctx.max_fundamental_age_days` calendar days old at t is stale: every column
      except the keys is null, exactly as if nothing had been filed.

    The join goes through `pit.asof.asof_join`; a row whose usable_from or filing date is after t
    raises LookaheadError.
    """
    if dimension not in PERIOD_MONTHS:
        raise ValueError(f"unknown dimension {dimension!r}; expected one of {sorted(PERIOD_MONTHS)}")
    metrics = list(dict.fromkeys(metrics))
    lags = sorted({int(k) for k in lags})
    if not metrics or not lags or lags[0] < 0:
        raise ValueError("fundamentals_asof needs at least one metric and non-negative lags")
    months = PERIOD_MONTHS[dimension]
    keys = ["security_id", "fiscal_period_end"]

    fund = ctx.fundamentals.lazy().filter(pl.col("dimension") == dimension)
    periods, cells = pl.collect_all([
        fund.group_by(keys).agg(pl.col("available_date").min()),
        fund.filter(pl.col("metric").is_in(metrics)).select(
            *keys, "metric", "value", pl.col("available_date").alias("avail")
        ),
    ])
    wide = cells.pivot(on="metric", index=keys, values=["value", "avail"])
    wide = wide.with_columns(
        [pl.lit(None, pl.Float64).alias(f"value_{m}") for m in metrics if f"value_{m}" not in wide.columns]
        + [pl.lit(None, pl.Date).alias(f"avail_{m}") for m in metrics if f"avail_{m}" not in wide.columns]
    )
    wide = add_usable_from(
        periods.join(wide, on=keys, how="left").sort(keys), ctx.calendar, ctx.fundamental_lag
    )

    period_end, filed = pl.col("fiscal_period_end"), pl.col("available_date")
    value_cols = [f"value_{m}" for m in metrics]
    avail_cols = [f"avail_{m}" for m in metrics]
    lookup = wide.select("security_id", "fiscal_period_end", *value_cols, *avail_cols).sort(
        "fiscal_period_end", "security_id"
    )
    values = []
    for k in lags:
        if k == 0:
            for m in metrics:
                known = pl.col(f"avail_{m}") <= filed
                values.append(pl.when(known).then(pl.col(f"value_{m}")).alias(m))
            continue
        # the lagged period is found by date, not by position, so one missing period does not shift
        # every deeper lag onto the wrong row
        sfx = f"_lag{k}"
        target = period_end.dt.offset_by(f"-{k * months}mo").alias("_target")
        matched = (
            wide.select("security_id", "fiscal_period_end", target)
            .sort("_target", "security_id")
            .join_asof(
                lookup.rename({c: c + sfx for c in ["fiscal_period_end", *value_cols, *avail_cols]}),
                left_on="_target",
                right_on="fiscal_period_end" + sfx,
                by="security_id",
                strategy="nearest",
                tolerance=f"{LAG_TOLERANCE_DAYS}d",
                check_sortedness=False,
            )
            .drop("_target")
        )
        wide = wide.join(matched, on=keys, how="left", maintain_order="left")
        in_place = pl.col("fiscal_period_end" + sfx) < period_end
        for m in metrics:
            known = pl.col(f"avail_{m}{sfx}") <= filed
            values.append(pl.when(in_place & known).then(pl.col(f"value_{m}{sfx}")).alias(f"{m}{sfx}"))
    names = [e.meta.output_name() for e in values]
    # the supersede step below is positional: rows must be in fiscal_period_end order per security
    wide = wide.sort(keys)

    # A period is the latest one from its usable_from until a later period becomes usable. If a later
    # period was usable first (or on the same day), this one never is the latest and is dropped, which
    # leaves usable_from strictly increasing with the period: the as-of join then finds the latest period.
    superseded_on = pl.col("usable_from").fill_null(_FAR).cum_min(reverse=True).shift(-1).over("security_id")
    snapshots = (
        wide.select(*keys, "available_date", "usable_from", superseded_on.fill_null(_FAR).alias("_next"), *values)
        .filter(pl.col("usable_from") < pl.col("_next"))
        .drop("_next")
    )

    out = asof_join(ctx.entries.select("security_id", "date"), snapshots)
    # asof_join has checked usable_from; this checks the filing date itself, whatever produced usable_from
    assert_no_lookahead(out, "date", "available_date")
    fresh = (pl.col("date") - filed).dt.total_days() <= ctx.max_fundamental_age_days
    return out.select(
        "security_id",
        "date",
        pl.when(fresh).then(period_end).alias("fiscal_period_end"),
        pl.when(fresh).then(filed).alias("src_date"),
        *[pl.when(fresh).then(pl.col(c)).alias(c) for c in names],
    )


def _month_ends(ctx: FeatureContext) -> pl.DataFrame:
    return month_end_dates(ctx.calendar).to_frame("date")


def month_end_prices(ctx: FeatureContext) -> pl.DataFrame:
    """Prices on each month-end trading day. Columns: security_id, date, adj_close, close, close_unadj.
    A security without a price on that exact day has no row (nothing is carried forward)."""
    return (
        ctx.prices.select("security_id", "date", "adj_close", "close", "close_unadj")
        .join(_month_ends(ctx), on="date", how="semi")
        .sort("security_id", "date")
    )


def months_back(ctx: FeatureContext, k: int) -> pl.DataFrame:
    """Columns: date, date_k. For every month-end trading day, the month-end trading day of the
    calendar month k months earlier (null if that month is before the calendar)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    ends = _month_ends(ctx).with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
    return (
        ends.with_columns(pl.col("month").dt.offset_by(f"-{k}mo"))
        .join(ends.rename({"date": "date_k"}), on="month", how="left")
        .select("date", "date_k")
        .sort("date")
    )


def market_cap_at(ctx: FeatureContext) -> pl.DataFrame:
    """Market cap as of each entry: the latest positive value dated <= t and at most 7 calendar days
    old, else null. Columns: security_id, date, market_cap, market_cap_date (the date of that value)."""
    mcap = (
        ctx.marketcap.filter(pl.col("market_cap").is_finite() & (pl.col("market_cap") > 0))
        .select("security_id", pl.col("date").alias("market_cap_date"), "market_cap")
        .sort("security_id", "market_cap_date")
    )
    return (
        ctx.entries.select("security_id", "date")
        .sort("security_id", "date")
        .join_asof(
            mcap,
            left_on="date",
            right_on="market_cap_date",
            by="security_id",
            strategy="backward",
            tolerance=MCAP_TOLERANCE,
            check_sortedness=False,
        )
        .select("security_id", "date", "market_cap", "market_cap_date")
    )


def sector_map(ctx: FeatureContext) -> pl.DataFrame:
    """Columns: security_id, sector, famaindustry (current-only classifications; vendor limitation)."""
    return (
        ctx.securities.select("security_id", "sector", "famaindustry")
        .unique("security_id", keep="first", maintain_order=True)
        .sort("security_id")
    )

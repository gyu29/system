"""Small builders for feature tests: core-table-shaped frames from a few literals.

Usage (every frame a test does not pass to `make_ctx` is empty):

    from conftest import business_days
    from feature_helpers import d, make_ctx, make_filings, make_fundamentals, make_marketcap, make_prices
    from feature_helpers import period_ends

    cal = business_days(d(2019, 1, 1), d(2020, 12, 31))
    # per security: a flat price, {date: price} change points, or one price per trading day
    prices = make_prices({1: {d(2019, 1, 1): 10.0, d(2020, 6, 1): 12.0}, 2: 20.0}, cal)
    fund = make_fundamentals([
        # security_id, dimension, fiscal_period_end, available_date (the filing date), {metric: value}
        (1, "ARQ", d(2019, 12, 31), d(2020, 2, 14), {"equity": 500.0, "assets": 2000.0}),
        # eight quarters at once, each filed 40 days after the period end
        *make_filings(2, "ARQ", period_ends(d(2019, 12, 31), 8), lag_days=40, eps=[1, 1, 2, 2, 3, 3, 4, 4]),
    ])
    ctx = make_ctx(
        prices=prices,
        fundamentals=fund,
        marketcap=make_marketcap(prices, {1: 1.0e9, 2: 2.0e9}),
        securities=make_securities({1: ("Technology", "Software"), 2: "Energy"}),
        entries=[(1, d(2020, 3, 31)), (2, d(2020, 3, 31))],   # default: every (security, month end) with a price
    )
    out = value.compute(ctx)   # one row per entry

Filings are usable from the next trading day (`fundamental_lag=1`), as in production.
"""

from __future__ import annotations

import calendar as _calendar
import datetime as dt
import numbers
from collections.abc import Iterable, Mapping, Sequence

import polars as pl
from conftest import business_days  # noqa: F401  (re-exported: the trading calendar of every test)

from ltforecast.features.panel import SCHEMAS, FeatureContext, context_from_frames
from ltforecast.universe import month_end_dates

d = dt.date


def _expand(value, days: pl.Series) -> list[float | None]:
    """One number per day from a scalar, {date: value} change points (carried forward; null before
    the first one) or a sequence with exactly one value per day."""
    if isinstance(value, Mapping):
        steps = pl.DataFrame(
            {"date": list(value), "v": [None if v is None else float(v) for v in value.values()]},
            schema={"date": pl.Date, "v": pl.Float64},
        )
        return days.to_frame("date").join_asof(steps.sort("date"), on="date")["v"].to_list()
    if isinstance(value, numbers.Real):
        return [float(value)] * len(days)
    values = [None if v is None else float(v) for v in value]
    if len(values) != len(days):
        raise ValueError(f"expected {len(days)} daily values, got {len(values)}")
    return values


def make_prices(
    spec: Mapping[int, object],
    calendar: pl.Series,
    *,
    close: Mapping[int, object] | None = None,
    volume: object = 1.0e6,
    splits: Mapping[int, Mapping[dt.date, float]] | None = None,
    dividends: Mapping[int, Mapping[dt.date, float]] | None = None,
    first: Mapping[int, dt.date] | None = None,
    last: Mapping[int, dt.date] | None = None,
) -> pl.DataFrame:
    """A prices_daily-shaped frame.

    `spec[security_id]` is the total-return price (`adj_close`, which drives `ret`), given as
    - a number: flat over the whole calendar;
    - {date: price} change points: rows start at the first change point and carry each price forward;
    - a sequence: one price per trading day from the start of the calendar (or `first[security_id]`).
    `first` / `last` cut a security's rows to a listing / delisting date.

    `close` (split-adjusted, same shapes, per security) defaults to adj_close; open = high = low = close.
    `volume` is a number for everyone or {security_id: number | change points | sequence}.
    `splits` {security_id: {date: factor}} set `split_factor` on that day and make `close_unadj` equal
    close x the factors of all LATER splits (a 2:1 split: unadjusted 8 -> 6 while close goes 4 -> 6).
    `dividends` {security_id: {date: amount}} only fill the `dividend` column.
    """
    close, splits, dividends, first, last = (x or {} for x in (close, splits, dividends, first, last))
    frames = []
    for sid, value in spec.items():
        start = first.get(sid) or (min(value) if isinstance(value, Mapping) else calendar[0])
        days = calendar.filter((calendar >= start) & (calendar <= last.get(sid, calendar[-1])))
        if not isinstance(value, (Mapping, numbers.Real)):
            days = days.head(len(value))
        vol = volume.get(sid, 1.0e6) if isinstance(volume, Mapping) else volume
        df = pl.DataFrame({
            "security_id": pl.Series([sid] * len(days), dtype=pl.Int64),
            "date": days,
            "adj_close": pl.Series(_expand(value, days), dtype=pl.Float64),
            "volume": pl.Series(_expand(vol, days), dtype=pl.Float64),
        })
        df = df.with_columns(
            pl.Series("close", _expand(close[sid], days), dtype=pl.Float64) if sid in close else pl.col("adj_close").alias("close")
        )
        later = pl.lit(1.0)
        for day, factor in splits.get(sid, {}).items():
            later = later * pl.when(pl.col("date") < day).then(float(factor)).otherwise(1.0)
        df = df.with_columns(
            (pl.col("close") * later).alias("close_unadj"),
            pl.col("date").replace_strict(dict(dividends.get(sid, {})), default=0.0, return_dtype=pl.Float64).alias("dividend"),
            pl.col("date").replace_strict(dict(splits.get(sid, {})), default=1.0, return_dtype=pl.Float64).alias("split_factor"),
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1).alias("ret"),
        )
        frames.append(df)
    out = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMAS["prices"])
    return out.with_columns(
        pl.col("close").alias("open"), pl.col("close").alias("high"), pl.col("close").alias("low")
    ).select(list(SCHEMAS["prices"])).sort("security_id", "date")


def make_marketcap(prices: pl.DataFrame, caps: Mapping[int, object]) -> pl.DataFrame:
    """marketcap_daily on each security's price dates. `caps[security_id]` is dollars as a number,
    {date: cap} change points or one value per price row; securities not named get no rows."""
    frames = []
    for sid, value in caps.items():
        days = prices.filter(pl.col("security_id") == sid)["date"].sort()
        frames.append(pl.DataFrame({
            "security_id": pl.Series([sid] * len(days), dtype=pl.Int64),
            "date": days,
            "market_cap": pl.Series(_expand(value, days), dtype=pl.Float64),
        }))
    out = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMAS["marketcap"])
    return out.drop_nulls("market_cap").sort("security_id", "date")


def make_fundamentals(rows: Iterable[tuple]) -> pl.DataFrame:
    """fundamentals_pit (long) from (security_id, dimension, fiscal_period_end, available_date,
    {metric: value}) tuples. A metric given as None is left out, like a field missing from a filing."""
    long = [
        (sid, dim, metric, period, available, float(value))
        for sid, dim, period, available, metrics in rows
        for metric, value in metrics.items()
        if value is not None
    ]
    schema = {k: SCHEMAS["fundamentals"][k] for k in
              ("security_id", "dimension", "metric", "fiscal_period_end", "available_date", "value")}
    return pl.DataFrame(long, schema=schema, orient="row").sort("security_id", "metric", "available_date")


def period_ends(last: dt.date, n: int, months: int = 3) -> list[dt.date]:
    """`n` month-end dates `months` apart, oldest first, ending at the month of `last`
    (quarters by default; months=12 for fiscal years)."""
    out = []
    for i in range(n - 1, -1, -1):
        m = last.year * 12 + (last.month - 1) - i * months
        year, month = divmod(m, 12)
        out.append(dt.date(year, month + 1, _calendar.monthrange(year, month + 1)[1]))
    return out


def make_filings(
    security_id: int, dimension: str, ends: Sequence[dt.date], *, lag_days: int = 45, **metrics
) -> list[tuple]:
    """Rows for `make_fundamentals`: one filing per fiscal period end, filed `lag_days` calendar days
    later. Each metric is a number (every period) or one value per period (None = not reported)."""
    series = {
        m: [v] * len(ends) if isinstance(v, numbers.Real) or v is None else list(v) for m, v in metrics.items()
    }
    for m, vals in series.items():
        if len(vals) != len(ends):
            raise ValueError(f"{m}: expected {len(ends)} values, got {len(vals)}")
    return [
        (security_id, dimension, end, end + dt.timedelta(days=lag_days), {m: vals[i] for m, vals in series.items()})
        for i, end in enumerate(ends)
    ]


def make_securities(spec: Mapping[int, object] | Iterable[int]) -> pl.DataFrame:
    """securities from {security_id: sector} or {security_id: (sector, famaindustry)} (or just ids:
    no classification). Tickers are S<id>; everything else is null or a common-stock default."""
    if not isinstance(spec, Mapping):
        spec = {sid: (None, None) for sid in spec}
    rows = []
    for sid, value in spec.items():
        sector, fama = value if isinstance(value, tuple) else (value, None)
        rows.append({"security_id": sid, "ticker": f"S{sid}", "name": f"S{sid} Inc", "sector": sector,
                     "famaindustry": fama, "category": "Domestic Common Stock", "exchange": "NYSE"})
    base = pl.DataFrame(schema=SCHEMAS["securities"])
    if not rows:
        return base
    cols = ("security_id", "ticker", "name", "sector", "famaindustry", "category", "exchange")
    df = pl.DataFrame(rows, schema={k: SCHEMAS["securities"][k] for k in cols})
    return pl.concat([base, df], how="diagonal").sort("security_id")


def make_macro(series: Mapping[str, Mapping[dt.date, float]], *, lag_days: int | Mapping[str, int] = 1) -> pl.DataFrame:
    """macro_pit from {series: {observation_date: value}}; each observation becomes available
    `lag_days` calendar days later (a number, or {series: days})."""
    rows = []
    for name, obs in series.items():
        lag = lag_days[name] if isinstance(lag_days, Mapping) else lag_days
        rows += [(name, day, day + dt.timedelta(days=lag), float(v), True) for day, v in obs.items()]
    return pl.DataFrame(rows, schema=SCHEMAS["macro"], orient="row").sort("series", "available_date", "observation_date")


def make_benchmark(calendar: pl.Series, rets: object = 0.0) -> pl.DataFrame:
    """benchmark_daily over the calendar. `rets` is a number (every day), {date: return} (zero on the
    other days) or one return per day; bench_level = cumulative product of 1 + bench_ret."""
    if isinstance(rets, Mapping):
        values = calendar.replace_strict(dict(rets), default=0.0, return_dtype=pl.Float64).to_list()
    else:
        values = _expand(rets, calendar)
    return pl.DataFrame({"date": calendar, "bench_ret": pl.Series(values, dtype=pl.Float64)}).with_columns(
        pl.lit(1, pl.Int64).alias("n_members"),
        (1 + pl.col("bench_ret")).cum_prod().alias("bench_level"),
    )


def make_universe(
    prices: pl.DataFrame, *, marketcap: pl.DataFrame | None = None, exclude: Iterable = ()
) -> pl.DataFrame:
    """universe: one row per (month-end trading day, security with a price that day), all members
    except those in `exclude` (security ids, or (security_id, date) pairs). market_cap is the value
    on that day when `marketcap` is given; adv and mcap_threshold are null."""
    exclude = list(exclude)
    out_ids = [x for x in exclude if not isinstance(x, tuple)]
    out_pairs = pl.DataFrame(
        [x for x in exclude if isinstance(x, tuple)], schema={"security_id": pl.Int64, "date": pl.Date}, orient="row"
    ).with_columns(pl.lit(True).alias("_out"))
    ends = month_end_dates(prices["date"].unique().sort()).to_frame("date")
    u = prices.select("date", "security_id", "close_unadj").join(ends, on="date", how="semi")
    if marketcap is not None:
        u = u.join(marketcap.select("security_id", "date", "market_cap"), on=["security_id", "date"], how="left")
    u = u.join(out_pairs, on=["security_id", "date"], how="left").with_columns(
        pl.lit(True).alias("is_common"),
        (~pl.col("security_id").is_in(out_ids) & pl.col("_out").is_null()).alias("in_universe"),
    )
    return (
        pl.concat([pl.DataFrame(schema=SCHEMAS["universe"]), u.drop("_out")], how="diagonal")
        .select(list(SCHEMAS["universe"]))
        .sort("date", "security_id")
    )


def make_entries(pairs: Iterable[tuple[int, dt.date]]) -> pl.DataFrame:
    """The output grid from (security_id, date) pairs."""
    return pl.DataFrame(list(pairs), schema={"security_id": pl.Int64, "date": pl.Date}, orient="row")


def make_ctx(
    *,
    prices: pl.DataFrame | None = None,
    fundamentals: pl.DataFrame | Iterable[tuple] | None = None,
    marketcap: pl.DataFrame | None = None,
    securities: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    benchmark: pl.DataFrame | None = None,
    universe: pl.DataFrame | None = None,
    entries: pl.DataFrame | Iterable[tuple[int, dt.date]] | None = None,
    calendar: pl.Series | None = None,
    fundamental_lag: int = 1,
    max_fundamental_age_days: int = 550,
) -> FeatureContext:
    """A FeatureContext for a test. Frames left out are empty (with the core-table schema).

    - `entries` (pairs or a frame) is the output grid. Default: the members of `universe`, which
      itself defaults to `make_universe(prices)`: every (security, month end) that has a price.
      With `entries` but no `universe`, the universe is exactly the entries.
    - `calendar` defaults to the distinct price dates; pass it when there are no prices.
    - `fundamentals` may be the tuples `make_fundamentals` takes.
    """
    if fundamentals is not None and not isinstance(fundamentals, pl.DataFrame):
        fundamentals = make_fundamentals(fundamentals)
    if entries is not None and not isinstance(entries, pl.DataFrame):
        entries = make_entries(entries)
    if entries is None and universe is None:
        if prices is None:
            raise ValueError("make_ctx needs `entries`, `universe` or `prices` to define the output grid")
        universe = make_universe(prices, marketcap=marketcap)
    return context_from_frames(
        prices=prices,
        universe=universe,
        entries=entries,
        fundamentals=fundamentals,
        marketcap=marketcap,
        securities=securities,
        macro=macro,
        benchmark=benchmark,
        calendar=calendar,
        fundamental_lag=fundamental_lag,
        max_fundamental_age_days=max_fundamental_age_days,
    )

"""Market context features (plan 3.1): one value per date, the same for every stock on that date.

- term_spread    DGS10 - DGS3MO (percentage points). For each series: the latest observation that had
                 been published by t (available_date <= t).
- credit_spread  BAA10Y (Moody's Baa yield minus the 10-year Treasury yield), chosen the same way.
- mkt_ret_12m    bench_level(t) / bench_level(252 trading days before t) - 1. Null when the benchmark
                 had no members on the first day of the window (n_members = 0: the level is a
                 placeholder there), when t has fewer than 252 trading days behind it, or when any
                 benchmark day of the window is missing.
- xs_dispersion  standard deviation (ddof = 1) across the universe members at t (the rows of
                 `ctx.entries` on that date) of adj_close(t) / adj_close(21 trading days before t) - 1.
                 A member needs a positive price on both days (nothing is carried forward); at least
                 MIN_DISPERSION_MEMBERS members with a return are required.

Macro point-in-time rules:
- Only rows with available_date <= t are seen, so an observation dated before t but released after
  it does not exist yet. Among the rows seen, the one used has the greatest observation_date; if that
  observation was revised by t, the latest revision published by t is used.
- The series are daily. An observation dated more than MACRO_MAX_AGE_DAYS calendar days before t is
  stale and gives null (the series stopped, or the data was not refreshed).

"Trading days before t" are rows of `ctx.calendar`; a date that is not on the calendar gets nulls for
the two price-based features.

`src_date_context` is the latest input date behind any non-null feature: t when mkt_ret_12m or
xs_dispersion exists, else the latest available_date of the macro observations used; null when all
four features are null.
"""

from __future__ import annotations

import polars as pl

from ltforecast.features.panel import FeatureContext

FAMILIES = ("context",)
FEATURES = {"context": ("term_spread", "credit_spread", "mkt_ret_12m", "xs_dispersion")}

MACRO_MAX_AGE_DAYS = 10
MARKET_WINDOW_DAYS = 252
DISPERSION_WINDOW_DAYS = 21
MIN_DISPERSION_MEMBERS = 20


def _finite(e: pl.Expr) -> pl.Expr:
    """Null instead of NaN or inf."""
    return pl.when(e.is_finite()).then(e)


def _positive(col: str) -> pl.Expr:
    c = pl.col(col)
    return pl.when(c.is_finite() & (c > 0)).then(c)


def _calendar(ctx: FeatureContext) -> pl.DataFrame:
    """Columns: date, _i (position on the trading calendar)."""
    return pl.DataFrame({"date": ctx.calendar}).with_row_index("_i").with_columns(pl.col("_i").cast(pl.Int64))


def _macro_asof(ctx: FeatureContext, dates: pl.DataFrame, series: str) -> pl.DataFrame:
    """Columns: date, <series> (the value known at each date, null if none or stale) and
    <series>_available (the available_date of that value, null when the value is)."""
    obs, avail = pl.col("observation_date"), pl.col("available_date")
    known = (
        ctx.macro.filter((pl.col("series") == series) & pl.col("value").is_finite())
        .select("observation_date", "available_date", "value")
        .drop_nulls()
        .sort("available_date", "observation_date")
        # in order of publication, keep a row only if it is (still) the newest observation: a revision
        # of an older observation never replaces a newer one
        .filter(obs == obs.cum_max())
        .unique("available_date", keep="last", maintain_order=True)
    )
    fresh = (pl.col("date") - obs).dt.total_days() <= MACRO_MAX_AGE_DAYS
    return dates.join_asof(known, left_on="date", right_on="available_date", strategy="backward").select(
        "date",
        pl.when(fresh).then(pl.col("value")).alias(series),
        pl.when(fresh).then(avail).alias(f"{series}_available"),
    )


def _market_return(ctx: FeatureContext, cal: pl.DataFrame) -> pl.DataFrame:
    """Columns: date, mkt_ret_12m, for every calendar day."""
    k = MARKET_WINDOW_DAYS
    level = _positive("bench_level")
    days = pl.col("_level").is_not_null().cast(pl.Int64).cum_sum()
    return (
        cal.join(
            ctx.benchmark.select("date", "bench_level", "n_members").unique("date", keep="first"),
            on="date",
            how="left",
        )
        .sort("_i")
        .with_columns(level.alias("_level"))
        .select(
            "date",
            _finite(
                pl.when((pl.col("n_members").shift(k) > 0) & (days - days.shift(k) == k))
                .then(pl.col("_level") / pl.col("_level").shift(k) - 1)
            ).alias("mkt_ret_12m"),
        )
    )


def _dispersion(ctx: FeatureContext, cal: pl.DataFrame, dates: pl.DataFrame) -> pl.DataFrame:
    """Columns: date, xs_dispersion, for the entry dates that are on the calendar."""
    # each entry date and the trading day 21 rows earlier
    window = (
        dates.join(cal, on="date")
        .with_columns(pl.col("_i") - DISPERSION_WINDOW_DAYS)
        .join(cal.rename({"date": "_start"}), on="_i")
        .select("date", "_start")
    )
    px = ctx.prices.lazy().select("security_id", "date", _positive("adj_close").alias("_px"))
    ret = _finite(pl.col("_px") / pl.col("_px0") - 1)
    return (
        ctx.entries.lazy()
        .select("security_id", "date")
        .join(window.lazy(), on="date")
        .join(px, on=["security_id", "date"])
        .join(
            px.rename({"date": "_start", "_px": "_px0"}), on=["security_id", "_start"]
        )
        .select("date", "security_id", ret.alias("_ret"))
        .drop_nulls()
        .collect()
        # eager and in a fixed row order: the lazy engine sums floats in a run-dependent order, which
        # changes the last bit of the std between two identical calls
        .sort("date", "security_id")
        .group_by("date", maintain_order=True)
        .agg(pl.len().alias("_n"), pl.col("_ret").std(ddof=1).alias("_sd"))
        .select(
            "date",
            _finite(pl.when(pl.col("_n") >= MIN_DISPERSION_MEMBERS).then(pl.col("_sd"))).alias("xs_dispersion"),
        )
    )


def compute(ctx: FeatureContext) -> pl.DataFrame:
    """One row per entry, sorted by security_id, date. Columns: security_id, date, term_spread,
    credit_spread, mkt_ret_12m, xs_dispersion (Float64, null when undefined), src_date_context."""
    keys = ["security_id", "date"]
    dates = ctx.entries.select("date").unique().sort("date")
    cal = _calendar(ctx)

    by_date = dates
    for series in ("DGS10", "DGS3MO", "BAA10Y"):
        by_date = by_date.join(_macro_asof(ctx, dates, series), on="date", how="left")
    by_date = (
        by_date.join(_market_return(ctx, cal), on="date", how="left")
        .join(_dispersion(ctx, cal, dates), on="date", how="left")
        .with_columns(
            _finite(pl.col("DGS10") - pl.col("DGS3MO")).alias("term_spread"),
            pl.col("BAA10Y").alias("credit_spread"),
        )
    )
    has = lambda name: pl.col(name).is_not_null()  # noqa: E731
    by_date = by_date.select(
        "date",
        *[pl.col(c).cast(pl.Float64) for c in FEATURES["context"]],
        pl.max_horizontal(
            pl.when(has("mkt_ret_12m") | has("xs_dispersion")).then(pl.col("date")),
            pl.when(has("term_spread")).then(pl.max_horizontal("DGS10_available", "DGS3MO_available")),
            pl.when(has("credit_spread")).then(pl.col("BAA10Y_available")),
        ).alias("src_date_context"),
    )
    out = ctx.entries.select(keys).join(by_date, on="date", how="left").sort(keys)
    if out.height != ctx.entries.height:
        raise AssertionError(f"context: {out.height} rows for {ctx.entries.height} entries")
    return out

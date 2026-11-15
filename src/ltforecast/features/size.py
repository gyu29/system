"""Size and liquidity features (plan 3.1), from prices and market caps dated <= t only.

- log_mcap  ln(market_cap(t)), market cap in dollars (`panel.market_cap_at`: the latest positive value
            dated <= t and at most 7 calendar days old).
- amihud    mean over the trailing 252 trading days of |ret| / (close x volume), x 1e6: the absolute
            return per million dollars traded. `ret` is the daily total return; `close` and `volume`
            are both split-adjusted, so their product is the true dollar volume. Days with zero (or
            missing) volume, a non-positive close or no return are skipped; at least MIN_AMIHUD_DAYS
            usable days are required.
- turnover  mean daily share volume over the trailing 63 trading days / shares outstanding, where
            shares outstanding = market_cap(t) / close(t). Both sides are in split-adjusted shares.
            Zero-volume days count as zeros; at least MIN_TURNOVER_DAYS days with a volume are required.

Windows are days of the trading calendar up to and including t (not rows of the stock), and the stock
needs a price row at t itself. Raw values are never sign-flipped (the catalog carries the signs).

`src_date_size` is the latest input date behind any non-null feature: t when amihud or turnover
exists, else the date of the market cap used; null when all three are null.
"""

from __future__ import annotations

import polars as pl

from ltforecast.features.panel import FeatureContext, market_cap_at

FAMILIES = ("size",)
FEATURES = {"size": ("log_mcap", "amihud", "turnover")}

AMIHUD_WINDOW_DAYS = 252
MIN_AMIHUD_DAYS = 126
TURNOVER_WINDOW_DAYS = 63
MIN_TURNOVER_DAYS = 42
AMIHUD_SCALE = 1.0e6


def _finite(e: pl.Expr) -> pl.Expr:
    """Null instead of NaN or inf."""
    return pl.when(e.is_finite()).then(e)


def _positive(col: str) -> pl.Expr:
    c = pl.col(col)
    return pl.when(c.is_finite() & (c > 0)).then(c)


def _daily(ctx: FeatureContext) -> pl.DataFrame:
    """Per entry with a price row at t: _amihud, _mean_volume (already null below their minimum counts)
    and _close (the positive split-adjusted close at t)."""
    cal = pl.DataFrame({"date": ctx.calendar}).with_row_index("_i").with_columns(pl.col("_i").cast(pl.Int64))
    ret, vol = _finite(pl.col("ret")), _finite(pl.col("volume"))
    dollar_volume = _positive("close") * _positive("volume")
    count = lambda col, days: (  # noqa: E731
        pl.col(col).is_not_null().cast(pl.Int32).rolling_sum_by("_i", window_size=f"{days}i").over("security_id")
    )
    mean = lambda col, days: (  # noqa: E731
        pl.col(col).rolling_mean_by("_i", window_size=f"{days}i").over("security_id")
    )
    return (
        ctx.prices.lazy()
        .select(
            "security_id",
            "date",
            _positive("close").alias("_close"),
            _finite(ret.abs() / dollar_volume).alias("_illiq"),
            pl.when(vol >= 0).then(vol).alias("_vol"),
        )
        .join(cal.lazy(), on="date")
        .sort("security_id", "_i")
        .with_columns(
            pl.when(count("_illiq", AMIHUD_WINDOW_DAYS) >= MIN_AMIHUD_DAYS)
            .then(mean("_illiq", AMIHUD_WINDOW_DAYS) * AMIHUD_SCALE)
            .alias("_amihud"),
            pl.when(count("_vol", TURNOVER_WINDOW_DAYS) >= MIN_TURNOVER_DAYS)
            .then(mean("_vol", TURNOVER_WINDOW_DAYS))
            .alias("_mean_volume"),
        )
        .join(ctx.entries.lazy().select("security_id", "date"), on=["security_id", "date"], how="semi")
        .select("security_id", "date", "_close", "_amihud", "_mean_volume")
        .collect()
    )


def compute(ctx: FeatureContext) -> pl.DataFrame:
    """One row per entry, sorted by security_id, date. Columns: security_id, date, log_mcap, amihud,
    turnover (Float64, null when undefined), src_date_size."""
    keys = ["security_id", "date"]
    mcap = pl.col("market_cap")
    out = (
        market_cap_at(ctx)
        .join(_daily(ctx), on=keys, how="left")
        .with_columns(
            _finite(pl.when(mcap > 0).then(mcap).log()).alias("log_mcap"),
            _finite(pl.col("_amihud")).alias("amihud"),
            # shares outstanding = market cap / close, so volume / shares = volume x close / market cap
            _finite(pl.col("_mean_volume") * pl.col("_close") / pl.when(mcap > 0).then(mcap)).alias("turnover"),
        )
    )
    uses_prices = pl.col("amihud").is_not_null() | pl.col("turnover").is_not_null()
    uses_mcap = pl.col("log_mcap").is_not_null() | pl.col("turnover").is_not_null()
    out = out.select(
        *keys,
        *[pl.col(c).cast(pl.Float64) for c in FEATURES["size"]],
        pl.max_horizontal(
            pl.when(uses_prices).then(pl.col("date")), pl.when(uses_mcap).then(pl.col("market_cap_date"))
        ).alias("src_date_size"),
    ).sort(keys)
    if out.height != ctx.entries.height:
        raise AssertionError(f"size: {out.height} rows for {ctx.entries.height} entries")
    return out

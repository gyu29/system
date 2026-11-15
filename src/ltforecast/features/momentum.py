"""Momentum and long-term reversal features (plan 3.1), from prices dated <= t only.

momentum
- mom_12_1    P(t-1m) / P(t-12m) - 1
- mom_6_1     P(t-1m) / P(t-6m) - 1
- ind_mom_6m  equal-weight mean of P(t) / P(t-6m) - 1 over the universe members at t in the same industry
- high_52w    P(t) / max of P over the trailing 252 trading days up to and including t

reversal
- rev_36_12   P(t-12m) / P(t-36m) - 1

P is the total-return price (`adj_close`). "t-km" is the last trading day of the calendar month k
months before t (`panel.months_back`). A month return needs a positive price on BOTH of its dates:
nothing is carried forward, so a stock listed 8 months ago has a null mom_12_1 and a stock with no
print on a month end has null returns through that month end.

Conventions worth knowing:
- ind_mom_6m: the industry is `famaindustry`; a stock with no famaindustry is grouped by `sector`
  instead (with the other such stocks of that sector, never mixed into a Fama industry). The members
  are the rows of `ctx.entries` on that date. The mean INCLUDES the stock itself and needs at least
  MIN_INDUSTRY_NAMES members with a 6-month return; a member without a return of its own (recently
  listed) still receives its industry's mean. No industry and no sector -> null.
- high_52w: the window is 252 days of the trading calendar (not 252 rows of the stock), and needs
  at least MIN_HIGH_DAYS prices in it, including one at t.
- Raw values are never sign-flipped: rev_36_12 is the plain past return (the catalog carries sign -1).

`src_date_<family>` is the latest price date that entered any non-null feature of the family
(t-1 month if only the skip-month returns exist, t-12 months for reversal), null if all are null.
"""

from __future__ import annotations

import polars as pl

from ltforecast.features.panel import FeatureContext, month_end_prices, months_back, sector_map

FAMILIES = ("momentum", "reversal")
FEATURES = {
    "momentum": ("mom_12_1", "mom_6_1", "ind_mom_6m", "high_52w"),
    "reversal": ("rev_36_12",),
}

HIGH_WINDOW_DAYS = 252
MIN_HIGH_DAYS = 126
MIN_INDUSTRY_NAMES = 5
_MONTHS = (1, 6, 12, 36)


def _positive(col: str) -> pl.Expr:
    """The price if it is a finite positive number, else null (NaN, inf, zero and negatives are missing)."""
    c = pl.col(col)
    return pl.when(c.is_finite() & (c > 0)).then(c)


def _finite(e: pl.Expr) -> pl.Expr:
    return pl.when(e.is_finite()).then(e)


def _ret(end: str, start: str) -> pl.Expr:
    return _finite(pl.col(end) / pl.col(start) - 1)


def _month_end_panel(ctx: FeatureContext) -> pl.DataFrame:
    """Per entry: date_<k> and the price on it (p_<k>) for k in _MONTHS, plus p_0 (the price at t)."""
    px = month_end_prices(ctx).select("security_id", "date", _positive("adj_close").alias("p")).drop_nulls("p")
    out = ctx.entries.select("security_id", "date").join(
        px.rename({"p": "p_0"}), on=["security_id", "date"], how="left"
    )
    for k in _MONTHS:
        out = out.join(months_back(ctx, k).rename({"date_k": f"date_{k}"}), on="date", how="left").join(
            px.rename({"date": f"date_{k}", "p": f"p_{k}"}), on=["security_id", f"date_{k}"], how="left"
        )
    return out


def _industry_momentum(ctx: FeatureContext, panel: pl.DataFrame) -> pl.DataFrame:
    """security_id, date, ind_mom_6m for every row of `panel` (the entries)."""
    sectors = sector_map(ctx).select(
        "security_id",
        pl.when(pl.col("famaindustry").is_not_null())
        .then(pl.concat_str(pl.lit("fama:"), pl.col("famaindustry")))
        .when(pl.col("sector").is_not_null())
        .then(pl.concat_str(pl.lit("sector:"), pl.col("sector")))
        .alias("_group"),
    )
    r6 = pl.col("_r6")
    enough = r6.count().over("date", "_group") >= MIN_INDUSTRY_NAMES
    return (
        panel.select("security_id", "date", _ret("p_0", "p_6").alias("_r6"))
        .join(sectors, on="security_id", how="left")
        .with_columns(
            pl.when(pl.col("_group").is_not_null() & enough).then(r6.mean().over("date", "_group")).alias("ind_mom_6m")
        )
        .select("security_id", "date", _finite(pl.col("ind_mom_6m")).alias("ind_mom_6m"))
    )


def _high_52w(ctx: FeatureContext) -> pl.DataFrame:
    """security_id, date, high_52w on the entry rows that have a price at t."""
    cal = pl.DataFrame({"date": ctx.calendar}).with_row_index("_i").with_columns(pl.col("_i").cast(pl.Int64))
    window = f"{HIGH_WINDOW_DAYS}i"  # rows with calendar index in (i - 252, i]
    p = pl.col("p")
    return (
        ctx.prices.lazy()
        .select("security_id", "date", _positive("adj_close").alias("p"))
        .drop_nulls("p")
        .join(cal.lazy(), on="date")
        .sort("security_id", "_i")
        .with_columns(
            p.rolling_max_by("_i", window_size=window).over("security_id").alias("_max"),
            p.is_not_null().cast(pl.Int32).rolling_sum_by("_i", window_size=window).over("security_id").alias("_n"),
        )
        .join(ctx.entries.lazy().select("security_id", "date"), on=["security_id", "date"], how="semi")
        .select(
            "security_id",
            "date",
            pl.when(pl.col("_n") >= MIN_HIGH_DAYS).then(_finite(p / pl.col("_max"))).alias("high_52w"),
        )
        .collect()
    )


def compute(ctx: FeatureContext) -> pl.DataFrame:
    """One row per entry, sorted by security_id, date. Columns: security_id, date, mom_12_1, mom_6_1,
    ind_mom_6m, high_52w, rev_36_12 (Float64, null when undefined), src_date_momentum,
    src_date_reversal."""
    panel = _month_end_panel(ctx)
    keys = ["security_id", "date"]
    out = (
        panel.with_columns(
            _ret("p_1", "p_12").alias("mom_12_1"),
            _ret("p_1", "p_6").alias("mom_6_1"),
            _ret("p_12", "p_36").alias("rev_36_12"),
        )
        .join(_industry_momentum(ctx, panel), on=keys, how="left")
        .join(_high_52w(ctx), on=keys, how="left")
    )
    skip_month = pl.col("mom_12_1").is_not_null() | pl.col("mom_6_1").is_not_null()
    through_t = pl.col("ind_mom_6m").is_not_null() | pl.col("high_52w").is_not_null()
    out = out.select(
        *keys,
        *[pl.col(c).cast(pl.Float64) for fam in FAMILIES for c in FEATURES[fam]],
        pl.max_horizontal(
            pl.when(skip_month).then(pl.col("date_1")), pl.when(through_t).then(pl.col("date"))
        ).alias("src_date_momentum"),
        pl.when(pl.col("rev_36_12").is_not_null()).then(pl.col("date_12")).alias("src_date_reversal"),
    ).sort(keys)
    if out.height != ctx.entries.height:
        raise AssertionError(f"momentum: {out.height} rows for {ctx.entries.height} entries")
    return out

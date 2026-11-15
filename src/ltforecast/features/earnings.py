"""Earnings-dynamics features (plan 3.1): sue, eps_growth_stability.

Both read SF1 `eps` through `panel.fundamentals_asof`, so a value is only ever taken from a filing
that was usable at t. Lags are looked up by date: a lag is null only when the period it points at is
missing or was not yet known at the filing.

EPS and splits
--------------
SF1 `eps` is as first reported, and later splits do not restate it, so a 2:1 split halves EPS with
unchanged earnings. A split made after the period end but before the statements are issued IS in
the filed figure (ASC 260-10-55-12). Convention used here (the same one features/investment.py uses
for share counts):

- the EPS filed on day F (its `available_date`) reflects every split with ex-date < F and none on or
  after F; a split on the filing day itself is taken as not yet in the figures;
- so EPS is put on a common basis by multiplying it by the product of `prices.split_factor` over the
  security's price rows dated < F (`split_adjusted_eps_context`).

That is the basis before any split rather than the basis of the latest period. The two differ by one
factor per (security, latest period), and both features are ratios of EPS to EPS, so the result is
the same. Each value reads only splits dated before its own filing date. `fundamentals_asof` only
serves filings with available_date <= t, so nothing after t is used, and a split after the latest
filing changes nothing.

Residual: a company that does not restate for a split made between the period end and the filing
(for example because the split takes effect after issuance) is misadjusted for that filing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import polars as pl

from ltforecast.features.panel import FeatureContext, fundamentals_asof

FAMILIES = ("earnings",)
FEATURES = ("sue", "eps_growth_stability")
METRICS = ("eps",)  # the SF1 indicators this module reads (for make_context)

SUE_WINDOW = 8        # year-over-year changes in the denominator: the 8 BEFORE the latest one
SUE_MIN_CHANGES = 6   # at least this many of them must exist
ANNUAL_YEARS = 4      # fiscal years of annual EPS; all are required (3 year-over-year changes)
# A standard deviation this small relative to the EPS level is floating-point noise around zero
# (as-reported EPS has two or three decimals, so a real one is many orders of magnitude larger).
ZERO_STD_REL_TOL = 1e-9

_FILING_MARKER = "__filing__"
_KEYS = ["security_id", "date"]
_COLS = ["security_id", "dimension", "metric", "fiscal_period_end", "available_date", "value"]


def split_adjusted_eps_context(ctx: FeatureContext, dimensions: Sequence[str] = ("ARQ", "ARY")) -> FeatureContext:
    """`ctx` with fundamentals cut down to split-adjusted `eps` of `dimensions`.

    Each EPS is multiplied by the cumulative split factor of the splits dated before its own filing
    date (see the module docstring). One marker row per fiscal period carries the period's filing date (the earliest
    available_date of any metric), so `fundamentals_asof` on the result dates filings exactly as it
    does on `ctx`.
    """
    fund = ctx.fundamentals.lazy().filter(pl.col("dimension").is_in(list(dimensions)))
    marker, eps = pl.collect_all([
        fund.group_by("security_id", "dimension", "fiscal_period_end").agg(pl.col("available_date").min()),
        fund.filter(pl.col("metric") == "eps").select(_COLS),
    ])
    marker = marker.select(
        "security_id", "dimension", pl.lit(_FILING_MARKER).alias("metric"), "fiscal_period_end",
        "available_date", pl.lit(None, pl.Float64).alias("value"),
    )
    splits = (
        ctx.prices.lazy()
        .filter(pl.col("split_factor").is_not_null() & (pl.col("split_factor") != 1.0))
        .select("security_id", pl.col("date").alias("split_date"), "split_factor")
        .sort("security_id", "split_date")
        .with_columns(pl.col("split_factor").cum_prod().over("security_id").alias("_cum"))
        .collect()
    )
    eps = (
        eps.sort("security_id", "available_date")
        .join_asof(
            splits, left_on="available_date", right_on="split_date", by="security_id",
            strategy="backward", allow_exact_matches=False, check_sortedness=False,
        )
        .with_columns((pl.col("value") * pl.col("_cum").fill_null(1.0)).alias("value"))
        .select(_COLS)
    )
    return dataclasses.replace(ctx, fundamentals=pl.concat([marker, eps]))


def _lag(metric: str, k: int) -> pl.Expr:
    return pl.col(metric if k == 0 else f"{metric}_lag{k}")


def _std(values: Sequence[pl.Expr]) -> tuple[pl.Expr, pl.Expr]:
    """(sample standard deviation across the non-null `values` of a row, how many there are)."""
    n = pl.sum_horizontal([v.is_not_null() for v in values])
    mean = pl.sum_horizontal(values) / n
    ss = pl.sum_horizontal([(v - mean) ** 2 for v in values])
    return pl.when(n >= 2).then((ss / (n - 1)).sqrt()), n


def _finite(e: pl.Expr) -> pl.Expr:
    return pl.when(e.is_finite()).then(e).cast(pl.Float64)


def compute(ctx: FeatureContext) -> pl.DataFrame:
    """Columns: security_id, date, sue, eps_growth_stability (Float64), src_date_earnings.

    One row per row of `ctx.entries`, sorted by security_id, date.

    - sue (Bernard & Thomas 1989, seasonal random walk): (EPS of the latest usable quarter - EPS four
      quarters earlier) / sample std (ddof=1) of the 8 PRIOR year-over-year changes (lags 1..8; the
      latest change is excluded so a large surprise does not inflate its own scale, which would cap
      sue near sqrt(8) and make it non-monotone). Null if the latest change is missing, if fewer than
      6 of the 8 prior changes exist, or if their std is zero. Reads quarterly EPS (ARQ) at lags 0..12.
    - eps_growth_stability: sample std (ddof=1) of the 3 year-over-year changes in annual EPS (ARY)
      over the last 4 fiscal years / the mean |EPS| of those 4 years. Null unless all 4 years exist
      and that mean is > 0. A low value is a steady earnings path (0 = the same change every year).
    - src_date_earnings: the latest filing date behind the non-null features of the row.

    EPS is split-adjusted first (module docstring).
    """
    adj = split_adjusted_eps_context(ctx)
    arq = fundamentals_asof(adj, "ARQ", ["eps"], range(SUE_WINDOW + 5)).rename({"src_date": "_src_q"})
    ary = fundamentals_asof(adj, "ARY", ["eps"], range(ANNUAL_YEARS)).rename({"src_date": "_src_y"})
    ary = ary.rename({c: f"a_{c}" for c in ary.columns if c.startswith("eps")})

    q_changes = [_lag("eps", j) - _lag("eps", j + 4) for j in range(SUE_WINDOW + 1)]
    q_std, q_n = _std(q_changes[1:])  # prior changes only
    q_scale = pl.max_horizontal([_lag("eps", j).abs() for j in range(SUE_WINDOW + 5)])
    sue = pl.when(
        q_changes[0].is_not_null() & (q_n >= SUE_MIN_CHANGES) & (q_std > ZERO_STD_REL_TOL * q_scale)
    ).then(q_changes[0] / q_std)

    years = [_lag("a_eps", j) for j in range(ANNUAL_YEARS)]
    y_changes = [years[j] - years[j + 1] for j in range(ANNUAL_YEARS - 1)]
    y_std, y_n = _std(y_changes)
    y_level = pl.sum_horizontal([y.abs() for y in years]) / ANNUAL_YEARS
    stability = pl.when((y_n == ANNUAL_YEARS - 1) & (y_level > 0)).then(y_std / y_level)

    out = (
        ctx.entries.select(_KEYS)
        .join(arq.drop("fiscal_period_end"), on=_KEYS, how="left")
        .join(ary.drop("fiscal_period_end"), on=_KEYS, how="left")
        .with_columns(_finite(sue).alias("sue"), _finite(stability).alias("eps_growth_stability"))
    )
    src = pl.max_horizontal(
        pl.when(pl.col("sue").is_not_null()).then(pl.col("_src_q")),
        pl.when(pl.col("eps_growth_stability").is_not_null()).then(pl.col("_src_y")),
    )
    return out.select(*_KEYS, *FEATURES, src.alias("src_date_earnings")).sort(_KEYS)

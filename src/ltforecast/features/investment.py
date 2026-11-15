"""Investment / issuance features (plan 3.1): asset_growth, net_issuance, capex_growth.

Definitions are those of config/features.yaml. All three compare the latest quarterly filing usable
at t with the period four quarters earlier (`panel.fundamentals_asof`, lag 4, looked up by date), so
they are null until five quarters of filings exist, and a lag is null only when the period it points at
is missing or was not yet known at the filing.

Share counts and splits
-----------------------
SF1 share counts (`sharesbas`, `shareswa`) are as first reported, and later splits do not restate
them. A split made after the period end but before the filing IS in the filed count: weighted shares
are restated for it (ASC 260-10-55-12), and the cover-page `sharesbas` is measured near the filing
date. Convention used here (plan 1.3 "first reported"):

- a count filed on day F (its `available_date`) reflects every split with ex-date < F and none on
  or after F; a split on the filing day itself is taken as not yet in the figures;
- so a count is put on a common basis by dividing it by the product of `prices.split_factor` over
  the security's price rows dated < F (`split_adjusted_context`).

Consequences: a split between the two filings is removed (100 shares -> 2:1 split -> 200 shares is
net_issuance 0), including one between the latest period end and its filing. A split after the
latest filing, even one before t, changes neither count and needs no adjustment. Each value reads
only splits dated before its own filing date. `fundamentals_asof` only serves filings with
available_date <= t, so nothing after t is used. Residual: a count not yet on the basis of a split
made before its filing (a vendor or company that does not restate) shows that split as issuance.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import polars as pl

from ltforecast.features.panel import FeatureContext, fundamentals_asof

FAMILIES: dict[str, tuple[str, ...]] = {"investment": ("asset_growth", "net_issuance", "capex_growth")}

_FILING_MARKER = "__filing__"
_KEYS = ["security_id", "date"]


def split_adjusted_context(
    ctx: FeatureContext, dimension: str, share_metrics: Sequence[str], other_metrics: Sequence[str] = ()
) -> FeatureContext:
    """`ctx` with fundamentals cut down to one dimension and `share_metrics` put on a common share basis.

    Each share metric is divided by the cumulative split factor of the splits dated before its own
    filing date (see the module docstring), so ratios of the adjusted counts across periods ignore splits. `other_metrics`
    are kept unchanged. One marker row per fiscal period carries the period's filing date (the earliest
    available_date of any metric), so `fundamentals_asof` on the result dates filings exactly as it does
    on `ctx`.
    """
    fund = ctx.fundamentals.lazy().filter(pl.col("dimension") == dimension)
    cols = ["security_id", "dimension", "metric", "fiscal_period_end", "available_date", "value"]
    marker, shares, others = pl.collect_all([
        fund.group_by("security_id", "dimension", "fiscal_period_end").agg(pl.col("available_date").min()),
        fund.filter(pl.col("metric").is_in(list(share_metrics))).select(cols),
        fund.filter(pl.col("metric").is_in(list(other_metrics))).select(cols),
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
    shares = (
        shares.sort("security_id", "available_date")
        .join_asof(
            splits, left_on="available_date", right_on="split_date", by="security_id",
            strategy="backward", allow_exact_matches=False, check_sortedness=False,
        )
        .with_columns((pl.col("value") / pl.col("_cum").fill_null(1.0)).alias("value"))
        .select(cols)
    )
    return dataclasses.replace(ctx, fundamentals=pl.concat([marker, shares, others]))


def _finite(e: pl.Expr) -> pl.Expr:
    return pl.when(e.is_finite()).then(e).cast(pl.Float64)


def src_date(features: dict[str, Sequence[str]], name: str) -> pl.Expr:
    """Latest filing date among the sources of the features that are not null (null if all are).
    `features` maps each feature column to the src-date columns of the filings it reads."""
    used = [pl.when(pl.col(f).is_not_null()).then(pl.max_horizontal(srcs)) for f, srcs in features.items()]
    return pl.max_horizontal(used).alias(name)


def compute(ctx: FeatureContext) -> pl.DataFrame:
    """One row per entry: security_id, date, asset_growth, net_issuance, capex_growth, src_date_investment.

    - asset_growth: assets (ARQ) / assets four quarters earlier - 1; null if the earlier value is <= 0.
    - net_issuance: log(split-adjusted sharesbas (ARQ) / the same four quarters earlier); null unless
      both counts are > 0.
    - capex_growth: |capex (ART)| / |capex (ART) four quarters earlier| - 1; null if the earlier value is 0.
    - src_date_investment: the latest filing date behind the non-null features of the row.
    """
    arq = fundamentals_asof(
        split_adjusted_context(ctx, "ARQ", ["sharesbas"], ["assets"]), "ARQ", ["assets", "sharesbas"], (0, 4)
    ).rename({"src_date": "_src_q"})
    art = fundamentals_asof(ctx, "ART", ["capex"], (0, 4)).rename({"src_date": "_src_t"})

    a0, a4 = pl.col("assets"), pl.col("assets_lag4")
    s0, s4 = pl.col("sharesbas"), pl.col("sharesbas_lag4")
    c0, c4 = pl.col("capex").abs(), pl.col("capex_lag4").abs()
    out = (
        ctx.entries.select(_KEYS)
        .join(arq.drop("fiscal_period_end"), on=_KEYS, how="left")
        .join(art.drop("fiscal_period_end"), on=_KEYS, how="left")
        .with_columns(
            _finite(pl.when(a4 > 0).then(a0 / a4 - 1)).alias("asset_growth"),
            _finite(pl.when((s0 > 0) & (s4 > 0)).then((s0 / s4).log())).alias("net_issuance"),
            _finite(pl.when(c4 > 0).then(c0 / c4 - 1)).alias("capex_growth"),
        )
    )
    sources = {"asset_growth": ["_src_q"], "net_issuance": ["_src_q"], "capex_growth": ["_src_t"]}
    return out.select(*_KEYS, *FAMILIES["investment"], src_date(sources, "src_date_investment")).sort(_KEYS)

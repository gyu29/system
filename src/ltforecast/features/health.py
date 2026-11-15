"""Financial-health features (plan 3.1; Campbell, Hilscher & Szilagyi 2008; Altman 1968).

    debt_to_assets     debt (ARQ) / assets (ARQ)
    cash_to_assets     cashneq (ARQ) / assets (ARQ)
    interest_coverage  ebit (ART) / |intexp (ART)|
    altman_z           1.2 workingcapital/assets + 1.4 retearn/assets + 3.3 ebit/assets
                       + 0.6 market_cap(t)/liabilities + 1.0 revenue/assets

Balance-sheet items come from the latest usable quarterly filing (ARQ), EBIT, interest expense and
revenue from the latest trailing-twelve-month figures (ART), both through `panel.fundamentals_asof`.
The market value of equity in the Z-score is the market cap at the prediction date
(`panel.market_cap_at`), not the one on the balance-sheet date.

Conventions
- Total assets <= 0 is not a balance sheet: every ratio over assets is null.
- interest_coverage is null when interest expense is zero (a firm with no debt service has no
  coverage ratio; it is not "infinitely safe" in the data). The absolute value makes the sign of
  the vendor's `intexp` irrelevant; a negative EBIT gives a negative coverage, which is kept.
- altman_z needs all five terms: it is null if any input is missing, if assets <= 0 or if total
  liabilities <= 0. A high score means low distress risk.
- Output is Float64 with null for missing, never NaN or inf.
"""

from __future__ import annotations

import polars as pl

from ltforecast.features.panel import FeatureContext, fundamentals_asof, market_cap_at

FAMILIES = ("health",)
FEATURES = ("debt_to_assets", "interest_coverage", "altman_z", "cash_to_assets")
ARQ_METRICS = ("debt", "assets", "cashneq", "workingcapital", "retearn", "liabilities")
ART_METRICS = ("ebit", "intexp", "revenue")
METRICS = ARQ_METRICS + ART_METRICS  # the SF1 indicators this module reads (for make_context)
SRC_DATE = "src_date_health"

# which features read which input: decides what counts as "used" for the source date
_USES_ARQ = ("debt_to_assets", "cash_to_assets", "altman_z")
_USES_ART = ("interest_coverage", "altman_z")
_USES_MCAP = ("altman_z",)
_KEYS = ["security_id", "date"]


def _ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """num / den where the denominator is strictly positive, else null."""
    return pl.when(den > 0).then(num / den)


def _any_set(cols: tuple[str, ...]) -> pl.Expr:
    return pl.any_horizontal([pl.col(c).is_not_null() for c in cols])


def compute(ctx: FeatureContext) -> pl.DataFrame:
    """Columns: security_id, date, the four FEATURES (Float64), src_date_health.

    One row per row of `ctx.entries`, sorted by security_id, date. `src_date_health` is the latest
    date among the inputs that produced a non-null feature (filing dates, and the market-cap date
    when altman_z is set); it is null when every feature is null and never after `date`.
    """
    arq = fundamentals_asof(ctx, "ARQ", ARQ_METRICS).select(
        *_KEYS, pl.col("src_date").alias("_arq_date"), *ARQ_METRICS
    )
    art = fundamentals_asof(ctx, "ART", ART_METRICS).select(
        *_KEYS, pl.col("src_date").alias("_art_date"), *ART_METRICS
    )
    assets = pl.col("assets")
    altman = (
        1.2 * _ratio(pl.col("workingcapital"), assets)
        + 1.4 * _ratio(pl.col("retearn"), assets)
        + 3.3 * _ratio(pl.col("ebit"), assets)
        + 0.6 * _ratio(pl.col("market_cap"), pl.col("liabilities"))
        + 1.0 * _ratio(pl.col("revenue"), assets)
    )
    raw = {
        "debt_to_assets": _ratio(pl.col("debt"), assets),
        "interest_coverage": _ratio(pl.col("ebit"), pl.col("intexp").abs()),
        "altman_z": altman,
        "cash_to_assets": _ratio(pl.col("cashneq"), assets),
    }
    return (
        ctx.entries.select(_KEYS)
        .join(market_cap_at(ctx), on=_KEYS, how="left")
        .join(arq, on=_KEYS, how="left")
        .join(art, on=_KEYS, how="left")
        .with_columns([e.cast(pl.Float64).alias(name) for name, e in raw.items()])
        # a NaN or inf in the vendor data must not survive as a feature value
        .with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).alias(c) for c in FEATURES])
        .with_columns(
            pl.max_horizontal(
                pl.when(_any_set(_USES_MCAP)).then(pl.col("market_cap_date")),
                pl.when(_any_set(_USES_ARQ)).then(pl.col("_arq_date")),
                pl.when(_any_set(_USES_ART)).then(pl.col("_art_date")),
            ).alias(SRC_DATE)
        )
        .select(*_KEYS, *FEATURES, SRC_DATE)
        .sort(_KEYS)
    )

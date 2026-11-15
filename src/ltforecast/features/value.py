"""Value features (plan 3.1): fundamentals from the latest usable filing over the price at t.

    book_to_price      equity (ARQ) / market_cap(t)
    earnings_yield     netinc (ART) / market_cap(t)
    fcf_yield          fcf (ART) / market_cap(t)
    ebitda_to_ev       ebitda (ART) / EV(t)
    sales_to_ev        revenue (ART) / EV(t)
    shareholder_yield  -(ncfdiv + ncfcommon) (ART) / market_cap(t)

    EV(t) = market_cap(t) + debt (ARQ) - cashneq (ARQ)

Market cap and EV are measured at the prediction date (`panel.market_cap_at`: the latest positive
value dated <= t, at most 7 days old); balance-sheet items come from the latest quarterly filing
(ARQ), flows from the latest trailing-twelve-month figures (ART), both through
`panel.fundamentals_asof` (usable from the trading day after the filing date, null once stale).

Conventions
- Negative numerators are kept. A loss gives a negative earnings yield, and negative book equity
  gives a negative book-to-price: the stock then ranks at the bottom of the cross-section, next to
  the most expensive ones. That is deliberate: negative-equity firms are mostly distressed or highly
  levered, and dropping them (the Fama-French convention) would turn a real state into a missing one.
- EV(t) <= 0 (net cash larger than the market cap) makes the EV ratios meaningless: null.
- Sharadar cash-flow signs: `ncfdiv` is negative when dividends are paid, `ncfcommon` is negative for
  net buybacks and positive for net issuance. shareholder_yield is the cash handed back, so a dividend
  payer and a repurchaser are positive and a net issuer is negative.
- No input is guessed: a feature is null when any of its inputs is missing (a filing without `debt`
  gives null EV ratios, one without `ncfdiv` a null shareholder yield).
- Output is Float64 with null for missing, never NaN or inf.
"""

from __future__ import annotations

import polars as pl

from ltforecast.features.panel import FeatureContext, fundamentals_asof, market_cap_at

FAMILIES = ("value",)
FEATURES = (
    "book_to_price", "earnings_yield", "fcf_yield", "ebitda_to_ev", "sales_to_ev", "shareholder_yield",
)
ARQ_METRICS = ("equity", "debt", "cashneq")
ART_METRICS = ("netinc", "fcf", "ebitda", "revenue", "ncfdiv", "ncfcommon")
METRICS = ARQ_METRICS + ART_METRICS  # the SF1 indicators this module reads (for make_context)
SRC_DATE = "src_date_value"

# which features read which input: decides what counts as "used" for the source date
_USES_ARQ = ("book_to_price", "ebitda_to_ev", "sales_to_ev")
_USES_ART = ("earnings_yield", "fcf_yield", "ebitda_to_ev", "sales_to_ev", "shareholder_yield")
_KEYS = ["security_id", "date"]


def _ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """num / den where the denominator is strictly positive, else null."""
    return pl.when(den > 0).then(num / den)


def _any_set(cols: tuple[str, ...]) -> pl.Expr:
    return pl.any_horizontal([pl.col(c).is_not_null() for c in cols])


def compute(ctx: FeatureContext) -> pl.DataFrame:
    """Columns: security_id, date, the six FEATURES (Float64), src_date_value.

    One row per row of `ctx.entries`, sorted by security_id, date. `src_date_value` is the latest
    date among the inputs that produced a non-null feature (filing dates and the market-cap date);
    it is null when every feature is null and never after `date`.
    """
    arq = fundamentals_asof(ctx, "ARQ", ARQ_METRICS).select(
        *_KEYS, pl.col("src_date").alias("_arq_date"), *ARQ_METRICS
    )
    art = fundamentals_asof(ctx, "ART", ART_METRICS).select(
        *_KEYS, pl.col("src_date").alias("_art_date"), *ART_METRICS
    )
    mcap = pl.col("market_cap")
    ev = mcap + pl.col("debt") - pl.col("cashneq")
    raw = {
        "book_to_price": _ratio(pl.col("equity"), mcap),
        "earnings_yield": _ratio(pl.col("netinc"), mcap),
        "fcf_yield": _ratio(pl.col("fcf"), mcap),
        "ebitda_to_ev": _ratio(pl.col("ebitda"), ev),
        "sales_to_ev": _ratio(pl.col("revenue"), ev),
        "shareholder_yield": _ratio(-(pl.col("ncfdiv") + pl.col("ncfcommon")), mcap),
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
                pl.when(_any_set(FEATURES)).then(pl.col("market_cap_date")),
                pl.when(_any_set(_USES_ARQ)).then(pl.col("_arq_date")),
                pl.when(_any_set(_USES_ART)).then(pl.col("_art_date")),
            ).alias(SRC_DATE)
        )
        .select(*_KEYS, *FEATURES, SRC_DATE)
        .sort(_KEYS)
    )

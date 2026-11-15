"""Profitability / quality features (plan 3.1): gross_profitability, roic, op_margin_stability,
piotroski_f, accruals.

Definitions are those of config/features.yaml. Flows are trailing twelve months (ART), balance-sheet
items come from the latest quarter (ARQ), and the two multi-year features read fiscal years (ARY).
Every input is the latest filing usable at t (`panel.fundamentals_asof`).

Piotroski F-score (Piotroski 2000), latest fiscal year (0) against the one before (1). Each signal
is 1 when true:

  1. return on assets > 0             netinc0 / assets0 > 0
  2. operating cash flow > 0          ncfo0 / assets0 > 0
  3. return on assets up              netinc0 / assets0 > netinc1 / assets1
  4. cash flow exceeds earnings       ncfo0 > netinc0
  5. leverage down                    debtnc0 / assets0 < debtnc1 / assets1
  6. current ratio up                 assetsc0 / liabilitiesc0 > assetsc1 / liabilitiesc1
  7. no new shares                    split-adjusted shareswa0 <= shareswa1
  8. gross margin up                  gp0 / revenue0 > gp1 / revenue1
  9. asset turnover up                revenue0 / assets0 > revenue1 / assets1

Ratios use the same year's closing assets (the paper scales by opening assets, which would need a
third year). The score is null unless all nine signals can be computed: every input present in both
years (except the earlier year's operating cash flow, which no signal reads) and assets, current
liabilities, revenue and share counts > 0 in both. Companies that do not
report current assets / liabilities (banks, insurers) therefore have no F-score. Share counts are
split-adjusted as in features/investment.py, so a split alone is not an issuance.
"""

from __future__ import annotations

import polars as pl

from ltforecast.features.investment import _finite, split_adjusted_context, src_date
from ltforecast.features.panel import FeatureContext, fundamentals_asof

FAMILIES: dict[str, tuple[str, ...]] = {
    "quality": ("gross_profitability", "roic", "op_margin_stability", "piotroski_f", "accruals")
}

TAX_RATE = 0.21        # flat statutory rate in the roic definition
MARGIN_YEARS = 5       # op_margin_stability window (fiscal years)
MARGIN_MIN_YEARS = 3

_KEYS = ["security_id", "date"]
_ARY = ["netinc", "ncfo", "assets", "debtnc", "assetsc", "liabilitiesc", "gp", "revenue", "opinc"]


def _lag(metric: str, k: int) -> pl.Expr:
    return pl.col(metric if k == 0 else f"{metric}_lag{k}")


def _margin_std() -> pl.Expr:
    """Sample std (ddof=1) of opinc / revenue over the fiscal years of the window that have a margin
    (revenue > 0); null with fewer than MARGIN_MIN_YEARS of them."""
    margins = [
        pl.when(_lag("revenue", k) > 0).then(_lag("opinc", k) / _lag("revenue", k)) for k in range(MARGIN_YEARS)
    ]
    n = pl.sum_horizontal([m.is_not_null().cast(pl.Int32) for m in margins])
    mean = pl.sum_horizontal(margins) / n
    ss = pl.sum_horizontal([(m - mean) ** 2 for m in margins])  # null margins are skipped
    return pl.when(n >= MARGIN_MIN_YEARS).then((ss / (n - 1)).sqrt())


def _piotroski() -> pl.Expr:
    c = _lag
    roa0, roa1 = c("netinc", 0) / c("assets", 0), c("netinc", 1) / c("assets", 1)
    signals = [
        roa0 > 0,
        c("ncfo", 0) / c("assets", 0) > 0,
        roa0 > roa1,
        c("ncfo", 0) > c("netinc", 0),
        c("debtnc", 0) / c("assets", 0) < c("debtnc", 1) / c("assets", 1),
        c("assetsc", 0) / c("liabilitiesc", 0) > c("assetsc", 1) / c("liabilitiesc", 1),
        c("shareswa", 0) <= c("shareswa", 1),
        c("gp", 0) / c("revenue", 0) > c("gp", 1) / c("revenue", 1),
        c("revenue", 0) / c("assets", 0) > c("revenue", 1) / c("assets", 1),
    ]
    positive = pl.all_horizontal(
        [c(m, k) > 0 for m in ("assets", "liabilitiesc", "revenue", "shareswa") for k in (0, 1)]
    )
    # a comparison with a null input is null, and a null term makes the whole sum null
    score = signals[0].cast(pl.Float64)
    for s in signals[1:]:
        score = score + s.cast(pl.Float64)
    return pl.when(positive).then(score)


def compute(ctx: FeatureContext) -> pl.DataFrame:
    """One row per entry: security_id, date, the five quality features, src_date_quality.

    - gross_profitability: gp (ART) / assets (ARQ); null if assets <= 0.
    - roic: ebit (ART) x (1 - 0.21) / invcap (ARQ); null if invcap <= 0.
    - op_margin_stability: std (ddof=1) of opinc / revenue (ARY) over the last 5 fiscal years, at least
      3 with revenue > 0. Low = stable.
    - piotroski_f: 0..9, see the module docstring; null unless all nine signals are computable.
    - accruals: (netinc - ncfo) (ART) / assets (ARQ); null if assets <= 0.
    - src_date_quality: the latest filing date behind the non-null features of the row.
    """
    arq = fundamentals_asof(ctx, "ARQ", ["assets", "invcap"]).select(
        *_KEYS, pl.col("src_date").alias("_src_q"), pl.col("assets").alias("_assets_q"), "invcap"
    )
    art = fundamentals_asof(ctx, "ART", ["gp", "ebit", "netinc", "ncfo"]).select(
        *_KEYS, pl.col("src_date").alias("_src_t"), pl.col("gp").alias("_gp_t"), "ebit",
        pl.col("netinc").alias("_netinc_t"), pl.col("ncfo").alias("_ncfo_t"),
    )
    ary = fundamentals_asof(
        split_adjusted_context(ctx, "ARY", ["shareswa"], _ARY), "ARY", [*_ARY, "shareswa"], range(MARGIN_YEARS)
    ).rename({"src_date": "_src_y"}).drop("fiscal_period_end")

    assets_q = pl.col("_assets_q")
    out = (
        ctx.entries.select(_KEYS)
        .join(arq, on=_KEYS, how="left")
        .join(art, on=_KEYS, how="left")
        .join(ary, on=_KEYS, how="left")
        .with_columns(
            _finite(pl.when(assets_q > 0).then(pl.col("_gp_t") / assets_q)).alias("gross_profitability"),
            _finite(pl.when(pl.col("invcap") > 0).then(pl.col("ebit") * (1 - TAX_RATE) / pl.col("invcap"))).alias("roic"),
            _finite(_margin_std()).alias("op_margin_stability"),
            _finite(_piotroski()).alias("piotroski_f"),
            _finite(pl.when(assets_q > 0).then((pl.col("_netinc_t") - pl.col("_ncfo_t")) / assets_q)).alias("accruals"),
        )
    )
    sources = {
        "gross_profitability": ["_src_q", "_src_t"],
        "roic": ["_src_q", "_src_t"],
        "op_margin_stability": ["_src_y"],
        "piotroski_f": ["_src_y"],
        "accruals": ["_src_q", "_src_t"],
    }
    return out.select(*_KEYS, *FAMILIES["quality"], src_date(sources, "src_date_quality")).sort(_KEYS)

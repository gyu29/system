"""Assemble the feature tables (plan section 3): family modules -> features_raw -> features.

`features_raw` keeps the unranked values and one `src_date_<family>` column per family (the latest
date of any input behind the row's non-null features), which the leakage test reads.
`features` is what the models see: `transform.transform` of the raw table.
"""

from __future__ import annotations

from types import ModuleType

import polars as pl

from ltforecast.features import context, earnings, health, investment, momentum, quality, risk, size, value
from ltforecast.features.catalog import Catalog, load_catalog
from ltforecast.features.leakage import SRC_PREFIX, assert_sources_not_after
from ltforecast.features.panel import FeatureContext, sector_map
from ltforecast.features.transform import transform

# Every family module exposes compute(ctx) -> one row per entry: security_id, date, its features and
# its src_date_<family> column(s).
MODULES: tuple[ModuleType, ...] = (value, quality, investment, momentum, earnings, risk, health, size, context)

# The SF1 indicators the family modules read, for `panel.make_context(metrics=...)`: fundamentals_pit
# holds every indicator in long form, and loading all of them does not fit in memory at full scale.
# tests/test_features_build.py checks that a context cut to this list gives the same features.
SF1_METRICS: tuple[str, ...] = tuple(sorted({
    *value.METRICS, *health.METRICS, *earnings.METRICS,
    # quality
    "assets", "invcap", "gp", "ebit", "netinc", "ncfo", "debtnc", "assetsc", "liabilitiesc", "revenue", "opinc",
    "shareswa",
    # investment
    "sharesbas", "capex",
}))

KEYS = ("date", "security_id")


def build_raw(ctx: FeatureContext, catalog: Catalog | None = None) -> pl.DataFrame:
    """One row per ctx.entries row, sorted by date, security_id.

    Columns: date, security_id, sector, every catalog feature (raw value, Float64, null when
    undefined), then the src_date_<family> columns. Raises ValueError if the modules do not produce
    exactly the catalog's features, and LookaheadError if any source date is after its row's date.
    """
    catalog = catalog or load_catalog()
    keys = ["security_id", "date"]
    out = ctx.entries.select(keys).join(sector_map(ctx).select("security_id", "sector"), on="security_id", how="left")
    for module in MODULES:
        part = module.compute(ctx)
        clash = sorted(set(part.columns) & set(out.columns) - set(keys))
        if clash:
            raise ValueError(f"{module.__name__} repeats columns {clash}")
        out = out.join(part, on=keys, how="left", validate="1:1")
    if out.height != ctx.entries.height:
        raise AssertionError(f"features: {out.height} rows for {ctx.entries.height} entries")

    names = catalog.names()
    src = [c for c in out.columns if c.startswith(SRC_PREFIX)]
    produced = [c for c in out.columns if c not in (*keys, "sector", *src)]
    if set(produced) != set(names):
        raise ValueError(
            f"family modules and catalog disagree: not computed {sorted(set(names) - set(produced))}, "
            f"not in the catalog {sorted(set(produced) - set(names))}"
        )
    out = out.select(*KEYS, "sector", *[pl.col(c).cast(pl.Float64) for c in names], *src).sort(KEYS)
    assert_sources_not_after(out)
    return out


def build_features(ctx: FeatureContext, catalog: Catalog | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(features_raw, features). `features` has date, security_id, sector and catalog.model_columns()."""
    catalog = catalog or load_catalog()
    raw = build_raw(ctx, catalog)
    return raw, transform(raw, catalog)

"""Raw features -> model inputs (plan 3.2): cross-sectional ranks, sector-relative ranks, has_ flags.

Ranks are centred on zero and never sign-flipped: a high rank means a high raw value. A feature's
`sign` in the catalog is used only by the factor composite and the explanations.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ltforecast.features.catalog import HAS_PREFIX, SR_SUFFIX, Catalog

KEYS = ("date", "security_id")


def _centered_rank(col: str, by: Sequence[str]) -> pl.Expr:
    # NaN and +-inf are not valid raw values; treat them as missing rather than as the largest value
    x = pl.when(pl.col(col).is_finite()).then(pl.col(col))
    return ((x.rank("average") - 0.5) / x.count() - 0.5).over(by)


def xs_rank(col: str) -> pl.Expr:
    """Rank of `col` among the non-null values on the same date: (average rank - 0.5) / n - 0.5,
    strictly inside (-0.5, 0.5); ties share their average rank; nulls stay null."""
    return _centered_rank(col, ["date"]).alias(col)


def sector_rank(col: str, min_group: int = 5) -> pl.Expr:
    """The same rank within (date, sector), named `<col>_sr`. Null when the stock has no sector or
    fewer than `min_group` stocks of its sector have a value on that date."""
    by = ["date", "sector"]
    x = pl.when(pl.col(col).is_finite()).then(pl.col(col))
    enough = pl.col("sector").is_not_null() & (x.count().over(by) >= min_group)
    return pl.when(enough).then(_centered_rank(col, by)).alias(col + SR_SUFFIX)


def has_flags(df: pl.DataFrame, catalog: Catalog) -> pl.DataFrame:
    """Adds `has_<family>` (Int8, 0/1) per stock family: 1 if any raw feature of the family is non-null."""
    return df.with_columns(
        pl.any_horizontal(pl.col(catalog.names(kind="stock", family=family)).is_not_null())
        .cast(pl.Int8)
        .alias(HAS_PREFIX + family)
        for family in catalog.stock_families()
    )


def transform(raw: pl.DataFrame, catalog: Catalog, *, min_group: int = 5) -> pl.DataFrame:
    """features_raw -> features.

    Columns: date, security_id, sector, then catalog.model_columns(): stock features ranked per date
    (same names), `<name>_sr` ranked within sector, `has_<family>` flags (from the raw values), and
    the context features unranked. Missing values stay null. Sorted by date, security_id.
    """
    stock, context = catalog.names(kind="stock"), catalog.names(kind="context")
    missing = [c for c in (*KEYS, "sector", *stock, *context) if c not in raw.columns]
    if missing:
        raise ValueError(f"raw features lack columns {missing}")
    relative = [name.removesuffix(SR_SUFFIX) for name in catalog.sector_relative_names()]
    out = has_flags(raw, catalog).with_columns(
        *[sector_rank(c, min_group) for c in relative],
        *[xs_rank(c) for c in stock],
        *[pl.col(c).cast(pl.Float64) for c in context],
    )
    return out.select(*KEYS, "sector", *catalog.model_columns()).sort(KEYS)


def impute_for_linear(df: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    """Null -> 0.0 in `cols`: zero is the cross-sectional median of a centred rank."""
    return df.with_columns(pl.col(list(cols)).fill_null(0.0))

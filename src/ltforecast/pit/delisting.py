"""Delisting classification and assumed delisting returns (plan 1.3).

Sharadar gives no delisting return, so one is assumed: performance-related delistings lose
`performance_return` (Shumway 1997: -30%), cash mergers exit at the last price.
"""

from __future__ import annotations

import polars as pl

from ltforecast.config import DelistingRules

GENERIC_DELIST_ACTION = "delisted"


def delist_actions(rules: DelistingRules) -> tuple[str, ...]:
    return (*rules.performance_actions, *rules.merger_actions, GENERIC_DELIST_ACTION)


def classify(reason: pl.Expr, rules: DelistingRules) -> pl.Expr:
    """'performance' | 'merger' for a delist reason; unknown reasons fall back to rules.unknown_as."""
    return (
        pl.when(reason.is_in(rules.performance_actions))
        .then(pl.lit("performance"))
        .when(reason.is_in(rules.merger_actions))
        .then(pl.lit("merger"))
        .otherwise(pl.lit(rules.unknown_as))
    )


def delisting_events(securities: pl.DataFrame, rules: DelistingRules) -> pl.DataFrame:
    """One row per delisted security: security_id, delisted_date, delist_reason, delist_class, delist_return."""
    cls = classify(pl.col("delist_reason"), rules)
    return securities.filter(pl.col("delisted_date").is_not_null()).select(
        "security_id",
        "delisted_date",
        "delist_reason",
        cls.alias("delist_class"),
        pl.when(cls == "merger")
        .then(pl.lit(rules.merger_return))
        .otherwise(pl.lit(rules.performance_return))
        .alias("delist_return"),
    )

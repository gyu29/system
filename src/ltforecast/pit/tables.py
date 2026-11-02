"""Raw Sharadar/FRED tables -> core point-in-time tables (plan 1.2)."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from ltforecast.config import DelistingRules
from ltforecast.pit.delisting import GENERIC_DELIST_ACTION, delist_actions

SF1_ID_COLS = ("ticker", "dimension", "calendardate", "datekey", "reportperiod", "lastupdated", "fiscalperiod")


def build_securities(tickers: pl.DataFrame, actions: pl.DataFrame, rules: DelistingRules) -> pl.DataFrame:
    sec = tickers.filter(pl.col("table") == "SEP")
    dupes = sec.filter(pl.col("ticker").is_duplicated())
    if dupes.height:
        raise ValueError(f"tickers are not unique in TICKERS(table=SEP): {dupes['ticker'].unique().to_list()[:10]}")

    # Prefer a specific delisting reason over the generic "delisted" action; latest wins.
    reasons = (
        actions.filter(pl.col("action").is_in(delist_actions(rules)))
        .with_columns((pl.col("action") != GENERIC_DELIST_ACTION).alias("_specific"))
        .sort("ticker", "_specific", "date")
        .group_by("ticker")
        .agg(pl.col("action").last().alias("delist_reason"))
    )
    return (
        sec.join(reasons, on="ticker", how="left")
        .select(
            pl.col("permaticker").alias("security_id"),
            "ticker",
            "name",
            pl.col("secfilings").str.extract(r"CIK=(\d+)").cast(pl.Int64, strict=False).alias("cik"),
            "category",
            "exchange",
            "sector",
            "industry",
            "siccode",
            "famaindustry",
            pl.col("firstpricedate").alias("first_price_date"),
            pl.col("lastpricedate").alias("last_price_date"),
            pl.when(pl.col("isdelisted") == "Y").then(pl.col("lastpricedate")).alias("delisted_date"),
            pl.when(pl.col("isdelisted") == "Y").then(pl.col("delist_reason")).alias("delist_reason"),
            pl.col("relatedtickers").alias("ticker_history"),
        )
        .sort("security_id")
    )


def _ids(securities: pl.DataFrame) -> pl.DataFrame:
    return securities.select("ticker", "security_id")


def build_prices(sep: pl.DataFrame | pl.LazyFrame, actions: pl.DataFrame, securities: pl.DataFrame) -> pl.DataFrame:
    divs = (
        actions.filter(pl.col("action") == "dividend")
        .group_by("ticker", "date")
        .agg(pl.col("value").sum().alias("dividend"))
    )
    splits = (
        actions.filter(pl.col("action") == "split")
        .group_by("ticker", "date")
        .agg(pl.col("value").product().alias("split_factor"))
    )
    return (
        sep.lazy()
        .join(_ids(securities).lazy(), on="ticker", how="inner")
        .join(divs.lazy(), on=["ticker", "date"], how="left")
        .join(splits.lazy(), on=["ticker", "date"], how="left")
        .select(
            "security_id",
            "date",
            "open",
            "high",
            "low",
            "close",
            pl.col("closeunadj").alias("close_unadj"),
            pl.col("closeadj").alias("adj_close"),
            "volume",
            pl.col("dividend").fill_null(0.0),
            pl.col("split_factor").fill_null(1.0),
        )
        .sort("security_id", "date")
        .with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1).over("security_id").alias("ret"))
        .collect()
    )


def build_fundamentals(
    sf1: pl.DataFrame | pl.LazyFrame,
    securities: pl.DataFrame,
    dimensions: Iterable[str],
    metrics: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Long-format, first-reported fundamentals keyed by SEC filing date (datekey)."""
    sf1 = sf1.lazy().filter(pl.col("dimension").is_in(list(dimensions)))
    value_cols = [c for c in sf1.collect_schema().names() if c not in SF1_ID_COLS]
    if metrics is not None:
        value_cols = [c for c in value_cols if c in set(metrics)]
    return (
        sf1.join(_ids(securities).lazy(), on="ticker", how="inner")
        .unpivot(
            index=["security_id", "dimension", "reportperiod", "datekey"],
            on=value_cols,
            variable_name="metric",
            value_name="value",
        )
        .drop_nulls("value")
        .rename({"reportperiod": "fiscal_period_end", "datekey": "available_date"})
        # Amendments re-file the same period later; keep the first value the market saw.
        .sort("available_date")
        .unique(["security_id", "dimension", "metric", "fiscal_period_end"], keep="first", maintain_order=True)
        .select("security_id", "dimension", "metric", "fiscal_period_end", "available_date", "value")
        .sort("security_id", "metric", "available_date")
        .collect()
    )


def build_marketcap(daily: pl.DataFrame | pl.LazyFrame, securities: pl.DataFrame, unit: float) -> pl.DataFrame:
    return (
        daily.lazy()
        .join(_ids(securities).lazy(), on="ticker", how="inner")
        .select("security_id", "date", (pl.col("marketcap") * unit).alias("market_cap"))
        .drop_nulls("market_cap")
        .sort("security_id", "date")
        .collect()
    )


def build_macro(frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(list(frames)).sort("series", "available_date", "observation_date")

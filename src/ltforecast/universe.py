"""Monthly investable universe, decided on each date from data available on that date (plan 0, 1.3)."""

from __future__ import annotations

import datetime as dt

import polars as pl

from ltforecast.config import UniverseRules


def month_end_dates(calendar: pl.Series) -> pl.Series:
    """Last trading day of each calendar month."""
    return (
        calendar.to_frame("date")
        .group_by(pl.col("date").dt.truncate("1mo").alias("month"))
        .agg(pl.col("date").max())
        .sort("date")["date"]
    )


def cpi_asof(macro: pl.DataFrame, series: str, dates: pl.Series) -> pl.DataFrame:
    """CPI level as known on each date (latest observation already released)."""
    cpi = macro.filter(pl.col("series") == series).sort("available_date")
    if cpi.is_empty():
        raise ValueError(f"no {series} rows in macro_pit; ingest FRED first")
    return (
        dates.to_frame("date")
        .sort("date")
        .join_asof(
            cpi.select("available_date", pl.col("value").alias("cpi")),
            left_on="date",
            right_on="available_date",
            strategy="backward",
        )
        .drop("available_date")
    )


def mcap_thresholds(macro: pl.DataFrame, rules: UniverseRules, dates: pl.Series) -> pl.DataFrame:
    """Market-cap floor in nominal dollars of each date: min_market_cap deflated from cpi_base_date."""
    base = cpi_asof(macro, rules.cpi_series, pl.Series([rules.cpi_base_date]))["cpi"][0]
    return cpi_asof(macro, rules.cpi_series, dates).select(
        "date", (rules.min_market_cap * pl.col("cpi") / base).alias("mcap_threshold")
    )


def build_universe(
    prices: pl.LazyFrame | pl.DataFrame,
    securities: pl.DataFrame,
    marketcap: pl.LazyFrame | pl.DataFrame,
    macro: pl.DataFrame,
    rules: UniverseRules,
    start: dt.date,
    end: dt.date | None = None,
) -> pl.DataFrame:
    prices = prices.lazy()
    calendar = prices.select(pl.col("date").unique().sort()).collect().to_series()
    month_ends = month_end_dates(calendar)
    mask = month_ends >= start
    if end is not None:
        mask &= month_ends <= end
    month_ends = month_ends.filter(mask)

    w = rules.adv_window_days
    # SEP close and volume are both split-adjusted, so their product is the true dollar volume.
    daily = (
        prices.select("security_id", "date", "close", "close_unadj", "volume")
        .sort("security_id", "date")
        .with_columns(
            (pl.col("close") * pl.col("volume"))
            .rolling_mean(window_size=w, min_samples=max(1, (2 * w) // 3))
            .over("security_id")
            .alias("adv")
        )
        .filter(pl.col("date").is_in(month_ends.implode()))
        .select("security_id", "date", "close_unadj", "adv")
        .collect()
    )

    mcap = marketcap.lazy().select("security_id", "date", "market_cap").collect()
    daily = daily.sort("security_id", "date").join_asof(
        mcap.sort("security_id", "date"),
        on="date",
        by="security_id",
        strategy="backward",
        tolerance="7d",
        check_sortedness=False,
    )

    common = securities.select("security_id", pl.col("category").is_in(rules.categories).alias("is_common"))
    out = (
        daily.join(common, on="security_id", how="left")
        .join(mcap_thresholds(macro, rules, month_ends), on="date", how="left")
        .with_columns(
            (
                pl.col("is_common").fill_null(False)
                & (pl.col("close_unadj") > rules.min_price).fill_null(False)
                & (pl.col("market_cap") > pl.col("mcap_threshold")).fill_null(False)
                & (pl.col("adv") > rules.min_adv).fill_null(False)
            ).alias("in_universe")
        )
    )
    return out.select(
        "date", "security_id", "close_unadj", "market_cap", "adv", "mcap_threshold", "is_common", "in_universe"
    ).sort("date", "security_id")

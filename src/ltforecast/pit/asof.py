"""As-of joins that only ever see data that was public on the join date.

A row with `available_date` d becomes usable on the `lag`-th trading day *after* d
(lag=1: a 10-Q filed on Tuesday is first usable on Wednesday's close).
"""

from __future__ import annotations

import polars as pl


class LookaheadError(AssertionError):
    pass


def trading_calendar(prices: pl.LazyFrame | pl.DataFrame) -> pl.Series:
    return prices.lazy().select(pl.col("date").unique().sort()).collect().to_series()


def usable_from(dates: pl.Series, calendar: pl.Series, lag: int) -> pl.Series:
    """Trading day on which data published on `dates` may first be used. Null if beyond the calendar."""
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if lag == 0:
        idx = calendar.search_sorted(dates, side="left")
    else:
        idx = calendar.search_sorted(dates, side="right") + (lag - 1)
    in_range = idx < len(calendar)
    out = calendar.gather(idx.clip(upper_bound=len(calendar) - 1))
    return pl.select(pl.when(pl.lit(in_range)).then(pl.lit(out))).to_series().alias("usable_from")


def add_usable_from(df: pl.DataFrame, calendar: pl.Series, lag: int, col: str = "available_date") -> pl.DataFrame:
    return df.with_columns(usable_from(df[col], calendar, lag))


def asof_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    by: str = "security_id",
    date_col: str = "date",
    tolerance: str | None = None,
) -> pl.DataFrame:
    """Attach to each left row the latest right row with usable_from <= left date."""
    if "usable_from" not in right.columns:
        raise ValueError("right side needs a usable_from column; call add_usable_from first")
    out = left.sort(by, date_col).join_asof(
        right.drop_nulls("usable_from").sort(by, "usable_from"),
        left_on=date_col,
        right_on="usable_from",
        by=by,
        strategy="backward",
        tolerance=tolerance,
        check_sortedness=False,
    )
    assert_no_lookahead(out, date_col, "usable_from")
    return out


def assert_no_lookahead(df: pl.DataFrame, date_col: str, source_col: str) -> None:
    bad = df.filter(pl.col(source_col) > pl.col(date_col))
    if bad.height:
        raise LookaheadError(f"{bad.height} rows use {source_col} after {date_col}:\n{bad.head(5)}")

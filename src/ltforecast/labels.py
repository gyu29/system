"""Forward-return labels for each (security, month end) in the universe (plan section 2).

Returns are total returns from SEP `closeadj`, measured from the month-end close over a fixed number
of trading days. A stock that delists inside the window earns its assumed delisting return
(pit/delisting.py) on the next trading day; the proceeds are then held in the benchmark until the
window ends, so a delisting adds no excess return after it happens.

Benchmark: a daily cap-weighted index of the universe. On each day its members are the stocks in the
universe at the previous month end, weighted by their market cap on their previous trading day.
"""

from __future__ import annotations

import polars as pl

from ltforecast.config import Dates, DelistingRules, LabelRules
from ltforecast.pit.asof import trading_calendar, usable_from
from ltforecast.pit.delisting import delisting_events

TRADING_DAYS_PER_YEAR = 252


def delisting_returns(securities: pl.DataFrame, rules: DelistingRules, calendar: pl.Series) -> pl.DataFrame:
    """The assumed delisting return, realized on the first trading day after the last price.
    Columns: security_id, date, delist_return. Delistings at the end of the calendar are dropped."""
    ev = delisting_events(securities, rules)
    return (
        ev.with_columns(usable_from(ev["delisted_date"], calendar, lag=1).alias("date"))
        .drop_nulls("date")
        .select("security_id", "date", "delist_return")
    )


def benchmark_index(
    prices: pl.DataFrame,
    universe: pl.DataFrame,
    marketcap: pl.DataFrame,
    delist_rets: pl.DataFrame,
    calendar: pl.Series,
) -> pl.DataFrame:
    """Daily cap-weighted universe return. Columns: date, bench_ret, n_members, bench_level."""
    period_starts = universe.select(pl.col("date").unique().sort().alias("period_start"))
    members = universe.filter(pl.col("in_universe")).select(pl.col("date").alias("period_start"), "security_id")
    daily = (
        pl.concat([
            prices.select("security_id", "date", "ret"),
            delist_rets.select("security_id", "date", pl.col("delist_return").alias("ret")),
        ])
        .sort("security_id", "date")
        .join_asof(
            marketcap.select("security_id", "date", "market_cap").sort("security_id", "date"),
            on="date",
            by="security_id",
            strategy="backward",
            tolerance="7d",
            check_sortedness=False,
        )
        # weight = market cap at the previous close, so a day's own move doesn't set its weight
        .with_columns(pl.col("market_cap").shift(1).over("security_id").alias("weight"))
        .sort("date")
        # the month-end close belongs to the period that ends on it, not the one it starts
        .join_asof(period_starts, left_on="date", right_on="period_start", allow_exact_matches=False)
        .join(members, on=["period_start", "security_id"], how="inner")
        .filter(pl.col("ret").is_not_null() & (pl.col("weight") > 0))
    )
    agg = daily.group_by("date").agg(
        ((pl.col("weight") * pl.col("ret")).sum() / pl.col("weight").sum()).alias("bench_ret"),
        pl.len().alias("n_members"),
    )
    return (
        calendar.to_frame("date")
        .join(agg, on="date", how="left")
        .with_columns(pl.col("bench_ret").fill_null(0.0), pl.col("n_members").fill_null(0))
        .with_columns((1 + pl.col("bench_ret")).cum_prod().alias("bench_level"))
    )


def forward_returns(
    prices: pl.DataFrame,
    entries: pl.DataFrame,
    bench: pl.DataFrame,
    delist_rets: pl.DataFrame,
    calendar: pl.Series,
    horizon: int,
    stale_days: int,
    vol_min_obs: int,
) -> pl.DataFrame:
    """Return over `horizon` trading days after each entry (security_id, date).

    Columns: security_id, date, label_end, fwd_ret, bench_ret, realized_vol. All null when the window
    runs past the calendar, or when the stock stops trading without a delisting (a data gap).
    """
    cal = calendar.to_frame("date").with_row_index("d").with_columns(pl.col("d").cast(pl.Int64))
    px = (
        prices.select("security_id", "date", "adj_close", (pl.col("ret") + 1).log().alias("lr"))
        .sort("security_id", "date")
        .with_columns(
            # running sums give the variance of log returns between any two rows
            pl.col("lr").fill_null(0.0).cum_sum().over("security_id").alias("s1"),
            pl.col("lr").pow(2).fill_null(0.0).cum_sum().over("security_id").alias("s2"),
            pl.col("lr").is_not_null().cum_sum().over("security_id").alias("n"),
        )
        .join(cal, on="date")
        .drop("lr")
    )
    levels = bench.select("date", "bench_level")

    start = (
        entries.select("security_id", "date")
        .join(px, on=["security_id", "date"], how="left")
        .rename({"adj_close": "px_t", "s1": "s1_t", "s2": "s2_t", "n": "n_t", "d": "d_t"})
        .with_columns((pl.col("d_t") + horizon).alias("d_end"))
        .join(cal.rename({"d": "d_end", "date": "label_end"}), on="d_end", how="left")
    )
    complete = (
        start.drop_nulls(["label_end", "px_t"])
        .sort("label_end")
        .join_asof(
            px.rename({"date": "last_date", "d": "d_last", "adj_close": "px_end", "s1": "s1_e", "s2": "s2_e", "n": "n_e"}),
            left_on="label_end",
            right_on="last_date",
            by="security_id",
            strategy="backward",
            check_sortedness=False,
        )
        .join(delist_rets.rename({"date": "delist_date"}), on="security_id", how="left")
        .join(levels.rename({"bench_level": "lvl_t"}), on="date", how="left")
        .join(levels.rename({"date": "label_end", "bench_level": "lvl_end"}), on="label_end", how="left")
        .join(levels.rename({"date": "delist_date", "bench_level": "lvl_del"}), on="delist_date", how="left")
    )

    delisted = pl.col("delist_date").is_not_null() & (pl.col("delist_date") <= pl.col("label_end"))
    stale = pl.col("d_last") < pl.col("d_end") - stale_days
    gross = (
        pl.when(delisted)
        .then(pl.col("px_end") / pl.col("px_t") * (1 + pl.col("delist_return")) * pl.col("lvl_end") / pl.col("lvl_del"))
        .when(stale)
        .then(None)
        .otherwise(pl.col("px_end") / pl.col("px_t"))
    )
    n = pl.col("n_e") - pl.col("n_t")
    s1 = pl.col("s1_e") - pl.col("s1_t")
    var = ((pl.col("s2_e") - pl.col("s2_t")) - s1.pow(2) / n) / (n - 1)
    vol = pl.when(n >= vol_min_obs).then((var.clip(lower_bound=0.0) * TRADING_DAYS_PER_YEAR).sqrt())

    out = complete.with_columns(gross.alias("gross")).select(
        "security_id",
        "date",
        "label_end",
        (pl.col("gross") - 1).alias("fwd_ret"),
        pl.when(pl.col("gross").is_not_null()).then(pl.col("lvl_end") / pl.col("lvl_t") - 1).alias("bench_ret"),
        pl.when(pl.col("gross").is_not_null()).then(vol).alias("realized_vol"),
    )
    return (
        entries.select("security_id", "date")
        .join(out, on=["security_id", "date"], how="left")
        .sort("date", "security_id")
    )


def period(dates: Dates) -> pl.Expr:
    """dev | gap | holdout | live, by prediction date. Holdout rows are for Phase 9 only."""
    d = pl.col("date")
    return (
        pl.when(d <= dates.dev_end)
        .then(pl.lit("dev"))
        .when(d < dates.holdout_start)
        .then(pl.lit("gap"))
        .when(d <= dates.holdout_end)
        .then(pl.lit("holdout"))
        .otherwise(pl.lit("live"))
        .alias("period")
    )


def build_labels(
    prices: pl.DataFrame,
    universe: pl.DataFrame,
    marketcap: pl.DataFrame,
    securities: pl.DataFrame,
    rules: LabelRules,
    delisting: DelistingRules,
    dates: Dates,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(labels, benchmark_daily). One label row per in-universe (security_id, month end).

    Per horizon h: fwd_ret_h, bench_ret_h, excess_h = log(1+fwd) - log(1+bench), excess_w_h (winsorized
    per date), rank_h (per-date percentile of excess_h in (0, 1)), realized_vol_h, label_end_h.
    `target` is rank of the primary horizon: the training label.
    """
    calendar = trading_calendar(prices)
    delist_rets = delisting_returns(securities, delisting, calendar)
    bench = benchmark_index(prices, universe, marketcap, delist_rets, calendar)
    entries = universe.filter(pl.col("in_universe")).select("security_id", "date")

    out = entries.with_columns(period(dates))
    lo, hi = rules.winsorize
    for name, h in rules.horizons.items():
        fr = forward_returns(
            prices, entries, bench, delist_rets, calendar, h, rules.stale_price_days, rules.vol_min_obs
        )
        ex = (pl.col("fwd_ret") + 1).log() - (pl.col("bench_ret") + 1).log()
        fr = fr.with_columns(ex.alias("excess")).with_columns(
            pl.col("excess")
            .clip(pl.col("excess").quantile(lo).over("date"), pl.col("excess").quantile(hi).over("date"))
            .alias("excess_w"),
            ((pl.col("excess").rank("average") - 0.5) / pl.col("excess").count()).over("date").alias("rank"),
        )
        cols = ["fwd_ret", "bench_ret", "excess", "excess_w", "rank", "realized_vol", "label_end"]
        out = out.join(
            fr.select("security_id", "date", *[pl.col(c).alias(f"{c}_{name}") for c in cols]),
            on=["security_id", "date"],
            how="left",
        )
    out = out.with_columns(pl.col(f"rank_{rules.primary}").alias("target")).sort("date", "security_id")
    return out, bench

"""Automated data-quality checks run on every refresh (plan 1.4)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ltforecast.config import Config
from ltforecast.pit.asof import add_usable_from, trading_calendar
from ltforecast.store import Store


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # pass | warn | fail
    detail: str


def _result(name: str, ok: bool, detail: str, *, severity: str = "fail") -> CheckResult:
    return CheckResult(name, "pass" if ok else severity, detail)


def check_no_future_available(fund: pl.DataFrame, macro: pl.DataFrame, today: dt.date) -> CheckResult:
    early = fund.filter(pl.col("available_date") < pl.col("fiscal_period_end")).height
    future = fund.filter(pl.col("available_date") > today).height
    macro_early = macro.filter(pl.col("available_date") < pl.col("observation_date")).height
    ok = early == future == macro_early == 0
    return _result(
        "no_future_available_date",
        ok,
        f"filed before period end: {early}, filed after today: {future}, macro released before observed: {macro_early}",
    )


def check_price_jumps(prices: pl.DataFrame, max_move: float) -> CheckResult:
    """Daily total-return moves beyond max_move on days with no split or dividend."""
    jumps = prices.filter(
        (pl.col("ret").abs() > max_move) & (pl.col("split_factor") == 1.0) & (pl.col("dividend") == 0.0)
    )
    frac = jumps.height / max(prices.height, 1)
    sample = jumps.sort(pl.col("ret").abs(), descending=True).head(5).select("security_id", "date", "ret").rows()
    return _result("price_jumps", frac < 1e-4, f"{jumps.height} moves > {max_move:.0%} without action; top: {sample}", severity="warn")


def check_split_consistency(prices: pl.DataFrame, tol: float = 0.01) -> CheckResult:
    """close_unadj/close may only change on split days; otherwise the split adjustment is broken."""
    ratio = (pl.col("close_unadj") / pl.col("close"))
    bad = (
        prices.sort("security_id", "date")
        .with_columns((ratio / ratio.shift(1) - 1).over("security_id").abs().alias("drift"))
        .filter((pl.col("drift") > tol) & (pl.col("split_factor") == 1.0))
    )
    return _result("split_consistency", bad.height == 0, f"{bad.height} unexplained adjustment-ratio changes", severity="warn")


def universe_counts(universe: pl.DataFrame) -> pl.DataFrame:
    return universe.group_by("date").agg(pl.col("in_universe").sum().alias("n")).sort("date")


def check_universe_counts(universe: pl.DataFrame, max_change: float) -> CheckResult:
    counts = universe_counts(universe).with_columns((pl.col("n") / pl.col("n").shift(1) - 1).alias("chg"))
    jumps = counts.filter(pl.col("chg").abs() > max_change)
    return _result(
        "universe_count_stability",
        jumps.is_empty(),
        f"{jumps.height} months with |count change| > {max_change:.0%}: {jumps.select('date', 'n', 'chg').head(5).rows()}",
    )


def check_universe_size(universe: pl.DataFrame, lo: int = 2000, hi: int = 3500) -> CheckResult:
    counts = universe_counts(universe)["n"]
    med = counts.median() if counts.len() else 0
    return _result("universe_size", lo <= med <= hi, f"median {med:.0f} stocks/month (expected {lo}-{hi})", severity="warn")


def check_delisted_present(universe: pl.DataFrame, securities: pl.DataFrame) -> CheckResult:
    members = universe.filter("in_universe").select("security_id").unique()
    delisted = members.join(securities.filter(pl.col("delisted_date").is_not_null()), on="security_id").height
    frac = delisted / max(members.height, 1)
    return _result("delisted_present", delisted > 0, f"{delisted} of {members.height} universe members later delisted ({frac:.0%})")


def coverage_report(universe: pl.DataFrame, fund: pl.DataFrame, calendar: pl.Series, lag: int) -> pl.DataFrame:
    """Per year: universe size and % of members with market cap, ADV and a quarterly filing in the last 120 days."""
    members = universe.filter("in_universe")
    filings = add_usable_from(
        fund.filter(pl.col("dimension") == "ARQ").select("security_id", "available_date").unique(), calendar, lag
    ).drop_nulls("usable_from")
    with_filing = members.sort("security_id", "date").join_asof(
        filings.sort("security_id", "usable_from").select("security_id", "usable_from"),
        left_on="date",
        right_on="usable_from",
        by="security_id",
        strategy="backward",
        tolerance="120d",
        check_sortedness=False,
    )
    return (
        with_filing.group_by(pl.col("date").dt.year().alias("year"))
        .agg(
            pl.len().alias("member_months"),
            pl.col("market_cap").is_not_null().mean().alias("pct_market_cap"),
            pl.col("adv").is_not_null().mean().alias("pct_adv"),
            pl.col("usable_from").is_not_null().mean().alias("pct_recent_10q"),
        )
        .sort("year")
    )


def spot_check_sample(prices: pl.DataFrame, securities: pl.DataFrame, n: int = 20, seed: int = 0) -> pl.DataFrame:
    """Random (ticker, date, close) rows to compare by hand against an independent source."""
    ids = prices.select("security_id").unique().sample(min(n, prices["security_id"].n_unique()), seed=seed)
    return (
        prices.join(ids, on="security_id")
        .group_by("security_id")
        .agg(pl.all().sample(1, seed=seed).first())
        .join(securities.select("security_id", "ticker"), on="security_id")
        .select("ticker", "date", "close_unadj", "close", "adj_close", "volume")
        .sort("ticker")
    )


def run_all(store: Store, cfg: Config, report_dir: Path | None = None) -> list[CheckResult]:
    prices = store.scan("prices_daily").collect()
    securities = store.scan("securities").collect()
    fund = store.scan("fundamentals_pit").collect()
    macro = store.scan("macro_pit").collect()
    universe = store.scan("universe").collect()
    q = cfg.quality

    results = [
        check_no_future_available(fund, macro, dt.date.today()),
        check_price_jumps(prices, q.max_daily_move_without_action),
        check_split_consistency(prices),
        check_universe_counts(universe, q.max_monthly_count_change),
        check_universe_size(universe),
        check_delisted_present(universe, securities),
    ]

    report_dir = report_dir or store.root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    coverage_report(universe, fund, trading_calendar(prices), cfg.pit.fundamental_lag_trading_days).write_csv(
        report_dir / "coverage.csv"
    )
    spot_check_sample(prices, securities).write_csv(report_dir / "spot_check.csv")
    return results

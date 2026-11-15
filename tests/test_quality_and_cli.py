import datetime as dt

import polars as pl
import pytest
from typer.testing import CliRunner

from ltforecast import quality
from ltforecast.cli import app
from ltforecast.store import Store

d = dt.date


def test_universe_count_drop_fails():
    u = pl.DataFrame({
        "date": [d(2020, 1, 31)] * 10 + [d(2020, 2, 28)] * 10,
        "security_id": list(range(10)) * 2,
        "in_universe": [True] * 10 + [True] * 5 + [False] * 5,
    })
    assert quality.check_universe_counts(u, 0.10).status == "fail"
    assert quality.check_universe_counts(u.filter(pl.col("date") == d(2020, 1, 31)), 0.10).status == "pass"


def test_universe_count_warm_up_months_are_ignored():
    # empty first month (ADV window not yet filled), then a stable universe: not a coverage jump
    u = pl.DataFrame({
        "date": [d(2020, 1, 31)] * 10 + [d(2020, 2, 28)] * 10 + [d(2020, 3, 31)] * 10,
        "security_id": list(range(10)) * 3,
        "in_universe": [False] * 10 + [True] * 20,
    })
    assert quality.check_universe_counts(u, 0.10).status == "pass"
    # a universe that later empties out still fails
    gone = u.with_columns(pl.when(pl.col("date") == d(2020, 3, 31)).then(False).otherwise(pl.col("in_universe")).alias("in_universe"))
    assert quality.check_universe_counts(gone, 0.10).status == "fail"


def test_price_jump_without_action_is_flagged():
    p = pl.DataFrame({
        "security_id": [1, 1, 2], "date": [d(2020, 1, 2), d(2020, 1, 3), d(2020, 1, 3)],
        "ret": [None, 0.9, 0.9], "split_factor": [1.0, 1.0, 2.0], "dividend": [0.0, 0.0, 0.0],
    })
    r = quality.check_price_jumps(p, 0.5)
    assert r.status == "warn" and r.detail.startswith("1 moves")


def test_filing_before_period_end_fails():
    fund = pl.DataFrame({"fiscal_period_end": [d(2020, 3, 31)], "available_date": [d(2020, 3, 1)]})
    macro = pl.DataFrame({"observation_date": [d(2020, 1, 1)], "available_date": [d(2020, 2, 15)]})
    assert quality.check_no_future_available(fund, macro, d(2026, 1, 1)).status == "fail"


def _share_frames(counts: list[float], split_day: dt.date) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Security 1 files ARQ sharesbas on 2020-02-14, 05-15, 08-14 (and an amendment of the first, ignored);
    it splits 2:1 on `split_day`."""
    filed = [d(2020, 2, 14), d(2020, 5, 15), d(2020, 8, 14), d(2020, 9, 30)]
    periods = [d(2019, 12, 31), d(2020, 3, 31), d(2020, 6, 30), d(2019, 12, 31)]
    fund = pl.DataFrame({
        "security_id": [1] * 5, "dimension": ["ARQ"] * 4 + ["ART"], "metric": ["sharesbas"] * 5,
        "fiscal_period_end": periods + [d(2020, 3, 31)], "available_date": filed + [d(2020, 5, 15)],
        "value": [*counts, 999.0, 5.0],
    })
    prices = pl.DataFrame({
        "security_id": [1, 1, 1], "date": [d(2020, 1, 2), split_day, d(2020, 9, 1)], "split_factor": [1.0, 2.0, 1.0],
    })
    return fund, prices


@pytest.mark.parametrize(
    ("counts", "split_day", "status"),
    [
        ([100.0, 100.0, 210.0], d(2020, 6, 1), "pass"),   # split between the 2nd and 3rd filing, in the 3rd count
        ([100.0, 200.0, 210.0], d(2020, 4, 15), "pass"),  # split after a period end, before its filing: in that count
        ([100.0, 100.0, 210.0], d(2020, 5, 15), "pass"),  # split on a filing day: not yet in that day's count
        ([100.0, 100.0, 105.0], d(2020, 6, 1), "warn"),   # restated for the split: ratio 1.05, not 2
        ([200.0, 200.0, 210.0], d(2020, 6, 1), "warn"),   # earlier count restated to the later basis
    ],
)
def test_share_basis_check(counts, split_day, status):
    fund, prices = _share_frames(counts, split_day)
    r = quality.check_share_basis(fund, prices)
    assert r.status == status, r.detail
    assert r.detail.startswith("1 split events")


def test_share_basis_check_skips_without_splits():
    fund, prices = _share_frames([100.0, 100.0, 105.0], d(2020, 6, 1))
    r = quality.check_share_basis(fund, prices.with_columns(pl.lit(1.0).alias("split_factor")))
    assert r.status == "skip"


def test_end_to_end_build_and_check(cfg, raw_tickers, raw_sep, raw_sf1, raw_daily, raw_actions, macro):
    store = Store(cfg.data_dir)
    raws = {"TICKERS": raw_tickers, "SEP": raw_sep, "SF1": raw_sf1, "DAILY": raw_daily, "ACTIONS": raw_actions}
    for name, df in raws.items():
        store.write_raw("sharadar", name, df)
    store.write_raw("fred", "OBSERVATIONS", macro)
    config = cfg.data_dir.parent / "config" / "universe.yaml"

    runner = CliRunner()
    built = runner.invoke(app, ["build", "--config", str(config)])
    assert built.exit_code == 0, built.output
    assert "securities: 6 (2 delisted)" in built.output

    # partitioned by year, readable from DuckDB
    assert {p.name for p in store.core_path("prices_daily").iterdir()} == {"year=2019", "year=2020"}
    n = store.duckdb().execute("select count(distinct security_id) from universe where in_universe").fetchone()[0]
    assert n == 3

    checked = runner.invoke(app, ["check", "--config", str(config)])
    # Synthetic universe is tiny, so size/stability checks fail; the run itself must still complete.
    assert "[PASS] no_future_available_date" in checked.output
    assert "[PASS] delisted_present" in checked.output
    assert (cfg.data_dir / "reports" / "coverage.csv").exists()
    assert (cfg.data_dir / "reports" / "spot_check.csv").exists()

    labelled = runner.invoke(app, ["labels", "--config", str(config)])
    assert labelled.exit_code == 0, labelled.output
    lab = store.scan("labels").collect()
    bench = dict(store.scan("benchmark_daily").select("date", "bench_level").collect().iter_rows())

    # CCC (bankrupt 2020-06-30, flat at $20): the 12m window from 2019-07-31 contains the delisting.
    # It earns -30% on the next trading day, then holds the benchmark to the window end.
    ccc = lab.filter((pl.col("security_id") == 103) & (pl.col("date") == d(2019, 7, 31))).row(0, named=True)
    end = ccc["label_end_12m"]
    assert ccc["fwd_ret_12m"] == pytest.approx(0.7 * bench[end] / bench[d(2020, 7, 1)] - 1)
    assert lab.filter(pl.col("label_end_12m") > d(2020, 12, 31)).height == 0

    bar = store.scan("barrier_labels").collect()
    n_entries = bar.select("security_id", "date").n_unique()
    assert n_entries > 0 and bar.height == 25 * n_entries
    assert set(bar["outcome"]) <= {"tp", "sl", "time"}

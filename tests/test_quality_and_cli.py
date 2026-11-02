import datetime as dt

import polars as pl
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

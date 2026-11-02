import datetime as dt

import polars as pl
import pytest
from conftest import START, business_days

from ltforecast.pit import tables
from ltforecast.universe import build_universe, mcap_thresholds, month_end_dates

d = dt.date


@pytest.fixture
def universe(raw_tickers, raw_actions, raw_sep, raw_daily, macro, cfg):
    sec = tables.build_securities(raw_tickers, raw_actions, cfg.delisting)
    prices = tables.build_prices(raw_sep, raw_actions, sec)
    mcap = tables.build_marketcap(raw_daily, sec, cfg.universe.marketcap_unit)
    return build_universe(prices, sec, mcap, macro, cfg.universe, START)


def members(u: pl.DataFrame, date: dt.date) -> set[int]:
    return set(u.filter((pl.col("date") == date) & pl.col("in_universe"))["security_id"])


def test_month_ends_are_last_trading_days():
    me = month_end_dates(business_days())
    assert me.len() == 24
    assert me[0] == d(2019, 1, 31)
    assert me[2] == d(2019, 3, 29)  # March 31 2019 was a Sunday


def test_membership(universe):
    # Jan 2019 has fewer than adv_window*2/3 trading days, so nobody has an ADV yet
    assert members(universe, d(2019, 1, 31)) == set()
    # AAA passes on its *unadjusted* $8 even though its split-adjusted close is $4
    assert members(universe, d(2019, 6, 28)) == {101, 103, 104}
    # after the split AAA trades at $6 and stays in; CCC (bankrupt 2020-06-30) is in on its last day
    assert members(universe, d(2020, 6, 30)) == {101, 103, 104}
    assert members(universe, d(2020, 7, 31)) == {101, 104}
    assert members(universe, d(2020, 9, 30)) == {101}


def test_exclusion_reasons(universe):
    def row(sid):
        return universe.filter((pl.col("date") == d(2019, 6, 28)) & (pl.col("security_id") == sid)).row(0, named=True)

    assert row(102)["close_unadj"] == 3.0                       # BBB: price
    assert row(105)["is_common"] is False                       # EEE: ETF
    assert row(106)["market_cap"] < row(106)["mcap_threshold"]  # FFF: size


def test_mcap_threshold_is_deflated(macro, cfg):
    t = mcap_thresholds(macro, cfg.universe, pl.Series([d(2019, 1, 31), d(2020, 12, 31)]))
    early, base = t["mcap_threshold"].to_list()
    assert base == pytest.approx(300e6)
    assert early < base


def test_cpi_uses_only_released_values(macro, cfg):
    # On 2019-02-01 the latest CPI released (45-day lag) is the 2018-12 observation
    t = mcap_thresholds(macro, cfg.universe, pl.Series([d(2019, 2, 1)]))
    known = macro.filter(pl.col("available_date") <= d(2019, 2, 1)).sort("available_date")[-1]
    assert known["observation_date"][0] == d(2018, 12, 1)
    base = macro.filter(pl.col("available_date") <= d(2020, 12, 31)).sort("available_date")["value"][-1]
    assert t["mcap_threshold"][0] == pytest.approx(300e6 * known["value"][0] / base)

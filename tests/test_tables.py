import datetime as dt

import polars as pl
import pytest

from ltforecast.pit import tables
from ltforecast.pit.delisting import delisting_events

d = dt.date


@pytest.fixture
def securities(raw_tickers, raw_actions, cfg):
    return tables.build_securities(raw_tickers, raw_actions, cfg.delisting)


def test_securities(securities):
    assert securities.height == 6  # SF1-table duplicate row dropped
    by = {r["ticker"]: r for r in securities.iter_rows(named=True)}
    assert by["AAA"]["cik"] == 101
    assert by["AAA"]["delisted_date"] is None and by["AAA"]["delist_reason"] is None
    assert by["CCC"]["delisted_date"] == d(2020, 6, 30)
    assert by["CCC"]["delist_reason"] == "bankruptcyliquidation"
    assert by["DDD"]["delist_reason"] == "acquisitionby"  # specific reason beats generic 'delisted'


def test_duplicate_tickers_rejected(raw_tickers, raw_actions, cfg):
    dup = pl.concat([raw_tickers, raw_tickers.filter(pl.col("ticker") == "AAA", pl.col("table") == "SEP")])
    with pytest.raises(ValueError, match="not unique"):
        tables.build_securities(dup, raw_actions, cfg.delisting)


def test_delisting_events(securities, cfg):
    ev = {r["security_id"]: r for r in delisting_events(securities, cfg.delisting).iter_rows(named=True)}
    assert set(ev) == {103, 104}
    assert ev[103]["delist_class"] == "performance" and ev[103]["delist_return"] == pytest.approx(-0.30)
    assert ev[104]["delist_class"] == "merger" and ev[104]["delist_return"] == 0.0


def test_unknown_delisting_is_conservative(raw_tickers, raw_actions, cfg):
    actions = raw_actions.filter(pl.col("action") != "bankruptcyliquidation")
    sec = tables.build_securities(raw_tickers, actions, cfg.delisting)
    ev = delisting_events(sec, cfg.delisting).filter(pl.col("security_id") == 103).row(0, named=True)
    assert ev["delist_reason"] is None and ev["delist_class"] == "performance"


def test_prices(raw_sep, raw_actions, securities):
    p = tables.build_prices(raw_sep, raw_actions, securities)
    aaa = p.filter(pl.col("security_id") == 101)
    assert aaa.filter(pl.col("date") == d(2019, 6, 14))["dividend"][0] == 0.5
    assert aaa.filter(pl.col("date") == d(2020, 3, 2))["split_factor"][0] == 2.0
    assert aaa["dividend"].sum() == 0.5 and aaa["split_factor"].product() == 2.0
    assert aaa["ret"][0] is None  # first day has no prior close
    # delisted names keep their history up to the last trading day
    assert p.filter(pl.col("security_id") == 103)["date"].max() == d(2020, 6, 30)


def test_fundamentals_first_reported_only(raw_sf1, securities, cfg):
    f = tables.build_fundamentals(raw_sf1, securities, cfg.pit.sf1_dimensions)
    assert set(f["dimension"]) == {"ARQ"}
    rev = f.filter((pl.col("security_id") == 101) & (pl.col("metric") == "revenue")).sort("fiscal_period_end")
    assert rev["value"].to_list() == [100.0, 120.0]
    assert rev["available_date"].to_list() == [d(2019, 5, 1), d(2019, 8, 2)]
    # null sharesbas for Q2 is dropped, not stored
    assert f.filter((pl.col("metric") == "sharesbas") & (pl.col("security_id") == 101)).height == 1


def test_marketcap_units(raw_daily, securities, cfg):
    m = tables.build_marketcap(raw_daily, securities, cfg.universe.marketcap_unit)
    assert m.filter(pl.col("security_id") == 101)["market_cap"][0] == 1.0e9

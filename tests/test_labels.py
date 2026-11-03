"""Hand-computed labels (plan section 2 unit test).

Five stocks in the universe on 2020-01-31, 5-trading-day horizon (window Feb 3 - Feb 7):

A  100 -> 102, 104, 106, 108, 110                      mcap 400
B  50 flat, 45 on Feb 7                                 mcap 100
S  2:1 split on Feb 5: unadjusted 200 -> 100, total-return price 100 flat, 105 on Feb 7   mcap 200
D  bankrupt: last price Feb 4 (20 flat); -30% delisting return realized Feb 5             mcap 100
M  acquired: 25 -> 30 on Feb 3, last price Feb 5; 0% delisting return realized Feb 6       mcap 200
X  not in the universe: doubles on Feb 3 and must not move the benchmark                  mcap 1000
"""

import datetime as dt
import math
import statistics

import polars as pl
import pytest
from conftest import business_days

from ltforecast.config import Dates, LabelRules
from ltforecast.labels import benchmark_index, build_labels, delisting_returns, forward_returns, period

d = dt.date
CAL = business_days(d(2020, 1, 1), d(2020, 3, 31))
JAN, FEB, MAR = d(2020, 1, 31), d(2020, 2, 28), d(2020, 3, 31)

# security_id: (price changes, last price date, market cap)
STOCKS = {
    1: ({d(2020, 1, 1): 100, d(2020, 2, 3): 102, d(2020, 2, 4): 104, d(2020, 2, 5): 106, d(2020, 2, 6): 108,
         d(2020, 2, 7): 110}, None, 400.0),
    2: ({d(2020, 1, 1): 50, d(2020, 2, 7): 45}, None, 100.0),
    3: ({d(2020, 1, 1): 100, d(2020, 2, 7): 105}, None, 200.0),
    4: ({d(2020, 1, 1): 20}, d(2020, 2, 4), 100.0),
    5: ({d(2020, 1, 1): 25, d(2020, 2, 3): 30}, d(2020, 2, 5), 200.0),
    6: ({d(2020, 1, 1): 10, d(2020, 2, 3): 20}, None, 1000.0),
}


def make_prices(stocks) -> pl.DataFrame:
    frames = []
    for sid, (changes, last, _) in stocks.items():
        days = CAL.filter(CAL <= last) if last else CAL
        steps = pl.DataFrame({"date": list(changes), "adj_close": [float(v) for v in changes.values()]})
        frames.append(
            days.to_frame()
            .join_asof(steps.sort("date"), on="date")
            .with_columns(pl.lit(sid, pl.Int64).alias("security_id"))
        )
    df = pl.concat(frames).with_columns(
        pl.col("adj_close").alias("close"),
        pl.col("adj_close").alias("open"),
        # S: the raw price halves on the split day; labels must only see the total-return price
        pl.when((pl.col("security_id") == 3) & (pl.col("date") < d(2020, 2, 5)))
        .then(pl.col("adj_close") * 2)
        .otherwise(pl.col("adj_close"))
        .alias("close_unadj"),
    )
    return df.sort("security_id", "date").with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1).over("security_id").alias("ret")
    )


def make_universe(prices: pl.DataFrame) -> pl.DataFrame:
    return (
        prices.filter(pl.col("date").is_in([JAN, FEB, MAR]))
        .select("date", "security_id", (pl.col("security_id") != 6).alias("in_universe"))
    )


@pytest.fixture
def prices():
    return make_prices(STOCKS)


@pytest.fixture
def marketcap(prices):
    caps = pl.DataFrame({"security_id": list(STOCKS), "market_cap": [s[2] for s in STOCKS.values()]})
    return prices.select("security_id", "date").join(caps, on="security_id")


@pytest.fixture
def securities():
    return pl.DataFrame({
        "security_id": [1, 2, 3, 4, 5, 6],
        "delisted_date": [None, None, None, d(2020, 2, 4), d(2020, 2, 5), None],
        "delist_reason": [None, None, None, "bankruptcyliquidation", "acquisitionby", None],
    })


def rules(**kw) -> LabelRules:
    base = dict(horizons={"1w": 5}, primary="1w", winsorize=(0.01, 0.99), vol_min_obs=3, stale_price_days=5,
                barriers=None)
    return LabelRules(**{**base, **kw})


@pytest.fixture
def built(prices, marketcap, securities, cfg):
    return build_labels(prices, make_universe(prices), marketcap, securities, rules(), cfg.delisting, cfg.dates)


# Daily benchmark returns, Feb 3-7, from the table above (weights = previous day's market cap)
R = {
    d(2020, 2, 3): (400 * 0.02 + 200 * 0.2) / 1000,
    d(2020, 2, 4): 400 * (2 / 102) / 1000,
    d(2020, 2, 5): (400 * (2 / 104) - 100 * 0.30) / 1000,      # D's delisting return
    d(2020, 2, 6): 400 * (2 / 106) / 900,                       # D gone; M's 0% delisting return
    d(2020, 2, 7): (400 * (2 / 108) - 100 * 0.10 + 200 * 0.05) / 700,  # D and M gone
}


def test_delisting_return_lands_on_next_trading_day(securities, cfg):
    dr = delisting_returns(securities, cfg.delisting, CAL).sort("security_id")
    assert dr.rows() == [(4, d(2020, 2, 5), -0.30), (5, d(2020, 2, 6), 0.0)]


def test_benchmark_is_cap_weighted_over_members(prices, marketcap, securities, cfg):
    dr = delisting_returns(securities, cfg.delisting, CAL)
    bench = benchmark_index(prices, make_universe(prices), marketcap, dr, CAL)
    got = dict(bench.filter(pl.col("date").is_in(list(R))).select("date", "bench_ret").iter_rows())
    assert got == pytest.approx(R)
    # Jan 31 closes out January; there is no December membership, so January has no benchmark
    assert bench.filter(pl.col("date") <= JAN)["n_members"].sum() == 0


def test_hand_computed_forward_returns(built):
    lab, _ = built
    jan = {r["security_id"]: r for r in lab.filter(pl.col("date") == JAN).iter_rows(named=True)}
    assert set(jan) == {1, 2, 3, 4, 5}  # X is not in the universe

    bench = math.prod(1 + r for r in R.values()) - 1
    expected = {
        1: 110 / 100 - 1,
        2: 45 / 50 - 1,
        3: 105 / 100 - 1,                                                 # split invisible
        4: 0.70 * (1 + R[d(2020, 2, 6)]) * (1 + R[d(2020, 2, 7)]) - 1,  # -30%, then benchmark
        5: 30 / 25 * 1.0 * (1 + R[d(2020, 2, 7)]) - 1,                  # cash-out, then benchmark
    }
    for sid, fwd in expected.items():
        row = jan[sid]
        assert row["fwd_ret_1w"] == pytest.approx(fwd), sid
        assert row["bench_ret_1w"] == pytest.approx(bench)
        assert row["excess_1w"] == pytest.approx(math.log1p(fwd) - math.log1p(bench))
        assert row["label_end_1w"] == d(2020, 2, 7)

    # ranks: D < B < S < A < M, at (rank - 0.5) / n
    assert {sid: jan[sid]["rank_1w"] for sid in jan} == pytest.approx({4: 0.1, 2: 0.3, 3: 0.5, 1: 0.7, 5: 0.9})
    assert all(r["target"] == r["rank_1w"] for r in jan.values())


def test_realized_vol(built):
    lab, _ = built
    a = lab.filter((pl.col("date") == JAN) & (pl.col("security_id") == 1)).row(0, named=True)
    path = [100, 102, 104, 106, 108, 110]
    lr = [math.log(b / a) for a, b in zip(path, path[1:])]
    assert a["realized_vol_1w"] == pytest.approx(statistics.stdev(lr) * math.sqrt(252))
    # D has only two returns before delisting: fewer than vol_min_obs
    dd = lab.filter((pl.col("date") == JAN) & (pl.col("security_id") == 4)).row(0, named=True)
    assert dd["realized_vol_1w"] is None and dd["fwd_ret_1w"] is not None


def test_incomplete_window_is_null(built):
    lab, _ = built
    assert lab.filter(pl.col("date") == FEB)["fwd_ret_1w"].is_not_null().all()
    mar = lab.filter(pl.col("date") == MAR)
    assert mar.height == 3 and mar["fwd_ret_1w"].is_null().all() and mar["target"].is_null().all()


def test_winsorize_per_date(prices, marketcap, securities, cfg):
    lab, _ = build_labels(prices, make_universe(prices), marketcap, securities, rules(winsorize=(0.25, 0.75)),
                          cfg.delisting, cfg.dates)
    jan = {r["security_id"]: r for r in lab.filter(pl.col("date") == JAN).iter_rows(named=True)}
    assert jan[4]["excess_w_1w"] == pytest.approx(jan[2]["excess_1w"])  # D clipped up to B
    assert jan[5]["excess_w_1w"] == pytest.approx(jan[1]["excess_1w"])  # M clipped down to A
    assert jan[3]["excess_w_1w"] == jan[3]["excess_1w"]


def test_price_gap_without_delisting_is_null():
    # G stops trading after Feb 4 with no delisting: a data gap, not an outcome
    p = make_prices({7: ({d(2020, 1, 1): 40}, d(2020, 2, 4), 100.0)})
    u = make_universe(p).with_columns(pl.lit(True).alias("in_universe"))
    mcap = p.select("security_id", "date", pl.lit(100.0).alias("market_cap"))
    empty = pl.DataFrame(schema={"security_id": pl.Int64, "date": pl.Date, "delist_return": pl.Float64})
    bench = benchmark_index(p, u, mcap, empty, CAL)
    entries = u.filter(pl.col("date") == JAN).select("security_id", "date")

    def fwd(stale_days):
        return forward_returns(p, entries, bench, empty, CAL, 5, stale_days, 3)["fwd_ret"][0]

    assert fwd(stale_days=2) is None   # last price 3 trading days before the window end
    assert fwd(stale_days=5) == 0.0


def test_period():
    dates = Dates(d(2000, 1, 1), d(2021, 10, 31), d(2022, 11, 1), d(2025, 10, 31))
    df = pl.DataFrame({"date": [d(2021, 10, 29), d(2022, 3, 31), d(2022, 11, 30), d(2025, 10, 31), d(2025, 11, 28)]})
    assert df.select(period(dates))["period"].to_list() == ["dev", "gap", "holdout", "holdout", "live"]

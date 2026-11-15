"""Size and liquidity features: hand-computed values, null rules, sign conventions and
point-in-time behaviour."""

from __future__ import annotations

import math

import polars as pl
import pytest
from feature_helpers import business_days, d, make_ctx, make_marketcap, make_prices
from polars.testing import assert_frame_equal

from ltforecast.features import size
from ltforecast.features.catalog import load_catalog

CAL = business_days(d(2019, 1, 1), d(2020, 12, 31))
T = d(2020, 3, 31)        # prediction date: the last trading day of March 2020
T2 = d(2020, 6, 30)       # a later prediction date
I_T = CAL.to_list().index(T)

FEATURES = ["log_mcap", "amihud", "turnover"]

# The total-return price alternates 10, 11, 10, 11, ...: the daily return is +0.1 on odd calendar
# positions and -1/11 on even ones, so any 252-day window holds 126 of each.
ZIGZAG = [10.0 if j % 2 == 0 else 11.0 for j in range(len(CAL))]
UP, DOWN = 0.1, 1.0 / 11.0
CLOSE = 20.0              # split-adjusted close (deliberately not the total-return price)
VOLUME = 1.0e5            # shares a day -> $2m a day
MCAP = 1.0e9              # -> 50m shares outstanding at a $20 close


def frames(volume=VOLUME, close=CLOSE, caps=MCAP, spec=None, **kwargs):
    spec = {1: ZIGZAG} if spec is None else spec
    prices = make_prices(spec, CAL, close={sid: close for sid in spec}, volume=volume, **kwargs)
    caps = caps if isinstance(caps, dict) and set(caps) <= set(spec) else {sid: caps for sid in spec}
    return prices, make_marketcap(prices, caps)


def compute(prices, marketcap, entries=((1, T),), **kwargs):
    return size.compute(make_ctx(prices=prices, marketcap=marketcap, entries=list(entries), **kwargs))


def row(out: pl.DataFrame, sid: int = 1, date=T) -> dict:
    rows = out.filter((pl.col("security_id") == sid) & (pl.col("date") == date)).to_dicts()
    assert len(rows) == 1
    return rows[0]


def assert_clean(out: pl.DataFrame) -> None:
    """Float64 features with null for missing, never NaN or inf."""
    for c in FEATURES:
        assert out.schema[c] == pl.Float64
        assert out[c].drop_nulls().is_finite().all(), c


def test_module_matches_the_catalog():
    assert size.FAMILIES == ("size",)
    assert list(size.FEATURES["size"]) == load_catalog().names(family="size")


def test_hand_computed_values():
    out = compute(*frames())
    r = row(out)
    dollar_volume = 20.0 * 1.0e5                                           # close x volume = $2m a day
    assert r["log_mcap"] == pytest.approx(math.log(1.0e9))                 # 20.7233
    # mean |ret| / dollar volume over 126 up days and 126 down days, per $1m traded
    assert r["amihud"] == pytest.approx((126 * UP + 126 * DOWN) / 252 / dollar_volume * 1.0e6)   # 0.04773
    shares = 1.0e9 / 20.0                                                  # market cap / close = 50m
    assert r["turnover"] == pytest.approx(1.0e5 / shares)                  # 0.002 of the shares a day
    assert r["src_date_size"] == T
    assert_clean(out)


def test_one_row_per_entry_and_schema():
    prices, mcap = frames()
    ctx = make_ctx(prices=prices, marketcap=mcap, entries=[(1, T), (1, T2), (99, T)])   # 99: no data at all
    out = size.compute(ctx)
    assert out.height == ctx.entries.height == 3
    assert out.select("security_id", "date").is_unique().all()
    assert out.columns == ["security_id", "date", *FEATURES, "src_date_size"]
    assert out.schema["src_date_size"] == pl.Date
    assert_frame_equal(out.select("security_id", "date"), ctx.entries.select("security_id", "date"))
    assert out.filter(pl.col("src_date_size") > pl.col("date")).is_empty()
    assert_clean(out)
    r = row(out, 99)
    assert all(r[c] is None for c in FEATURES) and r["src_date_size"] is None


def test_amihud_skips_zero_volume_days_and_needs_126_usable_days():
    # volume only on the up days (odd positions): 126 usable days in the window, all with |ret| = 0.1
    volume = [VOLUME if j % 2 == 1 else 0.0 for j in range(len(CAL))]
    out = compute(*frames(volume={1: volume}))
    r = row(out)
    assert r["amihud"] == pytest.approx(UP / (20.0 * 1.0e5) * 1.0e6)       # 0.05, not inf and not diluted
    assert_clean(out)

    up_day = I_T if I_T % 2 == 1 else I_T - 1                              # one more silent day: 125 left
    volume[up_day] = 0.0
    assert row(compute(*frames(volume={1: volume})))["amihud"] is None


def test_turnover_counts_zero_volume_days_as_zero():
    volume = [VOLUME if j % 2 == 1 else 0.0 for j in range(len(CAL))]
    r = row(compute(*frames(volume={1: volume})))
    traded_days = sum(1 for j in range(I_T - 62, I_T + 1) if j % 2 == 1)   # of the last 63 days
    assert traded_days in (31, 32)
    assert r["turnover"] == pytest.approx(traded_days * 1.0e5 / 63 / (1.0e9 / 20.0))


def test_turnover_window_is_63_trading_days_including_t():
    # 9e9 shares a day until 63 days before t (must not count), then 42 days of 1e5 and 21 days of 4e5
    volume = {CAL[0]: 9.0e9, CAL[I_T - 62]: 1.0e5, CAL[I_T - 20]: 4.0e5}
    r = row(compute(*frames(volume={1: volume})))
    mean_volume = (42 * 1.0e5 + 21 * 4.0e5) / 63                           # 2e5
    assert r["turnover"] == pytest.approx(mean_volume / (1.0e9 / 20.0))    # 0.004


def test_amihud_window_is_252_trading_days_including_t():
    base = row(compute(*frames()))["amihud"]
    # one share traded (a huge price impact) on the day just outside the window: no effect
    outside = {CAL[0]: VOLUME, CAL[I_T - 252]: 1.0, CAL[I_T - 251]: VOLUME}
    assert row(compute(*frames(volume={1: outside})))["amihud"] == pytest.approx(base)
    # the same on the oldest day inside the window
    inside = {CAL[0]: VOLUME, CAL[I_T - 251]: 1.0, CAL[I_T - 250]: VOLUME}
    ret = UP if (I_T - 251) % 2 == 1 else DOWN
    others = 126 * UP + 126 * DOWN - ret                                   # the 251 ordinary days
    expected = (others / (20.0 * 1.0e5) + ret / (20.0 * 1.0)) / 252 * 1.0e6
    assert row(compute(*frames(volume={1: inside})))["amihud"] == pytest.approx(expected)


def test_dollar_volume_uses_the_split_adjusted_close():
    base = row(compute(*frames()))
    # a 2:1 split after t doubles close_unadj on every earlier day; close and volume are already adjusted
    prices, mcap = frames(splits={1: {d(2020, 9, 1): 2.0}})
    assert prices.filter(pl.col("date") == T)["close_unadj"].item() == 40.0
    r = row(compute(prices, mcap))
    assert r["amihud"] == base["amihud"] and r["turnover"] == base["turnover"]


def test_market_cap_is_the_latest_value_at_most_a_week_old():
    # last market cap on Monday 2020-03-30 (none at t): used, and dated
    r = row(compute(*frames(caps={1: {CAL[0]: 1.0e9, d(2020, 3, 30): 2.0e9, T: None}})))
    assert r["log_mcap"] == pytest.approx(math.log(2.0e9))
    assert r["turnover"] == pytest.approx(1.0e5 / (2.0e9 / 20.0))

    # last market cap on Thursday 2020-03-19, 12 days before t: stale
    r = row(compute(*frames(caps={1: {CAL[0]: 1.0e9, d(2020, 3, 20): None}})))
    assert r["log_mcap"] is None and r["turnover"] is None
    assert r["amihud"] is not None                                         # needs no market cap
    assert r["src_date_size"] == T


@pytest.mark.parametrize("bad", [0.0, -1.0e9, float("nan")])
def test_non_positive_market_cap_is_missing(bad):
    out = compute(*frames(caps={1: bad}))
    r = row(out)
    assert r["log_mcap"] is None and r["turnover"] is None
    assert_clean(out)


def test_non_positive_close_at_t_gives_null_turnover():
    out = compute(*frames(close={CAL[0]: 20.0, T: 0.0}))
    r = row(out)
    assert r["turnover"] is None                                           # shares outstanding undefined
    # amihud loses the day at t (an up day or a down day) and averages the other 251
    lost = UP if I_T % 2 == 1 else DOWN
    assert r["amihud"] == pytest.approx((126 * UP + 126 * DOWN - lost) / 251 / (20.0 * 1.0e5) * 1.0e6)
    assert_clean(out)


def test_insufficient_history():
    # listed 2019-11-01: 108 prices < 126 for amihud, >= 42 volumes for turnover
    prices, mcap = frames(spec={1: {d(2019, 11, 1): 10.0, d(2019, 11, 4): 11.0}, 2: ZIGZAG})
    assert prices.filter((pl.col("security_id") == 1) & (pl.col("date") <= T)).height == 108
    r = row(compute(prices, mcap))
    assert r["amihud"] is None
    assert r["turnover"] == pytest.approx(1.0e5 / (1.0e9 / 20.0))
    assert r["log_mcap"] == pytest.approx(math.log(1.0e9))

    # listed 2020-03-02: 22 prices -> only the market cap
    prices, mcap = frames(spec={1: {d(2020, 3, 2): 10.0, d(2020, 3, 3): 11.0}, 2: ZIGZAG})
    r = row(compute(prices, mcap))
    assert r["amihud"] is None and r["turnover"] is None
    assert r["log_mcap"] == pytest.approx(math.log(1.0e9))
    assert r["src_date_size"] == T                                         # the market cap is dated t


def test_no_price_row_at_t():
    prices, mcap = frames()
    prices, mcap = prices.filter(pl.col("date") != T), mcap.filter(pl.col("date") != T)
    r = row(compute(prices, mcap, calendar=CAL))
    assert r["amihud"] is None and r["turnover"] is None
    assert r["log_mcap"] == pytest.approx(math.log(1.0e9))                 # yesterday's market cap
    assert r["src_date_size"] == d(2020, 3, 30)                            # the only input used


def test_sign_conventions_are_raw_values():
    # stock 2: a tenth of the volume and a tenth of the market cap
    prices, mcap = frames(spec={1: ZIGZAG, 2: ZIGZAG}, volume={1: 1.0e5, 2: 1.0e4}, caps={1: 1.0e9, 2: 1.0e8})
    out = compute(prices, mcap, entries=[(1, T), (2, T)])
    big, small = row(out, 1), row(out, 2)
    assert small["amihud"] == pytest.approx(10 * big["amihud"])            # less liquid = higher
    assert small["log_mcap"] == pytest.approx(big["log_mcap"] - math.log(10))
    assert small["turnover"] == pytest.approx(big["turnover"])             # 1e4 / 5m = 1e5 / 50m shares
    cat = load_catalog()
    assert [cat.spec(n).sign for n in FEATURES] == [-1, 1, -1]             # signs live in the catalog only


def test_delisted_stock_has_features_at_its_last_month_end():
    last = d(2020, 2, 28)
    n_days = CAL.to_list().index(last) + 1
    prices, mcap = frames(spec={1: ZIGZAG[:n_days], 2: ZIGZAG}, last={1: last})
    out = compute(prices, mcap, entries=[(1, last), (1, T)])
    r = row(out, 1, last)
    assert r["amihud"] == pytest.approx((UP + DOWN) / 2 / (20.0 * 1.0e5) * 1.0e6)
    assert r["turnover"] == pytest.approx(0.002) and r["log_mcap"] == pytest.approx(math.log(1.0e9))
    assert r["src_date_size"] == last
    gone = row(out, 1, T)                                                  # a month after the last price
    assert all(gone[c] is None for c in FEATURES) and gone["src_date_size"] is None


def test_point_in_time_later_data_does_not_change_features_at_t():
    prices, mcap = frames()
    entries = [(1, T), (1, T2)]
    base = compute(prices, mcap, entries)

    after = pl.col("date") > T
    scaled = lambda col, k: pl.when(after).then(pl.col(col) * k).otherwise(pl.col(col)).alias(col)  # noqa: E731
    changed = compute(
        prices.with_columns(scaled("adj_close", 3.0), scaled("close", 5.0), scaled("volume", 7.0), scaled("ret", 9.0)),
        mcap.with_columns(scaled("market_cap", 2.0)),
        entries,
    )
    at_t = pl.col("date") == T
    assert base.filter(at_t).select(FEATURES).null_count().sum_horizontal().item() == 0
    assert_frame_equal(base.filter(at_t), changed.filter(at_t))
    for c in FEATURES:                                                     # the change is real at the later date
        assert row(changed, 1, T2)[c] != row(base, 1, T2)[c]


def test_point_in_time_dropping_everything_after_t_changes_nothing():
    prices, mcap = frames()
    full = compute(prices, mcap)
    cut = compute(prices.filter(pl.col("date") <= T), mcap.filter(pl.col("date") <= T))
    assert_frame_equal(full, cut)


def test_point_in_time_a_market_cap_dated_after_t_is_never_used():
    prices, mcap = frames()
    only_later = mcap.filter(pl.col("date") > T)                           # first market cap at t+1
    r = row(compute(prices, only_later))
    assert r["log_mcap"] is None and r["turnover"] is None
    assert r["amihud"] is not None and r["src_date_size"] == T


def test_point_in_time_the_day_at_t_is_the_last_one_used():
    prices, mcap = frames()
    base = row(compute(prices, mcap))

    def with_volume(day, shares):
        p = prices.with_columns(
            pl.when(pl.col("date") == day).then(shares).otherwise(pl.col("volume")).alias("volume")
        )
        return row(compute(p, mcap))

    assert with_volume(CAL[I_T + 1], 5.0e7) == base                        # t+1: invisible
    on_t = with_volume(T, 6.4e6)                                           # t itself: in both windows
    assert on_t["turnover"] == pytest.approx((62 * 1.0e5 + 6.4e6) / 63 / (1.0e9 / 20.0))   # 0.004
    assert on_t["amihud"] < base["amihud"]

"""Momentum and reversal features: hand-computed values, null rules, sign conventions and
point-in-time behaviour."""

from __future__ import annotations

import math

import polars as pl
import pytest
from feature_helpers import business_days, d, make_ctx, make_prices, make_securities
from polars.testing import assert_frame_equal

from ltforecast.features import momentum
from ltforecast.features.catalog import load_catalog

CAL = business_days(d(2016, 1, 1), d(2020, 12, 31))
T = d(2020, 3, 31)        # prediction date: the last trading day of March 2020
T2 = d(2020, 6, 30)       # a later prediction date
M1 = d(2020, 2, 28)       # t - 1 month  (Feb 29 2020 is a Saturday)
M6 = d(2019, 9, 30)       # t - 6 months
M12 = d(2019, 3, 29)      # t - 12 months (Mar 30/31 2019 is a weekend)
M36 = d(2017, 3, 31)      # t - 36 months
I_T = CAL.to_list().index(T)

FEATURES = ["mom_12_1", "mom_6_1", "ind_mom_6m", "high_52w", "rev_36_12"]

# stock 1: the price on each date that matters, carried forward in between
PATH = {
    d(2016, 1, 1): 50.0,
    M36: 40.0,
    d(2019, 3, 1): 200.0,     # a high more than 252 trading days before t: outside the 52-week window
    M12: 60.0,
    M6: 80.0,
    d(2019, 12, 16): 110.0,   # the 52-week high
    M1: 90.0,
    T: 99.0,
}

# 6-month returns (t-6m to t) of the software names: mean = (0.1 + 0.2 + 0.3 - 0.1 + 0.5) / 5 = 0.2
SOFTWARE = {11: 0.10, 12: 0.20, 13: 0.30, 14: -0.10, 15: 0.50}


def row(out: pl.DataFrame, sid: int = 1, date=T) -> dict:
    rows = out.filter((pl.col("security_id") == sid) & (pl.col("date") == date)).to_dicts()
    assert len(rows) == 1
    return rows[0]


def six_month(ret: float) -> dict:
    """A price path worth 100 at t-6 months and 100 x (1 + ret) at t."""
    return {d(2016, 1, 1): 100.0, T: 100.0 * (1 + ret)}


def assert_clean(out: pl.DataFrame) -> None:
    """Float64 features with null for missing, never NaN or inf."""
    for c in FEATURES:
        assert out.schema[c] == pl.Float64
        assert out[c].drop_nulls().is_finite().all(), c


def industry_ctx(extra_prices=None, extra_sec=None, entries=None, software=None):
    spec = {sid: six_month(r) for sid, r in (SOFTWARE if software is None else software).items()}
    spec.update(extra_prices or {})
    sec = {sid: ("Technology", "Software") for sid in spec}
    sec.update(extra_sec or {})
    prices = make_prices(spec, CAL)
    entries = [(sid, T) for sid in spec] if entries is None else entries
    return make_ctx(prices=prices, securities=make_securities(sec), entries=entries)


def test_module_matches_the_catalog():
    cat = load_catalog()
    assert momentum.FAMILIES == ("momentum", "reversal")
    for family in momentum.FAMILIES:
        assert list(momentum.FEATURES[family]) == cat.names(family=family)


def test_hand_computed_price_features():
    out = momentum.compute(make_ctx(prices=make_prices({1: PATH}, CAL), entries=[(1, T)]))
    r = row(out)
    assert r["mom_12_1"] == pytest.approx(90.0 / 60.0 - 1)     # P(t-1m) / P(t-12m) - 1 = 0.5
    assert r["mom_6_1"] == pytest.approx(90.0 / 80.0 - 1)      # P(t-1m) / P(t-6m) - 1 = 0.125
    assert r["rev_36_12"] == pytest.approx(60.0 / 40.0 - 1)    # P(t-12m) / P(t-36m) - 1 = 0.5
    assert r["high_52w"] == pytest.approx(99.0 / 110.0)        # P(t) / 52-week high = 0.9 (the 200 is too old)
    assert r["ind_mom_6m"] is None                             # no classification
    assert r["src_date_momentum"] == T                         # high_52w used the price at t
    assert r["src_date_reversal"] == M12                       # the reversal window ends at t-12 months
    assert_clean(out)


def test_one_row_per_entry_and_schema():
    prices = make_prices({1: PATH, 2: 10.0}, CAL)
    entries = [(1, T), (1, T2), (2, T), (99, T)]               # 99 has no prices at all
    ctx = make_ctx(prices=prices, entries=entries)
    out = momentum.compute(ctx)
    assert out.height == ctx.entries.height == 4
    assert out.select("security_id", "date").is_unique().all()
    assert out.columns == ["security_id", "date", *FEATURES, "src_date_momentum", "src_date_reversal"]
    assert out.schema["src_date_momentum"] == pl.Date and out.schema["src_date_reversal"] == pl.Date
    assert_frame_equal(out.select("security_id", "date"), ctx.entries.select("security_id", "date"))
    assert_clean(out)
    for c in ("src_date_momentum", "src_date_reversal"):
        assert out.filter(pl.col(c) > pl.col("date")).is_empty()
    r = row(out, 99)
    assert all(r[c] is None for c in FEATURES)
    assert r["src_date_momentum"] is None and r["src_date_reversal"] is None


def test_industry_momentum_is_the_equal_weight_mean_including_the_stock_itself():
    out = momentum.compute(industry_ctx())
    mean = (0.10 + 0.20 + 0.30 - 0.10 + 0.50) / 5              # 0.2
    for sid in SOFTWARE:                                       # every member gets the same value,
        assert row(out, sid)["ind_mom_6m"] == pytest.approx(mean)   # which includes its own return
        assert row(out, sid)["src_date_momentum"] == T


def test_industry_momentum_needs_five_names_with_a_return():
    four = {sid: SOFTWARE[sid] for sid in (11, 12, 13, 14)}
    out = momentum.compute(industry_ctx(software=four))
    assert all(row(out, sid)["ind_mom_6m"] is None for sid in four)

    # a fifth member that was listed 4 months ago has no 6-month return: still only four returns
    young = {16: {d(2019, 12, 2): 100.0}}
    out = momentum.compute(industry_ctx(software=four, extra_prices=young))
    assert all(row(out, sid)["ind_mom_6m"] is None for sid in (11, 12, 13, 14, 16))


def test_a_member_without_its_own_return_gets_the_industry_mean():
    out = momentum.compute(industry_ctx(extra_prices={16: {d(2019, 12, 2): 100.0}}))
    assert row(out, 16)["ind_mom_6m"] == pytest.approx(0.2)    # the mean of the five others
    assert row(out, 11)["ind_mom_6m"] == pytest.approx(0.2)    # 16 adds nothing to the mean


def test_industry_momentum_uses_only_universe_members_on_that_date():
    outsider = {17: six_month(9.0)}                            # same industry, a +900% return
    members = [(sid, T) for sid in SOFTWARE]
    out = momentum.compute(industry_ctx(extra_prices=outsider, entries=members))
    assert row(out, 11)["ind_mom_6m"] == pytest.approx(0.2)    # 17 is not an entry at t: ignored

    out = momentum.compute(industry_ctx(extra_prices=outsider))   # now 17 is a member
    assert row(out, 11)["ind_mom_6m"] == pytest.approx((1.0 + 9.0) / 6)


def test_industry_falls_back_to_sector_when_famaindustry_is_missing():
    energy = {31: 0.00, 32: 0.05, 33: 0.10, 34: 0.15, 35: 0.20}     # sector Energy, no famaindustry
    extra = {sid: six_month(r) for sid, r in energy.items()}
    extra[36] = six_month(5.0)       # sector Energy but famaindustry "Oil": its own (one-name) group
    extra[41] = six_month(0.3)       # in a sector-less, industry-less row
    extra[42] = six_month(0.3)       # not in securities at all
    sec = {sid: ("Energy", None) for sid in energy}
    sec.update({36: ("Energy", "Oil"), 41: (None, None)})
    ctx = industry_ctx(extra_prices=extra)
    ctx = make_ctx(
        prices=ctx.prices,
        securities=make_securities({**{sid: ("Technology", "Software") for sid in SOFTWARE}, **sec}),
        entries=ctx.entries,
    )
    out = momentum.compute(ctx)
    for sid in energy:
        assert row(out, sid)["ind_mom_6m"] == pytest.approx((0.00 + 0.05 + 0.10 + 0.15 + 0.20) / 5)   # 0.1
    assert row(out, 36)["ind_mom_6m"] is None     # one name in "Oil"; never mixed into the sector group
    assert row(out, 41)["ind_mom_6m"] is None
    assert row(out, 42)["ind_mom_6m"] is None
    assert row(out, 11)["ind_mom_6m"] == pytest.approx(0.2)    # software is unaffected by the others


def test_high_52w_window_is_252_trading_days_including_t():
    def high(spike_day):
        path = {CAL[0]: 10.0, spike_day: 20.0, CAL[CAL.to_list().index(spike_day) + 1]: 10.0}
        return row(momentum.compute(make_ctx(prices=make_prices({1: path}, CAL), entries=[(1, T)])))["high_52w"]

    assert high(CAL[I_T - 251]) == pytest.approx(10.0 / 20.0)   # the 252nd day back, counting t: in the window
    assert high(CAL[I_T - 252]) == pytest.approx(1.0)           # one day older: out
    # a new high at t itself counts: the feature is 1.0 at the high and never above it
    path = {CAL[0]: 10.0, T: 30.0}
    assert row(momentum.compute(make_ctx(prices=make_prices({1: path}, CAL), entries=[(1, T)])))["high_52w"] == 1.0


def test_stock_listed_eight_months_ago():
    prices = make_prices({1: {d(2019, 7, 31): 10.0, M1: 12.0}}, CAL)
    r = row(momentum.compute(make_ctx(prices=prices, entries=[(1, T)])))
    assert r["mom_12_1"] is None                               # no price at t-12 months
    assert r["mom_6_1"] == pytest.approx(12.0 / 10.0 - 1)      # both ends exist: 0.2
    assert r["rev_36_12"] is None and r["src_date_reversal"] is None
    assert r["high_52w"] == pytest.approx(1.0)                 # about 175 prices >= MIN_HIGH_DAYS


def test_stock_listed_three_months_ago_has_nothing():
    prices = make_prices({1: {d(2020, 1, 2): 10.0}, 2: 10.0}, CAL)
    r = row(momentum.compute(make_ctx(prices=prices, entries=[(1, T), (2, T)])))
    n_prices = prices.filter((pl.col("security_id") == 1) & (pl.col("date") <= T)).height
    assert n_prices == 64 < momentum.MIN_HIGH_DAYS
    assert all(r[c] is None for c in FEATURES)
    assert r["src_date_momentum"] is None


def test_no_carry_forward_over_a_missing_month_end_price():
    full = make_prices({1: PATH}, CAL)
    gap = full.filter(pl.col("date") != M1)                    # no print on the t-1 month end
    r = row(momentum.compute(make_ctx(prices=gap, entries=[(1, T)], calendar=CAL)))
    assert r["mom_12_1"] is None and r["mom_6_1"] is None
    assert r["rev_36_12"] == pytest.approx(0.5)                # its two dates are intact
    assert r["src_date_momentum"] == T                         # only high_52w is left

    no_t = full.filter(pl.col("date") != T)                    # no print at t
    r = row(momentum.compute(make_ctx(prices=no_t, entries=[(1, T)], calendar=CAL)))
    assert r["high_52w"] is None
    assert r["mom_12_1"] == pytest.approx(0.5)
    assert r["src_date_momentum"] == M1                        # the latest price actually used


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf")])
def test_bad_prices_give_null_not_nan_or_inf(bad):
    prices = make_prices({1: PATH}, CAL).with_columns(
        pl.when(pl.col("date") == M12).then(bad).otherwise(pl.col("adj_close")).alias("adj_close")
    )
    out = momentum.compute(make_ctx(prices=prices, entries=[(1, T)]))
    r = row(out)
    assert r["mom_12_1"] is None                               # P(t-12m) is the denominator
    assert r["rev_36_12"] is None                              # and the numerator here
    assert r["mom_6_1"] == pytest.approx(0.125)
    assert r["high_52w"] == pytest.approx(0.9) and not math.isnan(r["high_52w"])
    assert_clean(out)


def test_sign_conventions_are_raw_returns():
    winner = {CAL[0]: 10.0, M36: 10.0, M12: 30.0, M1: 60.0}    # up 200% then up 100%
    loser = {CAL[0]: 10.0, M36: 10.0, M12: 5.0, M1: 4.0}       # down 50% then down 20%
    out = momentum.compute(make_ctx(prices=make_prices({1: winner, 2: loser}, CAL), entries=[(1, T), (2, T)]))
    w, lo = row(out, 1), row(out, 2)
    assert w["mom_12_1"] == pytest.approx(1.0) and lo["mom_12_1"] == pytest.approx(-0.2)
    # rev_36_12 is the plain past return (positive for the past winner); the catalog carries sign -1
    assert w["rev_36_12"] == pytest.approx(2.0) and lo["rev_36_12"] == pytest.approx(-0.5)
    assert lo["high_52w"] == pytest.approx(4.0 / 5.0) and w["high_52w"] == pytest.approx(1.0)
    cat = load_catalog()
    assert [cat.spec(n).sign for n in FEATURES] == [1, 1, 1, 1, -1]


def test_skip_month_returns_ignore_the_latest_month():
    base = row(momentum.compute(make_ctx(prices=make_prices({1: PATH}, CAL), entries=[(1, T)])))
    moved = row(momentum.compute(make_ctx(prices=make_prices({1: {**PATH, T: 45.0}}, CAL), entries=[(1, T)])))
    for c in ("mom_12_1", "mom_6_1", "rev_36_12"):
        assert moved[c] == base[c]                             # the price at t is not in these windows
    assert moved["high_52w"] == pytest.approx(45.0 / 110.0)    # but it is in this one


def test_delisted_stock_has_features_up_to_its_last_month_end():
    path = {d(2016, 1, 1): 10.0, d(2019, 2, 28): 20.0, d(2020, 1, 31): 30.0}
    prices = make_prices({1: path, 2: 10.0}, CAL, last={1: M1})   # stock 2 keeps the calendar going
    out = momentum.compute(make_ctx(prices=prices, entries=[(1, M1), (1, T)]))
    r = row(out, 1, M1)                                        # months back from Feb 2020
    assert r["mom_12_1"] == pytest.approx(30.0 / 20.0 - 1)     # P(2020-01-31) / P(2019-02-28) - 1
    assert r["mom_6_1"] == pytest.approx(30.0 / 20.0 - 1)      # P(2020-01-31) / P(2019-08-30) - 1
    assert r["rev_36_12"] == pytest.approx(20.0 / 10.0 - 1)    # P(2019-02-28) / P(2017-02-28) - 1
    assert r["high_52w"] == pytest.approx(1.0)
    assert r["src_date_momentum"] == M1
    assert row(out, 1, T)["high_52w"] is None                  # no price at t after the delisting
    assert_clean(out)


def pit_frames():
    spec = {1: PATH, **{sid: six_month(r) for sid, r in SOFTWARE.items()}}
    prices = make_prices(spec, CAL)
    securities = make_securities({sid: ("Technology", "Software") for sid in spec})
    return prices, securities, [(sid, day) for sid in spec for day in (T, T2)]


def test_point_in_time_later_prices_do_not_change_features_at_t():
    prices, securities, entries = pit_frames()
    base = momentum.compute(make_ctx(prices=prices, securities=securities, entries=entries))

    after = pl.col("date") > T
    changed_prices = prices.with_columns(
        pl.when(after).then(pl.col("adj_close") * 3.0).otherwise(pl.col("adj_close")).alias("adj_close"),
        pl.when(after).then(pl.col("close") * 3.0).otherwise(pl.col("close")).alias("close"),
        pl.when(after).then(pl.lit(9.0)).otherwise(pl.col("ret")).alias("ret"),
    )
    changed = momentum.compute(make_ctx(prices=changed_prices, securities=securities, entries=entries))

    at_t = pl.col("date") == T
    assert base.filter(at_t).height == 6
    assert base.filter(at_t).select(FEATURES).null_count().sum_horizontal().item() == 0   # every feature is live
    assert_frame_equal(base.filter(at_t), changed.filter(at_t))
    # the change is real: the later date sees it
    assert row(changed, 1, T2)["mom_12_1"] != row(base, 1, T2)["mom_12_1"]
    assert row(changed, 1, T2)["ind_mom_6m"] != row(base, 1, T2)["ind_mom_6m"]


def test_point_in_time_dropping_everything_after_t_changes_nothing():
    prices, securities, entries = pit_frames()
    at_t = [e for e in entries if e[1] == T]
    full = momentum.compute(make_ctx(prices=prices, securities=securities, entries=at_t))
    cut = momentum.compute(
        make_ctx(prices=prices.filter(pl.col("date") <= T), securities=securities, entries=at_t)
    )
    assert_frame_equal(full, cut)
    assert row(full)["ind_mom_6m"] is not None and row(full)["high_52w"] is not None


def test_point_in_time_each_window_ends_where_documented():
    """A price change at t+1 moves nothing; a change at t moves only the features whose window
    includes t (ind_mom_6m, high_52w)."""
    prices, securities, entries = pit_frames()
    at_t = [e for e in entries if e[1] == T]
    base = momentum.compute(make_ctx(prices=prices, securities=securities, entries=at_t))

    def bump(day):
        p = prices.with_columns(
            pl.when((pl.col("date") == day) & (pl.col("security_id") == 1))
            .then(pl.col("adj_close") * 0.5)
            .otherwise(pl.col("adj_close"))
            .alias("adj_close")
        )
        return momentum.compute(make_ctx(prices=p, securities=securities, entries=at_t))

    assert_frame_equal(base, bump(CAL[I_T + 1]))
    on_t = bump(T)
    for c in ("mom_12_1", "mom_6_1", "rev_36_12"):
        assert on_t[c].to_list() == base[c].to_list()
    assert row(on_t)["high_52w"] == pytest.approx(49.5 / 110.0)
    # stock 1's 6-month return drops from 99/80 - 1 to 49.5/80 - 1, and with it the six-name mean
    assert row(base, 11)["ind_mom_6m"] == pytest.approx((1.0 + 99.0 / 80.0 - 1) / 6)
    assert row(on_t, 11)["ind_mom_6m"] == pytest.approx((1.0 + 49.5 / 80.0 - 1) / 6)

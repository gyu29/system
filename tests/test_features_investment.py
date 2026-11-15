"""Investment / issuance features on hand-built filings: values, split handling, null rules, point-in-time.

The trading calendar is every weekday of 2016-2020. Quarterly filings arrive 45 calendar days after the
period end and are usable from the next trading day: the quarter ending 2019-12-31 is filed on Friday
2020-02-14 and first used on Monday 2020-02-17. T = Friday 2020-02-28 is a month end.
"""

import math

import polars as pl
import pytest
from conftest import business_days
from feature_helpers import d, make_ctx, make_filings, make_prices, period_ends
from polars.testing import assert_frame_equal

from ltforecast.features import investment
from ltforecast.features.catalog import load_catalog

CAL = business_days(d(2016, 1, 1), d(2020, 12, 31))
Q5 = period_ends(d(2019, 12, 31), 5)   # 2018-12-31 ... 2019-12-31: the latest quarter and four before it
FILED = d(2020, 2, 14)
T = d(2020, 2, 28)
FEATURES = ["asset_growth", "net_issuance", "capex_growth"]


def filings(sid=1, ends=Q5, *, assets=None, sharesbas=None, capex=None, lag_days=45) -> list[tuple]:
    """ARQ rows (assets, sharesbas) and ART rows (capex), one value per period end."""
    rows = make_filings(sid, "ARQ", ends, lag_days=lag_days, assets=assets, sharesbas=sharesbas)
    return rows + make_filings(sid, "ART", ends, lag_days=lag_days, capex=capex)


def run(rows, entries=((1, T),), *, splits=None, prices=None) -> pl.DataFrame:
    if prices is None:
        ids = sorted({r[0] for r in rows} | {e[0] for e in entries})
        prices = make_prices({sid: 10.0 for sid in ids}, CAL, splits=splits)
    return investment.compute(make_ctx(prices=prices, fundamentals=rows, entries=list(entries), calendar=CAL))


def one(rows, t=T, **kw) -> dict:
    return run(rows, ((1, t),), **kw).row(0, named=True)


def test_families_match_the_catalog():
    assert investment.FAMILIES == {"investment": tuple(load_catalog().names(family="investment"))}


def test_hand_computed_values():
    rows = filings(
        assets=[1000, 1100, 1150, 1200, 1250], sharesbas=[100, 100, 105, 105, 110], capex=[-50, -55, -60, -70, -80]
    )
    out = one(rows)
    assert out["asset_growth"] == pytest.approx(1250 / 1000 - 1)        # 0.25
    assert out["net_issuance"] == pytest.approx(math.log(110 / 100))    # 0.0953...
    assert out["capex_growth"] == pytest.approx(80 / 50 - 1)            # 0.6
    assert out["src_date_investment"] == FILED


def test_sign_conventions():
    # shrinking assets, a buyback and a capex cut are all negative
    out = one(filings(assets=[1000, 0, 0, 0, 800], sharesbas=[100, 0, 0, 0, 90], capex=[-50, 0, 0, 0, -40]))
    assert out["asset_growth"] == pytest.approx(-0.2)
    assert out["net_issuance"] == pytest.approx(math.log(0.9))
    assert out["capex_growth"] == pytest.approx(-0.2)
    # capex reported as a positive number gives the same growth as the usual negative outflow
    assert one(filings(capex=[50, 0, 0, 0, 80]))["capex_growth"] == pytest.approx(0.6)
    # spending cut to nothing is -100%
    assert one(filings(capex=[-50, 0, 0, 0, 0]))["capex_growth"] == pytest.approx(-1.0)


def test_output_shape_one_row_per_entry():
    rows = filings(1, assets=[1000] * 5, sharesbas=[100] * 5, capex=[-50] * 5) + filings(
        2, Q5[:2], assets=[10, 20], sharesbas=[1, 1], capex=[-1, -1]
    )
    entries = [(sid, day) for sid in (1, 2, 3) for day in (d(2019, 12, 31), d(2020, 1, 31), T, d(2020, 3, 31))]
    ctx = make_ctx(prices=make_prices({1: 10.0, 2: 10.0, 3: 10.0}, CAL), fundamentals=rows, entries=entries, calendar=CAL)
    out = investment.compute(ctx)
    assert out.columns == ["security_id", "date", *FEATURES, "src_date_investment"]
    assert out.height == ctx.entries.height == 12
    assert out.select("security_id", "date").is_unique().all()
    assert_frame_equal(out.select("security_id", "date"), ctx.entries)
    assert [out.schema[c] for c in FEATURES] == [pl.Float64] * 3
    assert out.schema["src_date_investment"] == pl.Date
    for c in FEATURES:
        assert not out[c].is_nan().any() and not out[c].is_infinite().any()
    # security 1 has values only once the fifth quarter is usable; 2 (two quarters) and 3 (no filings) never
    filled = out.filter(pl.col("asset_growth").is_not_null())
    assert filled.select("security_id", "date").rows() == [(1, T), (1, d(2020, 3, 31))]
    empty = out.filter(pl.col("asset_growth").is_null())
    assert empty.select(pl.all_horizontal(pl.col(*FEATURES, "src_date_investment").is_null()).all()).item()
    assert (out.drop_nulls("src_date_investment")["src_date_investment"] <= out.drop_nulls("src_date_investment")["date"]).all()


# ---- splits ----

def test_split_between_the_two_measurements_is_not_issuance():
    # 100 shares at 2018-12-31, a 2:1 split on 2019-06-03, 200 shares at 2019-12-31: nothing was issued
    rows = filings(sharesbas=[100, 100, 200, 200, 200])
    assert one(rows, splits={1: {d(2019, 6, 3): 2.0}})["net_issuance"] == 0.0
    # without the split in the price table the same counts are a doubling
    assert one(rows)["net_issuance"] == pytest.approx(math.log(2))


def test_split_plus_real_issuance():
    # 2:1 split, then 10% more shares: 100 -> 200 -> 220, net issuance log(220 / (100 x 2)) = log(1.1)
    rows = filings(sharesbas=[100, 100, 200, 210, 220])
    assert one(rows, splits={1: {d(2019, 6, 3): 2.0}})["net_issuance"] == pytest.approx(math.log(1.1))


def test_reverse_split_and_two_splits():
    # 1:4 reverse split on 2019-05-01: 400 -> 100 shares (the 2019-03-31 count, filed 2019-05-15, is post-split)
    assert one(filings(sharesbas=[400, 100, 100, 100, 100]), splits={1: {d(2019, 5, 1): 0.25}})["net_issuance"] == 0.0
    # 2:1 on 2019-03-01 then 3:1 on 2019-09-03: 100 -> 600 shares, and 660 after a 10% issue
    two = {1: {d(2019, 3, 1): 2.0, d(2019, 9, 3): 3.0}}
    assert one(filings(sharesbas=[100, 200, 200, 600, 660]), splits=two)["net_issuance"] == pytest.approx(math.log(1.1))


@pytest.mark.parametrize(
    "split_day",
    [
        d(2020, 2, 14),   # on the filing date (the filed figures were fixed before it)
        d(2020, 2, 20),   # after the filing, before the prediction date
        T,                # on the prediction date
        d(2020, 3, 16),   # after the prediction date
    ],
)
def test_split_after_the_latest_measurement_changes_nothing(split_day):
    # both counts were filed before the split (on or before 2020-02-14), so both are on the pre-split basis
    rows = filings(sharesbas=[100, 100, 100, 100, 110])
    out = one(rows, splits={1: {split_day: 2.0}})
    assert out["net_issuance"] == pytest.approx(math.log(1.1))
    assert out == one(rows)


def test_split_between_the_period_end_and_the_filing_is_already_in_the_filed_count():
    # a 2:1 split on 2020-01-15, after the 2019-12-31 period end but before its 2020-02-14 filing: the
    # filing reports the count on the post-split basis (ASC 260, cover-page count), 100 -> 200 shares
    split = {1: {d(2020, 1, 15): 2.0}}
    assert one(filings(sharesbas=[100, 100, 100, 100, 200]), splits=split)["net_issuance"] == 0.0
    # ... and 220 is a 10% issue on top of the split
    out = one(filings(sharesbas=[100, 100, 100, 100, 220]), splits=split)
    assert out["net_issuance"] == pytest.approx(math.log(1.1))


def test_split_before_the_filing_of_the_older_period_is_already_in_its_count():
    # a 2:1 split on 2019-01-15, after the 2018-12-31 period end, before its 2019-02-14 filing: that
    # filing and every later one report 200 shares, so nothing was issued over the year
    assert one(filings(sharesbas=[200, 200, 200, 200, 200]), splits={1: {d(2019, 1, 15): 2.0}})["net_issuance"] == 0.0


def test_split_on_the_period_end_counts_as_before_the_measurement():
    # convention: a count filed on day F reflects every split with ex-date < F
    rows = filings(sharesbas=[100, 100, 100, 100, 200])
    assert one(rows, splits={1: {d(2019, 12, 31): 2.0}})["net_issuance"] == 0.0
    # a split on the earlier period end is already in the earlier count: 200 -> 200
    rows = filings(sharesbas=[200, 200, 200, 200, 200])
    assert one(rows, splits={1: {d(2018, 12, 31): 2.0}})["net_issuance"] == 0.0


def test_split_before_both_measurements_and_other_securities_split():
    rows = filings(sharesbas=[200, 200, 200, 200, 220])
    assert one(rows, splits={1: {d(2018, 6, 1): 2.0}})["net_issuance"] == pytest.approx(math.log(1.1))
    # a split of security 2 does not touch security 1
    out = run(rows + filings(2, sharesbas=[100, 100, 200, 200, 200]), [(1, T), (2, T)], splits={2: {d(2019, 6, 3): 2.0}})
    assert out["net_issuance"].to_list() == [pytest.approx(math.log(1.1)), 0.0]


def test_split_adjusted_context_keeps_filing_dates_and_other_metrics():
    rows = filings(assets=[1000] * 5, sharesbas=[100, 100, 200, 200, 200], capex=[-1] * 5)
    ctx = make_ctx(
        prices=make_prices({1: 10.0}, CAL, splits={1: {d(2019, 6, 3): 2.0}}), fundamentals=rows, entries=[(1, T)], calendar=CAL
    )
    adj = investment.split_adjusted_context(ctx, "ARQ", ["sharesbas"], ["assets"]).fundamentals
    assert adj["dimension"].unique().to_list() == ["ARQ"]
    shares = adj.filter(pl.col("metric") == "sharesbas").sort("fiscal_period_end")
    assert shares["value"].to_list() == [100.0, 100.0, 100.0, 100.0, 100.0]   # 200 / 2 after the split
    assert adj.filter(pl.col("metric") == "assets")["value"].to_list() == [1000.0] * 5
    filed = adj.group_by("fiscal_period_end").agg(pl.col("available_date").min()).sort("fiscal_period_end")
    assert filed["available_date"].to_list() == [e + (FILED - Q5[-1]) for e in Q5]


# ---- null rules ----

def test_null_until_five_quarters_exist():
    rows = filings(ends=Q5[1:], assets=[1, 2, 3, 4], sharesbas=[1, 1, 1, 1], capex=[-1, -1, -1, -1])
    out = one(rows)
    assert [out[c] for c in FEATURES] == [None, None, None]
    assert out["src_date_investment"] is None


def test_null_only_when_the_quarter_four_back_is_missing():
    # 2018-12-31 (four quarters before the latest) was never filed: nothing to compare with, hence null
    ends = [d(2018, 9, 30), d(2019, 3, 31), d(2019, 6, 30), d(2019, 9, 30), d(2019, 12, 31)]
    out = one(filings(ends=ends, assets=[1, 2, 3, 4, 5], sharesbas=[1] * 5, capex=[-1] * 5))
    assert [out[c] for c in FEATURES] == [None, None, None]


def test_a_skipped_intermediate_quarter_does_not_shift_the_lag():
    # 2019-03-31 is missing, but 2018-12-31 (four quarters back) exists: same result as with no gap
    gap = Q5[:1] + Q5[2:]
    with_gap = one(filings(ends=gap, assets=[100, 3, 4, 125], sharesbas=[10, 1, 1, 11], capex=[-2, -1, -1, -3]))
    full = one(filings(assets=[100, 7, 3, 4, 125], sharesbas=[10, 9, 1, 1, 11], capex=[-2, -9, -1, -1, -3]))
    assert all(with_gap[c] is not None for c in FEATURES)
    assert [with_gap[c] for c in FEATURES] == pytest.approx([full[c] for c in FEATURES])
    assert with_gap["asset_growth"] == pytest.approx(0.25)


@pytest.mark.parametrize("bad", [0.0, -100.0])
def test_zero_or_negative_denominators_are_null(bad):
    out = one(filings(assets=[bad, 1, 1, 1, 1250], sharesbas=[bad, 1, 1, 1, 110], capex=[0.0, -1, -1, -1, -80]))
    assert [out[c] for c in FEATURES] == [None, None, None]
    assert out["src_date_investment"] is None
    # a zero or negative share count in the latest quarter has no log either
    assert one(filings(sharesbas=[100, 1, 1, 1, bad]))["net_issuance"] is None


def test_missing_metric_only_nulls_its_own_feature():
    out = one(filings(assets=[1000, 1, 1, 1, 1250], sharesbas=[100, 1, 1, 1, None], capex=[None, -1, -1, -1, -80]))
    assert out["asset_growth"] == pytest.approx(0.25)
    assert out["net_issuance"] is None and out["capex_growth"] is None
    assert out["src_date_investment"] == FILED


def test_stale_filing_is_null():
    rows = filings(ends=period_ends(d(2018, 3, 31), 5), assets=[1000, 1, 1, 1, 1250], sharesbas=[100] * 5, capex=[-5] * 5)
    assert one(rows, t=d(2018, 6, 29))["asset_growth"] == pytest.approx(0.25)
    out = one(rows)   # filed 2018-05-15, more than 550 days before 2020-02-28
    assert [out[c] for c in FEATURES] == [None, None, None]


# ---- point in time ----

BASE = filings(assets=[1000, 1100, 1150, 1200, 1250], sharesbas=[100, 100, 105, 105, 110], capex=[-50, -55, -60, -70, -80])
PIT_ENTRIES = [(1, d(2020, 1, 31)), (1, FILED), (1, T)]


def test_filing_is_not_used_on_its_own_date():
    six = period_ends(d(2019, 12, 31), 6)
    vals = dict(assets=[900, 1000, 1100, 1150, 1200, 1250], sharesbas=[90, 100, 100, 105, 105, 110], capex=[-40, -50, -55, -60, -70, -80])
    with_new = filings(ends=six, **vals)
    without = filings(ends=six[:5], **{k: v[:5] for k, v in vals.items()})
    # on the filing date itself (2020-02-14) the features still come from the quarter before it
    on_day = one(with_new, t=FILED)
    assert on_day == one(without, t=FILED)
    assert on_day["asset_growth"] == pytest.approx(1200 / 900 - 1)
    assert on_day["src_date_investment"] == d(2019, 11, 14)
    # the next trading day they move to the new filing
    after = one(with_new, t=d(2020, 2, 17))
    assert after["asset_growth"] == pytest.approx(0.25)
    assert after["net_issuance"] == pytest.approx(math.log(1.1))
    assert after["capex_growth"] == pytest.approx(0.6)
    assert after["src_date_investment"] == FILED


@pytest.mark.parametrize(
    "later",
    [
        # the next quarter, filed after the last prediction date
        filings(ends=[d(2020, 3, 31)], assets=[9999], sharesbas=[9999], capex=[-9999]),
        # a filing made on the last prediction date itself
        filings(ends=[d(2020, 1, 31)], assets=[9999], sharesbas=[9999], capex=[-9999], lag_days=28),
    ],
    ids=["later-filing", "filing-on-t"],
)
def test_later_filings_do_not_change_earlier_dates(later):
    assert later[0][3] >= T
    assert_frame_equal(run(BASE + later, PIT_ENTRIES), run(BASE, PIT_ENTRIES))
    # ... and the test has teeth: the same rows do change a later date
    after = [(1, d(2020, 6, 30))]
    assert run(BASE + later, after).row(0) != run(BASE, after).row(0)


def test_year_ago_quarter_first_filed_after_the_prediction_date_is_not_used():
    # the 2018-12-31 quarter reaches the market only on 2020-03-10: at T there is no year-ago value
    recent = filings(ends=Q5[1:], assets=[1100, 1150, 1200, 1250], sharesbas=[100, 105, 105, 110], capex=[-55, -60, -70, -80])
    late = filings(ends=Q5[:1], assets=[1000], sharesbas=[100], capex=[-50], lag_days=435)
    assert late[0][3] == d(2020, 3, 10)
    out = run(recent + late, PIT_ENTRIES)
    assert_frame_equal(out, run(recent, PIT_ENTRIES))
    assert out.select(pl.all_horizontal(pl.col(*FEATURES).is_null()).all()).item()


def test_later_prices_and_splits_do_not_change_earlier_dates():
    base = run(BASE, PIT_ENTRIES)
    changed = make_prices(
        {1: {CAL[0]: 10.0, d(2020, 3, 2): 500.0}}, CAL, splits={1: {d(2020, 3, 2): 2.0, d(2020, 6, 1): 5.0}}
    )
    assert_frame_equal(run(BASE, PIT_ENTRIES, prices=changed), base)
    # prices cut off at the prediction date give the same answer as the full history
    cut = make_prices({1: 10.0}, CAL).filter(pl.col("date") <= T)
    assert_frame_equal(run(BASE, PIT_ENTRIES, prices=cut), base)


def test_src_date_never_after_the_prediction_date():
    entries = [(1, day) for day in CAL.filter((CAL >= d(2019, 6, 1)) & (CAL <= d(2020, 6, 30))).to_list()]
    out = run(BASE, entries).drop_nulls("src_date_investment")
    assert out.height > 0
    assert (out["src_date_investment"] < out["date"]).all()
    assert out["src_date_investment"].unique().to_list() == [FILED]
    assert out["date"].min() == d(2020, 2, 17)

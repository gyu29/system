"""Earnings features: hand-computed values, null rules, split adjustment and point-in-time behaviour."""

from __future__ import annotations

import math

import polars as pl
import pytest
from feature_helpers import business_days, d, make_ctx, make_filings, make_prices, period_ends

from ltforecast.features import earnings
from ltforecast.features.catalog import load_catalog

CAL = business_days(d(2015, 1, 1), d(2020, 12, 31))
T = d(2020, 4, 30)  # a Thursday; the prediction date of most tests
QUARTERS = period_ends(d(2019, 12, 31), 12)        # 2017-03-31 .. 2019-12-31
QUARTERS_14 = period_ends(d(2019, 12, 31), 14)     # 2016-09-30 .. 2019-12-31
YEARS = period_ends(d(2019, 12, 31), 4, months=12)  # 2016-12-31 .. 2019-12-31
SPLIT_DAY = d(2018, 8, 20)  # after the 6th quarter is filed (2018-08-09), before the 7th ends; between the 2nd and 3rd year

# quarterly eps, oldest first. Year-over-year changes, oldest first:
# 0.1, 0.3, 0.2, 0.4 (2018 vs 2017) then 0.1, 0.2, 0.3, 0.5 (2019 vs 2018)
EPS_Q = [1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 1.2, 1.4, 1.2, 1.5, 1.5, 1.9]
CHANGES_Q = [0.1, 0.3, 0.2, 0.4, 0.1, 0.2, 0.3, 0.5]
PRIOR_Q = CHANGES_Q[:-1]  # the denominator of sue excludes the latest change
# annual eps, oldest first. Changes: +0.5, -0.3, +0.8
EPS_Y = [1.0, 1.5, 1.2, 2.0]


def sample_std(xs: list[float]) -> float:
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def quarterly(eps, sid: int = 1, ends=QUARTERS) -> list[tuple]:
    # revenue is always reported, so a period whose eps is None still exists as a filed period
    return make_filings(sid, "ARQ", ends, lag_days=40, eps=eps, revenue=100.0)


def annual(eps, sid: int = 1, ends=YEARS) -> list[tuple]:
    return make_filings(sid, "ARY", ends, lag_days=60, eps=eps, revenue=400.0)


def run(fund, *, entries=((1, T),), **kw) -> pl.DataFrame:
    kw.setdefault("calendar", CAL)
    return earnings.compute(make_ctx(fundamentals=list(fund), entries=list(entries), **kw))


def one(fund, col: str, **kw):
    out = run(fund, **kw)
    assert out.height == 1
    return out[col][0]


def split_prices(day=SPLIT_DAY, factor: float = 2.0) -> pl.DataFrame:
    return make_prices({1: 10.0}, CAL, splits={1: {day: factor}})


# ---- shape ----


def test_module_matches_the_catalog():
    assert earnings.FAMILIES == ("earnings",)
    assert tuple(load_catalog().names(family="earnings")) == earnings.FEATURES


def test_one_row_per_entry_with_float_columns_and_no_nan():
    month_ends = [d(2019, 12, 31), d(2020, 1, 31), d(2020, 2, 28), d(2020, 3, 31), T]
    entries = [(sid, day) for sid in (1, 2, 3) for day in month_ends]
    fund = quarterly(EPS_Q) + annual(EPS_Y) + quarterly(0.0, sid=2) + annual(0.0, sid=2)  # 3 has no filings
    ctx = make_ctx(fundamentals=fund, entries=entries, calendar=CAL)
    out = earnings.compute(ctx)

    assert out.columns == ["security_id", "date", "sue", "eps_growth_stability", "src_date_earnings"]
    assert out.schema["sue"] == pl.Float64 and out.schema["eps_growth_stability"] == pl.Float64
    assert out.schema["src_date_earnings"] == pl.Date
    assert out.height == ctx.entries.height == 15
    assert out.select("security_id", "date").is_unique().all()
    assert out.select("security_id", "date").equals(ctx.entries.select("security_id", "date"))
    for col in earnings.FEATURES:
        assert out[col].drop_nulls().is_finite().all()
    # all-zero eps: zero std (sue) and zero mean |eps| (stability) are null, never NaN or inf
    assert out.filter(pl.col("security_id") != 1).select(pl.all_horizontal(pl.col(earnings.FEATURES).is_null()).all()).item()
    assert out.filter(pl.col("security_id") != 1)["src_date_earnings"].null_count() == 10
    assert (out["src_date_earnings"].drop_nulls() <= out.drop_nulls("src_date_earnings")["date"]).all()


# ---- sue ----


def test_sue_hand_computed():
    # numerator: 1.9 - 1.4 = 0.5. Denominator: the 7 PRIOR changes (0.1, 0.3, 0.2, 0.4, 0.1, 0.2, 0.3;
    # the latest one excluded) have sum 1.6 and sum of squares 0.44, so squared deviations sum to
    # 0.44 - 1.6^2 / 7 = 0.52 / 7, std = sqrt(0.52 / 42) = 0.111270 and sue = 0.5 / 0.111270
    assert sample_std(PRIOR_Q) == pytest.approx(math.sqrt(0.52 / 42))
    sue = one(quarterly(EPS_Q), "sue")
    assert sue == pytest.approx(0.5 / math.sqrt(0.52 / 42))
    assert sue == pytest.approx(4.49359, abs=1e-5)


def test_sue_full_window_reads_eight_prior_changes_from_lag_twelve():
    # 13 quarters: one more change at the start (Q4 2017 - Q4 2016 = 1.0 - 0.9 = 0.1), so the
    # denominator is the 8 prior changes 0.1, 0.1, 0.3, 0.2, 0.4, 0.1, 0.2, 0.3: sum 1.7, sum of squares
    # 0.45, std = sqrt((0.45 - 1.7^2 / 8) / 7) = 0.112599, sue = 0.5 / 0.112599
    eps13 = [0.9] + EPS_Q
    expected = 0.5 / math.sqrt((0.45 - 1.7**2 / 8) / 7)
    assert expected == pytest.approx(4.44053, abs=1e-5)
    assert one(quarterly(eps13, ends=QUARTERS_14[1:]), "sue") == pytest.approx(expected)
    # a 14th, older quarter is outside the window: an absurd value there changes nothing
    eps14 = [1000.0] + eps13
    assert one(quarterly(eps14, ends=QUARTERS_14), "sue") == pytest.approx(expected)


def test_sue_after_a_skipped_intermediate_quarter_uses_seven_of_eight_changes():
    # 13 quarters with Q2 2019 (two quarters before the latest) never filed. Lags are looked up by
    # date, so only the change that needs Q2 2019 (Q2 2019 - Q2 2018 = 0.2) drops out; the numerator
    # and the other 7 prior changes are unaffected
    eps13 = [0.9] + EPS_Q
    ends = QUARTERS_14[1:]
    gap_ends, gap_eps = ends[:10] + ends[11:], eps13[:10] + eps13[11:]
    assert d(2019, 6, 30) not in gap_ends
    prior = [0.1, 0.1, 0.3, 0.2, 0.4, 0.1, 0.3]
    assert one(quarterly(gap_eps, ends=gap_ends), "sue") == pytest.approx(0.5 / sample_std(prior))


def test_sue_is_monotone_in_the_surprise_with_no_cap():
    # same history, a growing beat K: sue = K / std(prior changes), so it rises without bound and a
    # 100x beat ranks above a 10x beat (the current change must not inflate its own scale)
    sues = [one(quarterly(EPS_Q[:-1] + [1.4 + k]), "sue") for k in (0.3, 1.0, 3.0, 10.0, 100.0)]
    assert sues == pytest.approx([k / sample_std(PRIOR_Q) for k in (0.3, 1.0, 3.0, 10.0, 100.0)])
    assert all(a < b for a, b in zip(sues, sues[1:]))
    assert sues[-1] > 800.0


def test_sue_sign_positive_for_a_beat_negative_for_a_miss():
    miss = EPS_Q[:-1] + [0.9]  # latest change 0.9 - 1.4 = -0.5
    assert one(quarterly(miss), "sue") == pytest.approx(-0.5 / sample_std(PRIOR_Q))
    assert one(quarterly(miss), "sue") < 0 < one(quarterly(EPS_Q), "sue")


def test_sue_uses_the_quarter_usable_at_each_date():
    # Q4 2019 is filed on 2020-02-09 (a Sunday) and is usable from Monday 2020-02-10. On 2020-01-31 the
    # latest quarter is Q3 2019: 11 quarters, 7 changes, numerator 1.5 - 1.2 = 0.3 over the 6 prior
    out = run(quarterly(EPS_Q), entries=[(1, d(2020, 1, 31)), (1, d(2020, 2, 28))])
    assert out["sue"][0] == pytest.approx(0.3 / sample_std(CHANGES_Q[:6]))
    assert out["sue"][1] == pytest.approx(0.5 / sample_std(PRIOR_Q))
    assert out["src_date_earnings"].to_list() == [d(2019, 11, 9), d(2020, 2, 9)]


def test_sue_needs_six_changes():
    # 11 quarters -> the latest change + 6 prior ones; 10 quarters -> 5 prior changes -> null
    assert one(quarterly(EPS_Q[1:], ends=QUARTERS[1:]), "sue") == pytest.approx(0.5 / sample_std(CHANGES_Q[1:7]))
    assert one(quarterly(EPS_Q[2:], ends=QUARTERS[2:]), "sue") is None


def test_sue_skips_a_missing_change_in_the_denominator_but_not_in_the_numerator():
    # eps not reported in Q3 2017 (index 2): the prior change 0.2 (Q3 2018 vs Q3 2017) is missing, 6 remain
    hole = EPS_Q[:2] + [None] + EPS_Q[3:]
    assert one(quarterly(hole), "sue") == pytest.approx(0.5 / sample_std([0.1, 0.3, 0.4, 0.1, 0.2, 0.3]))
    # a second missing prior change (Q2 2017, index 1) leaves 5 -> null
    assert one(quarterly(EPS_Q[:1] + [None, None] + EPS_Q[3:]), "sue") is None
    # eps not reported in Q4 2018 (index 7): the latest change itself is missing
    assert one(quarterly(EPS_Q[:7] + [None] + EPS_Q[8:]), "sue") is None
    # eps not reported in the latest quarter
    assert one(quarterly(EPS_Q[:-1] + [None]), "sue") is None


def test_sue_null_after_a_skipped_quarter():
    # no filing at all for Q4 2018, the quarter four before the latest: the latest change has nothing
    # to compare with, so sue is null (rather than comparing Q4 2019 with Q3 2018)
    ends = QUARTERS[:7] + QUARTERS[8:]
    assert one(quarterly(EPS_Q[:7] + EPS_Q[8:], ends=ends), "sue") is None


def test_sue_null_when_the_changes_have_zero_std():
    flat = [1.0] * 12                               # every change exactly 0
    ramp = [1.0 + 0.1 * i for i in range(12)]       # every change 0.4 up to rounding noise
    assert one(quarterly(flat), "sue") is None
    assert one(quarterly(ramp), "sue") is None


def test_sue_null_without_filings_or_when_the_filing_is_stale():
    assert one([], "sue") is None
    assert one(annual(EPS_Y), "sue") is None  # only annual filings
    # the last filing is dated 2020-02-09, 81 days before T
    assert one(quarterly(EPS_Q), "sue", max_fundamental_age_days=81) is not None
    out = run(quarterly(EPS_Q), max_fundamental_age_days=80)
    assert out["sue"][0] is None and out["src_date_earnings"][0] is None


def test_sue_split_adjusted_two_for_one_gives_zero_change():
    # true (pre-split basis) eps: the latest quarter equals the one a year earlier -> surprise 0,
    # while the other changes (0.1, 0.3, 0.2, 0.4, 0.1, 0.2, 0.3) keep the std positive
    true_eps = EPS_Q[:-1] + [1.4]
    # 2:1 split on 2019-07-01, between Q4 2018 and Q4 2019: the quarters ending after it (Q3 and Q4
    # 2019) are reported per new share, i.e. halved, so as reported Q4 falls from 1.4 to 0.7
    split_day = d(2019, 7, 1)
    reported = true_eps[:10] + [e / 2 for e in true_eps[10:]]
    assert reported[-1] == 0.7 and reported[7] == 1.4 and reported[10] == 0.75

    adjusted = one(quarterly(reported), "sue", prices=split_prices(split_day))
    assert adjusted == pytest.approx(0.0, abs=1e-12)
    assert adjusted == pytest.approx(one(quarterly(true_eps), "sue"), abs=1e-12)
    # without the split record the halving reads as a collapse in earnings: the changes are
    # 0.1, 0.3, 0.2, 0.4, 0.1, 0.2, then 0.75 - 1.2 = -0.45 and (latest) 0.7 - 1.4 = -0.7
    raw = one(quarterly(reported), "sue")
    assert raw == pytest.approx(-0.7 / sample_std([0.1, 0.3, 0.2, 0.4, 0.1, 0.2, -0.45]))
    assert raw < -1.0


def test_sue_split_adjusted_matches_the_unsplit_company():
    for factor in (2.0, 3.0, 0.1):  # 2:1, 3:1 and a 1:10 reverse split
        reported = EPS_Q[:6] + [e / factor for e in EPS_Q[6:]]
        sue = one(quarterly(reported), "sue", prices=split_prices(factor=factor))
        assert sue == pytest.approx(0.5 / sample_std(PRIOR_Q))


def test_sue_identical_earnings_across_a_split_have_no_surprise():
    # the same earnings every quarter, halved per share by the split: every adjusted change is 0,
    # so there is no surprise and no spread to scale it by -> null (not a large negative number)
    reported = [2.0] * 6 + [1.0] * 6
    assert one(quarterly(reported), "sue", prices=split_prices()) is None
    assert one(quarterly(reported), "sue") is not None  # unadjusted: spurious


def test_sue_split_after_the_latest_period_end_changes_nothing():
    base = one(quarterly(EPS_Q), "sue")
    # every reported quarter ends before the split, so all are on the same share basis already
    assert one(quarterly(EPS_Q), "sue", prices=split_prices(d(2020, 3, 2))) == base
    assert one(quarterly(EPS_Q), "sue", prices=split_prices(d(2020, 6, 1))) == base  # after T


def test_sue_split_between_the_period_end_and_the_filing_is_already_in_the_filed_eps():
    # a company restates the EPS of a period for a split made before its statements are issued
    # (ASC 260-10-55-12), so a split after the period end but before the filing is already in that
    # filing's EPS. Nothing economic happens: sue must equal that of the unsplit company.
    base = one(quarterly(EPS_Q), "sue")
    assert base == pytest.approx(0.5 / sample_std(PRIOR_Q))
    # 2:1 split on 2020-01-15, between the 2019-12-31 period end and its 2020-02-09 filing
    reported = EPS_Q[:-1] + [EPS_Q[-1] / 2]
    assert one(quarterly(reported), "sue", prices=split_prices(d(2020, 1, 15))) == pytest.approx(base, abs=1e-12)
    # a year later: 2:1 split on 2019-01-15, between 2018-12-31 and its 2019-02-09 filing, so the
    # 2018-12-31 quarter (index 7) and every later one are reported per new share
    reported = EPS_Q[:7] + [e / 2 for e in EPS_Q[7:]]
    assert one(quarterly(reported), "sue", prices=split_prices(d(2019, 1, 15))) == pytest.approx(base, abs=1e-12)


def test_sue_split_on_the_filing_date_is_not_in_the_filed_eps():
    # filed 41 days after the period end, so the 2019-12-31 quarter is filed on Monday 2020-02-10. The
    # filed figures are fixed before the filing day, so a split dated on it is not in them ...
    def filed_41(eps):
        return make_filings(1, "ARQ", QUARTERS, lag_days=41, eps=eps, revenue=100.0)

    base = one(filed_41(EPS_Q), "sue")
    assert base == pytest.approx(0.5 / sample_std(PRIOR_Q))
    assert one(filed_41(EPS_Q), "sue", prices=split_prices(d(2020, 2, 10))) == base
    # ... while one dated the trading day before is
    halved = EPS_Q[:-1] + [EPS_Q[-1] / 2]
    assert one(filed_41(halved), "sue", prices=split_prices(d(2020, 2, 7))) == pytest.approx(base, abs=1e-12)


def test_sue_point_in_time():
    base = run(quarterly(EPS_Q))
    q1 = d(2020, 3, 31)

    def with_q1(filed, eps=9.0):
        return quarterly(EPS_Q) + [(1, "ARQ", q1, filed, {"eps": eps})]

    # a filing after T, and one made on T itself (usable from the next trading day), change nothing
    for filed in (d(2020, 5, 8), T):
        for eps in (9.0, -9.0):
            assert run(with_q1(filed, eps)).equals(base)
    # a later restatement of an old quarter, made after T, changes nothing either
    restated = quarterly(EPS_Q[:7] + [None] + EPS_Q[8:]) + [(1, "ARQ", QUARTERS[7], d(2020, 5, 8), {"eps": 5.0})]
    assert run(restated)["sue"][0] is None
    # control: the same filing one trading day before T is used
    used = run(with_q1(d(2020, 4, 29)))
    # 13 quarters: latest change 9.0 - 1.2, scaled by all 8 earlier changes
    assert used["sue"][0] == pytest.approx((9.0 - 1.2) / sample_std(CHANGES_Q))
    assert used["src_date_earnings"][0] == d(2020, 4, 29)


# ---- eps_growth_stability ----


def test_eps_growth_stability_hand_computed():
    # changes +0.5, -0.3, +0.8: mean 1/3, squared deviations (1/6)^2 + (19/30)^2 + (14/30)^2 = 0.646667,
    # std = sqrt(0.646667 / 2) = 0.568624; mean |eps| = (1.0 + 1.5 + 1.2 + 2.0) / 4 = 1.425
    std = sample_std([0.5, -0.3, 0.8])
    assert std == pytest.approx(math.sqrt(0.97 / 3))
    value = one(annual(EPS_Y), "eps_growth_stability")
    assert value == pytest.approx(std / 1.425)
    assert value == pytest.approx(0.399034, abs=1e-6)


def test_eps_growth_stability_uses_absolute_eps_and_is_lower_for_steadier_growth():
    # losses: changes -1.5, +2.5, -3.0 ; mean |eps| = (1 + 0.5 + 2 + 1) / 4 = 1.125
    erratic = one(annual([1.0, -0.5, 2.0, -1.0]), "eps_growth_stability")
    assert erratic == pytest.approx(sample_std([-1.5, 2.5, -3.0]) / 1.125)
    # the same change every year: std 0 -> 0.0, the steadiest possible value (not null)
    steady = one(annual([1.0, 1.25, 1.5, 1.75]), "eps_growth_stability")
    assert steady == 0.0
    assert 0.0 <= steady < one(annual(EPS_Y), "eps_growth_stability") < erratic


def test_eps_growth_stability_null_rules():
    # three fiscal years are only two changes
    assert one(annual(EPS_Y[1:], ends=YEARS[1:]), "eps_growth_stability") is None
    # a year without eps
    assert one(annual([1.0, None, 1.2, 2.0]), "eps_growth_stability") is None
    # a skipped fiscal year: 2015, 2017, 2018, 2019
    gap = [d(2015, 12, 31)] + YEARS[1:]
    assert one(annual(EPS_Y, ends=gap), "eps_growth_stability") is None
    # mean |eps| of zero
    assert one(annual([0.0, 0.0, 0.0, 0.0]), "eps_growth_stability") is None
    assert one([], "eps_growth_stability") is None
    assert one(quarterly(EPS_Q), "eps_growth_stability") is None  # only quarterly filings


def test_eps_growth_stability_uses_the_last_four_years():
    # a fifth, older year is ignored
    ends = period_ends(d(2019, 12, 31), 5, months=12)
    assert one(annual([50.0] + EPS_Y, ends=ends), "eps_growth_stability") == pytest.approx(
        sample_std([0.5, -0.3, 0.8]) / 1.425
    )


def test_eps_growth_stability_split_adjusted():
    # 2:1 split on 2018-08-20: fiscal 2018 and 2019 are reported per new share
    reported = [1.0, 1.5, 0.6, 1.0]
    adjusted = one(annual(reported), "eps_growth_stability", prices=split_prices())
    assert adjusted == pytest.approx(sample_std([0.5, -0.3, 0.8]) / 1.425)
    # unadjusted: changes +0.5, -0.9, +0.4 over a mean |eps| of 1.025
    assert one(annual(reported), "eps_growth_stability") == pytest.approx(sample_std([0.5, -0.9, 0.4]) / 1.025)
    # a split after the latest fiscal year end, before or after T, changes nothing
    base = one(annual(EPS_Y), "eps_growth_stability")
    for day in (d(2020, 3, 2), d(2020, 6, 1)):
        assert one(annual(EPS_Y), "eps_growth_stability", prices=split_prices(day)) == base


def test_eps_growth_stability_split_between_year_end_and_filing():
    # 2:1 split on 2020-01-15, after the FY2019 year end, before its 2020-02-29 filing: FY2019 EPS is
    # reported per new share and the result equals the unsplit company's
    base = one(annual(EPS_Y), "eps_growth_stability")
    reported = EPS_Y[:-1] + [EPS_Y[-1] / 2]
    assert one(annual(reported), "eps_growth_stability", prices=split_prices(d(2020, 1, 15))) == pytest.approx(base)


def test_eps_growth_stability_point_in_time():
    # fiscal 2019 is filed 2020-02-29 (a Saturday), usable from Monday 2020-03-02
    early = period_ends(d(2019, 12, 31), 5, months=12)
    five = [0.4] + EPS_Y
    out = run(annual(five, ends=early), entries=[(1, d(2020, 2, 28)), (1, d(2020, 3, 31))])
    # on 2020-02-28 the last four years are 2015..2018: changes +0.6, +0.5, -0.3 over mean 4.1 / 4
    assert out["eps_growth_stability"][0] == pytest.approx(sample_std([0.6, 0.5, -0.3]) / 1.025)
    assert out["eps_growth_stability"][1] == pytest.approx(sample_std([0.5, -0.3, 0.8]) / 1.425)
    assert out["src_date_earnings"].to_list() == [d(2019, 3, 1), d(2020, 2, 29)]

    base = run(annual(EPS_Y))
    fy2020 = d(2020, 12, 31)
    for filed in (d(2020, 5, 8), T):  # after T, and on T itself
        for eps in (9.0, -9.0):
            assert run(annual(EPS_Y) + [(1, "ARY", fy2020, filed, {"eps": eps})]).equals(base)
    used = run(annual(EPS_Y) + [(1, "ARY", fy2020, d(2020, 4, 29), {"eps": 9.0})])
    # control: years 1.5, 1.2, 2.0, 9.0 -> changes -0.3, +0.8, +7.0 over mean 13.7 / 4
    assert used["eps_growth_stability"][0] == pytest.approx(sample_std([-0.3, 0.8, 7.0]) / 3.425)


# ---- src_date ----


def test_src_date_is_the_latest_filing_behind_a_non_null_feature():
    both = run(quarterly(EPS_Q) + annual(EPS_Y))
    assert both["src_date_earnings"][0] == d(2020, 2, 29)  # annual filing, later than the 10-Q of 02-09
    assert run(quarterly(EPS_Q))["src_date_earnings"][0] == d(2020, 2, 9)
    # sue is null (zero std), so the quarterly filing is not a source even though it exists
    only_annual = run(quarterly([1.0] * 12, ends=period_ends(d(2020, 3, 31), 12)) + annual(EPS_Y))
    assert only_annual["sue"][0] is None
    assert only_annual["src_date_earnings"][0] == d(2020, 2, 29)
    assert run(quarterly([1.0] * 12))["src_date_earnings"][0] is None


def test_split_adjustment_keeps_the_filing_date_of_other_metrics():
    # the period's filing is dated by its earliest metric (revenue here); eps arrives in an amendment
    # after T and must not be used, and must not move the filing date either
    fund = quarterly(EPS_Q[:-1], ends=QUARTERS[:-1]) + [
        (1, "ARQ", QUARTERS[-1], d(2020, 2, 9), {"revenue": 100.0}),
        (1, "ARQ", QUARTERS[-1], d(2020, 5, 8), {"eps": 1.9}),
    ]
    out = run(fund, prices=split_prices(d(2020, 6, 1)))
    assert out["sue"][0] is None and out["src_date_earnings"][0] is None

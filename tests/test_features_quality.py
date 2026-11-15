"""Profitability / quality features on hand-built filings: values, null rules, point-in-time.

The trading calendar is every weekday of 2014-2020. Quarterly (ARQ) and trailing-twelve-month (ART)
rows for the quarter ending 2019-12-31 are filed on Friday 2020-02-14. Fiscal years (ARY) are filed 60
calendar days after the year end: FY2019 on Saturday 2020-02-29, first used on Monday 2020-03-02.
T = Tuesday 2020-03-31 is a month end.
"""

import math

import polars as pl
import pytest
from conftest import business_days
from feature_helpers import d, make_ctx, make_filings, make_prices, period_ends
from polars.testing import assert_frame_equal

from ltforecast.features import quality
from ltforecast.features.catalog import load_catalog

CAL = business_days(d(2014, 1, 1), d(2020, 12, 31))
Q_END = d(2019, 12, 31)
Q_FILED = d(2020, 2, 14)
Y5 = period_ends(d(2019, 12, 31), 5, months=12)   # FY2015 ... FY2019
Y_FILED = d(2020, 2, 29)
T = d(2020, 3, 31)
FEATURES = ["gross_profitability", "roic", "op_margin_stability", "piotroski_f", "accruals"]

# FY2018, the year every F-score below is compared against:
# ROA 50/1000 = 0.05, leverage 300/1000 = 0.30, current ratio 400/200 = 2.0, gross margin 300/1000 = 0.30,
# asset turnover 1000/1000 = 1.0, 100 shares
PRIOR = dict(netinc=50, ncfo=60, assets=1000, debtnc=300, assetsc=400, liabilitiesc=200, shareswa=100, gp=300, revenue=1000)
# FY2019 passing all nine signals
GOOD = dict(netinc=80, ncfo=100, assets=1000, debtnc=250, assetsc=450, liabilitiesc=200, shareswa=100, gp=352, revenue=1100)
# FY2019 failing all nine
BAD = dict(netinc=-20, ncfo=-30, assets=1000, debtnc=350, assetsc=380, liabilitiesc=200, shareswa=110, gp=270, revenue=950)


def quarter(sid=1, *, end=Q_END, lag_days=45, assets=1000, invcap=800, gp=300, ebit=200, netinc=120, ncfo=150) -> list[tuple]:
    """The ARQ (assets, invcap) and ART (gp, ebit, netinc, ncfo) rows of one quarterly filing."""
    return make_filings(sid, "ARQ", [end], lag_days=lag_days, assets=assets, invcap=invcap) + make_filings(
        sid, "ART", [end], lag_days=lag_days, gp=gp, ebit=ebit, netinc=netinc, ncfo=ncfo
    )


def years(sid=1, ends=Y5, *, lag_days=60, **metrics) -> list[tuple]:
    """ARY rows: each metric is one value per fiscal year (or a number for every year)."""
    return make_filings(sid, "ARY", ends, lag_days=lag_days, **metrics)


def two_years(latest: dict, prior: dict = PRIOR, sid=1, **kw) -> list[tuple]:
    """FY2018 = `prior`, FY2019 = `latest` (None drops a metric from that year's filing)."""
    return years(sid, Y5[-2:], **{m: [prior.get(m), latest.get(m)] for m in PRIOR}, **kw)


def run(rows, entries=((1, T),), *, splits=None, prices=None) -> pl.DataFrame:
    if prices is None:
        ids = sorted({r[0] for r in rows} | {e[0] for e in entries})
        prices = make_prices({sid: 10.0 for sid in ids}, CAL, splits=splits)
    return quality.compute(make_ctx(prices=prices, fundamentals=rows, entries=list(entries), calendar=CAL))


def one(rows, t=T, **kw) -> dict:
    return run(rows, ((1, t),), **kw).row(0, named=True)


def test_families_match_the_catalog():
    assert quality.FAMILIES == {"quality": tuple(load_catalog().names(family="quality"))}


# ---- hand-computed values ----

def test_ratios_from_the_latest_quarter():
    out = one(quarter(assets=1000, invcap=800, gp=300, ebit=200, netinc=120, ncfo=150))
    assert out["gross_profitability"] == pytest.approx(300 / 1000)            # 0.30
    assert out["roic"] == pytest.approx(200 * (1 - 0.21) / 800)               # 0.1975
    assert out["accruals"] == pytest.approx((120 - 150) / 1000)               # -0.03
    assert out["op_margin_stability"] is None and out["piotroski_f"] is None  # no fiscal-year filings
    assert out["src_date_quality"] == Q_FILED


def test_sign_conventions():
    # earnings above cash flow are positive accruals (the bad direction); an operating loss is a negative roic
    out = one(quarter(netinc=150, ncfo=120, ebit=-200, gp=-50))
    assert out["accruals"] == pytest.approx(0.03)
    assert out["roic"] == pytest.approx(-0.1975)
    assert out["gross_profitability"] == pytest.approx(-0.05)


def test_op_margin_stability_five_years():
    # margins 0.10, 0.12, 0.14, 0.12, 0.12: mean 0.12, squared deviations 0.0004 + 0 + 0.0004 + 0 + 0 = 0.0008,
    # variance 0.0008 / (5 - 1) = 0.0002, std = 0.014142...
    out = one(years(opinc=[10, 12, 14, 12, 12], revenue=100))
    assert out["op_margin_stability"] == pytest.approx(math.sqrt(0.0002))
    assert out["src_date_quality"] == Y_FILED
    # a constant margin is perfectly stable
    assert one(years(opinc=[10, 20, 30, 40, 50], revenue=[100, 200, 300, 400, 500]))["op_margin_stability"] == pytest.approx(0.0)


def test_op_margin_stability_window_and_minimum():
    # three years, margins 0.10, 0.20, 0.30: mean 0.20, variance (0.01 + 0 + 0.01) / 2 = 0.01, std 0.10
    assert one(years(ends=Y5[-3:], opinc=[10, 20, 30], revenue=100))["op_margin_stability"] == pytest.approx(0.1)
    # two years are not enough
    assert one(years(ends=Y5[-2:], opinc=[10, 30], revenue=100))["op_margin_stability"] is None
    # a sixth, older year is outside the window: FY2014's 0.90 margin is ignored
    six = period_ends(d(2019, 12, 31), 6, months=12)
    out = one(years(ends=six, opinc=[90, 10, 12, 14, 12, 12], revenue=100))
    assert out["op_margin_stability"] == pytest.approx(math.sqrt(0.0002))


def test_op_margin_stability_skips_years_without_a_margin():
    # FY2016 has zero revenue and FY2017 no operating income: the three remaining margins 0.10, 0.20, 0.30 count
    out = one(years(opinc=[10, 5, None, 20, 30], revenue=[100, 0, 100, 100, 100]))
    assert out["op_margin_stability"] == pytest.approx(0.1)
    # with one more year missing only two margins are left
    assert one(years(opinc=[10, 5, None, None, 30], revenue=[100, 0, 100, 100, 100]))["op_margin_stability"] is None
    # negative revenue is no denominator either
    assert one(years(ends=Y5[-3:], opinc=[10, 20, 30], revenue=[100, -100, 100]))["op_margin_stability"] is None


def test_piotroski_all_nine():
    # ROA 80/1000 = 0.08 > 0 | CFO 100 > 0 | ROA 0.08 > 0.05 | CFO 100 > NI 80 | leverage 0.25 < 0.30 |
    # current ratio 450/200 = 2.25 > 2.0 | shares 100 <= 100 | gross margin 352/1100 = 0.32 > 0.30 |
    # turnover 1100/1000 = 1.1 > 1.0
    out = one(two_years(GOOD))
    assert out["piotroski_f"] == 9.0
    assert out["src_date_quality"] == Y_FILED


def test_piotroski_zero():
    # ROA -0.02 | CFO -30 | ROA down | CFO -30 < NI -20 | leverage 0.35 > 0.30 | current ratio 1.9 < 2.0 |
    # shares 110 > 100 | gross margin 270/950 = 0.284 < 0.30 | turnover 0.95 < 1.0
    assert one(two_years(BAD))["piotroski_f"] == 0.0


def test_piotroski_mixed():
    # ROA 0.04 > 0 (1) | CFO 60 > 0 (1) | ROA 0.04 < 0.05 (0) | CFO 60 > NI 40 (1) | leverage 0.25 < 0.30 (1) |
    # current ratio 1.9 < 2.0 (0) | shares 105 > 100 (0) | gross margin 0.32 > 0.30 (1) | turnover 1.1 > 1.0 (1)
    mixed = dict(netinc=40, ncfo=60, assets=1000, debtnc=250, assetsc=380, liabilitiesc=200, shareswa=105, gp=352, revenue=1100)
    assert one(two_years(mixed))["piotroski_f"] == 6.0


@pytest.mark.parametrize(
    ("change", "lost"),
    [
        (dict(netinc=-10, ncfo=5), 2),          # ROA <= 0 and ROA down (CFO 5 still above NI -10)
        (dict(ncfo=-5), 2),                     # CFO <= 0 and CFO < NI
        (dict(ncfo=70), 1),                     # CFO > 0 but below NI 80
        (dict(debtnc=300), 1),                  # leverage unchanged at 0.30 is not "down"
        (dict(assetsc=400), 1),                 # current ratio unchanged is not "up"
        (dict(shareswa=100.5), 1),              # any new shares
        (dict(gp=330), 1),                      # gross margin unchanged at 0.30
        (dict(revenue=1000, gp=320), 1),        # turnover unchanged at 1.0 (gross margin still up: 0.32)
    ],
)
def test_piotroski_each_signal(change, lost):
    assert one(two_years({**GOOD, **change}))["piotroski_f"] == 9.0 - lost


def test_piotroski_fewer_shares_passes_and_split_is_not_issuance():
    assert one(two_years({**GOOD, "shareswa": 95}))["piotroski_f"] == 9.0
    # 100 -> 200 shares is an issuance ...
    doubled = two_years({**GOOD, "shareswa": 200})
    assert one(doubled)["piotroski_f"] == 8.0
    # ... unless a 2:1 split between the two year ends explains it
    assert one(doubled, splits={1: {d(2019, 6, 3): 2.0}})["piotroski_f"] == 9.0
    # a split after the latest year end but before its filing (2020-02-29) is already in the filed
    # count: FY2019 reports 200 shares on the post-split basis, which is not an issuance
    assert one(doubled, splits={1: {d(2020, 1, 15): 2.0}})["piotroski_f"] == 9.0
    # a split after the latest filing (before the prediction date) changes neither count
    assert one(two_years(GOOD), splits={1: {d(2020, 3, 10): 2.0}})["piotroski_f"] == 9.0


@pytest.mark.parametrize("metric", sorted(PRIOR))
def test_piotroski_null_unless_all_nine_are_computable(metric):
    # a metric missing from the latest year leaves at least one signal undefined
    assert one(two_years({**GOOD, metric: None}))["piotroski_f"] is None
    # so does one missing from the year before, except operating cash flow: no signal reads last year's
    prior_missing = one(two_years(GOOD, {**PRIOR, metric: None}))["piotroski_f"]
    assert prior_missing == (9.0 if metric == "ncfo" else None)


@pytest.mark.parametrize("metric", ["assets", "liabilitiesc", "revenue", "shareswa"])
@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_piotroski_null_on_zero_or_negative_denominators(metric, bad):
    assert one(two_years({**GOOD, metric: bad}))["piotroski_f"] is None
    assert one(two_years(GOOD, {**PRIOR, metric: bad}))["piotroski_f"] is None


def test_piotroski_needs_two_consecutive_fiscal_years():
    assert one(years(ends=Y5[-1:], **GOOD))["piotroski_f"] is None
    # FY2018 missing: FY2017 is not "the year before"
    gap = years(ends=[Y5[-3], Y5[-1]], **{m: [PRIOR[m], GOOD[m]] for m in PRIOR})
    assert one(gap)["piotroski_f"] is None


# ---- null rules for the ratios ----

@pytest.mark.parametrize("bad", [0.0, -100.0])
def test_zero_or_negative_denominators_are_null(bad):
    out = one(quarter(assets=bad, invcap=bad))
    assert [out[c] for c in FEATURES] == [None] * 5
    assert out["src_date_quality"] is None
    # each denominator only nulls the features that use it
    out = one(quarter(invcap=bad))
    assert out["roic"] is None and out["gross_profitability"] == pytest.approx(0.3)
    out = one(quarter(assets=bad))
    assert out["gross_profitability"] is None and out["accruals"] is None and out["roic"] == pytest.approx(0.1975)


def test_missing_inputs_are_null():
    out = one(quarter(gp=None, ncfo=None))
    assert out["gross_profitability"] is None and out["accruals"] is None
    assert out["roic"] == pytest.approx(0.1975)
    # trailing-twelve-month rows without a balance sheet: nothing to scale by
    art_only = make_filings(1, "ART", [Q_END], gp=300, ebit=200, netinc=120, ncfo=150)
    out = one(art_only)
    assert [out[c] for c in FEATURES] == [None] * 5 and out["src_date_quality"] is None


def test_stale_filings_are_null():
    rows = quarter(end=d(2018, 3, 31)) + years(ends=period_ends(d(2017, 12, 31), 5, months=12), opinc=[10, 12, 14, 12, 12], revenue=100)
    fresh = one(rows, t=d(2018, 6, 29))
    assert fresh["gross_profitability"] == pytest.approx(0.3)
    assert fresh["op_margin_stability"] == pytest.approx(math.sqrt(0.0002))
    out = one(rows)   # filed in spring 2018, more than 550 days before 2020-03-31
    assert [out[c] for c in FEATURES] == [None] * 5


# ---- output shape ----

def test_output_shape_one_row_per_entry():
    rows = (
        quarter(1) + two_years(GOOD)                       # 1: everything except the 5-year std
        + quarter(2, assets=0, invcap=0)                   # 2: zero denominators only
        + years(3, opinc=[10, 12, 14, 12, 12], revenue=100)  # 3: fiscal years only
    )
    dates = [d(2019, 12, 31), d(2020, 1, 31), d(2020, 2, 28), T]
    entries = [(sid, day) for sid in (1, 2, 3, 4) for day in dates]
    ctx = make_ctx(prices=make_prices({s: 10.0 for s in (1, 2, 3, 4)}, CAL), fundamentals=rows, entries=entries, calendar=CAL)
    out = quality.compute(ctx)
    assert out.columns == ["security_id", "date", *FEATURES, "src_date_quality"]
    assert out.height == ctx.entries.height == 16
    assert out.select("security_id", "date").is_unique().all()
    assert_frame_equal(out.select("security_id", "date"), ctx.entries)
    assert [out.schema[c] for c in FEATURES] == [pl.Float64] * 5
    assert out.schema["src_date_quality"] == pl.Date
    for c in FEATURES:
        assert not out[c].is_nan().any() and not out[c].is_infinite().any()
    known = out.drop_nulls("src_date_quality")
    assert (known["src_date_quality"] < known["date"]).all()
    # the source date is null exactly when every feature is
    all_null = pl.all_horizontal(pl.col(*FEATURES).is_null())
    assert out.select((all_null == pl.col("src_date_quality").is_null()).all()).item()
    assert out.filter(pl.col("security_id").is_in([2, 4])).select(all_null.all()).item()


def test_src_date_is_the_latest_filing_actually_used():
    # FY2015-FY2019 margins; the quarter is filed 2020-02-14, FY2018 on 2019-03-01, FY2019 on 2020-02-29
    rows = quarter() + years(opinc=[10, 12, 14, 12, 12], revenue=100)
    out = run(rows, [(1, d(2020, 1, 31)), (1, d(2020, 2, 28)), (1, T)])
    assert out["src_date_quality"].to_list() == [d(2019, 3, 1), Q_FILED, Y_FILED]
    # at 2020-01-31 only the fiscal years to FY2018 are known: margins 0.10, 0.12, 0.14, 0.12 (mean 0.12),
    # variance 0.0008 / 3
    assert out["op_margin_stability"][0] == pytest.approx(math.sqrt(0.0008 / 3))
    assert out["gross_profitability"].to_list() == [None, pytest.approx(0.3), pytest.approx(0.3)]


# ---- point in time ----

BASE = quarter() + two_years(GOOD) + years(ends=Y5[:3], opinc=[10, 12, 14], revenue=100)
PIT_ENTRIES = [(1, d(2020, 1, 31)), (1, d(2020, 2, 28)), (1, T)]
LATER = {
    "later-quarter": quarter(end=d(2020, 3, 31), assets=5, invcap=5, gp=999, ebit=999, netinc=999, ncfo=-999),
    "later-fiscal-year": years(ends=[d(2020, 12, 31)], lag_days=1, **BAD),
    "quarter-filed-on-t": quarter(end=d(2020, 2, 29), lag_days=31, assets=5, invcap=5, gp=999, ebit=999, netinc=999, ncfo=-999),
    "fiscal-year-filed-on-t": years(ends=[d(2020, 2, 29)], lag_days=31, **BAD),
}


@pytest.mark.parametrize("name", sorted(LATER))
def test_later_filings_do_not_change_earlier_dates(name):
    later = LATER[name]
    assert all(row[3] >= T for row in later)
    assert_frame_equal(run(BASE + later, PIT_ENTRIES), run(BASE, PIT_ENTRIES))
    # ... and the test has teeth: the same rows do change a later date
    after = [(1, d(2021, 1, 29))] if name == "later-fiscal-year" else [(1, d(2020, 6, 30))]
    cal = business_days(d(2014, 1, 1), d(2021, 12, 31))

    def at_later(rows):
        ctx = make_ctx(prices=make_prices({1: 10.0}, cal), fundamentals=rows, entries=after, calendar=cal)
        return quality.compute(ctx).row(0)

    assert at_later(BASE + later) != at_later(BASE)


def test_filing_is_not_used_on_its_own_date():
    # everything for 2019 is filed 91 days after the year end: on T = 2020-03-31 itself
    older = quarter(end=d(2019, 9, 30), assets=2000, invcap=1000, gp=200, ebit=100, netinc=10, ncfo=50) + two_years(
        PRIOR, {**PRIOR, "netinc": 40}, lag_days=91
    )[:1]
    new = quarter(lag_days=91) + two_years(GOOD, lag_days=91)[1:]
    assert {row[3] for row in new} == {T}
    on_day = one(older + new)
    assert on_day == one(older)
    assert on_day["gross_profitability"] == pytest.approx(200 / 2000)
    assert on_day["roic"] == pytest.approx(100 * 0.79 / 1000)
    assert on_day["accruals"] == pytest.approx((10 - 50) / 2000)
    assert on_day["piotroski_f"] is None   # FY2018 alone has nothing to be compared with
    assert on_day["src_date_quality"] == d(2019, 11, 14)
    # the next trading day the new filings are in
    after = one(older + new, t=d(2020, 4, 1))
    assert after["gross_profitability"] == pytest.approx(0.3)
    assert after["piotroski_f"] == 9.0
    assert after["src_date_quality"] == T


def test_later_prices_and_splits_do_not_change_earlier_dates():
    base = run(BASE, PIT_ENTRIES)
    assert base["piotroski_f"].to_list() == [None, None, 9.0]
    changed = make_prices(
        {1: {CAL[0]: 10.0, d(2020, 4, 1): 500.0}}, CAL, splits={1: {d(2020, 4, 1): 2.0, d(2020, 6, 1): 5.0}}
    )
    assert_frame_equal(run(BASE, PIT_ENTRIES, prices=changed), base)
    cut = make_prices({1: 10.0}, CAL).filter(pl.col("date") <= T)
    assert_frame_equal(run(BASE, PIT_ENTRIES, prices=cut), base)


def test_src_date_never_after_the_prediction_date_on_every_trading_day():
    entries = [(1, day) for day in CAL.filter((CAL >= d(2019, 1, 1)) & (CAL <= d(2020, 6, 30))).to_list()]
    out = run(BASE, entries).drop_nulls("src_date_quality")
    assert out.height > 0
    assert (out["src_date_quality"] < out["date"]).all()
    # the F-score appears on the first trading day after the FY2019 filing (Saturday 2020-02-29)
    assert out.filter(pl.col("piotroski_f").is_not_null())["date"].min() == d(2020, 3, 2)
    assert out.filter(pl.col("gross_profitability").is_not_null())["date"].min() == d(2020, 2, 17)

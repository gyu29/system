"""Value features: hand-computed values, sign conventions, null rules and point-in-time behaviour."""

from __future__ import annotations

import math

import polars as pl
import pytest
from feature_helpers import business_days, d, make_ctx, make_marketcap, make_prices

from ltforecast.features import value
from ltforecast.features.catalog import load_catalog

CAL = business_days(d(2019, 1, 1), d(2020, 12, 31))
T = d(2020, 3, 31)       # prediction date (a Tuesday, the last trading day of March 2020)
T2 = d(2020, 6, 30)      # a later prediction date
Q4 = d(2019, 12, 31)     # fiscal period end
FILED = d(2020, 2, 14)   # a Friday: usable from Monday 2020-02-17

ARQ = {"equity": 500.0, "debt": 300.0, "cashneq": 100.0}
ART = {"netinc": 80.0, "fcf": 60.0, "ebitda": 150.0, "revenue": 1000.0, "ncfdiv": -20.0, "ncfcommon": -30.0}
MCAP = 1000.0


def filing(sid=1, period=Q4, filed=FILED, arq=None, art=None):
    """The quarterly and the trailing-twelve-month rows of one filing; overrides replace metrics
    (None drops a metric from the filing)."""
    return [
        (sid, "ARQ", period, filed, {**ARQ, **(arq or {})}),
        (sid, "ART", period, filed, {**ART, **(art or {})}),
    ]


def ctx_for(fundamentals, caps=None, entries=((1, T),), prices=None, **kwargs):
    prices = make_prices({sid: 10.0 for sid in {e[0] for e in entries}}, CAL) if prices is None else prices
    caps = {sid: MCAP for sid in {e[0] for e in entries}} if caps is None else caps
    return make_ctx(
        prices=prices, fundamentals=fundamentals, marketcap=make_marketcap(prices, caps), entries=list(entries),
        **kwargs,
    )


def row(out: pl.DataFrame, sid: int = 1, date=T) -> dict:
    rows = out.filter((pl.col("security_id") == sid) & (pl.col("date") == date)).to_dicts()
    assert len(rows) == 1
    return rows[0]


def test_module_matches_the_catalog():
    assert value.FAMILIES == ("value",)
    assert list(value.FEATURES) == load_catalog().names(family="value")


def test_hand_computed_values():
    r = row(value.compute(ctx_for(filing())))
    ev = 1000.0 + 300.0 - 100.0                                       # market cap + debt - cash = 1200
    assert r["book_to_price"] == pytest.approx(500.0 / 1000.0)        # 0.5
    assert r["earnings_yield"] == pytest.approx(80.0 / 1000.0)        # 0.08
    assert r["fcf_yield"] == pytest.approx(60.0 / 1000.0)             # 0.06
    assert r["ebitda_to_ev"] == pytest.approx(150.0 / ev)             # 0.125
    assert r["sales_to_ev"] == pytest.approx(1000.0 / ev)             # 0.8333
    assert r["shareholder_yield"] == pytest.approx((20.0 + 30.0) / 1000.0)  # 0.05: dividends + buybacks
    assert r["src_date_value"] == T                                   # the market cap is dated t


def test_market_cap_is_measured_at_the_prediction_date():
    # 800 until the day before t, 2000 at t, 5000 the day after: only the value at t counts
    caps = {1: {CAL[0]: 800.0, T: 2000.0, d(2020, 4, 1): 5000.0}}
    r = row(value.compute(ctx_for(filing(), caps=caps)))
    assert r["book_to_price"] == pytest.approx(500.0 / 2000.0)
    assert r["earnings_yield"] == pytest.approx(80.0 / 2000.0)
    assert r["fcf_yield"] == pytest.approx(60.0 / 2000.0)
    assert r["ebitda_to_ev"] == pytest.approx(150.0 / (2000.0 + 300.0 - 100.0))
    assert r["sales_to_ev"] == pytest.approx(1000.0 / (2000.0 + 300.0 - 100.0))
    assert r["shareholder_yield"] == pytest.approx(50.0 / 2000.0)


def test_balance_sheet_is_quarterly_and_flows_are_trailing_twelve_months():
    # the same metric names under the other dimension must be ignored
    fund = [
        (1, "ARQ", Q4, FILED, {**ARQ, "netinc": 1.0, "fcf": 1.0, "ebitda": 1.0, "revenue": 1.0, "ncfdiv": -1.0, "ncfcommon": -1.0}),
        (1, "ART", Q4, FILED, {**ART, "equity": 1.0, "debt": 1.0, "cashneq": 1.0}),
    ]
    r = row(value.compute(ctx_for(fund)))
    assert r["book_to_price"] == pytest.approx(0.5)
    assert r["earnings_yield"] == pytest.approx(0.08)
    assert r["fcf_yield"] == pytest.approx(0.06)
    assert r["ebitda_to_ev"] == pytest.approx(0.125)
    assert r["sales_to_ev"] == pytest.approx(1000.0 / 1200.0)
    assert r["shareholder_yield"] == pytest.approx(0.05)


def test_shareholder_yield_sign_conventions():
    # Sharadar: ncfdiv < 0 when dividends are paid; ncfcommon < 0 for buybacks, > 0 for issuance
    fund = [
        *filing(1, art={"ncfdiv": -40.0, "ncfcommon": 0.0}),     # dividend payer
        *filing(2, art={"ncfdiv": 0.0, "ncfcommon": -70.0}),     # buyback
        *filing(3, art={"ncfdiv": 0.0, "ncfcommon": 120.0}),     # net issuer
        *filing(4, art={"ncfdiv": -10.0, "ncfcommon": 30.0}),    # pays a dividend but issues more
        *filing(5, art={"ncfdiv": 0.0, "ncfcommon": 0.0}),       # neither
    ]
    out = value.compute(ctx_for(fund, entries=[(i, T) for i in range(1, 6)]))
    assert row(out, 1)["shareholder_yield"] == pytest.approx(40.0 / 1000.0)            # +0.04
    assert row(out, 2)["shareholder_yield"] == pytest.approx(70.0 / 1000.0)            # +0.07
    assert row(out, 3)["shareholder_yield"] == pytest.approx(-120.0 / 1000.0)          # -0.12
    assert row(out, 4)["shareholder_yield"] == pytest.approx(-(-10.0 + 30.0) / 1000.0)  # -0.02
    assert row(out, 5)["shareholder_yield"] == 0.0


def test_negative_earnings_and_negative_equity_are_kept():
    fund = filing(arq={"equity": -200.0}, art={"netinc": -50.0, "fcf": -25.0, "ebitda": -60.0})
    r = row(value.compute(ctx_for(fund)))
    assert r["book_to_price"] == pytest.approx(-200.0 / 1000.0)   # negative equity ranks at the bottom
    assert r["earnings_yield"] == pytest.approx(-50.0 / 1000.0)   # a loss is a valid (bad) yield
    assert r["fcf_yield"] == pytest.approx(-25.0 / 1000.0)
    assert r["ebitda_to_ev"] == pytest.approx(-60.0 / 1200.0)


@pytest.mark.parametrize("cash, expect_null", [(1300.0, True), (1500.0, True), (1299.0, False)])
def test_ev_ratios_are_null_when_ev_is_not_positive(cash, expect_null):
    # EV = 1000 + 300 - cash: zero at 1300, negative above
    r = row(value.compute(ctx_for(filing(arq={"cashneq": cash}))))
    if expect_null:
        assert r["ebitda_to_ev"] is None and r["sales_to_ev"] is None
    else:
        assert r["ebitda_to_ev"] == pytest.approx(150.0 / 1.0)
        assert r["sales_to_ev"] == pytest.approx(1000.0 / 1.0)
    # the market-cap ratios do not depend on EV
    assert r["book_to_price"] == pytest.approx(0.5)
    assert r["earnings_yield"] == pytest.approx(0.08)


@pytest.mark.parametrize(
    "dim, metric, nulled",
    [
        ("arq", "equity", {"book_to_price"}),
        ("arq", "debt", {"ebitda_to_ev", "sales_to_ev"}),
        ("arq", "cashneq", {"ebitda_to_ev", "sales_to_ev"}),
        ("art", "netinc", {"earnings_yield"}),
        ("art", "fcf", {"fcf_yield"}),
        ("art", "ebitda", {"ebitda_to_ev"}),
        ("art", "revenue", {"sales_to_ev"}),
        ("art", "ncfdiv", {"shareholder_yield"}),
        ("art", "ncfcommon", {"shareholder_yield"}),
    ],
)
def test_a_missing_input_nulls_only_the_features_that_use_it(dim, metric, nulled):
    r = row(value.compute(ctx_for(filing(**{dim: {metric: None}}))))
    assert {f for f in value.FEATURES if r[f] is None} == nulled


def test_no_market_cap_means_no_value_features():
    out = value.compute(ctx_for(filing(), caps={}))
    r = row(out)
    assert all(r[f] is None for f in value.FEATURES)
    assert r["src_date_value"] is None


def test_zero_or_negative_market_cap_is_not_used():
    # a non-positive market cap is bad data: the last positive value within 7 days is used instead
    caps = {1: {CAL[0]: 1000.0, T: 0.0}}
    r = row(value.compute(ctx_for(filing(), caps=caps)))
    assert r["book_to_price"] == pytest.approx(0.5)
    assert r["src_date_value"] == d(2020, 3, 30)
    # none within 7 days: null
    caps = {1: {CAL[0]: 1000.0, d(2020, 3, 20): -5.0}}
    r = row(value.compute(ctx_for(filing(), caps=caps)))
    assert all(r[f] is None for f in value.FEATURES)


def test_no_filing_yet_and_stale_filing_are_null():
    r = row(value.compute(ctx_for([])))
    assert all(r[f] is None for f in value.FEATURES) and r["src_date_value"] is None
    # 46 days between the filing and t
    r = row(value.compute(ctx_for(filing(), max_fundamental_age_days=45)))
    assert all(r[f] is None for f in value.FEATURES) and r["src_date_value"] is None
    r = row(value.compute(ctx_for(filing(), max_fundamental_age_days=46)))
    assert all(r[f] is not None for f in value.FEATURES)


def test_nan_and_inf_inputs_become_null():
    fund = filing(arq={"equity": math.nan}, art={"revenue": math.inf, "netinc": -math.inf})
    out = value.compute(ctx_for(fund))
    r = row(out)
    assert r["book_to_price"] is None and r["sales_to_ev"] is None and r["earnings_yield"] is None
    assert r["fcf_yield"] == pytest.approx(0.06)
    assert_clean(out)


def test_src_date_is_the_latest_input_used():
    # market cap missing on t and the day before: the value of Friday 2020-03-27 is used
    prices = make_prices({1: 10.0}, CAL)
    mc = make_marketcap(prices, {1: MCAP}).filter(~pl.col("date").is_in([T, d(2020, 3, 30)]))
    ctx = make_ctx(prices=prices, fundamentals=filing(), marketcap=mc, entries=[(1, T)])
    assert row(value.compute(ctx))["src_date_value"] == d(2020, 3, 27)
    # a filing later than the market-cap date wins: filed Monday 2020-03-30, usable at t
    ctx = make_ctx(prices=prices, fundamentals=filing(filed=d(2020, 3, 30)), marketcap=mc, entries=[(1, T)])
    assert row(value.compute(ctx))["src_date_value"] == d(2020, 3, 30)


def test_a_filing_is_usable_from_the_next_trading_day():
    entries = [(1, FILED), (1, d(2020, 2, 17))]   # the filing day (Friday) and the Monday after
    out = value.compute(ctx_for(filing(), entries=entries))
    on_day, after = row(out, date=FILED), row(out, date=d(2020, 2, 17))
    assert all(on_day[f] is None for f in value.FEATURES) and on_day["src_date_value"] is None
    assert after["book_to_price"] == pytest.approx(0.5)
    assert after["src_date_value"] == d(2020, 2, 17)


# ---- point in time: nothing dated after t (or filed on t) may change the value at t ----

NEW_ARQ = {"equity": 900.0, "debt": 50.0, "cashneq": 400.0}
NEW_ART = {"netinc": -40.0, "fcf": 10.0, "ebitda": 300.0, "revenue": 2000.0, "ncfdiv": -5.0, "ncfcommon": 100.0}
PIT_ENTRIES = ((1, T), (1, T2))


def _pit_ctx(case: str | None):
    fund = filing()
    caps = {1: MCAP}
    prices = None
    if case == "later_filing":      # the next quarter, filed six weeks after t
        fund += filing(period=d(2020, 3, 31), filed=d(2020, 5, 11), arq=NEW_ARQ, art=NEW_ART)
    elif case == "filed_on_t":      # a newer period filed on the prediction date itself: usable at t+1
        fund += filing(period=d(2020, 1, 31), filed=T, arq=NEW_ARQ, art=NEW_ART)
    elif case == "later_marketcap":
        caps = {1: {CAL[0]: MCAP, d(2020, 4, 1): 4000.0}}
    elif case == "later_price":
        prices = make_prices({1: {CAL[0]: 10.0, d(2020, 4, 1): 55.0}}, CAL)
    return ctx_for(fund, caps=caps, entries=PIT_ENTRIES, prices=prices)


@pytest.mark.parametrize("feature", value.FEATURES)
@pytest.mark.parametrize("case", ["later_filing", "filed_on_t", "later_marketcap", "later_price"])
def test_point_in_time(feature, case):
    base, changed = value.compute(_pit_ctx(None)), value.compute(_pit_ctx(case))
    assert row(base)[feature] is not None
    assert row(changed)[feature] == row(base)[feature]
    assert row(changed)["src_date_value"] == row(base)["src_date_value"] == T
    if case != "later_price":   # value features do not read prices; the others must show up later
        assert row(changed, date=T2)[feature] != pytest.approx(row(base, date=T2)[feature])


def test_a_filing_made_on_the_prediction_date_is_used_the_day_after():
    fund = filing() + filing(period=d(2020, 1, 31), filed=T, arq=NEW_ARQ, art=NEW_ART)
    out = value.compute(ctx_for(fund, entries=[(1, T), (1, d(2020, 4, 1))]))
    assert row(out)["book_to_price"] == pytest.approx(500.0 / 1000.0)
    assert row(out, date=d(2020, 4, 1))["book_to_price"] == pytest.approx(900.0 / 1000.0)
    assert row(out, date=d(2020, 4, 1))["shareholder_yield"] == pytest.approx(-(-5.0 + 100.0) / 1000.0)


# ---- shape ----

def assert_clean(out: pl.DataFrame) -> None:
    assert out.columns == ["security_id", "date", *value.FEATURES, "src_date_value"]
    for f in value.FEATURES:
        assert out[f].dtype == pl.Float64
        assert out[f].is_nan().sum() == 0 and out[f].is_infinite().sum() == 0
    assert out["src_date_value"].dtype == pl.Date
    assert out.filter(pl.col("src_date_value") > pl.col("date")).is_empty()


def test_one_row_per_entry():
    # 1: full data; 2: no market cap; 3: no fundamentals; 4: listed mid-sample
    prices = make_prices({1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}, CAL, first={4: d(2020, 5, 4)})
    fund = (
        filing(1) + filing(1, period=d(2020, 3, 31), filed=d(2020, 5, 11), arq=NEW_ARQ, art=NEW_ART)
        + filing(2) + filing(4, period=d(2020, 3, 31), filed=d(2020, 5, 11))
    )
    ctx = make_ctx(prices=prices, fundamentals=fund, marketcap=make_marketcap(prices, {1: MCAP, 3: MCAP, 4: MCAP}))
    out = value.compute(ctx)
    assert ctx.entries.height == 24 * 3 + 8
    assert out.height == ctx.entries.height
    assert out.select("security_id", "date").is_unique().all()
    assert out.select("security_id", "date").equals(ctx.entries.select("security_id", "date"))
    assert_clean(out)
    assert out.filter(pl.col("security_id").is_in([2, 3]))["src_date_value"].null_count() == 48
    assert out.filter(pl.col("security_id") == 1)["book_to_price"].drop_nulls().unique().sort().to_list() == [0.5, 0.9]
    # every set feature has a source date, and a source date implies some feature is set
    any_set = pl.any_horizontal([pl.col(f).is_not_null() for f in value.FEATURES])
    assert out.filter(any_set != pl.col("src_date_value").is_not_null()).is_empty()


def test_empty_fundamentals_and_marketcap_give_all_null_rows():
    ctx = make_ctx(prices=make_prices({1: 10.0}, CAL))
    out = value.compute(ctx)
    assert out.height == ctx.entries.height == 24
    assert_clean(out)
    assert all(out[f].null_count() == 24 for f in value.FEATURES)

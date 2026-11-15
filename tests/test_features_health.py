"""Financial-health features: hand-computed values, null rules and point-in-time behaviour."""

from __future__ import annotations

import math

import polars as pl
import pytest
from feature_helpers import business_days, d, make_ctx, make_marketcap, make_prices

from ltforecast.features import health
from ltforecast.features.catalog import load_catalog

CAL = business_days(d(2019, 1, 1), d(2020, 12, 31))
T = d(2020, 3, 31)       # prediction date (a Tuesday, the last trading day of March 2020)
T2 = d(2020, 6, 30)      # a later prediction date
Q4 = d(2019, 12, 31)     # fiscal period end
FILED = d(2020, 2, 14)   # a Friday: usable from Monday 2020-02-17

ARQ = {
    "debt": 300.0, "assets": 2000.0, "cashneq": 100.0, "workingcapital": 400.0, "retearn": 600.0,
    "liabilities": 1500.0,
}
ART = {"ebit": 120.0, "intexp": 30.0, "revenue": 1000.0}
MCAP = 1000.0
# 1.2 x 400/2000 + 1.4 x 600/2000 + 3.3 x 120/2000 + 0.6 x 1000/1500 + 1.0 x 1000/2000
# = 0.24 + 0.42 + 0.198 + 0.4 + 0.5
Z = 1.758


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
    assert health.FAMILIES == ("health",)
    assert list(health.FEATURES) == load_catalog().names(family="health")


def test_hand_computed_values():
    r = row(health.compute(ctx_for(filing())))
    assert r["debt_to_assets"] == pytest.approx(300.0 / 2000.0)      # 0.15
    assert r["cash_to_assets"] == pytest.approx(100.0 / 2000.0)      # 0.05
    assert r["interest_coverage"] == pytest.approx(120.0 / 30.0)     # 4.0
    assert r["altman_z"] == pytest.approx(
        1.2 * 400.0 / 2000.0 + 1.4 * 600.0 / 2000.0 + 3.3 * 120.0 / 2000.0 + 0.6 * 1000.0 / 1500.0 + 1000.0 / 2000.0
    )
    assert r["altman_z"] == pytest.approx(Z)
    assert r["src_date_health"] == T                                 # altman_z uses the market cap at t


def test_altman_z_uses_the_market_cap_at_the_prediction_date():
    # 800 until the day before t, 3000 at t, 9000 the day after
    caps = {1: {CAL[0]: 800.0, T: 3000.0, d(2020, 4, 1): 9000.0}}
    r = row(health.compute(ctx_for(filing(), caps=caps)))
    assert r["altman_z"] == pytest.approx(0.24 + 0.42 + 0.198 + 0.6 * 3000.0 / 1500.0 + 0.5)   # 2.558
    # the pure balance-sheet ratios do not move with the price
    assert r["debt_to_assets"] == pytest.approx(0.15)
    assert r["cash_to_assets"] == pytest.approx(0.05)
    assert r["interest_coverage"] == pytest.approx(4.0)


def test_balance_sheet_is_quarterly_and_flows_are_trailing_twelve_months():
    # the same metric names under the other dimension must be ignored
    fund = [
        (1, "ARQ", Q4, FILED, {**ARQ, "ebit": 1.0, "intexp": 1.0, "revenue": 1.0}),
        (1, "ART", Q4, FILED, {**ART, **{m: 1.0 for m in ARQ}}),
    ]
    r = row(health.compute(ctx_for(fund)))
    assert r["debt_to_assets"] == pytest.approx(0.15)
    assert r["cash_to_assets"] == pytest.approx(0.05)
    assert r["interest_coverage"] == pytest.approx(4.0)
    assert r["altman_z"] == pytest.approx(Z)


def test_interest_coverage_signs_and_zero_interest():
    fund = [
        *filing(1, art={"intexp": -30.0}),               # interest booked as a negative number
        *filing(2, art={"ebit": -60.0}),                 # operating loss
        *filing(3, art={"intexp": 0.0}),                 # no interest expense
        *filing(4, art={"ebit": 0.0}),
    ]
    out = health.compute(ctx_for(fund, entries=[(i, T) for i in range(1, 5)]))
    assert row(out, 1)["interest_coverage"] == pytest.approx(120.0 / 30.0)    # |intexp|
    assert row(out, 2)["interest_coverage"] == pytest.approx(-60.0 / 30.0)    # -2.0, kept
    assert row(out, 3)["interest_coverage"] is None
    assert row(out, 4)["interest_coverage"] == 0.0
    # zero interest expense leaves everything else in place
    assert row(out, 3)["altman_z"] == pytest.approx(Z)
    assert row(out, 3)["debt_to_assets"] == pytest.approx(0.15)


def test_negative_inputs_are_kept_in_altman_z():
    # working-capital deficit, accumulated deficit and an operating loss
    fund = filing(arq={"workingcapital": -200.0, "retearn": -1000.0}, art={"ebit": -100.0})
    r = row(health.compute(ctx_for(fund)))
    expected = 1.2 * -200.0 / 2000.0 + 1.4 * -1000.0 / 2000.0 + 3.3 * -100.0 / 2000.0 + 0.6 * 1000.0 / 1500.0 + 0.5
    assert expected == pytest.approx(-0.12 - 0.7 - 0.165 + 0.4 + 0.5)   # -0.085
    assert r["altman_z"] == pytest.approx(expected)


@pytest.mark.parametrize("assets", [0.0, -50.0])
def test_non_positive_assets_null_every_asset_ratio(assets):
    r = row(health.compute(ctx_for(filing(arq={"assets": assets}))))
    assert r["debt_to_assets"] is None and r["cash_to_assets"] is None and r["altman_z"] is None
    assert r["interest_coverage"] == pytest.approx(4.0)
    assert r["src_date_health"] == FILED   # only the filing was used, not the market cap


@pytest.mark.parametrize("liabilities", [0.0, -10.0])
def test_non_positive_liabilities_null_altman_z(liabilities):
    r = row(health.compute(ctx_for(filing(arq={"liabilities": liabilities}))))
    assert r["altman_z"] is None
    assert r["debt_to_assets"] == pytest.approx(0.15)


def test_zero_debt_and_zero_cash_are_values_not_missing():
    r = row(health.compute(ctx_for(filing(arq={"debt": 0.0, "cashneq": 0.0}))))
    assert r["debt_to_assets"] == 0.0 and r["cash_to_assets"] == 0.0


@pytest.mark.parametrize(
    "dim, metric, nulled",
    [
        ("arq", "debt", {"debt_to_assets"}),
        ("arq", "cashneq", {"cash_to_assets"}),
        ("arq", "assets", {"debt_to_assets", "cash_to_assets", "altman_z"}),
        ("arq", "workingcapital", {"altman_z"}),
        ("arq", "retearn", {"altman_z"}),
        ("arq", "liabilities", {"altman_z"}),
        ("art", "ebit", {"interest_coverage", "altman_z"}),
        ("art", "intexp", {"interest_coverage"}),
        ("art", "revenue", {"altman_z"}),
    ],
)
def test_a_missing_input_nulls_only_the_features_that_use_it(dim, metric, nulled):
    r = row(health.compute(ctx_for(filing(**{dim: {metric: None}}))))
    assert {f for f in health.FEATURES if r[f] is None} == nulled


def test_no_market_cap_nulls_only_altman_z():
    r = row(health.compute(ctx_for(filing(), caps={})))
    assert r["altman_z"] is None
    assert r["debt_to_assets"] == pytest.approx(0.15)
    assert r["cash_to_assets"] == pytest.approx(0.05)
    assert r["interest_coverage"] == pytest.approx(4.0)
    assert r["src_date_health"] == FILED


def test_no_filing_yet_and_stale_filing_are_null():
    r = row(health.compute(ctx_for([])))
    assert all(r[f] is None for f in health.FEATURES) and r["src_date_health"] is None
    # 46 days between the filing and t
    r = row(health.compute(ctx_for(filing(), max_fundamental_age_days=45)))
    assert all(r[f] is None for f in health.FEATURES) and r["src_date_health"] is None
    r = row(health.compute(ctx_for(filing(), max_fundamental_age_days=46)))
    assert all(r[f] is not None for f in health.FEATURES)


def test_nan_and_inf_inputs_become_null():
    out = health.compute(ctx_for(filing(arq={"debt": math.nan, "retearn": math.inf}, art={"intexp": math.nan})))
    r = row(out)
    assert r["debt_to_assets"] is None and r["altman_z"] is None and r["interest_coverage"] is None
    assert r["cash_to_assets"] == pytest.approx(0.05)
    assert_clean(out)


def test_a_filing_is_usable_from_the_next_trading_day():
    entries = [(1, FILED), (1, d(2020, 2, 17))]   # the filing day (Friday) and the Monday after
    out = health.compute(ctx_for(filing(), entries=entries))
    on_day, after = row(out, date=FILED), row(out, date=d(2020, 2, 17))
    assert all(on_day[f] is None for f in health.FEATURES) and on_day["src_date_health"] is None
    assert after["debt_to_assets"] == pytest.approx(0.15)
    assert after["altman_z"] == pytest.approx(Z)
    assert after["src_date_health"] == d(2020, 2, 17)


# ---- point in time: nothing dated after t (or filed on t) may change the value at t ----

NEW_ARQ = {
    "debt": 900.0, "assets": 3000.0, "cashneq": 600.0, "workingcapital": -300.0, "retearn": 150.0,
    "liabilities": 2500.0,
}
NEW_ART = {"ebit": -90.0, "intexp": 60.0, "revenue": 2400.0}
PIT_ENTRIES = ((1, T), (1, T2))
USES_MARKET_CAP = {"altman_z"}


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


@pytest.mark.parametrize("feature", health.FEATURES)
@pytest.mark.parametrize("case", ["later_filing", "filed_on_t", "later_marketcap", "later_price"])
def test_point_in_time(feature, case):
    base, changed = health.compute(_pit_ctx(None)), health.compute(_pit_ctx(case))
    assert row(base)[feature] is not None
    assert row(changed)[feature] == row(base)[feature]
    assert row(changed)["src_date_health"] == row(base)["src_date_health"] == T
    # the change must show up at the later date, or the test proves nothing
    if case in ("later_filing", "filed_on_t") or (case == "later_marketcap" and feature in USES_MARKET_CAP):
        assert row(changed, date=T2)[feature] != pytest.approx(row(base, date=T2)[feature])
    else:
        assert row(changed, date=T2)[feature] == row(base, date=T2)[feature]


def test_a_filing_made_on_the_prediction_date_is_used_the_day_after():
    fund = filing() + filing(period=d(2020, 1, 31), filed=T, arq=NEW_ARQ, art=NEW_ART)
    out = health.compute(ctx_for(fund, entries=[(1, T), (1, d(2020, 4, 1))]))
    assert row(out)["debt_to_assets"] == pytest.approx(300.0 / 2000.0)
    nxt = row(out, date=d(2020, 4, 1))
    assert nxt["debt_to_assets"] == pytest.approx(900.0 / 3000.0)
    assert nxt["cash_to_assets"] == pytest.approx(600.0 / 3000.0)
    assert nxt["interest_coverage"] == pytest.approx(-90.0 / 60.0)
    assert nxt["altman_z"] == pytest.approx(
        1.2 * -300.0 / 3000.0 + 1.4 * 150.0 / 3000.0 + 3.3 * -90.0 / 3000.0 + 0.6 * 1000.0 / 2500.0 + 2400.0 / 3000.0
    )


# ---- shape ----

def assert_clean(out: pl.DataFrame) -> None:
    assert out.columns == ["security_id", "date", *health.FEATURES, "src_date_health"]
    for f in health.FEATURES:
        assert out[f].dtype == pl.Float64
        assert out[f].is_nan().sum() == 0 and out[f].is_infinite().sum() == 0
    assert out["src_date_health"].dtype == pl.Date
    assert out.filter(pl.col("src_date_health") > pl.col("date")).is_empty()


def test_one_row_per_entry():
    # 1: full data; 2: no market cap; 3: no fundamentals; 4: listed mid-sample
    prices = make_prices({1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}, CAL, first={4: d(2020, 5, 4)})
    fund = (
        filing(1) + filing(1, period=d(2020, 3, 31), filed=d(2020, 5, 11), arq=NEW_ARQ, art=NEW_ART)
        + filing(2) + filing(4, period=d(2020, 3, 31), filed=d(2020, 5, 11))
    )
    ctx = make_ctx(prices=prices, fundamentals=fund, marketcap=make_marketcap(prices, {1: MCAP, 3: MCAP, 4: MCAP}))
    out = health.compute(ctx)
    assert ctx.entries.height == 24 * 3 + 8
    assert out.height == ctx.entries.height
    assert out.select("security_id", "date").is_unique().all()
    assert out.select("security_id", "date").equals(ctx.entries.select("security_id", "date"))
    assert_clean(out)
    two = out.filter(pl.col("security_id") == 2)
    assert two["altman_z"].null_count() == 24                           # no market cap
    assert two["debt_to_assets"].drop_nulls().unique().to_list() == [0.15]
    assert two["src_date_health"].drop_nulls().unique().to_list() == [FILED]
    assert out.filter(pl.col("security_id") == 3)["src_date_health"].null_count() == 24
    assert out.filter(pl.col("security_id") == 1)["debt_to_assets"].drop_nulls().unique().sort().to_list() == [0.15, 0.3]
    # every set feature has a source date, and a source date implies some feature is set
    any_set = pl.any_horizontal([pl.col(f).is_not_null() for f in health.FEATURES])
    assert out.filter(any_set != pl.col("src_date_health").is_not_null()).is_empty()


def test_empty_fundamentals_and_marketcap_give_all_null_rows():
    ctx = make_ctx(prices=make_prices({1: 10.0}, CAL))
    out = health.compute(ctx)
    assert out.height == ctx.entries.height == 24
    assert_clean(out)
    assert all(out[f].null_count() == 24 for f in health.FEATURES)

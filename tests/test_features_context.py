"""Context features (features/context.py): hand-computed values, null rules and point-in-time behaviour.

The trading calendar is every weekday of 2019-2020 (524 days). T = 2020-12-31 (a Thursday month end,
the last calendar day) is the usual prediction date; T1 = 2020-06-30 is an earlier one for the
point-in-time tests, which change inputs dated after T1 and compare the row at T1.
"""

import datetime as dt
import math

import polars as pl
import pytest
from conftest import business_days
from feature_helpers import d, make_benchmark, make_ctx, make_macro, make_prices

from ltforecast.features import context
from ltforecast.features.catalog import load_catalog

CAL = business_days(d(2019, 1, 1), d(2020, 12, 31))
DAYS = CAL.to_list()
T = d(2020, 12, 31)
T1 = d(2020, 6, 30)
IT, IT1 = DAYS.index(T), DAYS.index(T1)
FEATURES = ["term_spread", "credit_spread", "mkt_ret_12m", "xs_dispersion"]
MEMBERS = list(range(1, 21))  # 20 stocks: the minimum for xs_dispersion


def run(*, entries=((1, T),), **frames) -> pl.DataFrame:
    return context.compute(make_ctx(entries=list(entries), calendar=CAL, **frames))


def row(out: pl.DataFrame, date=T, security_id=1) -> dict:
    return out.row(by_predicate=(pl.col("date") == date) & (pl.col("security_id") == security_id), named=True)


def macro(dgs10=None, dgs3mo=None, baa10y=None, *, lag_days=1) -> pl.DataFrame:
    series = {"DGS10": dgs10, "DGS3MO": dgs3mo, "BAA10Y": baa10y}
    return make_macro({k: v for k, v in series.items() if v is not None}, lag_days=lag_days)


def jump_prices(returns: dict[int, float], on: dt.date, *, extra=None) -> pl.DataFrame:
    """Each stock is flat at 100 and steps to 100 x (1 + r) on `on`."""
    spec = {sid: {DAYS[0]: 100.0, on: 100.0 * (1.0 + r)} for sid, r in returns.items()}
    return make_prices({**spec, **(extra or {})}, CAL)


def members(date=T, ids=MEMBERS) -> list[tuple[int, dt.date]]:
    return [(sid, date) for sid in ids]


# ---------------------------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------------------------

def test_module_matches_the_catalog():
    assert context.FAMILIES == ("context",)
    assert list(context.FEATURES["context"]) == load_catalog().names(family="context") == FEATURES


def test_one_row_per_entry_same_value_for_every_stock_on_a_date():
    returns = {sid: sid / 100 for sid in MEMBERS}
    out = run(
        entries=members(T) + members(T1, MEMBERS[:5]),
        prices=jump_prices(returns, DAYS[IT - 3]),
        benchmark=make_benchmark(CAL, 0.001),
        macro=macro({d(2020, 12, 30): 0.9, d(2020, 6, 29): 0.7}, {d(2020, 12, 30): 0.1, d(2020, 6, 29): 0.2},
                    {d(2020, 12, 30): 2.0, d(2020, 6, 29): 3.0}),
    )
    assert out.columns == ["security_id", "date", *FEATURES, "src_date_context"]
    assert out.height == 25
    assert out.select("security_id", "date").is_unique().all()
    assert out["security_id"].to_list() == sorted(out["security_id"].to_list())
    assert [out.schema[c] for c in FEATURES] == [pl.Float64] * 4
    assert out.schema["src_date_context"] == pl.Date
    for c in FEATURES:
        assert out[c].is_finite().all()                 # no null here, and never NaN / inf
        assert out.group_by("date").agg(pl.col(c).n_unique())[c].to_list() == [1, 1]
    assert (out["src_date_context"] <= out["date"]).all()
    # T1 has only 5 members (below the minimum of 20) but dispersion is still defined at T: check T1
    assert out.filter(pl.col("date") == T1)["term_spread"].to_list() == pytest.approx([0.5] * 5)


def test_no_inputs_gives_nulls_for_every_entry():
    out = run(entries=[(1, T), (2, T), (1, T1)])
    assert out.height == 3
    assert out.select(pl.all_horizontal(pl.col(*FEATURES, "src_date_context").is_null()).all()).item()
    assert [out.schema[c] for c in FEATURES] == [pl.Float64] * 4


# ---------------------------------------------------------------------------------------------
# term_spread, credit_spread
# ---------------------------------------------------------------------------------------------

def test_term_spread_is_10y_minus_3m_from_the_latest_published_observations():
    # published the next day: the 12-29 observations are known on 12-30, the 12-31 ones only on 1-1
    out = run(macro=macro(
        dgs10={d(2020, 12, 28): 0.90, d(2020, 12, 29): 0.94, d(2020, 12, 31): 5.0},
        dgs3mo={d(2020, 12, 28): 0.20, d(2020, 12, 29): 0.09, d(2020, 12, 31): 1.0},
    ))
    r = row(out)
    assert r["term_spread"] == pytest.approx(0.94 - 0.09)
    assert r["credit_spread"] is None            # no BAA10Y rows
    assert r["src_date_context"] == d(2020, 12, 30)


def test_term_spread_sign_inverted_curve_is_negative():
    out = run(macro=macro(dgs10={d(2020, 12, 29): 1.5}, dgs3mo={d(2020, 12, 29): 2.25}))
    assert row(out)["term_spread"] == pytest.approx(1.5 - 2.25) == pytest.approx(-0.75)


def test_term_spread_series_are_chosen_independently():
    # the 3-month yield for 12-29 is missing: its latest observation is 12-28, the 10-year one 12-29
    out = run(macro=macro(dgs10={d(2020, 12, 28): 0.90, d(2020, 12, 29): 0.94}, dgs3mo={d(2020, 12, 28): 0.20}))
    r = row(out)
    assert r["term_spread"] == pytest.approx(0.94 - 0.20)
    assert r["src_date_context"] == d(2020, 12, 30)   # the later of the two releases used


def test_term_spread_null_when_one_leg_is_missing():
    out = run(macro=macro(dgs10={d(2020, 12, 29): 0.94}, baa10y={d(2020, 12, 29): 2.3}))
    r = row(out)
    assert r["term_spread"] is None
    assert r["credit_spread"] == pytest.approx(2.3)


def test_credit_spread_is_the_latest_published_baa10y():
    out = run(macro=macro(baa10y={d(2020, 12, 24): 2.41, d(2020, 12, 28): 2.36, d(2020, 12, 31): 9.0}))
    r = row(out)
    assert r["credit_spread"] == pytest.approx(2.36)
    assert r["term_spread"] is None
    assert r["src_date_context"] == d(2020, 12, 29)


def test_macro_released_on_the_prediction_date_is_used_one_day_later_is_not():
    obs = {d(2020, 12, 29): 1.0, d(2020, 12, 30): 2.0}
    assert row(run(macro=macro(baa10y=obs, lag_days=1)))["credit_spread"] == 2.0   # 12-30 known on 12-31 = T
    assert row(run(macro=macro(baa10y=obs, lag_days=2)))["credit_spread"] == 1.0   # 12-30 known on 1-1 > T


def test_macro_observed_before_but_released_after_the_date_is_not_used():
    # a 5-day publication lag: at T1 = 6-30 the newest release is the 6-25 observation (out 6-30);
    # 6-26 (out 7-1) and 6-29 (out 7-4) are dated before T1 but were not public yet
    lag = dict(lag_days=5)
    base = dict(
        dgs10={d(2020, 6, 24): 0.70, d(2020, 6, 25): 0.68, d(2020, 6, 26): 0.64, d(2020, 6, 29): 0.63},
        dgs3mo={d(2020, 6, 24): 0.15, d(2020, 6, 25): 0.14, d(2020, 6, 26): 0.13, d(2020, 6, 29): 0.12},
        baa10y={d(2020, 6, 24): 2.50, d(2020, 6, 25): 2.45, d(2020, 6, 26): 2.40, d(2020, 6, 29): 2.35},
    )
    out = run(entries=[(1, T1)], macro=macro(**base, **lag))
    r = row(out, T1)
    assert r["term_spread"] == pytest.approx(0.68 - 0.14)
    assert r["credit_spread"] == pytest.approx(2.45)
    assert r["src_date_context"] == T1

    # change everything released after T1: nothing moves
    late = d(2020, 6, 25)
    changed = {k: {day: (v + 7.0 if day > late else v) for day, v in obs.items()} for k, obs in base.items()}
    changed["dgs10"][d(2020, 6, 30)] = 99.0
    assert run(entries=[(1, T1)], macro=macro(**changed, **lag)).equals(out)
    # ... while a change to a release usable at T1 does
    used = {**base, "baa10y": {**base["baa10y"], late: 3.0}}
    assert row(run(entries=[(1, T1)], macro=macro(**used, **lag)), T1)["credit_spread"] == 3.0


def test_macro_revision_published_later_is_not_used_and_one_published_by_t_is():
    first = macro(baa10y={d(2020, 12, 28): 2.0, d(2020, 12, 29): 2.5})
    revise = lambda day, value, available: pl.DataFrame(  # noqa: E731
        [("BAA10Y", day, available, value, False)], schema=first.schema, orient="row"
    )
    # 12-29 revised to 4.0 on 1-4 (after T): still 2.5
    assert row(run(macro=pl.concat([first, revise(d(2020, 12, 29), 4.0, d(2021, 1, 4))])))["credit_spread"] == 2.5
    # 12-29 revised to 4.0 on 12-31 (= T): the revision is what was known at T
    r = row(run(macro=pl.concat([first, revise(d(2020, 12, 29), 4.0, T)])))
    assert (r["credit_spread"], r["src_date_context"]) == (4.0, T)
    # a revision of the OLDER observation (12-28) arriving last does not replace the 12-29 value
    r = row(run(macro=pl.concat([first, revise(d(2020, 12, 28), 9.0, T)])))
    assert (r["credit_spread"], r["src_date_context"]) == (2.5, d(2020, 12, 30))


def test_macro_stale_after_ten_calendar_days():
    # T = 12-31: an observation dated 12-21 is 10 days old (used), one dated 12-18 is 13 days old (stale)
    assert (T - d(2020, 12, 21)).days == 10
    fresh = row(run(macro=macro(baa10y={d(2020, 12, 21): 2.2})))
    assert (fresh["credit_spread"], fresh["src_date_context"]) == (2.2, d(2020, 12, 22))
    stale = row(run(macro=macro(baa10y={d(2020, 12, 18): 2.2})))
    assert (stale["credit_spread"], stale["src_date_context"]) == (None, None)
    # 11 days old, on a calendar with a Sunday observation
    assert row(run(macro=macro(baa10y={d(2020, 12, 20): 2.2})))["credit_spread"] is None
    # one stale leg makes the spread null; the fresh series are unaffected
    r = row(run(macro=macro(dgs10={d(2020, 12, 29): 0.9}, dgs3mo={d(2020, 12, 1): 0.1}, baa10y={d(2020, 12, 29): 2.0})))
    assert (r["term_spread"], r["credit_spread"]) == (None, 2.0)


def test_macro_missing_values_are_skipped():
    # FRED holidays: a null or NaN value is not an observation
    m = macro(baa10y={d(2020, 12, 28): 2.1, d(2020, 12, 29): float("nan")})
    assert row(run(macro=m))["credit_spread"] == 2.1
    m = m.with_columns(pl.when(pl.col("value").is_nan()).then(None).otherwise(pl.col("value")).alias("value"))
    assert row(run(macro=m))["credit_spread"] == 2.1


# ---------------------------------------------------------------------------------------------
# mkt_ret_12m
# ---------------------------------------------------------------------------------------------

def test_mkt_ret_12m_is_the_benchmark_return_over_252_trading_days():
    start = DAYS[IT - 252]
    rets = {
        start: -0.50,             # the return OF the start day is before the window
        DAYS[IT - 251]: 0.10,     # first day of the window
        DAYS[IT - 100]: -0.20,
        T: 0.05,                  # the close of t is included
    }
    r = row(run(benchmark=make_benchmark(CAL, rets)))
    assert r["mkt_ret_12m"] == pytest.approx(1.10 * 0.80 * 1.05 - 1) == pytest.approx(-0.076)
    assert r["src_date_context"] == T


def test_mkt_ret_12m_uses_levels_not_a_fixed_base():
    # every day +0.1%: (1.001)^252 - 1, whatever the level at the start of the window
    r = row(run(benchmark=make_benchmark(CAL, 0.001)))
    assert r["mkt_ret_12m"] == pytest.approx(1.001**252 - 1)


def test_mkt_ret_12m_null_without_252_days_of_history():
    bench = make_benchmark(CAL, 0.001)
    first_ok, too_early = DAYS[252], DAYS[251]
    out = run(entries=[(1, first_ok), (1, too_early)], benchmark=bench)
    assert row(out, first_ok)["mkt_ret_12m"] == pytest.approx(1.001**252 - 1)
    assert row(out, too_early)["mkt_ret_12m"] is None
    assert row(out, too_early)["src_date_context"] is None


def test_mkt_ret_12m_null_when_the_benchmark_had_no_members_at_the_window_start():
    start = DAYS[IT - 252]
    bench = make_benchmark(CAL, 0.001)
    empty_at = lambda day: bench.with_columns(  # noqa: E731
        pl.when(pl.col("date") == day).then(0).otherwise(pl.col("n_members")).alias("n_members")
    )
    assert row(run(benchmark=empty_at(start)))["mkt_ret_12m"] is None
    assert row(run(benchmark=empty_at(DAYS[IT - 253])))["mkt_ret_12m"] is not None   # the day before: fine


def test_mkt_ret_12m_null_when_the_window_is_incomplete():
    bench = make_benchmark(CAL, 0.001)
    without = lambda day: bench.filter(pl.col("date") != day)  # noqa: E731
    assert row(run(benchmark=without(DAYS[IT - 252])))["mkt_ret_12m"] is None   # no start level
    assert row(run(benchmark=without(DAYS[IT - 30])))["mkt_ret_12m"] is None    # a day inside is missing
    assert row(run(benchmark=without(T)))["mkt_ret_12m"] is None                # no level at t
    assert row(run(benchmark=without(DAYS[IT - 253])))["mkt_ret_12m"] is not None
    # a non-positive level is not a level
    broken = bench.with_columns(
        pl.when(pl.col("date") == DAYS[IT - 252]).then(0.0).otherwise(pl.col("bench_level")).alias("bench_level")
    )
    assert row(run(benchmark=broken))["mkt_ret_12m"] is None
    # a date that is not a trading day has no window
    saturday = d(2020, 12, 26)
    assert row(run(entries=[(1, saturday)], benchmark=bench), saturday)["mkt_ret_12m"] is None


def test_mkt_ret_12m_ignores_the_benchmark_after_the_date():
    rets = {DAYS[IT1 - 251]: 0.10, DAYS[IT1 - 5]: 0.02}
    out = run(entries=[(1, T1)], benchmark=make_benchmark(CAL, rets))
    assert row(out, T1)["mkt_ret_12m"] == pytest.approx(1.10 * 1.02 - 1)
    later = {**rets, DAYS[IT1 + 1]: 0.30, DAYS[IT1 + 40]: -0.25}
    assert run(entries=[(1, T1)], benchmark=make_benchmark(CAL, later)).equals(out)
    # ... and the return of the date itself is part of the window
    same_day = {**rets, T1: 0.30}
    assert row(run(entries=[(1, T1)], benchmark=make_benchmark(CAL, same_day)), T1)["mkt_ret_12m"] == pytest.approx(
        1.10 * 1.02 * 1.30 - 1
    )


# ---------------------------------------------------------------------------------------------
# xs_dispersion
# ---------------------------------------------------------------------------------------------

def sample_std(values) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def test_xs_dispersion_is_the_sample_std_of_21_day_returns():
    # stock k returns k% over the window: 1%, 2%, ..., 20%
    returns = {sid: sid / 100 for sid in MEMBERS}
    out = run(entries=members(), prices=jump_prices(returns, DAYS[IT - 20]))
    r = row(out)
    # std of 1..20 with ddof = 1 is sqrt(20 x 21 / 12) = sqrt(35), so the answer is 0.01 x sqrt(35)
    assert r["xs_dispersion"] == pytest.approx(0.01 * math.sqrt(35.0))
    assert r["xs_dispersion"] == pytest.approx(sample_std(list(returns.values())))
    assert r["src_date_context"] == T
    assert out["xs_dispersion"].n_unique() == 1


def test_xs_dispersion_window_is_21_trading_days():
    returns = {sid: sid / 100 for sid in MEMBERS}
    # stock 1 jumps 50% ON the start day (21 days back): already in the start price, so its return is 0
    spec = {1: {DAYS[0]: 100.0, DAYS[IT - 21]: 150.0}}
    out = run(entries=members(), prices=jump_prices({k: v for k, v in returns.items() if k != 1}, T, extra=spec))
    expected = sample_std([0.0, *[sid / 100 for sid in MEMBERS[1:]]])
    assert row(out)["xs_dispersion"] == pytest.approx(expected)
    # one day later (20 days back) it is inside the window
    spec = {1: {DAYS[0]: 100.0, DAYS[IT - 20]: 150.0}}
    out = run(entries=members(), prices=jump_prices({k: v for k, v in returns.items() if k != 1}, T, extra=spec))
    assert row(out)["xs_dispersion"] == pytest.approx(sample_std([0.5, *[sid / 100 for sid in MEMBERS[1:]]]))


def test_xs_dispersion_uses_total_return_prices():
    # adj_close (total return) rises 10% for the odd stocks; the split-adjusted close does not move
    returns = {sid: (0.10 if sid % 2 else 0.0) for sid in MEMBERS}
    prices = jump_prices(returns, DAYS[IT - 5]).with_columns(pl.lit(100.0).alias("close"), pl.lit(100.0).alias("close_unadj"))
    # ten returns of 0.10 and ten of 0: mean 0.05, sum of squares 20 x 0.05^2, variance 0.05 / 19
    assert row(run(entries=members(), prices=prices))["xs_dispersion"] == pytest.approx(math.sqrt(0.05 / 19))


def test_xs_dispersion_is_zero_when_every_member_has_the_same_return():
    out = run(entries=members(), prices=jump_prices({sid: 0.04 for sid in MEMBERS}, DAYS[IT - 3]))
    assert row(out)["xs_dispersion"] == pytest.approx(0.0, abs=1e-12)


def test_xs_dispersion_only_counts_universe_members():
    returns = {sid: sid / 100 for sid in MEMBERS}
    with_outsider = jump_prices({**returns, 99: 5.0}, DAYS[IT - 20])   # stock 99 is not in the entries at T
    out = run(entries=members() + [(99, T1)], prices=with_outsider)
    assert row(out)["xs_dispersion"] == pytest.approx(0.01 * math.sqrt(35.0))
    assert row(out, T1, 99)["xs_dispersion"] is None                   # one member at T1


def test_xs_dispersion_needs_twenty_members_with_a_return():
    returns = {sid: sid / 100 for sid in MEMBERS}
    prices = jump_prices(returns, DAYS[IT - 20])
    assert row(run(entries=members(ids=MEMBERS[:19]), prices=prices))["xs_dispersion"] is None
    # 20 members, but stock 20 listed inside the window: no start price, so only 19 returns
    late = prices.filter((pl.col("security_id") != 20) | (pl.col("date") > DAYS[IT - 21]))
    r = row(run(entries=members(), prices=late))
    assert (r["xs_dispersion"], r["src_date_context"]) == (None, None)
    # a zero start price is no price either
    zero = prices.with_columns(
        pl.when((pl.col("security_id") == 20) & (pl.col("date") == DAYS[IT - 21]))
        .then(0.0).otherwise(pl.col("adj_close")).alias("adj_close")
    )
    assert row(run(entries=members(), prices=zero))["xs_dispersion"] is None
    # 21 members, one without a price at t: the other 20 define it
    returns21 = {**returns, 21: 0.9}
    no_close = jump_prices(returns21, DAYS[IT - 20]).filter((pl.col("security_id") != 21) | (pl.col("date") < T))
    out = run(entries=members(ids=[*MEMBERS, 21]), prices=no_close)
    assert out.height == 21
    assert row(out, T, 21)["xs_dispersion"] == pytest.approx(0.01 * math.sqrt(35.0))


def test_xs_dispersion_ignores_prices_after_the_date():
    returns = {sid: sid / 100 for sid in MEMBERS}
    prices = jump_prices(returns, DAYS[IT1 - 10])
    out = run(entries=members(T1), prices=prices)
    assert row(out, T1)["xs_dispersion"] == pytest.approx(0.01 * math.sqrt(35.0))
    later = prices.with_columns(
        pl.when(pl.col("date") > T1).then(pl.col("adj_close") * (1 + pl.col("security_id") / 7)).otherwise(pl.col("adj_close")).alias("adj_close")
    )
    assert run(entries=members(T1), prices=later).equals(out)
    # ... and the close of the date itself is used
    at_t1 = prices.with_columns(
        pl.when((pl.col("date") == T1) & (pl.col("security_id") == 1)).then(200.0).otherwise(pl.col("adj_close")).alias("adj_close")
    )
    changed = row(run(entries=members(T1), prices=at_t1), T1)["xs_dispersion"]
    assert changed == pytest.approx(sample_std([1.0, *[sid / 100 for sid in MEMBERS[1:]]]))


# ---------------------------------------------------------------------------------------------
# all four together
# ---------------------------------------------------------------------------------------------

def full_inputs(shock: float = 0.0) -> dict:
    """Inputs for T1 and T; `shock` changes every input dated (or released) after T1."""
    returns = {sid: sid / 100 for sid in MEMBERS}
    prices = jump_prices(returns, DAYS[IT1 - 10])
    prices = prices.with_columns(
        pl.when(pl.col("date") > DAYS[IT - 5]).then(pl.col("adj_close") * (1 + shock * pl.col("security_id")))
        .otherwise(pl.col("adj_close")).alias("adj_close")
    )
    obs = lambda a, b, k=1.0: {  # noqa: E731
        d(2020, 6, 26): a, d(2020, 6, 29): b, d(2020, 6, 30): b + k * shock, d(2020, 12, 30): b + 1 + k * shock
    }
    return dict(
        prices=prices,
        benchmark=make_benchmark(CAL, {DAYS[IT1 - 100]: 0.10, DAYS[IT1 + 1]: shock, DAYS[IT - 1]: 0.01 + shock}),
        macro=macro(obs(0.70, 0.66, 2.0), obs(0.16, 0.14), obs(2.5, 2.4)),
    )


def test_changing_any_input_after_the_date_changes_nothing_at_that_date():
    entries = members(T1) + members(T)
    base = run(entries=entries, **full_inputs())
    shocked = run(entries=entries, **full_inputs(0.25))
    at = lambda out, day: out.filter(pl.col("date") == day)  # noqa: E731
    r = row(base, T1)
    assert r["term_spread"] == pytest.approx(0.66 - 0.14)       # the 6-29 observations, released 6-30 = T1
    assert r["credit_spread"] == pytest.approx(2.4)
    assert r["mkt_ret_12m"] == pytest.approx(0.10)
    assert r["xs_dispersion"] == pytest.approx(0.01 * math.sqrt(35.0))
    assert at(shocked, T1).equals(at(base, T1))
    # the shock is real: every feature at T moves
    for c in FEATURES:
        assert row(shocked)[c] != pytest.approx(row(base)[c]), c


def test_src_date_is_the_latest_input_used_and_never_after_the_date():
    entries = members(T1) + members(T)
    out = run(entries=entries, **full_inputs())
    assert out["src_date_context"].to_list() == out["date"].to_list()   # prices through the close of t
    # macro only: the release dates, not t
    only_macro = run(macro=macro({d(2020, 12, 23): 0.9}, {d(2020, 12, 28): 0.1}, {d(2020, 12, 24): 2.0}))
    assert row(only_macro)["src_date_context"] == d(2020, 12, 29)      # DGS3MO, released last
    # a stale series does not count as used
    partly = run(macro=macro({d(2020, 12, 23): 0.9}, {d(2020, 12, 1): 0.1}, {d(2020, 12, 24): 2.0}))
    r = row(partly)
    assert (r["term_spread"], r["credit_spread"], r["src_date_context"]) == (None, 2.0, d(2020, 12, 25))
    assert (out["src_date_context"] <= out["date"]).all()

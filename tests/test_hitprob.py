"""Tests for hitprob: Monte Carlo hit probabilities, the scaled hit table, the classifier and helpers."""

from __future__ import annotations

import datetime as dt
import time

import numpy as np
import polars as pl
import pytest

from ltforecast.hitprob import (
    hit_calibration,
    CLASSES,
    PROB_COLS,
    HitClassifier,
    HitClassifierParams,
    HitProb,
    HitTable,
    HitTableParams,
    annual_drift,
    assign_buckets,
    choose_method,
    expected_value,
    multiclass_brier,
    reward_risk,
    simulate,
    standardized_residuals,
    vol_bucket_edges,
)

RNG = np.random.default_rng(123)
GAUSS = RNG.standard_normal(200_000)
# fat-tailed shocks (student t, 4 dof) for the table-vs-simulation check
FAT = RNG.standard_t(4, 200_000)


def se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 1e-4) / n))


@pytest.fixture(scope="module")
def table() -> HitTable:
    return HitTable.build({0: GAUSS, 1: FAT}, HitTableParams(n_paths=10_000, seed=7), edges=np.array([0.3]))


# simulate -----------------------------------------------------------------------------------------


def test_zero_drift_symmetric_barriers_equal_probabilities():
    n = 20_000
    p = simulate(0.0, 0.3, np.exp(0.25), np.exp(-0.25), GAUSS, n_paths=n, seed=1)
    assert p.p_tp + p.p_sl + p.p_none == pytest.approx(1.0)
    # p_tp - p_sl has std <= sqrt((p_tp + p_sl) / n)
    assert abs(p.p_tp - p.p_sl) < 4 * np.sqrt((p.p_tp + p.p_sl) / n)
    assert p.p_tp > 0.2 and p.p_sl > 0.2


def test_conditional_on_exit_matches_two_barrier_brownian():
    # zero drift: P(up first | exit) = b / (a + b); barriers 0.5 and 0.25 horizon sigmas, so most paths
    # exit. Discrete daily monitoring shifts each barrier out by ~0.5826 daily sigmas (Broadie-Glasserman).
    n, sig = 40_000, 0.4
    up, dn = 0.2, 0.1
    p = simulate(0.0, sig, np.exp(up), np.exp(-dn), GAUSS, n_paths=n, seed=2)
    assert p.p_none < 0.05
    cond = p.p_tp / (p.p_tp + p.p_sl)
    shift = 0.5826 * sig / np.sqrt(252)
    assert cond == pytest.approx(dn / (up + dn), abs=0.035)
    assert cond == pytest.approx((dn + shift) / (up + dn + 2 * shift), abs=4 * se(cond, n) + 0.005)


def test_none_barriers_give_zero_on_that_side_and_raise_time_probability():
    both = simulate(0.05, 0.3, 1.3, 0.8, GAUSS, n_paths=10_000, seed=3)
    no_tp = simulate(0.05, 0.3, None, 0.8, GAUSS, n_paths=10_000, seed=3)
    no_sl = simulate(0.05, 0.3, 1.3, None, GAUSS, n_paths=10_000, seed=3)
    hold = simulate(0.05, 0.3, None, None, GAUSS, n_paths=10_000, seed=3)
    assert no_tp.p_tp == 0.0 and no_sl.p_sl == 0.0
    assert hold.p_none == 1.0 and hold.p_tp == hold.p_sl == 0.0
    assert no_tp.p_none > both.p_none and no_sl.p_none > both.p_none
    # same paths: removing a barrier can only add hits on the other side
    assert no_tp.p_sl >= both.p_sl and no_sl.p_tp >= both.p_tp
    # hold: E[simple return] = exp(mu + sigma^2 / 2) - 1 for Gaussian log returns
    assert hold.mean_ret_none == pytest.approx(np.expm1(0.05 + 0.045), abs=4 * 0.33 / 100)


def test_simulate_deterministic_and_validates():
    a = simulate(0.1, 0.3, 1.2, 0.85, FAT, n_paths=2000, seed=5)
    b = simulate(0.1, 0.3, 1.2, 0.85, FAT, n_paths=2000, seed=5)
    assert a == b
    with pytest.raises(ValueError):
        simulate(0.0, 0.3, 0.9, 0.85, GAUSS)        # tp below entry
    with pytest.raises(ValueError):
        simulate(0.0, 0.0, 1.2, 0.85, GAUSS)


def test_simulate_shocks_are_restandardized():
    # scaling or shifting the residual pool changes nothing
    a = simulate(0.0, 0.3, 1.2, 0.85, GAUSS[:5000], n_paths=3000, seed=9)
    b = simulate(0.0, 0.3, 1.2, 0.85, 3.0 * GAUSS[:5000] + 0.7, n_paths=3000, seed=9)
    assert a.p_tp == pytest.approx(b.p_tp) and a.p_sl == pytest.approx(b.p_sl)


# table --------------------------------------------------------------------------------------------


def test_table_zero_drift_symmetric_and_closed_form(table):
    sd = 0.3 / np.sqrt(252)
    p = table.lookup(0.0, 0.25 / sd, 0.25 / sd, 0)
    assert abs(p.p_tp[0] - p.p_sl[0]) < 0.02
    # P(up first | exit) = b / (a + b), a = 0.5, b = 0.25 horizon sigmas (in daily units * sqrt(252))
    rh = np.sqrt(252)
    p = table.lookup(0.0, 0.5 * rh, 0.25 * rh, 0)
    cond = p.p_tp[0] / (p.p_tp[0] + p.p_sl[0])
    assert p.p_none[0] < 0.05
    assert cond == pytest.approx(1 / 3, abs=0.035)


@pytest.mark.parametrize("bucket,shocks", [(0, GAUSS), (1, FAT)])
def test_table_matches_direct_simulation_at_random_points(table, bucket, shocks):
    rng = np.random.default_rng(11 + bucket)
    n_paths = 20_000
    for _ in range(6):
        mu = rng.uniform(-0.3, 0.4)
        sig = rng.uniform(0.15, 0.8)
        tp = np.exp(rng.uniform(0.3, 2.0) * sig)
        sl = np.exp(-rng.uniform(0.3, 1.5) * sig)
        direct = simulate(mu, sig, tp, sl, shocks, n_paths=n_paths, seed=int(rng.integers(1 << 30)))
        got = table.score(np.array([mu]), np.array([sig]), np.array([tp]), np.array([sl]), np.array([bucket]))
        for name in ("p_tp", "p_sl", "p_none"):
            d, t = getattr(direct, name), getattr(got, name)[0]
            # MC error of both estimates (10k-path table, 20k-path direct) + interpolation allowance
            tol = 4 * np.sqrt(d * (1 - d) * (1 / n_paths + 1 / 10_000)) + 0.02
            assert t == pytest.approx(d, abs=tol), (name, mu, sig, tp, sl)
        if direct.p_none > 0.1:
            assert got.mean_ret_none[0] == pytest.approx(direct.mean_ret_none, abs=0.03)


def test_table_none_barriers(table):
    mu, sig = np.array([0.08, 0.08]), np.array([0.3, 0.3])
    p = table.score(mu, sig, np.array([1.3, np.nan]), np.array([0.8, 0.8]), np.array([0, 0]))
    assert p.p_tp[1] == 0.0 and p.p_none[1] > p.p_none[0]
    q = table.score(mu, sig, None, None, np.array([0, 0]))
    np.testing.assert_allclose(q.p_none, 1.0)
    np.testing.assert_allclose(q.p_tp, 0.0)
    # hold with Gaussian shocks: E[ret] = exp(mu + sigma^2 / 2) - 1
    np.testing.assert_allclose(q.mean_ret_none, np.expm1(0.08 + 0.045), atol=0.01)
    r = table.lookup(0.0, None, 0.2 * np.sqrt(252), 0)
    assert r.p_tp[0] == 0.0 and r.p_sl[0] > 0.5


def test_table_probabilities_sum_to_one_and_monotone(table):
    sig = np.full(50, 0.4)
    tp = np.exp(np.linspace(0.05, 1.5, 50))
    p = table.score(np.zeros(50), sig, tp, np.full(50, 0.8), np.zeros(50, dtype=int))
    np.testing.assert_allclose(p.p_tp + p.p_sl + p.p_none, 1.0, atol=1e-12)
    # a farther target is hit less often
    assert np.all(np.diff(p.p_tp) <= 1e-9)


def test_table_score_buckets_from_edges(table):
    mu, sig = np.array([0.0, 0.0]), np.array([0.2, 0.5])
    tp, sl = np.array([1.2, 1.5]), np.array([0.85, 0.7])
    a = table.score(mu, sig, tp, sl)
    b = table.score(mu, sig, tp, sl, np.array([0, 1]))
    np.testing.assert_array_equal(a.as_array(), b.as_array())
    with pytest.raises(KeyError):
        table.score(mu, sig, tp, sl, np.array([0, 2]))


def test_table_build_deterministic_per_bucket():
    p = HitTableParams(n_paths=500, horizon=40, seed=3, theta_nodes=(-1.0, 0.0, 1.0),
                       barrier_nodes=(0.2, 0.5, 1.0, float("inf")))
    t1 = HitTable.build({0: GAUSS[:3000], 1: FAT[:3000]}, p)
    t2 = HitTable.build({1: FAT[:3000]}, p)
    np.testing.assert_array_equal(t1.tables[1], t2.tables[1])


def test_table_scores_400k_quickly(table):
    n = 400_000
    rng = np.random.default_rng(0)
    sig = rng.uniform(0.1, 1.0, n)
    mu = rng.normal(0.05, 0.2, n)
    tp = np.exp(1.0 * sig)
    sl = np.where(rng.random(n) < 0.1, np.nan, np.exp(-0.75 * sig))
    t0 = time.perf_counter()
    p = table.score(mu, sig, tp, sl)
    assert time.perf_counter() - t0 < 5.0
    assert np.isfinite(p.as_array()).all()


# residuals ----------------------------------------------------------------------------------------


def _panel(n_days: int, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n_days)]
    sids = [1, 2, 3, 4, 5, 6]
    vols = {s: 0.1 * s for s in sids}
    prices = pl.DataFrame({
        "security_id": np.repeat(sids, n_days),
        "date": dates * len(sids),
        "ret": np.concatenate([np.expm1(rng.standard_normal(n_days) * vols[s] / np.sqrt(252)) for s in sids]),
    })
    sigma = pl.DataFrame({
        "security_id": np.repeat(sids, n_days),
        "date": dates * len(sids),
        "sigma": np.concatenate([np.full(n_days, vols[s]) for s in sids]),
    })
    return prices, sigma


def test_standardized_residuals_hand_value():
    prices = pl.DataFrame({"security_id": [1, 1, 1], "date": [dt.date(2020, 1, d) for d in (1, 2, 3)],
                           "ret": [0.0, np.expm1(0.02), np.expm1(-0.01)]})
    sigma = pl.DataFrame({"security_id": [1, 1], "date": [dt.date(2020, 1, 1), dt.date(2020, 1, 2)],
                          "sigma": [0.2, 0.4]})
    z = standardized_residuals(prices, sigma, dt.date(2020, 1, 3), n_buckets=1)
    # day 1 has no earlier sigma; day 2 uses sigma(1/1) = 0.2, day 3 uses sigma(1/2) = 0.4
    np.testing.assert_allclose(z[0], [0.02 / (0.2 / np.sqrt(252)), -0.01 / (0.4 / np.sqrt(252))])


def test_standardized_residuals_point_in_time():
    prices, sigma = _panel(120)
    through = dt.date(2020, 3, 1)
    base = standardized_residuals(prices, sigma, through)
    # later rows with wild values must not change anything
    later_p = prices.with_columns(pl.col("date") + pl.duration(days=200), pl.lit(5.0).alias("ret"))
    later_s = sigma.with_columns(pl.col("date") + pl.duration(days=200), pl.lit(9.0).alias("sigma"))
    after = standardized_residuals(pl.concat([prices, later_p]), pl.concat([sigma, later_s]), through)
    assert base.keys() == after.keys()
    for b in base:
        np.testing.assert_array_equal(base[b], after[b])
    # terciles by vol: sids 1-2, 3-4, 5-6; standardized residuals have unit variance in each bucket
    for b in base:
        assert base[b].size > 0
        assert base[b].std() == pytest.approx(1.0, abs=0.15)


def test_vol_buckets():
    sigma = pl.DataFrame({"security_id": [1, 2, 3, 4], "date": [dt.date(2020, 1, 1)] * 3 + [dt.date(2021, 1, 1)],
                          "sigma": [0.1, 0.2, 0.3, 5.0]})
    edges = vol_bucket_edges(sigma, dt.date(2020, 6, 1), 2)
    np.testing.assert_allclose(edges, [0.2])
    np.testing.assert_array_equal(assign_buckets(np.array([0.1, 0.2, 0.25, np.nan]), edges), [0, 1, 1, -1])


def test_annual_drift():
    assert annual_drift(0.02, 0.06) == pytest.approx(0.02 + np.log(1.06))


# classifier ---------------------------------------------------------------------------------------


def _clf_frame(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    s = rng.uniform(0.1, 0.6, n)
    u = rng.random(n)
    outcome = np.where(u < 0.5 + 0.3 * np.tanh(x), "tp", np.where(u < 0.8 + 0.1 * np.tanh(x), "sl", "time"))
    dates = [dt.date(2010, 1, 31) + dt.timedelta(days=30 * (i % 20)) for i in range(n)]
    return pl.DataFrame({"date": dates, "security_id": np.arange(n), "f1": x, "f2": rng.standard_normal(n),
                         "sigma_hat": s, "outcome": outcome})


def test_hit_classifier_fit_predict():
    df = _clf_frame(600)
    params = HitClassifierParams(min_data_in_leaf=5, n_estimators=30, n_estimators_no_valid=30,
                                 early_stopping_rounds=10, num_threads=1)
    clf = HitClassifier(["f1", "f2"], params, seed=1)
    assert clf.cols == ["f1", "f2", "sigma_hat"]
    clf.fit(df[:400], df[400:500])
    p = clf.predict_proba(df[500:].drop("outcome"))
    assert p.shape == (100, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()
    # the planted signal: P(tp) rises with f1
    hi, lo = df[500:]["f1"].to_numpy() > 0.5, df[500:]["f1"].to_numpy() < -0.5
    assert p[hi, 0].mean() > p[lo, 0].mean()
    # deterministic and picklable
    import pickle

    clf2 = HitClassifier(["f1", "f2"], params, seed=1).fit(df[:400], df[400:500])
    np.testing.assert_array_equal(clf2.predict(df[500:]), p)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(clf)).predict(df[500:]), p)
    # no validation rows
    clf3 = HitClassifier(["f1"], params).fit(df[:400], None)
    assert clf3.predict(df[:5]).shape == (5, 3)


def test_hit_classifier_rejects_label_inputs():
    with pytest.raises(ValueError):
        HitClassifier(["f1", "fwd_ret_12m"])
    with pytest.raises(ValueError):
        HitClassifier(["f1", "outcome"])


# choosing and using -------------------------------------------------------------------------------


def test_multiclass_brier_literal():
    p = np.array([[1.0, 0.0, 0.0], [0.5, 0.25, 0.25]])
    # row 1: 0; row 2 (outcome sl): 0.25 + 0.5625 + 0.0625 = 0.875
    assert multiclass_brier(p, ["tp", "sl"]) == pytest.approx(0.875 / 2)


def _oof(p: np.ndarray) -> pl.DataFrame:
    n = len(p)
    return pl.DataFrame({"date": [dt.date(2015, 1, 31)] * n, "security_id": np.arange(n),
                         **{c: p[:, j] for j, c in enumerate(PROB_COLS)}})


def test_choose_method_picks_lower_brier():
    rng = np.random.default_rng(0)
    n = 300
    out = rng.choice(list(CLASSES), n)
    onehot = np.column_stack([out == c for c in CLASSES]).astype(float)
    good = 0.8 * onehot + 0.2 / 3
    flat = np.full((n, 3), 1 / 3)
    outcomes = pl.DataFrame({"date": [dt.date(2015, 1, 31)] * n, "security_id": np.arange(n), "outcome": out})
    m, tab = choose_method(_oof(good), _oof(flat), outcomes)
    assert m == "mc"
    assert tab["method"].to_list() == ["mc", "classifier", "average"]
    assert tab["brier"][0] < tab["brier"][2] < tab["brier"][1]
    m, _ = choose_method(_oof(flat), _oof(good), outcomes)
    assert m == "classifier"
    # opposite-direction errors: the average beats both
    over = 0.5 * onehot + 0.5 * np.roll(onehot, 1, axis=1)
    under = 0.5 * onehot + 0.5 * np.roll(onehot, 2, axis=1)
    m, tab = choose_method(_oof(over), _oof(under), outcomes)
    assert m == "average"


def test_expected_value_and_reward_risk_literal():
    # 0.4 * 0.25 + 0.3 * (-0.10) + 0.3 * 0.05 - 2 * 0.002 = 0.1 - 0.03 + 0.015 - 0.004 = 0.081
    assert expected_value(np.array([0.4, 0.3, 0.3]), 0.25, -0.10, 0.05, cost=0.002) == pytest.approx(0.081)
    hp = HitProb(np.array([0.4, 0.0]), np.array([0.3, 0.6]), np.array([0.3, 0.4]), np.array([0.05, -0.02]))
    ev = expected_value(hp, np.array([0.25, np.nan]), np.array([-0.10, -0.10]), hp.mean_ret_none, cost=0.0)
    np.testing.assert_allclose(ev, [0.085, -0.06 - 0.008])
    assert reward_risk(0.25, -0.10) == pytest.approx(2.5)
    np.testing.assert_allclose(reward_risk(np.array([0.3, np.nan, 0.2]), np.array([-0.15, -0.1, 0.0])),
                               [2.0, np.nan, np.nan])


# hit_calibration: criterion-6 calibration over the classes the chosen pair actually has -----------


def _hits(p_tp, p_sl, p_none, outcome):
    return pl.DataFrame({"p_tp": p_tp, "p_sl": p_sl, "p_none": p_none, "outcome": outcome})


def test_hit_calibration_no_tp_pair_checks_sl_and_none_not_vacuous_tp():
    # k_tp=None: p_tp = 0, no "tp" outcomes. 10 rows, 1 sl and 9 time; p_sl = 0.9 (badly wrong)
    out = ["sl"] + ["time"] * 9
    df = _hits([0.0] * 10, [0.9] * 10, [0.1] * 10, out)
    r = hit_calibration(df, k_tp=None, k_sl=1.5, bins=10, min_bin_rows=1)
    assert r["tested"] == ["sl", "time"]
    assert "tp" not in r["gaps"]
    # sl: one bin, mean_prob 0.9, realized 0.1 -> 0.8; time: mean 0.1, realized 0.9 -> 0.8
    assert r["gaps"]["sl"] == pytest.approx(0.8)
    assert r["gaps"]["time"] == pytest.approx(0.8)
    assert r["gap"] == pytest.approx(0.8)
    assert r["worst"] == "sl"                       # first in CLASSES order on ties
    assert r["applicable"] is True
    assert set(r["tables"]) == {"sl", "time"}


def test_hit_calibration_takes_max_over_classes():
    # both barriers: tp calibrated exactly, sl off by 0.3
    out = ["tp", "sl", "time", "time"] * 5      # 20 rows: 5 tp, 5 sl, 10 time
    df = _hits([0.25] * 20, [0.55] * 20, [0.2] * 20, out)
    r = hit_calibration(df, k_tp=1.0, k_sl=1.0, bins=10, min_bin_rows=1)
    assert r["tested"] == ["tp", "sl", "time"]
    assert r["gaps"]["tp"] == pytest.approx(0.0)
    assert r["gaps"]["sl"] == pytest.approx(0.30)
    assert r["gaps"]["time"] == pytest.approx(0.30)
    assert r["gap"] == pytest.approx(0.30)
    assert r["worst"] == "sl"


def test_hit_calibration_hold_pair_not_applicable():
    df = _hits([0.0] * 4, [0.0] * 4, [1.0] * 4, ["time"] * 4)
    r = hit_calibration(df, k_tp=None, k_sl=None, bins=10, min_bin_rows=1)
    assert r["applicable"] is False
    assert r["tested"] == []
    assert np.isnan(r["gap"]) and r["worst"] is None


def test_hit_calibration_identically_zero_column_untested():
    # TP barrier exists but every p_tp is 0 (degenerate) -> not counted as a pass
    df = _hits([0.0] * 4, [0.5] * 4, [0.5] * 4, ["sl", "sl", "time", "time"])
    r = hit_calibration(df, k_tp=2.0, k_sl=1.0, bins=10, min_bin_rows=1)
    assert r["tested"] == ["sl", "time"]
    assert r["gap"] == pytest.approx(0.0)


def test_hit_calibration_min_bin_rows_unmet_gives_nan():
    df = _hits([0.3] * 3, [0.3] * 3, [0.4] * 3, ["tp", "sl", "time"])
    r = hit_calibration(df, k_tp=1.0, k_sl=1.0, bins=10, min_bin_rows=50)
    assert r["applicable"] is True
    assert np.isnan(r["gap"]) and r["worst"] is None

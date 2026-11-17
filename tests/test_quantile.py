"""Quantile model (plan 6.3): coverage on heteroskedastic data, width vs vol, non-crossing,
early stopping, determinism, pickling and the walk-forward runner with three outputs.

Synthetic data: y = 0.2 * x + sigma * N(0, 1) with log sigma uniform on [log 0.1, log 0.8]; the
model sees x and three noisy copies of log sigma as lvol_21 / lvol_63 / lvol_252. The true
[q10, q90] interval has 80% coverage and width 2.563 * sigma.
"""

import datetime as dt
import pickle

import numpy as np
import polars as pl
import pytest
from conftest import business_days

from ltforecast.modelconfig import ValidationRules
from ltforecast.models.quantile import (
    QuantileModel,
    QuantileParams,
    pinball_loss,
    rearrange,
    width_vol_correlation,
)
from ltforecast.universe import month_end_dates
from ltforecast.validation.runner import walk_forward
from ltforecast.validation.splitter import walk_forward_folds

SMALL = QuantileParams(
    num_leaves=7, max_depth=3, min_data_in_leaf=30, learning_rate=0.05, lambda_l2=1.0,
    n_estimators=400, n_estimators_no_valid=60, early_stopping_rounds=30, num_threads=1,
)


def hetero(n: int, seed: int, per_date: int = 100) -> tuple[pl.DataFrame, np.ndarray]:
    r = np.random.default_rng(seed)
    x = r.uniform(-0.5, 0.5, n)
    lv = r.uniform(np.log(0.1), np.log(0.8), n)
    sigma = np.exp(lv)
    y = 0.2 * x + sigma * r.standard_normal(n)
    df = pl.DataFrame({
        "date": np.arange(n) // per_date,
        "security_id": np.arange(n),
        "x": x,
        "lvol_21": lv + 0.1 * r.standard_normal(n),
        "lvol_63": lv + 0.05 * r.standard_normal(n),
        "lvol_252": lv,
        "excess_w_12m": y,
    })
    return df, sigma


@pytest.fixture(scope="module")
def fitted():
    train, _ = hetero(3000, 1)
    valid, _ = hetero(1000, 2)
    return QuantileModel(["x"], params=SMALL).fit(train, valid)


# --- helpers ---------------------------------------------------------------------------------


def test_pinball_loss_hand_values():
    # y - q = [2, -1]; alpha 0.1: [0.2, 0.9] -> 0.55; alpha 0.9: [1.8, 0.1] -> 0.95
    assert pinball_loss(np.array([3.0, 0.0]), np.array([1.0, 1.0]), 0.1) == pytest.approx(0.55)
    assert pinball_loss(np.array([3.0, 0.0]), np.array([1.0, 1.0]), 0.9) == pytest.approx(0.95)


def test_rearrange_sorts_each_row():
    q = np.array([[0.3, 0.1, 0.2], [-1.0, 0.0, 1.0]])
    assert rearrange(q).tolist() == [[0.1, 0.2, 0.3], [-1.0, 0.0, 1.0]]


def test_width_vol_correlation_hand_values():
    # date 0: widths 1,2,3 vs sigma 1,2,3 -> +1; date 1: widths 3,2,1 vs sigma 1,2,4 -> -1; date 2 too small
    q = np.array([[0, 1], [0, 2], [0, 3], [0, 3], [0, 2], [0, 1], [0, 5]], dtype=float)
    sigma = np.array([1, 2, 3, 1, 2, 4, 1], dtype=float)
    dates = [0, 0, 0, 1, 1, 1, 2]
    assert width_vol_correlation(q, sigma, dates, min_n=3) == pytest.approx(0.0)
    assert width_vol_correlation(q[:3], sigma[:3], dates[:3], min_n=3) == pytest.approx(1.0)
    assert np.isnan(width_vol_correlation(q[6:], sigma[6:], dates[6:], min_n=3))


def test_width_vol_correlation_rejects_bad_shapes():
    with pytest.raises(ValueError):
        width_vol_correlation(np.zeros((3, 1)), np.zeros(3), [0, 0, 0])
    with pytest.raises(ValueError):
        width_vol_correlation(np.zeros((3, 2)), np.zeros(2), [0, 0, 0])


# --- model -----------------------------------------------------------------------------------


def test_columns_fixed_order_with_vol_inputs_appended():
    m = QuantileModel(["x", "lvol_63"])
    assert m.cols == ["x", "lvol_63", "lvol_21", "lvol_252"]


@pytest.mark.parametrize("alphas", [(0.9, 0.1), (0.1, 0.1), (0.0, 0.5), (0.5, 1.0), ()])
def test_bad_alphas_rejected(alphas):
    with pytest.raises(ValueError):
        QuantileModel(["x"], alphas=alphas)


def test_target_cannot_be_input():
    with pytest.raises(ValueError):
        QuantileModel(["x", "excess_w_12m"])


def test_coverage_and_width_on_fresh_heteroskedastic_data(fitted):
    test, sigma = hetero(3000, 3)
    q = fitted.predict(test.drop("excess_w_12m"))
    y = test["excess_w_12m"].to_numpy()
    assert q.shape == (3000, 3)
    coverage = float(((y >= q[:, 0]) & (y <= q[:, 2])).mean())
    assert 0.70 <= coverage <= 0.90
    assert width_vol_correlation(q, sigma, test["date"]) > 0.5
    # the median sits inside the band and roughly half the outcomes fall below it
    assert 0.4 <= float((y <= q[:, 1]).mean()) <= 0.6


def test_early_stopping_on_valid_pinball(fitted):
    assert len(fitted.best_iterations) == 3
    assert all(1 <= r < SMALL.n_estimators for r in fitted.best_iterations)
    assert set(fitted.valid_pinball) == {0.1, 0.5, 0.9}
    assert all(v > 0 for v in fitted.valid_pinball.values())
    # refit: the final booster was trained for exactly best_iteration rounds
    assert [b.current_iteration() for b in fitted.boosters] == fitted.best_iterations


def test_no_refit_keeps_early_stopped_booster():
    train, _ = hetero(1500, 1)
    valid, _ = hetero(500, 2)
    p = QuantileParams(**{**SMALL.__dict__, "refit": False})
    m = QuantileModel(["x"], params=p).fit(train, valid)
    test, _ = hetero(200, 3)
    raw = m.predict_raw(test)
    want = m.boosters[0].predict(m._x(test), num_iteration=m.best_iterations[0])
    np.testing.assert_allclose(raw[:, 0], want)


def test_without_valid_uses_fixed_rounds():
    train, _ = hetero(600, 1)
    m = QuantileModel(["x"], params=SMALL).fit(train, None)
    assert m.best_iterations == [SMALL.n_estimators_no_valid] * 3
    assert m.valid_pinball == {}


def test_non_crossing_even_when_raw_boosters_cross():
    # tiny data, close alphas, aggressive trees: the raw boosters cross on many rows
    train, _ = hetero(40, 0)
    p = QuantileParams(
        num_leaves=4, max_depth=2, min_data_in_leaf=2, learning_rate=0.5, feature_fraction=1.0,
        bagging_fraction=0.5, lambda_l2=0.0, n_estimators_no_valid=50, num_threads=1,
    )
    m = QuantileModel(["x"], alphas=(0.4, 0.5, 0.6), params=p).fit(train, None)
    test, _ = hetero(500, 9)
    raw = m.predict_raw(test)
    assert (np.diff(raw, axis=1) < 0).any(axis=1).mean() > 0.2
    q = m.predict(test)
    assert (np.diff(q, axis=1) >= 0).all()
    np.testing.assert_allclose(np.sort(raw, axis=1), q)


def test_null_inputs_and_null_targets():
    train, _ = hetero(800, 1)
    train = train.with_columns(
        pl.when(pl.col("security_id") % 7 == 0).then(None).otherwise(pl.col("lvol_21")).alias("lvol_21"),
        pl.when(pl.col("security_id") % 11 == 0).then(None).otherwise(pl.col("excess_w_12m")).alias("excess_w_12m"),
    )
    m = QuantileModel(["x"], params=SMALL).fit(train, None)
    q = m.predict(train.drop("excess_w_12m"))
    assert np.isfinite(q).all()


def test_missing_input_column_raises(fitted):
    test, _ = hetero(50, 3)
    with pytest.raises(ValueError, match="lvol_252"):
        fitted.predict(test.drop("lvol_252"))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        QuantileModel(["x"]).predict(hetero(10, 0)[0])


def test_deterministic_for_fixed_seed():
    train, _ = hetero(1500, 1)
    valid, _ = hetero(500, 2)
    test, _ = hetero(300, 3)
    a = QuantileModel(["x"], params=SMALL, seed=7).fit(train, valid).predict(test)
    b = QuantileModel(["x"], params=SMALL, seed=7).fit(train, valid).predict(test)
    np.testing.assert_array_equal(a, b)


def test_pickle_round_trip(fitted):
    test, _ = hetero(300, 3)
    clone = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(clone.predict(test), fitted.predict(test))
    assert clone.best_iterations == fitted.best_iterations and clone.cols == fitted.cols


# --- runner integration ----------------------------------------------------------------------


def test_walk_forward_three_outputs():
    d = dt.date
    rules = ValidationRules(
        first_test_year=2010, test_months=12, purge_months=12, embargo_months=1,
        min_train_months=60, inner_valid_months=24,
    )
    dates = month_end_dates(business_days(d(2000, 1, 1), d(2010, 12, 31))).to_list()
    n_sec = 25
    df, _ = hetero(len(dates) * n_sec, 5, per_date=n_sec)
    i = df["date"].to_numpy()
    ends = [dates[k + 12] if k + 12 < len(dates) else None for k in i]
    frame = df.with_columns(
        pl.Series("date", [dates[k] for k in i]),
        pl.Series("label_end_12m", ends, dtype=pl.Date),
        pl.Series("target", np.zeros(len(i))),
    ).with_columns(
        pl.when(pl.col("label_end_12m").is_null()).then(None).otherwise(pl.col(c)).alias(c)
        for c in ("excess_w_12m", "target")
    )
    folds = walk_forward_folds(dates, rules, d(2010, 12, 31))
    oof = walk_forward(
        lambda: QuantileModel(["x"], params=SMALL), frame, folds, rules,
        target="excess_w_12m", outputs=("q10", "q50", "q90"),
    )
    assert oof.columns == ["date", "security_id", "fold", "q10", "q50", "q90"]
    assert oof.height == 12 * n_sec
    assert ((oof["q10"] <= oof["q50"]) & (oof["q50"] <= oof["q90"])).all()

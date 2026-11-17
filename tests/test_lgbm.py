"""LightGBM model: planted signal, early stopping on daily IC, determinism, lambdarank, SHAP, pickling.

Panel: month-end-like dates 2000-01 .. 2007-12 (day 28) x 40 securities. f1 carries the signal,
f2 is noise, f3 is mostly null. target = per-date centered rank of f1 + noise (or pure noise).
"""

import datetime as dt
import pickle

import numpy as np
import polars as pl
import pytest

from ltforecast.modelconfig import ValidationRules
from ltforecast.models.lgbm import (
    LGBM,
    LGBMParams,
    SeedAverage,
    _ICScorer,
    group_avg_rank,
    mean_date_ic,
    per_date_centered_rank,
    rank_labels,
)
from ltforecast.validation.metrics import ic_by_date
from ltforecast.validation.runner import walk_forward
from ltforecast.validation.splitter import walk_forward_folds

COLS = ["f1", "f2", "f3"]
TINY = LGBMParams(
    num_leaves=7, max_depth=3, min_data_in_leaf=5, learning_rate=0.1, n_estimators=30,
    n_estimators_no_valid=20, early_stopping_rounds=10, num_threads=1,
)
DATES = [dt.date(2000 + m // 12, m % 12 + 1, 28) for m in range(96)]
RULES = ValidationRules(first_test_year=2005, test_months=12, purge_months=12, embargo_months=1,
                        min_train_months=24, inner_valid_months=12)


def panel(n_sec=40, noise=0.3, signal=True, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(DATES) * n_sec
    f1 = rng.uniform(-0.5, 0.5, n)
    f2 = rng.uniform(-0.5, 0.5, n)
    f3 = np.where(rng.uniform(size=n) < 0.7, np.nan, rng.normal(size=n))
    raw = (f1 if signal else 0.0) + noise * rng.normal(size=n)
    df = pl.DataFrame({
        "date": pl.Series(DATES, dtype=pl.Date).gather(np.repeat(np.arange(len(DATES)), n_sec)),
        "security_id": np.tile(np.arange(1, n_sec + 1), len(DATES)),
        "f1": f1, "f2": f2, "f3": f3, "raw": raw,
    }).with_columns(pl.col("f3").fill_nan(None))
    ends = {d: dt.date(d.year + 1, d.month, 28) for d in DATES}
    return df.with_columns(
        ((pl.col("raw").rank("average") - 0.5) / pl.len() - 0.5).over("date").alias("target"),
        pl.col("date").replace_strict(ends, return_dtype=pl.Date).alias("label_end_12m"),
        pl.lit("dev").alias("period"),
    ).drop("raw")


def split(df):
    tr = df.filter(pl.col("date") < DATES[48])
    va = df.filter(pl.col("date").is_between(DATES[48], DATES[71]))
    te = df.filter(pl.col("date") >= DATES[72])
    return tr, va, te


# ranking helpers -----------------------------------------------------------------------------------


def test_group_avg_rank_by_hand():
    x = np.array([3.0, 1.0, 1.0, 5.0, 2.0, 2.0, 2.0])
    g = np.array([0, 0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(group_avg_rank(x, g), [3, 1.5, 1.5, 4, 2, 2, 2])


def test_per_date_centered_rank_matches_transform_convention():
    dates = pl.Series([1, 1, 1, 1, 2, 2])
    out = per_date_centered_rank(np.array([10.0, 30.0, 20.0, 20.0, 5.0, -1.0]), dates)
    # date 1: ranks 1, 4, 2.5, 2.5 -> (r - .5)/4 - .5 ; date 2: ranks 2, 1 -> (r - .5)/2 - .5
    np.testing.assert_allclose(out, [-0.375, 0.375, 0.0, 0.0, 0.25, -0.25])


def test_rank_labels_deciles():
    g = np.zeros(20, dtype=int)
    t = np.arange(20, dtype=float)[::-1]
    lab = rank_labels(t, g)
    np.testing.assert_array_equal(lab, np.repeat(np.arange(10), 2)[::-1])
    assert lab.dtype == np.int32


def test_ic_scorer_matches_metrics_ic_by_date_with_ties_and_constant_dates():
    rng = np.random.default_rng(3)
    dates = np.repeat(np.arange(6), 25)
    pred = rng.integers(0, 5, len(dates)).astype(float)          # many ties
    target = rng.normal(size=len(dates))
    pred[dates == 2] = 1.0                                       # constant prediction -> IC 0
    df = pl.DataFrame({"date": dates, "p": pred, "t": target})
    want = ic_by_date(df, "p", "t", min_n=3)["ic"].mean()
    assert mean_date_ic(pred, target, pl.Series(dates)) == pytest.approx(want, abs=1e-12)
    assert _ICScorer(target, dates)(pred) == pytest.approx(want, abs=1e-12)


# the model ------------------------------------------------------------------------------------------


def test_planted_signal_is_learned_out_of_fold():
    df = panel()
    folds = walk_forward_folds(DATES, RULES, DATES[-1])
    assert [f.name for f in folds] == ["2005", "2006", "2007"]
    oof = walk_forward(lambda: LGBM(COLS, TINY), df, folds, RULES)
    scored = oof.join(df.select("date", "security_id", "target"), on=["date", "security_id"])
    assert ic_by_date(scored, "pred", "target")["ic"].mean() > 0.2


def test_deterministic_for_fixed_seed_and_seeds_differ():
    tr, va, te = split(panel())
    a = LGBM(COLS, TINY, seed=7).fit(tr, va).predict(te)
    b = LGBM(COLS, TINY, seed=7).fit(tr.sample(fraction=1.0, shuffle=True, seed=1), va).predict(te)
    c = LGBM(COLS, TINY, seed=8).fit(tr, va).predict(te)
    np.testing.assert_array_equal(a, b)        # same seed, rows handed in another order
    assert not np.array_equal(a, c)


def test_early_stopping_stops_before_cap_on_noise():
    tr, va, _ = split(panel(signal=False))
    p = LGBMParams(**{**TINY.__dict__, "n_estimators": 300, "early_stopping_rounds": 10})
    m = LGBM(COLS, p).fit(tr, va)
    assert m.best_iteration < 300
    assert m.booster.num_trees() == m.best_iteration   # refit on train+valid for best_iteration rounds


def test_refit_false_keeps_early_stopped_booster_and_refit_true_differs():
    tr, va, te = split(panel())
    keep = LGBM(COLS, LGBMParams(**{**TINY.__dict__, "refit": False})).fit(tr, va)
    refit = LGBM(COLS, TINY).fit(tr, va)
    assert keep.best_iteration == refit.best_iteration
    assert keep.booster.num_trees() == keep.best_iteration
    assert keep.valid_ic is not None and keep.valid_ic > 0.2
    assert not np.array_equal(keep.predict(te), refit.predict(te))


def test_no_valid_uses_fixed_rounds():
    tr, _, te = split(panel())
    m = LGBM(COLS, TINY).fit(tr, None)
    assert m.best_iteration == TINY.n_estimators_no_valid == m.booster.num_trees()
    assert m.valid_ic is None and m.predict(te).shape == (te.height,)


@pytest.mark.parametrize("objective", ["regression", "lambdarank"])
def test_both_objectives_run_and_rank_well(objective):
    tr, va, te = split(panel())
    m = LGBM(COLS, LGBMParams(**{**TINY.__dict__, "objective": objective})).fit(tr, va)
    pred = m.predict(te)
    assert pred.shape == (te.height,) and np.isfinite(pred).all()
    assert mean_date_ic(pred, te["target"].to_numpy(), te["date"]) > 0.4


def test_lambdarank_refuses_oversized_query():
    big = pl.DataFrame({"date": [DATES[0]] * 10_001, "security_id": np.arange(10_001), "f1": 0.0,
                        "f2": 0.0, "f3": 0.0, "target": np.linspace(-0.5, 0.5, 10_001)})
    with pytest.raises(ValueError, match="10000"):
        LGBM(COLS, LGBMParams(**{**TINY.__dict__, "objective": "lambdarank"})).fit(big, None)


@pytest.mark.parametrize("objective", ["regression", "lambdarank"])
def test_shap_additivity(objective):
    tr, va, te = split(panel())
    m = LGBM(COLS, LGBMParams(**{**TINY.__dict__, "objective": objective})).fit(tr, va)
    c = m.contributions(te)
    assert c.shape == (te.height, len(COLS))
    np.testing.assert_allclose(c.sum(axis=1) + m.expected_value, m.predict(te), atol=1e-9)
    assert np.abs(c[:, 0]).mean() > np.abs(c[:, 1]).mean()     # the signal feature dominates


def test_pickle_round_trip():
    tr, va, te = split(panel())
    m = LGBM(COLS, TINY).fit(tr, va)
    back = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(back.predict(te), m.predict(te))
    assert back.best_iteration == m.best_iteration and back.cols == COLS


def test_predict_ignores_label_columns_and_needs_inputs():
    tr, va, te = split(panel())
    m = LGBM(COLS, TINY).fit(tr, va)
    shown = te.drop("target", "label_end_12m", "period")
    np.testing.assert_array_equal(m.predict(shown), m.predict(te))
    with pytest.raises(KeyError, match="f3"):
        m.predict(shown.drop("f3"))
    with pytest.raises(ValueError, match="label"):
        LGBM(["f1", "excess_12m"])
    with pytest.raises(ValueError, match="objective"):
        LGBMParams(objective="binary")


def test_seed_average_is_mean_of_member_date_ranks():
    tr, va, te = split(panel())
    sa = SeedAverage(COLS, TINY, seeds=(0, 1, 2)).fit(tr, va)
    want = np.mean([per_date_centered_rank(m.predict(te), te["date"]) for m in sa.members], axis=0)
    np.testing.assert_allclose(sa.predict(te), want)
    assert len({m.seed for m in sa.members}) == 3 and len(sa.best_iteration) == 3
    assert np.abs(sa.predict(te)).max() < 0.5
    back = pickle.loads(pickle.dumps(sa))
    np.testing.assert_array_equal(back.predict(te), sa.predict(te))
    c = sa.contributions(te)
    raw_mean = np.mean([m.predict(te) for m in sa.members], axis=0)
    np.testing.assert_allclose(c.sum(axis=1) + sa.expected_value, raw_mean, atol=1e-9)
    assert sa.with_params(refit=False).members[0].params.refit is False


def test_refit_from_equals_fresh_refit_true_fit_and_needs_valid():
    tr, va, te = split(panel())
    keep = LGBM(COLS, LGBMParams(**{**TINY.__dict__, "refit": False})).fit(tr, va)
    ref = LGBM(COLS, TINY).fit(tr, va)
    out = keep.refit_from(tr, va)
    assert out is not keep and out.params.refit is True
    assert (out.best_iteration, out.valid_ic) == (ref.best_iteration, ref.valid_ic)
    assert out.booster.num_trees() == ref.booster.num_trees()
    np.testing.assert_array_equal(out.predict(te), ref.predict(te))
    assert out.expected_value == pytest.approx(ref.expected_value)
    np.testing.assert_array_equal(keep.predict(te), LGBM(COLS, LGBMParams(**{**TINY.__dict__, "refit": False}))
                                  .fit(tr, va).predict(te))  # source model untouched
    with pytest.raises(RuntimeError):
        LGBM(COLS, TINY).fit(tr, None).refit_from(tr, va)

"""The runner's refit frame `full` (train + the purged gap + valid) reaches only the refit step.

For every model: selection (early stopping, ridge alpha, ensemble weights, valid R2) is identical with
and without `full`; `full = train + valid` reproduces the old refit exactly; a larger `full` changes the
final model and equals a model fit directly on it.
"""

import dataclasses
import functools

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge as SkRidge
from test_ensemble import stub_frame
from test_hitprob import _clf_frame
from test_lgbm import DATES, TINY, panel
from test_quantile import SMALL, hetero
from test_vol import synthetic_panel

from ltforecast.hitprob import HitClassifier, HitClassifierParams
from ltforecast.models.baselines import Ridge
from ltforecast.models.ensemble import Ensemble, TrainOnlyRidge
from ltforecast.models.lgbm import LGBM, SeedAverage
from ltforecast.models.quantile import QuantileModel
from ltforecast.models.vol import VolModel

LCOLS = ["f1", "f2", "f3"]


def lgbm_frames():
    """inner train 2000-01..2002-12, a 12-month gap 2003, inner valid 2004, test 2005."""
    df = panel()
    tr = df.filter(pl.col("date") < DATES[36])
    gap = df.filter(pl.col("date").is_between(DATES[36], DATES[47]))
    va = df.filter(pl.col("date").is_between(DATES[48], DATES[59]))
    te = df.filter(pl.col("date").is_between(DATES[60], DATES[71])).drop("target")
    return tr, gap, va, te


def test_lgbm_refit_uses_full_and_early_stopping_does_not():
    tr, gap, va, te = lgbm_frames()
    old = LGBM(LCOLS, TINY, seed=3).fit(tr, va)
    same = LGBM(LCOLS, TINY, seed=3).fit(tr, va, full=pl.concat([tr, va]))
    new = LGBM(LCOLS, TINY, seed=3).fit(tr, va, full=pl.concat([tr, gap, va]))
    assert new.best_iteration == old.best_iteration and new.valid_ic == old.valid_ic
    np.testing.assert_array_equal(same.predict(te), old.predict(te))
    assert not np.array_equal(new.predict(te), old.predict(te))
    # refit_from with the same `full` is the same model as fitting with it
    sel = LGBM(LCOLS, dataclasses.replace(TINY, refit=False), seed=3).fit(tr, va)
    np.testing.assert_array_equal(sel.refit_from(tr, va, full=pl.concat([tr, gap, va])).predict(te), new.predict(te))
    # SeedAverage passes it to every member
    sa = SeedAverage(LCOLS, TINY, seeds=(3,)).fit(tr, va, full=pl.concat([tr, gap, va]))
    np.testing.assert_array_equal(sa.members[0].predict(te), new.predict(te))


def test_ridge_alpha_is_chosen_without_full_and_the_final_fit_uses_it():
    tr, gap, va, te = lgbm_frames()
    alphas = (0.1, 10.0, 1000.0)
    old = Ridge(LCOLS, alphas).fit(tr, va)
    full = pl.concat([tr, gap, va])
    new = Ridge(LCOLS, alphas).fit(tr, va, full=full)
    assert new.alpha_ == old.alpha_ and new.valid_ic_ == old.valid_ic_
    x = full.select(pl.col(LCOLS).fill_null(0.0).fill_nan(0.0)).to_numpy()
    ref = SkRidge(alpha=new.alpha_).fit(x, full["target"].to_numpy())
    np.testing.assert_allclose(new.coef_, ref.coef_, rtol=1e-10)
    np.testing.assert_allclose(Ridge(LCOLS, alphas).fit(tr, va, full=pl.concat([tr, va])).coef_, old.coef_)
    # the selection-step ridge stays train-only so its valid predictions are out of sample
    a = TrainOnlyRidge(LCOLS, alphas).fit(tr, va)
    b = TrainOnlyRidge(LCOLS, alphas).fit(tr, va, full=full)
    np.testing.assert_array_equal(a.coef_, b.coef_)


class Member:
    """Predicts one column; remembers what its fit received."""

    def __init__(self, col, made, refit):
        self.col, self.refit, self.name = col, refit, col
        made.append(self)

    def fit(self, train, valid, *, full=None):
        self.full = full
        return self

    def predict(self, df):
        return df[self.col].to_numpy()


def test_ensemble_weights_ignore_full_and_only_final_members_get_it():
    train, valid = stub_frame(8, 40, 0, 2000), stub_frame(4, 40, 1, 2002)
    gap = stub_frame(2, 40, 2, 2001)
    full = pl.concat([train, gap, valid])
    runs = {}
    for name, f in (("old", None), ("new", full)):
        made: list = []
        factories = {c: functools.partial(Member, c, made) for c in ("good", "bad1", "bad2")}
        ens = Ensemble(factories, step=0.1, shrinkage=0.5).fit(train, valid, full=f)
        runs[name] = (ens, made)
    (old, _), (new, made) = runs["old"], runs["new"]
    assert new.weights_ == old.weights_ and new.valid_ic_ == old.valid_ic_
    assert all(m.full is None for m in made if not m.refit)
    assert [m.col for m in made if m.refit] == ["good", "bad1", "bad2"]
    assert all(m.full is full for m in made if m.refit)


def test_quantile_rounds_come_from_valid_and_the_refit_uses_full():
    tr, _ = hetero(2000, 1)
    gap, _ = hetero(1000, 3)
    va, _ = hetero(800, 2)
    te, _ = hetero(300, 4)
    old = QuantileModel(["x"], params=SMALL).fit(tr, va)
    new = QuantileModel(["x"], params=SMALL).fit(tr, va, full=pl.concat([tr, gap, va]))
    same = QuantileModel(["x"], params=SMALL).fit(tr, va, full=pl.concat([tr, va]))
    assert new.best_iterations == old.best_iterations and new.valid_pinball == old.valid_pinball
    np.testing.assert_array_equal(same.predict(te), old.predict(te))
    assert not np.array_equal(new.predict(te), old.predict(te))


def test_vol_valid_r2_is_train_only_and_the_refit_uses_full():
    df = synthetic_panel(n_sec=60, n_dates=30)
    dates = df["date"].unique().sort()
    tr = df.filter(pl.col("date") < dates[15])
    gap = df.filter(pl.col("date").is_between(dates[15], dates[19]))
    va = df.filter(pl.col("date") >= dates[20])
    full = pl.concat([tr, gap, va])
    old = VolModel().fit(tr, va)
    new = VolModel().fit(tr, va, full=full)
    assert new.valid_r2_ == old.valid_r2_
    assert new.coefficients() == pytest.approx(VolModel().fit(full, None).coefficients(), rel=1e-12)
    assert new.coefficients() != old.coefficients()


def test_hit_classifier_rounds_come_from_valid_and_the_refit_uses_full():
    # chronological, as the runner hands it over (the classifier sorts its fit rows by date, security_id)
    df = _clf_frame(900).sort("date", "security_id")
    params = HitClassifierParams(min_data_in_leaf=5, n_estimators=30, n_estimators_no_valid=30,
                                 early_stopping_rounds=10, num_threads=1)
    tr, gap, va, te = df[:400], df[400:600], df[600:700], df[700:].drop("outcome")
    old = HitClassifier(["f1", "f2"], params, seed=1).fit(tr, va)
    new = HitClassifier(["f1", "f2"], params, seed=1).fit(tr, va, full=pl.concat([tr, gap, va]))
    same = HitClassifier(["f1", "f2"], params, seed=1).fit(tr, va, full=pl.concat([tr, va]))
    assert new.best_iteration == old.best_iteration
    np.testing.assert_array_equal(same.predict(te), old.predict(te))
    assert not np.array_equal(new.predict(te), old.predict(te))

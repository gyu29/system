"""Ensemble (plan 6.1): simplex weights on inner valid, shrinkage, refit on train + valid, rank blend."""

import datetime as dt
import functools
import pickle

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge as SkRidge

from ltforecast.features.catalog import load_catalog
from ltforecast.models.baselines import FactorComposite, Ridge
from ltforecast.models.ensemble import Ensemble, TrainOnlyRidge, default_factories, simplex_grid
from ltforecast.models.lgbm import LGBMParams, SeedAverage

CATALOG = load_catalog()
TINY = LGBMParams(num_leaves=7, max_depth=3, min_data_in_leaf=5, learning_rate=0.1, n_estimators=30,
                  n_estimators_no_valid=20, early_stopping_rounds=10, num_threads=1)


def dates(n, start=2000):
    return [dt.date(start + m // 12, m % 12 + 1, 28) for m in range(n)]


def stub_frame(n_dates=8, n_sec=40, seed=0, start=2000):
    """target random; `good` = target + small noise; `bad1`, `bad2` = pure noise."""
    rng = np.random.default_rng(seed)
    n = n_dates * n_sec
    target = rng.normal(size=n)
    return pl.DataFrame({
        "date": pl.Series(dates(n_dates, start), dtype=pl.Date).gather(np.repeat(np.arange(n_dates), n_sec)),
        "security_id": np.tile(np.arange(1, n_sec + 1), n_dates),
        "good": target + 0.3 * rng.normal(size=n),
        "bad1": rng.normal(size=n),
        "bad2": rng.normal(size=n),
        "target": target,
        "excess_12m": target,
    })


class Stub:
    """Predicts one column; records every fit (spy) and refuses to see labels at predict."""

    def __init__(self, col, log, refit):
        self.col, self.log, self.refit, self.name = col, log, refit, f"stub_{col}"

    def fit(self, train, valid, *, full=None):
        self.log.append((self.col, self.refit, train.height, None if valid is None else valid.height))
        self.full = full
        return self

    def predict(self, df):
        assert "target" not in df.columns and "excess_12m" not in df.columns
        return df[self.col].to_numpy()


def stub_factories(log, cols=("good", "bad1", "bad2")):
    return {c: functools.partial(Stub, c, log) for c in cols}


# simplex grid ------------------------------------------------------------------------------------------


def test_simplex_grid():
    g = simplex_grid(3, 0.1)
    assert g.shape == (66, 3)
    np.testing.assert_allclose(g.sum(axis=1), 1.0)
    assert len({tuple(r) for r in np.round(g, 9)}) == 66
    assert ((g >= 0) & (g <= 1)).all()
    np.testing.assert_allclose(simplex_grid(1, 0.25), [[1.0]])
    assert simplex_grid(2, 0.5).tolist() == [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
    with pytest.raises(ValueError):
        simplex_grid(3, 0.3)
    with pytest.raises(ValueError):
        Ensemble({"a": None}, shrinkage=1.5)


# weights -----------------------------------------------------------------------------------------------


def test_weights_favour_the_strong_member_then_shrink_exactly():
    log = []
    train, valid = stub_frame(seed=1), stub_frame(seed=2, start=2002)
    ens = Ensemble(stub_factories(log), step=0.1, shrinkage=0.5).fit(train, valid)
    wb = ens.w_best_
    assert wb["good"] > wb["bad1"] and wb["good"] > wb["bad2"] and wb["good"] >= 0.5
    assert sum(wb.values()) == pytest.approx(1.0)
    for k in ens.names:
        assert ens.weights_[k] == (1 - 0.5) * wb[k] + 0.5 * (1 / 3)
    assert sum(ens.weights_.values()) == pytest.approx(1.0)
    assert ens.member_valid_ic_["good"] > 0.8 > abs(ens.member_valid_ic_["bad1"])
    assert ens.valid_ic_ >= ens.member_valid_ic_["good"] - 1e-12


@pytest.mark.parametrize("s", [0.0, 0.25, 1.0])
def test_shrinkage_extremes(s):
    train, valid = stub_frame(seed=1), stub_frame(seed=2, start=2002)
    ens = Ensemble(stub_factories([]), shrinkage=s).fit(train, valid)
    for k in ens.names:
        assert ens.weights_[k] == (1 - s) * ens.w_best_[k] + s * (1 / 3)
    if s == 1.0:
        assert all(w == pytest.approx(1 / 3) for w in ens.weights_.values())


def test_no_valid_gives_equal_weights_and_no_selection_fit():
    log = []
    ens = Ensemble(stub_factories(log)).fit(stub_frame(), None)
    assert ens.weights_ == pytest.approx({"good": 1 / 3, "bad1": 1 / 3, "bad2": 1 / 3})
    assert ens.w_best_ is None and ens.valid_ic_ is None
    assert [(c, r, v) for c, r, _, v in log] == [("good", True, None), ("bad1", True, None), ("bad2", True, None)]
    empty = stub_frame().clear()
    assert Ensemble(stub_factories([])).fit(stub_frame(), empty).weights_["bad2"] == pytest.approx(1 / 3)


def test_members_selected_with_refit_false_then_refit_on_train_plus_valid():
    log = []
    train, valid = stub_frame(seed=1), stub_frame(seed=2, start=2002)
    ens = Ensemble(stub_factories(log)).fit(train, valid)
    assert [(c, r) for c, r, _, _ in log] == [
        ("good", False), ("bad1", False), ("bad2", False), ("good", True), ("bad1", True), ("bad2", True)
    ]
    assert all(t == train.height and v == valid.height for _, _, t, v in log)
    assert all(m.refit for m in ens.members_.values())
    assert list(ens.members_) == ["good", "bad1", "bad2"]


def test_ridge_member_selection_is_out_of_sample_and_final_is_refit():
    """refit=False ridge is fit on train only; the final ridge is B3 refit on train + valid."""
    rng = np.random.default_rng(3)
    cols = ["a", "b"]

    def frame(n_dates, start):
        n = n_dates * 30
        df = pl.DataFrame({
            "date": pl.Series(dates(n_dates, start), dtype=pl.Date).gather(np.repeat(np.arange(n_dates), 30)),
            "security_id": np.tile(np.arange(30), n_dates), "a": rng.uniform(-0.5, 0.5, n),
            "b": rng.uniform(-0.5, 0.5, n),
        })
        return df.with_columns((pl.col("a") + 0.5 * pl.Series(rng.normal(size=n))).alias("target"))

    train, valid = frame(6, 2000), frame(4, 2001)
    sel = TrainOnlyRidge(cols, (1.0, 100.0)).fit(train, valid)
    ref = SkRidge(alpha=sel.alpha_).fit(train.select(cols).to_numpy(), train["target"].to_numpy())
    np.testing.assert_allclose(sel.coef_, ref.coef_)
    fac = {"r": functools.partial(lambda refit: (Ridge if refit else TrainOnlyRidge)(cols, (1.0, 100.0)))}
    ens = Ensemble(fac).fit(train, valid)
    both = pl.concat([train, valid])
    full = SkRidge(alpha=ens.members_["r"].alpha_).fit(both.select(cols).to_numpy(), both["target"].to_numpy())
    assert type(ens.members_["r"]) is Ridge
    np.testing.assert_allclose(ens.members_["r"].coef_, full.coef_)
    assert ens.weights_ == {"r": 1.0}


# predict -----------------------------------------------------------------------------------------------


def test_predict_is_weighted_mean_of_per_date_centered_ranks():
    d1, d2 = dt.date(2020, 1, 31), dt.date(2020, 2, 28)
    df = pl.DataFrame({
        "date": [d1] * 4 + [d2] * 4,
        "security_id": [1, 2, 3, 4] * 2,
        "x": [10.0, 30.0, 20.0, 40.0, 1.0, 1.0, 5.0, 3.0],
        "y": [4.0, 3.0, 2.0, 1.0, 0.0, 9.0, 8.0, 7.0],
    })
    ens = Ensemble(stub_factories([], cols=("x", "y"))).fit(df.with_columns(target=pl.lit(0.0)), None)
    ens.weights_ = {"x": 0.7, "y": 0.3}
    # (rank - 0.5) / 4 - 0.5 per date; ties share the average rank
    rx = [-0.375, 0.125, -0.125, 0.375, -0.25, -0.25, 0.375, 0.125]
    ry = [0.375, 0.125, -0.125, -0.375, -0.375, 0.375, 0.125, -0.125]
    expect = [0.7 * a + 0.3 * b for a, b in zip(rx, ry)]
    np.testing.assert_allclose(ens.predict(df), expect)
    # row order does not matter
    perm = [5, 0, 7, 2, 1, 6, 3, 4]
    np.testing.assert_allclose(ens.predict(df[perm]), np.asarray(expect)[perm])
    with pytest.raises(KeyError):
        ens.predict(df.drop("date"))
    assert ens.predict(df.clear()).shape == (0,)


# default members, real models --------------------------------------------------------------------------


def catalog_frame(n_dates, n_sec, start, seed):
    rng = np.random.default_rng(seed)
    n = n_dates * n_sec
    cols = {c: rng.uniform(-0.5, 0.5, n) for c in CATALOG.model_columns()}
    for c in CATALOG.flag_names():
        cols[c] = np.ones(n)
    df = pl.DataFrame({
        "date": pl.Series(dates(n_dates, start), dtype=pl.Date).gather(np.repeat(np.arange(n_dates), n_sec)),
        "security_id": np.tile(np.arange(1, n_sec + 1), n_dates), **cols,
    })
    raw = df["mom_12_1"].to_numpy() + df["gross_profitability"].to_numpy() + 0.5 * rng.normal(size=n)
    return df.with_columns(pl.Series("raw", raw)).with_columns(
        ((pl.col("raw").rank() - 0.5) / pl.len() - 0.5).over("date").alias("target")
    ).drop("raw")


def test_default_factories_fit_predict_and_pickle():
    fac = default_factories(CATALOG, lgbm_params=TINY, seeds=(0, 1))
    assert list(fac) == ["lgbm", "ridge", "composite"]
    assert isinstance(fac["lgbm"](False), SeedAverage) and fac["lgbm"](False).params.refit is False
    assert fac["lgbm"](True).params.refit is True
    assert type(fac["ridge"](False)) is TrainOnlyRidge and type(fac["ridge"](True)) is Ridge
    assert isinstance(fac["composite"](True), FactorComposite)
    train, valid, test = catalog_frame(12, 30, 2000, 1), catalog_frame(6, 30, 2001, 2), catalog_frame(2, 30, 2003, 3)
    ens = Ensemble(fac).fit(train, valid)
    assert sum(ens.weights_.values()) == pytest.approx(1.0)
    assert all(w >= 0.5 / 3 - 1e-12 for w in ens.weights_.values())
    pred = ens.predict(test)
    assert pred.shape == (test.height,) and np.isfinite(pred).all() and (np.abs(pred) < 0.5).all()
    again = pickle.loads(pickle.dumps(ens))
    np.testing.assert_array_equal(again.predict(test), pred)
    # deterministic
    np.testing.assert_array_equal(Ensemble(fac).fit(train, valid).predict(test), pred)


# one early-stopping pass per seed ----------------------------------------------------------------------


def test_lgbm_member_runs_early_stopping_once_per_seed_and_matches_fresh_refit(monkeypatch):
    import lightgbm as lgb

    import ltforecast.models.lgbm as lgbm_mod

    calls = []
    real_train = lgb.train

    def spy(params, dtrain, *a, **kw):
        calls.append(("es" if kw.get("valid_sets") else "plain", params["seed"]))
        return real_train(params, dtrain, *a, **kw)

    cols = ["good", "bad1", "bad2"]
    tr, va = stub_frame(8, 40, 1), stub_frame(4, 40, 2, start=2001)
    seeds = (0, 1)
    fac = {"lgbm": functools.partial(lambda refit: SeedAverage(
        cols, LGBMParams(**{**TINY.__dict__, "refit": refit}), seeds, "lgbm"))}
    # reference: the old path, a fresh refit=True member (its own ES pass + refit)
    ref = fac["lgbm"](True).fit(tr, va)

    monkeypatch.setattr(lgbm_mod.lgb, "train", spy)
    e = Ensemble(fac, step=1.0).fit(tr, va)
    # 1 ES pass + 1 refit per seed (previously 2 ES passes + 1 refit)
    assert sorted(calls) == sorted([("es", s) for s in seeds] + [("plain", s) for s in seeds])
    final = e.members_["lgbm"]
    assert final.params.refit is True
    assert final.best_iteration == ref.best_iteration
    assert [m.valid_ic for m in final.members] == [m.valid_ic for m in ref.members]
    shown = va.drop("target", "excess_12m")
    np.testing.assert_array_equal(final.predict(shown), ref.predict(shown))
    for a, b in zip(final.members, ref.members):
        np.testing.assert_array_equal(a.predict(shown), b.predict(shown))
        assert a.expected_value == pytest.approx(b.expected_value)

"""Baselines B0-B3 (plan section 5) on tiny hand-built frames."""

import datetime as dt
import math
import pickle

import numpy as np
import polars as pl
import pytest
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge as SkRidge

from ltforecast.features.catalog import load_catalog
from ltforecast.models.baselines import (
    BaselineParams,
    FactorComposite,
    NoSkill,
    Ridge,
    SingleFactor,
    baseline_factories,
    composite_themes,
    ic_or_zero,
)
from ltforecast.modelconfig import ValidationRules
from ltforecast.validation.runner import walk_forward
from ltforecast.validation.splitter import walk_forward_folds

d = dt.date
CATALOG = load_catalog()
DAY = d(2020, 1, 31)


def ranked_frame(values: dict[str, list], n: int) -> pl.DataFrame:
    """One date, n stocks, every linear column of the catalog (null unless given), plus label columns."""
    cols = {c: pl.Series(c, values.get(c, [None] * n), dtype=pl.Float64) for c in CATALOG.linear_columns()}
    return pl.DataFrame({
        "date": [DAY] * n, "security_id": pl.Series(range(1, n + 1), dtype=pl.Int64), **cols,
        "target": pl.Series([0.1 * i for i in range(n)], dtype=pl.Float64),
    })


# b0 ---------------------------------------------------------------------------------------------


def test_no_skill_predicts_zero_and_has_zero_ic():
    df = ranked_frame({"mom_12_1": [0.1, -0.2, 0.3]}, 3)
    m = NoSkill().fit(df, df)
    pred = m.predict(df.drop("target"))
    assert m.name == "b0_no_skill"
    assert pred.dtype == np.float64 and pred.tolist() == [0.0, 0.0, 0.0]
    # a constant prediction has IC 0, not NaN
    assert ic_or_zero(pred, df["target"], df["date"], min_n=2) == 0.0


# b1 ---------------------------------------------------------------------------------------------


def test_single_factor_sign_flip_and_missing_to_zero():
    df = ranked_frame({"accruals": [0.25, None, -0.4, float("nan")]}, 4)
    plus = SingleFactor("accruals", 1).fit(df, None).predict(df)
    minus = SingleFactor("accruals", -1).predict(df)
    assert plus.tolist() == [0.25, 0.0, -0.4, 0.0]
    assert minus.tolist() == [-0.25, 0.0, 0.4, 0.0]
    assert SingleFactor("accruals", -1).name == "b1_accruals"


def test_single_factor_rejects_bad_sign_and_absent_column():
    with pytest.raises(ValueError, match="sign"):
        SingleFactor("mom_12_1", 0)
    with pytest.raises(ValueError, match="lacks columns"):
        SingleFactor("mom_12_1", 1).predict(pl.DataFrame({"x": [1.0]}))


# b2 ---------------------------------------------------------------------------------------------


def test_composite_themes_from_catalog():
    themes = composite_themes(CATALOG)
    assert list(themes) == ["value", "quality", "momentum", "low_vol", "low_investment"]
    assert themes["low_vol"] == (("vol_1y", -1), ("idio_vol", -1), ("beta_60m", -1))
    assert ("accruals", -1) in themes["quality"] and ("book_to_price", 1) in themes["value"]
    assert [c for c, _ in themes["momentum"]] == ["mom_12_1", "mom_6_1", "ind_mom_6m", "high_52w"]
    assert [c for c, _ in themes["low_investment"]] == ["asset_growth", "net_issuance", "capex_growth"]
    # max_ret_1m (risk) and the sector-relative variants are not in the composite
    used = {c for members in themes.values() for c, _ in members}
    assert "max_ret_1m" not in used and not any(c.endswith("_sr") for c in used)


def test_factor_composite_hand_computed():
    df = ranked_frame({
        # stock 1: value (0.4 + 0.2) / 2 = 0.3; quality (0.1 - 0.3) / 2 = -0.1; no momentum;
        # low-vol -0.2; low-investment +0.4 -> composite (0.3 - 0.1 - 0.2 + 0.4) / 4 = 0.1
        "book_to_price": [0.4, None, None],
        "earnings_yield": [0.2, None, float("nan")],  # stock 3: NaN counts as missing
        "gross_profitability": [0.1, None, None],
        "accruals": [0.3, None, None],
        "vol_1y": [0.2, None, None],
        "asset_growth": [-0.4, None, None],
        # stock 3: momentum (0.3 - 0.1) / 2 = 0.1; low-vol (0.2 - 0.1) / 2 = 0.05 -> (0.1 + 0.05) / 2
        "mom_12_1": [None, None, 0.3],
        "mom_6_1": [None, None, -0.1],
        "idio_vol": [None, None, -0.2],
        "beta_60m": [None, None, 0.1],
        # ignored: not in any theme
        "max_ret_1m": [0.45, 0.45, 0.45],
        "book_to_price_sr": [0.49, 0.49, 0.49],
        "sue": [0.3, 0.3, 0.3],
    }, 3)
    m = FactorComposite(CATALOG).fit(df, df)
    scores = m.theme_scores(df)
    first = scores.row(0, named=True)
    assert first["momentum"] is None
    assert [first[t] for t in ("value", "quality", "low_vol", "low_investment")] == pytest.approx([0.3, -0.1, -0.2, 0.4])
    assert all(v is None for v in scores.row(1))
    assert scores["momentum"].to_list() == [None, None, pytest.approx(0.1)]
    assert scores["low_vol"].to_list()[2] == pytest.approx(0.05)
    assert scores["value"].to_list()[2] is None
    pred = m.predict(df.drop("target"))
    assert pred.tolist() == pytest.approx([0.1, 0.0, 0.075])  # stock 2 has nothing -> 0


def test_factor_composite_missing_whole_theme_averages_the_other_four():
    # every theme has one feature at +0.2 (sign-adjusted) except momentum, which is all missing
    df = ranked_frame({"book_to_price": [0.2], "gross_profitability": [0.2], "vol_1y": [-0.2],
                       "asset_growth": [-0.2]}, 1)
    assert FactorComposite(CATALOG).predict(df).tolist() == pytest.approx([0.2])
    with_mom = df.with_columns(pl.lit(-0.3).alias("mom_12_1"))
    assert FactorComposite(CATALOG).predict(with_mom).tolist() == pytest.approx([(4 * 0.2 - 0.3) / 5])


def test_factor_composite_requires_its_columns():
    with pytest.raises(ValueError, match="lacks columns"):
        FactorComposite(CATALOG).predict(ranked_frame({}, 2).drop("vol_1y"))


# b3 ---------------------------------------------------------------------------------------------


def planted(n_dates: int, n: int, start_year: int, noise: float, seed: int) -> pl.DataFrame:
    """target = 2a - b + noise; b = a + small noise (collinear), c is pure noise."""
    rng = np.random.default_rng(seed)
    days = [d(start_year + i // 12, i % 12 + 1, 28) for i in range(n_dates)]
    a = rng.uniform(-0.5, 0.5, (n_dates, n))
    b = a + rng.normal(0, 0.05, (n_dates, n))
    c = rng.uniform(-0.5, 0.5, (n_dates, n))
    y = 2 * a - b + noise * rng.normal(size=(n_dates, n))
    return pl.DataFrame({
        "date": pl.Series([day for day in days for _ in range(n)]),
        "security_id": pl.Series(np.tile(np.arange(n), n_dates), dtype=pl.Int64),
        "a": a.ravel(), "b": b.ravel(), "c": c.ravel(), "target": y.ravel(),
    })


def spearman_mean(pred: np.ndarray, df: pl.DataFrame) -> float:
    """Independent per-date Spearman IC (scipy), averaged over dates."""
    out = []
    for day in df["date"].unique().sort():
        mask = (df["date"] == day).to_numpy()
        out.append(spearmanr(pred[mask], df["target"].to_numpy()[mask]).statistic)
    return float(np.mean(out))


ABC = ["a", "b", "c"]


@pytest.mark.parametrize("noise", [0.01, 0.3, 1.0])
def test_ridge_picks_best_inner_valid_ic_and_refits_on_train_plus_valid(noise):
    train, valid = planted(12, 40, 2000, noise, 1), planted(6, 40, 2002, noise, 2)
    alphas = [1e-3, 1.0, 100.0, 1e5]
    m = Ridge(ABC, alphas).fit(train, valid)

    xt, yt = train.select(ABC).to_numpy(), train["target"].to_numpy()
    ics = {a: spearman_mean(SkRidge(alpha=a).fit(xt, yt).predict(valid.select(ABC).to_numpy()), valid)
           for a in alphas}
    assert m.valid_ic_ == pytest.approx(ics)
    assert m.alpha_ == max(ics, key=ics.get)

    both = pl.concat([train, valid])
    fresh = SkRidge(alpha=m.alpha_).fit(both.select(ABC).to_numpy(), both["target"].to_numpy())
    np.testing.assert_allclose(m.coef_, fresh.coef_, rtol=1e-12, atol=1e-12)
    assert m.intercept_ == pytest.approx(float(fresh.intercept_), abs=1e-12)
    test = planted(3, 40, 2004, noise, 3)
    np.testing.assert_allclose(m.predict(test.drop("target")), fresh.predict(test.select(ABC).to_numpy()),
                               atol=1e-12)


def test_ridge_low_noise_prefers_least_shrinkage():
    # with almost no noise OLS recovers 2a - b exactly; heavy shrinkage pulls toward a + b and ranks worse
    m = Ridge(ABC, [1e-3, 1e5]).fit(planted(12, 40, 2000, 0.01, 1), planted(6, 40, 2002, 0.01, 2))
    assert m.alpha_ == 1e-3
    assert m.valid_ic_[1e-3] > 0.99 > m.valid_ic_[1e5]
    np.testing.assert_allclose(m.coef_, [2.0, -1.0, 0.0], atol=0.05)


def test_ridge_nulls_are_zero_and_ties_go_to_more_shrinkage():
    train, valid = planted(12, 20, 2000, 0.3, 4), planted(6, 20, 2002, 0.3, 5)
    holed = train.with_columns(pl.when(pl.col("security_id") < 5).then(None).otherwise(pl.col("c")).alias("c"))
    m = Ridge(ABC, [10.0]).fit(holed, valid)
    filled = holed.with_columns(pl.col("c").fill_null(0.0))
    both = pl.concat([filled, valid])
    fresh = SkRidge(alpha=10.0).fit(both.select(ABC).to_numpy(), both["target"].to_numpy())
    np.testing.assert_allclose(m.coef_, fresh.coef_, atol=1e-12)
    # a feature that is all null scores every alpha the same IC -> the largest alpha wins
    flat = Ridge(["c"], [1.0, 100.0]).fit(holed.with_columns(pl.lit(None, pl.Float64).alias("c")),
                                        valid.with_columns(pl.lit(None, pl.Float64).alias("c")))
    assert flat.valid_ic_ == {1.0: 0.0, 100.0: 0.0} and flat.alpha_ == 100.0


def test_ridge_errors():
    df = planted(6, 20, 2000, 0.3, 6)
    with pytest.raises(ValueError, match="validation set"):
        Ridge(ABC, [1.0, 10.0]).fit(df, None)
    with pytest.raises(ValueError, match="positive"):
        Ridge(ABC, [0.0])
    with pytest.raises(RuntimeError, match="not fitted"):
        Ridge(ABC).predict(df)
    assert Ridge(ABC, [5.0]).fit(df, None).alpha_ == 5.0  # one alpha needs no validation set


# helpers, factories, runner ---------------------------------------------------------------------


def test_ic_or_zero():
    days = [d(2020, 1, 31)] * 4 + [d(2020, 2, 28)] * 4
    pred = [1, 2, 3, 4, 5, 5, 5, 5]  # constant on the second date -> IC 0 there
    target = [1, 2, 4, 3, 1, 2, 3, None]
    assert ic_or_zero(pred, target, days, min_n=3) == pytest.approx((0.8 + 0.0) / 2)
    assert ic_or_zero(pred, target, days, min_n=4) == pytest.approx(0.8)  # second date has 3 pairs
    assert math.isnan(ic_or_zero(pred, target, days, min_n=5))


def test_baseline_factories():
    f = baseline_factories(CATALOG)
    assert list(f) == ["b0_no_skill", "b1_mom_12_1", "b1_gross_profitability", "b1_earnings_yield",
                       "b2_composite", "b3_ridge"]
    models = {k: make() for k, make in f.items()}
    assert all(m.name == k for k, m in models.items())
    assert models["b1_mom_12_1"].sign == 1 and models["b1_mom_12_1"].col == "mom_12_1"
    assert models["b3_ridge"].cols == CATALOG.linear_columns()
    assert models["b3_ridge"].alphas == (1.0, 10.0, 100.0, 1000.0, 10000.0)
    assert f["b3_ridge"]() is not models["b3_ridge"]  # a fresh model per call
    custom = baseline_factories(CATALOG, BaselineParams(single_factors=("accruals",), ridge_alphas=(3.0,)))
    assert custom["b1_accruals"]().sign == -1 and custom["b3_ridge"]().alphas == (3.0,)
    pickle.loads(pickle.dumps(f))


def test_baselines_pickle_after_fit():
    df = ranked_frame({c: [0.1 * i - 0.2 for i in range(5)] for c in CATALOG.linear_columns()}, 5)
    for make in baseline_factories(CATALOG, BaselineParams(ridge_alphas=(10.0,))).values():
        m = make().fit(df, df)
        back = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(back.predict(df), m.predict(df))


def test_ridge_through_walk_forward():
    # 2000-01 .. 2011-12, 25 stocks, target = a (+ noise); 12m label ends 12 dates later
    rules = ValidationRules(first_test_year=2010, test_months=12, purge_months=12, embargo_months=1,
                            min_train_months=60, inner_valid_months=24)
    frame = planted(144, 25, 2000, 0.3, 7)
    days = frame["date"].unique().sort().to_list()
    end = dict(zip(days, days[12:] + [None] * 12))
    frame = frame.with_columns(
        pl.col("date").replace_strict(end, return_dtype=pl.Date).alias("label_end_12m"),
        pl.when(pl.col("date") <= days[-13]).then(pl.col("target")).alias("target"),
    )
    folds = walk_forward_folds(days, rules, d(2011, 12, 31))
    seen = []
    oof = walk_forward(baseline_factories(CATALOG)["b0_no_skill"], frame, folds, rules)
    assert (oof["pred"] == 0.0).all()
    oof = walk_forward(lambda: Ridge(ABC, [1.0, 1000.0]), frame, folds, rules,
                       on_fold=lambda fold, m, shown: seen.append((m.alpha_, "target" in shown.columns)))
    assert [s[1] for s in seen] == [False, False]
    scored = oof.join(frame.select("date", "security_id", "target"), on=["date", "security_id"])
    assert ic_or_zero(scored["pred"], scored["target"], scored["date"]) > 0.5

"""Explanations (plan 6.4): additive member attributions, ensemble blend, driver text, importance stability."""

import numpy as np
import polars as pl
import pytest
from test_ensemble import CATALOG, TINY, catalog_frame

from ltforecast.explain import (
    Driver,
    FoldImportance,
    describe,
    ensemble_contributions,
    importance_by_fold,
    importance_stability,
    member_contributions,
    ordinal,
    percentile,
    top_drivers,
)
from ltforecast.models.baselines import FactorComposite, Ridge
from ltforecast.models.ensemble import Ensemble, default_factories
from ltforecast.models.lgbm import LGBM, SeedAverage


@pytest.fixture(scope="module")
def frames():
    train, valid, test = catalog_frame(12, 30, 2000, 1), catalog_frame(6, 30, 2001, 2), catalog_frame(2, 30, 2003, 3)
    # some missing values, including a whole theme (low_vol) on a few rows
    test = test.with_columns(
        pl.when(pl.col("security_id") <= 3).then(None).otherwise(pl.col(c)).alias(c)
        for c in ("vol_1y", "idio_vol", "beta_60m", "book_to_price", "mom_12_1")
    )
    return train, valid, test


@pytest.fixture(scope="module")
def ensemble(frames):
    train, valid, _ = frames
    return Ensemble(default_factories(CATALOG, lgbm_params=TINY, seeds=(0, 1))).fit(train, valid)


# member additivity -------------------------------------------------------------------------------------


def test_lgbm_shap_rows_sum_to_raw_minus_expected_value(frames):
    train, valid, test = frames
    cols = CATALOG.model_columns()
    m = LGBM(cols, TINY).fit(train, valid)
    a = member_contributions(m, test)
    assert a.cols == cols and a.values.shape == (test.height, len(cols))
    np.testing.assert_allclose(a.values.sum(axis=1), m.predict(test) - a.base, atol=1e-9)
    np.testing.assert_allclose(a.raw, m.predict(test), atol=1e-9)
    s = SeedAverage(cols, TINY, seeds=(0, 1)).fit(train, valid)
    sa = member_contributions(s, test)
    raw_mean = np.mean([mm.predict(test) for mm in s.members], axis=0)
    np.testing.assert_allclose(sa.raw, raw_mean, atol=1e-9)
    np.testing.assert_allclose(sa.values.sum(axis=1), raw_mean - sa.base, atol=1e-9)


def test_ridge_coef_times_value_sums_to_pred_minus_intercept(frames):
    train, valid, test = frames
    r = Ridge(CATALOG.linear_columns(), (10.0,)).fit(train, valid)
    a = member_contributions(r, test)
    np.testing.assert_allclose(a.values.sum(axis=1), r.predict(test) - r.intercept_, atol=1e-12)
    assert a.base == r.intercept_
    # a null value contributes exactly 0
    j = a.cols.index("mom_12_1")
    assert (a.values[:3, j] == 0).all()


def test_composite_shares_sum_to_composite_score(frames):
    _, _, test = frames
    c = FactorComposite(CATALOG)
    a = member_contributions(c, test)
    np.testing.assert_allclose(a.values.sum(axis=1), c.predict(test), atol=1e-12)
    # hand check one share: row 10, momentum theme fully available, 5 themes available
    mom = [col for col, _ in c.themes["momentum"]]
    j = a.cols.index("mom_6_1")
    sign = CATALOG.spec("mom_6_1").sign
    assert a.values[10, j] == pytest.approx(sign * test["mom_6_1"][10] / len(mom) / 5)
    # rows 0-2 miss the low-vol theme entirely: 4 themes; their low-vol columns attribute 0
    assert (a.values[:3, a.cols.index("vol_1y")] == 0).all()
    j2 = a.cols.index("roic")
    n_quality = len(c.themes["quality"])
    assert a.values[0, j2] == pytest.approx(CATALOG.spec("roic").sign * test["roic"][0] / n_quality / 4)


def test_unknown_member_type_raises(frames):
    with pytest.raises(TypeError):
        member_contributions(object(), frames[2])


# ensemble blend -----------------------------------------------------------------------------------------


def test_ensemble_contributions_are_weighted_standardized_member_sums(ensemble, frames):
    test = frames[2]
    contrib, names = ensemble_contributions(ensemble, test, CATALOG)
    assert contrib.shape == (test.height, len(names)) and len(set(names)) == len(names)
    assert names == [c for c in CATALOG.model_columns() if c in names]
    expect = np.zeros(test.height)
    for nm, m in ensemble.members_.items():
        a = member_contributions(m, test)
        sd = pl.DataFrame({"d": test["date"], "r": a.raw}).select(pl.col("r").std(ddof=0).over("d"))["r"].to_numpy()
        expect += ensemble.weights_[nm] * (a.raw - a.base) / sd
    np.testing.assert_allclose(contrib.sum(axis=1), expect, atol=1e-9)
    # the blended attribution tracks the ensemble score
    assert np.corrcoef(contrib.sum(axis=1), ensemble.predict(test))[0, 1] > 0.8


# drivers ------------------------------------------------------------------------------------------------


def test_percentile_formula_and_ordinals():
    assert percentile(0.47) == 97
    assert percentile(0.38) == 88
    assert percentile(-0.3) == 20
    assert percentile(-0.005) == 50       # 49.5 rounds half up
    assert percentile(0.0) == 50
    assert percentile(-0.4995) == 0
    assert percentile(None) is None and percentile(float("nan")) is None
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 100, 111)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "100th", "111th"
    ]


def test_top_drivers_text():
    names = ["gross_profitability", "fcf_yield_sr", "term_spread", "has_value", "accruals", "ebitda_to_ev",
             "credit_spread"]
    contrib = [0.5, 0.3, 0.2, 0.9, -0.4, -0.1, 0.0]
    values = [0.47, 0.38, 0.012, 1.0, -0.3, None, 0.03]
    pos, neg = top_drivers(contrib, names, values, CATALOG, k=3)
    assert [d.text for d in pos] == [
        "Gross profits/assets (97th pct)", "Free-cash-flow yield vs sector (88th pct)", "Market context: term spread"
    ]
    assert pos[0] == Driver("gross_profitability", "Gross profits/assets", 97, 0.5, "Gross profits/assets (97th pct)")
    assert pos[2].percentile is None
    assert [d.text for d in neg] == ["Accruals (20th pct)", "EBITDA/EV (missing)"]
    assert all(d.feature != "has_value" for d in pos + neg)
    p1, n1 = top_drivers(contrib, names, values, CATALOG, k=1)
    assert [d.feature for d in p1] == ["gross_profitability"] and [d.feature for d in n1] == ["accruals"]
    # has_* only, or all zero -> nothing
    assert top_drivers([1.0, 0.0], ["has_value", "accruals"], [1, 0.1], CATALOG) == ([], [])
    assert describe("has_quality", 1.0, CATALOG) is None
    assert describe("market_12m_unknown", 0.1, CATALOG)[2] == "market_12m_unknown"


def test_top_drivers_on_ensemble_rows(ensemble, frames):
    test = frames[2]
    contrib, names = ensemble_contributions(ensemble, test, CATALOG)
    vals = test.select(names).to_numpy()
    pos, neg = top_drivers(contrib[5], names, list(vals[5]), CATALOG)
    assert len(pos) <= 3 and len(neg) <= 3
    assert all(d.contribution > 0 for d in pos) and all(d.contribution < 0 for d in neg)
    assert [d.contribution for d in pos] == sorted((d.contribution for d in pos), reverse=True)


# importance ---------------------------------------------------------------------------------------------


def test_importance_by_fold_mean_abs():
    c = np.array([[1.0, -2.0, 0.0], [-3.0, 2.0, 0.5]])
    t = importance_by_fold({"2010": (c, ["a", "b", "c"]), "2011": (2 * c, ["a", "b", "c"])})
    assert t.columns == ["fold", "feature", "mean_abs"]
    assert t.filter(pl.col("fold") == "2010")["mean_abs"].to_list() == [2.0, 2.0, 0.25]
    assert t.filter(pl.col("fold") == "2011")["mean_abs"].to_list() == [4.0, 4.0, 0.5]
    assert importance_by_fold({}).height == 0


def test_importance_stability():
    feats = [f"f{i}" for i in range(10)]
    base = np.arange(10, 0, -1, dtype=float)

    def table(vectors):
        return pl.concat(
            pl.DataFrame({"fold": str(k), "feature": feats, "mean_abs": v}) for k, v in enumerate(vectors)
        )

    assert importance_stability(table([base, base, base * 3])) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    shuffled = importance_stability(table([base, rng.permutation(base), rng.permutation(base)]))
    assert shuffled < 0.9
    assert importance_stability(table([base, base[::-1]])) == pytest.approx(-1.0)
    assert np.isnan(importance_stability(table([base])))
    # a feature absent from one fold counts as 0 importance there
    partial = table([base]).vstack(pl.DataFrame({"fold": "1", "feature": feats[:9], "mean_abs": base[:9]}))
    assert importance_stability(partial) == pytest.approx(1.0)


def test_fold_importance_hook(ensemble, frames):
    test = frames[2]
    hook = FoldImportance(CATALOG, max_rows=40)

    class F:
        name = "2003"

    hook(F(), ensemble, test)
    t = hook.table()
    assert set(t["fold"]) == {"2003"} and t.height == len(hook.per_fold["2003"][1])
    assert hook.per_fold["2003"][0].shape[0] == 40
    assert (t["mean_abs"] >= 0).all()


def test_ensemble_contributions_subset_equals_full_frame_rows(ensemble, frames):
    test = frames[2]
    full, names = ensemble_contributions(ensemble, test, CATALOG)
    ids = test["security_id"].unique().sort().to_list()[:2]
    sub, sub_names = ensemble_contributions(ensemble, test, CATALOG, security_ids=ids)
    keep = test["security_id"].is_in(ids).to_numpy()
    assert sub_names == names and sub.shape == (int(keep.sum()), len(names))
    np.testing.assert_array_equal(sub, full[keep])
    # a frame cut down to the requested rows is refused once a minimum cross-section is set
    n_min = int(test.group_by("date").len()["len"].min())
    with pytest.raises(ValueError, match="partial cross-section"):
        ensemble_contributions(ensemble, test.filter(pl.Series(keep)), CATALOG, min_cross_section=n_min)
    ok, _ = ensemble_contributions(ensemble, test, CATALOG, security_ids=ids, min_cross_section=n_min)
    np.testing.assert_array_equal(ok, sub)

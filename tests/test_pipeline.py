"""pipeline.py helpers on tiny hand-built inputs: percentiles, levels, score wiring (stub models),
the step-9 evaluation and its markdown report. The full run is in test_e2e_train.py."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from ltforecast.config import BarrierRules
from ltforecast.dataset import label_columns
from ltforecast.hitprob import HitProb
from ltforecast.modelconfig import load_model_config
from ltforecast.pipeline import (
    SCORE_COLS,
    FinalModels,
    FINAL_FILES,
    _need_folds,
    check_labels_before_holdout,
    check_train_state,
    data_fingerprint,
    effective_config_hash,
    horizon_months,
    spread_deflated_sharpe,
    evaluate_forecast,
    fit_all_recorded,
    level_ratios,
    load_final,
    rank_pct,
    report_markdown,
    score,
)

RULES = BarrierRules(horizon="12m", k_tp=(1.0,), k_sl=(0.5,), tp_bounds=(0.10, 1.00), sl_bounds=(0.08, 0.40),
                     sigma_window=252, sigma_min_obs=126)


def test_rank_pct_is_the_per_date_percentile():
    df = pl.DataFrame({"date": [1, 1, 1, 2, 2], "pred": [0.3, 0.1, 0.3, 5.0, -1.0]})
    out = df.select(rank_pct("pred")).to_series().to_list()
    # date 1: ranks 2.5, 1, 2.5 of 3 -> (r - 0.5) / 3; date 2: ranks 2, 1 of 2
    assert out == pytest.approx([2 / 3, 1 / 6, 2 / 3, 0.75, 0.25])


def test_level_ratios_follow_the_guardrails():
    sigma = np.array([0.01, 0.30, 5.0])
    tp, sl = level_ratios(sigma, 1.0, 0.5, RULES)
    np.testing.assert_allclose(tp, [1.10, np.exp(0.30), 2.0])
    np.testing.assert_allclose(sl, [0.92, np.exp(-0.15), 0.60])
    assert level_ratios(sigma, None, None, RULES) == (None, None)


def test_fewer_than_three_folds_raises():
    with pytest.raises(ValueError, match="at least 3"):
        _need_folds([object(), object()])


# score --------------------------------------------------------------------------------------------


class NoLabels:
    """Stub model: fails if it is shown any label column; returns fixed values."""

    def __init__(self, fn):
        self.fn = fn
        self.seen = []

    def _check(self, df):
        assert not label_columns(df.columns), label_columns(df.columns)
        self.seen.append(df.columns)

    def predict(self, df):
        self._check(df)
        return self.fn(df)

    predict_proba = predict


class StubTable:
    def score(self, mu, sigma, tp, sl):
        n = len(sigma)
        return HitProb(np.full(n, 0.2), np.full(n, 0.3), np.full(n, 0.5), np.full(n, np.nan))


@pytest.fixture
def scoring_inputs():
    dates = [dt.date(2020, 1, 31)] * 3 + [dt.date(2020, 2, 28)] * 3
    sids = [1, 2, 3] * 2
    features = pl.DataFrame({"date": dates, "security_id": sids, "sector": ["A"] * 6, "f": [0.1, -0.2, 0.3] * 2,
                             "target": [0.9] * 6, "excess_12m": [0.5] * 6})
    raw = pl.DataFrame({"date": dates, "security_id": sids, "beta_60m": [1.0] * 6, "log_mcap": [20.0] * 6})
    days = pl.date_range(dt.date(2019, 1, 1), dt.date(2020, 3, 1), "1d", eager=True)
    prices = pl.DataFrame({"security_id": np.repeat([1, 2, 3], len(days)), "date": pl.concat([days] * 3),
                           "adj_close": np.tile(100 * np.exp(np.cumsum(np.full(len(days), 0.001))), 3)})
    securities = pl.DataFrame({"security_id": [1, 2, 3], "sector": ["A", "B", "A"]})
    return features, raw, prices, securities


def stub_models(isotonic=None, method="mc", min_cross_section=1) -> FinalModels:
    return FinalModels(
        ensemble=NoLabels(lambda df: df["f"].to_numpy()),
        vol=NoLabels(lambda df: np.full(df.height, 0.30)),
        quantile=NoLabels(lambda df: np.tile([-0.1, 0.0, 0.1], (df.height, 1))),
        isotonic=isotonic, conformal_offset=0.05, conformal_ok=True, k_tp=1.0, k_sl=None, verdict="risk_reference",
        barriers=RULES, hit_table=StubTable(),
        hit_classifier=NoLabels(lambda df: np.tile([0.4, 0.1, 0.5], (df.height, 1))),
        method=method, market_return=0.06, trained_through=dt.date(2019, 12, 31), model_columns=["f"],
        min_cross_section=min_cross_section,
    )


def test_score_wiring_without_labels(scoring_inputs):
    features, raw, prices, securities = scoring_inputs
    models = stub_models(method="average")
    out = score(models, features, raw, prices, securities, dates=[dt.date(2020, 2, 28)])
    assert out.columns == list(SCORE_COLS)
    assert out["date"].unique().to_list() == [dt.date(2020, 2, 28)] and out.height == 3
    assert out["rank_pct"].to_list() == pytest.approx([0.5, 1 / 6, 5 / 6])
    assert out["prob"].to_list() == [0.5] * 3 and not out["prob_calibrated"].any()  # no isotonic map
    # conformal offset widens the range around q50
    assert out["q10"].to_list() == pytest.approx([-0.15] * 3) and out["q90"].to_list() == pytest.approx([0.15] * 3)
    assert out["tp_level"].to_list() == pytest.approx([np.expm1(0.30)] * 3)
    assert out["sl_level"].null_count() == 3  # k_sl None: no stop
    # average of Monte Carlo (0.2, 0.3, 0.5) and classifier (0.4, 0.1, 0.5)
    assert out.select("p_tp", "p_sl", "p_none").row(0) == pytest.approx((0.3, 0.2, 0.5))
    assert (out["trained_through"] == dt.date(2019, 12, 31)).all()
    for m in (models.ensemble, models.vol, models.quantile, models.hit_classifier):
        assert m.seen  # every model was called, none was shown a label


def test_score_applies_the_isotonic_map(scoring_inputs):
    features, raw, prices, securities = scoring_inputs
    iso = SimpleNamespace(predict=lambda s: np.clip(s, 0.2, 0.8))
    out = score(stub_models(isotonic=iso), features, raw, prices, securities)
    assert out["prob_calibrated"].all()
    assert out["prob"].to_list() == pytest.approx(np.clip(out["rank_pct"].to_numpy(), 0.2, 0.8))


def test_score_ranks_against_the_full_cross_section_then_filters(scoring_inputs):
    features, raw, prices, securities = scoring_inputs
    iso = SimpleNamespace(predict=lambda s: np.clip(s, 0.2, 0.8))
    models = stub_models(isotonic=iso, min_cross_section=3)
    full = score(models, features, raw, prices, securities)
    sub = score(models, features, raw, prices, securities, security_ids=[3])
    # f = 0.1, -0.2, 0.3 -> security 3 ranks 3rd of 3 on each date: rank_pct (3 - 0.5) / 3 = 5/6, pred 0.3
    assert sub.height == 2 and sub["security_id"].to_list() == [3, 3]
    assert sub["rank_pct"].to_list() == pytest.approx([5 / 6] * 2)
    assert sub["pred"].to_list() == pytest.approx([0.3] * 2)
    assert sub["prob"].to_list() == pytest.approx([0.8] * 2)
    assert sub.equals(full.filter(pl.col("security_id") == 3))
    assert score(models, features, raw, prices, securities, security_ids=[99]).height == 0


def test_score_refuses_a_partial_cross_section(scoring_inputs):
    features, raw, prices, securities = scoring_inputs
    models = stub_models(min_cross_section=3)
    # one ticker alone would get pred 0 and rank_pct 0.5 whatever it is: refuse instead
    with pytest.raises(ValueError, match="cross-section"):
        score(models, features.filter(pl.col("security_id") == 3), raw, prices, securities)
    with pytest.raises(ValueError, match="cross-section"):
        score(models, features.filter(pl.col("security_id") != 2), raw, prices, securities)


# deflated Sharpe of overlapping spreads ------------------------------------------------------------


def test_horizon_months_from_the_outcome_name():
    assert horizon_months("excess_12m") == 12
    assert horizon_months("excess_w_6m") == 6
    assert horizon_months("target") == 1


def test_deflated_sharpe_uses_the_non_overlapping_sample_size():
    from scipy.stats import norm

    # 24 monthly spreads alternating 0.3, -0.1: mean 0.1, std(ddof=1) = 0.2 * sqrt(24/23)
    spread = np.tile([0.3, -0.1], 12)
    d = spread_deflated_sharpe(spread, n_trials=1, var_sr=None, horizon=12)
    sr = 0.1 / (0.2 * np.sqrt(24 / 23))
    assert d["spread_sharpe_monthly"] == pytest.approx(sr)
    assert d["n_months"] == 24 and d["n_eff"] == pytest.approx(2.0)  # 24 months of 12m windows = 2 years
    assert d["var_sr"] == pytest.approx(0.5)  # fallback 1 / n_eff
    # skew 0, kurt 1 for a two-point symmetric series: denom = 1 + 0 * sr^2; T - 1 = 1
    assert d["dsr"] == pytest.approx(norm.cdf(sr * 1.0 / np.sqrt(1.0)))
    one = spread_deflated_sharpe(spread, n_trials=1, var_sr=None, horizon=1)
    assert one["n_eff"] == 24


def test_deflated_sharpe_false_positive_rate_on_overlapping_zero_skill_spreads():
    # 12-month overlapping sums of iid monthly noise, T = 240, no skill: DSR > 0.95 should be ~5%
    rng = np.random.default_rng(0)
    eps = rng.normal(size=(1000, 240 + 11))
    c = np.cumsum(np.pad(eps, ((0, 0), (1, 0))), axis=1)
    spreads = c[:, 12:] - c[:, :-12]
    dsr = np.array([spread_deflated_sharpe(s, n_trials=1, var_sr=None, horizon=12)["dsr"] for s in spreads])
    assert np.mean(dsr > 0.95) < 0.09


# fit_final guards ----------------------------------------------------------------------------------


def test_fast_and_full_configs_hash_differently():
    full, fast = load_model_config(), load_model_config(fast=True)
    assert effective_config_hash(full) != effective_config_hash(fast)
    assert effective_config_hash(full) == effective_config_hash(load_model_config())


def test_fit_final_refuses_a_train_state_from_another_config_or_run():
    full, fast = load_model_config(), load_model_config(fast=True)
    state = {"effective_config_hash": effective_config_hash(fast), "fast": True, "dev_end": "2013-12-31"}
    check_train_state(state, fast, dt.date(2013, 12, 31))
    check_train_state(state, fast, dt.date(2012, 12, 31))  # a shorter refit is fine
    with pytest.raises(ValueError, match="fast"):
        check_train_state(state, full, dt.date(2013, 12, 31))
    with pytest.raises(ValueError, match="config"):
        check_train_state(state | {"effective_config_hash": "x"}, fast, dt.date(2013, 12, 31))
    with pytest.raises(ValueError, match="after"):
        check_train_state(state, fast, dt.date(2014, 6, 30))
    with pytest.raises(ValueError, match="re-run"):
        check_train_state({"fast": True}, fast, dt.date(2013, 12, 31))  # an old train_state.json


def _weekdays(start: dt.date, n: int) -> pl.Series:
    days = pl.date_range(start, start + dt.timedelta(days=3 * n), "1d", eager=True)
    return days.filter(days.dt.weekday() <= 5).head(n).alias("date")


def test_dev_label_windows_must_end_before_the_holdout():
    # calendar Mon 2021-10-25 .. Fri 2021-11-05; last dev date Fri 10-29 (index 4); +3 trading days = Wed 11-03
    cal = _weekdays(dt.date(2021, 10, 25), 10)
    assert cal[4] == dt.date(2021, 10, 29) and cal[7] == dt.date(2021, 11, 3) and cal[-1] == dt.date(2021, 11, 5)
    frame = pl.DataFrame({"date": [dt.date(2021, 10, 28), dt.date(2021, 10, 29)], "security_id": [1, 1],
                          "target": [0.5, None], "label_end_12m": [dt.date(2021, 11, 2), None]})
    check_labels_before_holdout(frame, cal, 3, dt.date(2021, 11, 4))  # one trading day of margin
    with pytest.raises(ValueError, match="2021-10-29"):  # window of the last dev date reaches the holdout
        check_labels_before_holdout(frame, cal, 3, dt.date(2021, 11, 3))
    # a stored label end inside the holdout is caught even if the calendar step says otherwise
    late = frame.with_columns(pl.Series("label_end_12m", [dt.date(2021, 11, 8), None]))
    with pytest.raises(ValueError, match="label_end_12m"):
        check_labels_before_holdout(late, cal, 3, dt.date(2021, 11, 4))
    # a window running off the calendar is safe only if no holdout price exists yet
    check_labels_before_holdout(frame, cal, 20, dt.date(2021, 11, 8))
    with pytest.raises(ValueError, match="holdout"):
        check_labels_before_holdout(frame, cal, 20, dt.date(2021, 11, 5))


class _SpyModel:
    """Records the rows its fit received (train + valid, or `full` when the runner passes one)."""

    name = "spy"

    def fit(self, train, valid, *, full=None):
        parts = [full] if full is not None else [train, valid]
        self.seen = pl.concat([p.select("date", "security_id") for p in parts])
        return self

    def predict(self, df):
        return np.zeros(df.height)


def test_final_fit_metadata_describes_the_rows_actually_fitted():
    # 30 month ends x 2 stocks, all labelled; inner split: valid = last 4 dates, train = first 30-4-3-1 = 22
    dates = pl.date_range(dt.date(2000, 1, 1), dt.date(2002, 6, 1), "1mo", eager=True).dt.month_end()
    frame = pl.DataFrame({"date": np.repeat(dates.to_numpy(), 2), "security_id": [1, 2] * 30}).with_columns(
        pl.col("date").cast(pl.Date), pl.lit(0.5).alias("target"),
        (pl.col("date").cast(pl.Date) + pl.duration(days=60)).alias("label_end_12m"))
    rules = load_model_config().validation
    rules = dataclasses.replace(rules, purge_months=3, embargo_months=1, inner_valid_months=4)
    model, meta = fit_all_recorded(_SpyModel, frame, rules)
    assert isinstance(model, _SpyModel)  # the recorder is unwrapped before the model is saved
    assert meta["fit_rows"] == model.seen.height
    assert meta["fit_date_min"] == model.seen["date"].min() and meta["fit_date_max"] == model.seen["date"].max()
    assert meta["labelled_rows"] == 60
    # the runner refits on `full`: the 4 purge/embargo months (2001-11 .. 2002-02) between inner train
    # and inner valid are fitted too, so all 60 labelled rows reach the final model
    assert meta["fit_rows"] == 60 and meta["fit_dates"] == 30
    assert meta["fit_dates"] == model.seen["date"].n_unique()


def _write_final(path, run_id="r1", through="2013-12-31", cols=("a", "b")):
    from ltforecast.models.io import save_model

    for name in FINAL_FILES:
        save_model({"name": name}, path / f"{name}.pkl", {"run_id": run_id, "trained_through": through,
                                                          "columns": list(cols)})
    meta = {"run_id": run_id, "trained_through": through, "feature_columns": list(cols), "k_tp": 1.0,
            "k_sl": 0.5, "verdict": "x", "hit_method": "mc", "market_return": 0.0, "conformal_offset": 0.0,
            "conformal_ok": True, "barriers": {"horizon": "12m", "tp_bounds": [0.1, 1.0], "sl_bounds": [0.08, 0.4]}}
    (path / "metadata.json").write_text(json.dumps(meta))


def test_load_final_refuses_artifacts_from_different_runs(tmp_path):
    from ltforecast.models.io import save_model

    _write_final(tmp_path)
    assert load_final(tmp_path).ensemble == {"name": "ensemble"}
    # a later refit that died after writing ensemble.pkl
    save_model({"name": "new"}, tmp_path / "ensemble.pkl", {"run_id": "r2", "trained_through": "2013-12-31",
                                                             "columns": ["a", "b"]})
    with pytest.raises(ValueError, match="ensemble"):
        load_final(tmp_path)
    for kw in ({"through": "2011-12-31"}, {"cols": ("x", "y")}):
        bad = tmp_path / str(len(kw.get("cols", ())))
        bad.mkdir()
        _write_final(bad)
        ref = json.loads((bad / "metadata.json").read_text())
        _write_final(bad, **kw)
        (bad / "metadata.json").write_text(json.dumps(ref))
        with pytest.raises(ValueError, match="does not match"):
            load_final(bad)
    old = tmp_path / "old"
    old.mkdir()
    _write_final(old)
    meta = json.loads((old / "metadata.json").read_text())
    del meta["run_id"]
    (old / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="run_id"):  # written before run ids: not trusted
        load_final(old)


def test_data_fingerprint_is_per_date_and_order_free():
    d1, d2 = dt.date(2020, 1, 31), dt.date(2020, 2, 28)
    df = pl.DataFrame({"date": [d1, d1, d2], "security_id": [1, 2, 1], "x": [0.1, 0.2, 0.3]})
    fp = data_fingerprint(df)
    assert set(fp) == {"2020-01-31", "2020-02-28"}
    assert data_fingerprint(df.reverse()) == fp
    changed = data_fingerprint(df.with_columns(pl.when(pl.col("date") == d2).then(0.4).otherwise(pl.col("x")).alias("x")))
    assert changed["2020-01-31"] == fp["2020-01-31"] and changed["2020-02-28"] != fp["2020-02-28"]


# evaluation and report ----------------------------------------------------------------------------


def eval_inputs(gain: float, seed: int = 0):
    """36 months x 60 stocks; target = rank of excess; the forecast has signal `gain`, B2 none."""
    rng = np.random.default_rng(seed)
    dates = [dt.date(2010 + m // 12, m % 12 + 1, 28) for m in range(36)]
    rows = []
    for d in dates:
        x = rng.normal(size=60)
        ex = gain * x + rng.normal(size=60)
        rows.append(pl.DataFrame({"date": [d] * 60, "security_id": np.arange(60), "x": x, "excess_12m": ex}))
    df = pl.concat(rows).with_columns(
        ((pl.col("excess_12m").rank() - 0.5) / pl.len()).over("date").alias("target"),
        pl.col("date").dt.year().cast(pl.String).alias("fold"),
    )
    n = df.height
    forecast = df.select(
        "date", "security_id", "fold", pl.col("x").alias("pred"),
        ((pl.col("x").rank() - 0.5) / pl.len()).over("date").alias("rank_pct"),
        pl.lit(0.5).alias("prob"), pl.lit(True).alias("prob_calibrated"),
        pl.lit(-1.0).alias("q10"), pl.lit(0.0).alias("q50"), pl.lit(1.0).alias("q90"), pl.lit(0.3).alias("sigma_hat"),
        pl.lit(0.2).alias("p_tp"), pl.lit(0.3).alias("p_sl"), pl.lit(0.5).alias("p_none"),
        pl.lit(0.3).alias("tp_level"), pl.lit(-0.15).alias("sl_level"),
    )
    labels = df.select("date", "security_id", "target", "excess_12m")
    outcome = np.where(np.arange(n) % 5 == 0, "tp", np.where(np.arange(n) % 5 == 1, "sl", "time"))
    barriers = df.select("date", "security_id").with_columns(pl.Series("outcome", outcome))
    b2 = df.select("date", "security_id", "fold", pl.Series("pred", rng.normal(size=n)))
    sel = SimpleNamespace(
        k_tp=1.0, k_sl=0.5, verdict="risk_reference", hold={"sharpe": 1.0, "max_dd": -0.2}, n_trials=29,
        holdout={"sharpe": 0.8, "max_dd": -0.22, "hold_sharpe": 1.0, "hold_max_dd": -0.2, "nw_tstat": -0.5,
                 "n_days": 500, "picks": [{"fold": "2010", "k_tp": 1.0, "k_sl": 0.5}]},
        summary=pl.DataFrame({"k_tp": [1.0], "k_sl": [0.5], "mean_sharpe": [0.9], "mean_max_dd": [-0.25],
                              "eligible": [False]}),
        variants=pl.DataFrame({"variant": ["hold"], "param": [None], "fold": ["2010"], "sharpe": [1.0],
                               "max_dd": [-0.2]}, schema_overrides={"param": pl.Float64}),
    )
    return forecast, labels, barriers, {"b2_composite": b2}, pl.DataFrame({"model": ["b2_composite"]}), sel


@pytest.fixture
def mcfg(tmp_path):
    import dataclasses

    m = load_model_config()
    return dataclasses.replace(m, experiments_dir=tmp_path / "exp",
                               calibration=dataclasses.replace(m.calibration, min_bin_rows=1))


@pytest.mark.parametrize(("gain", "beats"), [(0.0, False), (1.0, True)])
def test_evaluation_states_the_b2_verdict_and_all_six_criteria(mcfg, gain, beats):
    forecast, labels, barriers, b_oof, b_table, sel = eval_inputs(gain)
    report, checks, rec = evaluate_forecast(forecast, labels, barriers, b_oof, b_table, mcfg, sel=sel)
    assert [c.name for c in checks] == [
        "rank_ic", "spread_positive_years", "beats_baseline", "interval_coverage", "calibration", "tpsl_rules",
    ]
    assert all(c.status in ("pass", "fail") for c in checks)
    assert report["beats_b2"] is beats and rec == ("ship ensemble" if beats else "ship B2")
    # TP/SL verdict "risk_reference" fails criterion 6 whatever the calibration
    assert checks[-1].status == "fail"
    # P(TP first) = 0.2 everywhere and 1 in 5 outcomes is tp: calibrated
    assert report["tp_calibration_gap"] == pytest.approx(0.0)
    assert report["deflated_sharpe"]["n_trials"] == 1  # nothing logged in this empty experiments dir
    # 36 monthly 12m spreads overlap: 3 independent years
    assert report["deflated_sharpe"]["n_months"] == 36 and report["deflated_sharpe"]["n_eff"] == pytest.approx(3.0)

    report["provenance"] = {"synthetic": True}
    md = report_markdown(report)
    assert "SYNTHETIC DATA" in md
    assert f"Does the ensemble beat B2 (factor composite)? {'YES' if beats else 'NO'}." in md
    assert f"Recommendation: {rec}." in md
    for n in range(1, 7):
        assert f"\n| {n} | " in md
    # the verdict is shown with its out-of-sample (leave-one-fold-out) numbers, not only in-sample ones
    assert "held-out folds" in checks[-1].detail and "NW t -0.50" in checks[-1].detail
    assert ("Verdict test (pair picked on the other folds, scored on the held-out fold): net Sharpe 0.80 vs hold "
            "1.00, max drawdown -22.0% vs hold -20.0%, Newey-West t of the daily difference -0.50 over 500 days") in md
    assert report["exits"]["holdout"]["nw_tstat"] == -0.5
    report["provenance"] = {"synthetic": False}
    assert "SYNTHETIC" not in report_markdown(report)


def test_criterion_six_fails_a_no_tp_pair_with_miscalibrated_p_sl(mcfg):
    forecast, labels, barriers, b_oof, b_table, sel = eval_inputs(1.0)
    n = forecast.height
    # no TP barrier: p_tp = 0; the model says P(SL first) = 0.9 but SL is hit on 1 row in 10
    forecast = forecast.with_columns(pl.lit(0.0).alias("p_tp"), pl.lit(0.9).alias("p_sl"), pl.lit(0.1).alias("p_none"))
    barriers = barriers.with_columns(pl.Series("outcome", np.where(np.arange(n) % 10 == 0, "sl", "time")))
    sel.k_tp, sel.k_sl, sel.verdict = None, 0.5, "exit_orders"
    sel.summary = sel.summary.with_columns(pl.lit(None, dtype=pl.Float64).alias("k_tp"))
    report, checks, _ = evaluate_forecast(forecast, labels, barriers, b_oof, b_table, mcfg, sel=sel)
    hc = report["hit_calibration"]
    assert hc["tested"] == ["sl", "time"] and hc["worst"] in ("sl", "time")
    assert hc["gaps"]["sl"] == pytest.approx(0.8) and hc["gap"] == pytest.approx(0.8)
    assert checks[-1].status == "fail" and "P(SL hit first)" in checks[-1].detail
    assert "per class" in report_markdown(report | {"provenance": {"synthetic": True}})


def test_coverage_is_judged_on_dates_with_a_fitted_conformal_offset(mcfg):
    forecast, labels, barriers, b_oof, b_table, sel = eval_inputs(1.0)
    dates = forecast["date"].unique().sort().to_list()
    # the first year's ranges are useless (width 0); those dates had no conformal offset
    first = dates[:12]
    forecast = forecast.with_columns(
        pl.when(pl.col("date").is_in(first)).then(0.0).otherwise(pl.col("q10")).alias("q10"),
        pl.when(pl.col("date").is_in(first)).then(0.0).otherwise(pl.col("q90")).alias("q90"),
    )
    conformal = pl.DataFrame({"date": dates, "conformal_ok": [d not in first for d in dates]})
    report, _, _ = evaluate_forecast(forecast, labels, barriers, b_oof, b_table, mcfg, sel=sel, conformal=conformal)
    assert report["coverage_rows"] == "conformal_ok"
    ev = forecast.join(labels, on=["date", "security_id"])
    late = ev.filter(~pl.col("date").is_in(first))
    want = ((late["excess_12m"] >= late["q10"]) & (late["excess_12m"] <= late["q90"])).mean()
    assert report["coverage"] == pytest.approx(want)
    assert report["coverage_all_rows"] < report["coverage"]
    # no fitted date: criterion 4 falls back to all rows
    none_ok = conformal.with_columns(pl.lit(False).alias("conformal_ok"))
    r2, _, _ = evaluate_forecast(forecast, labels, barriers, b_oof, b_table, mcfg, sel=sel, conformal=none_ok)
    assert r2["coverage_rows"] == "all" and r2["coverage"] == pytest.approx(report["coverage_all_rows"])


def _stub_tune_run(ics: dict):
    """tune.run stand-in: mean_ic per candidate from `ics` keyed by (objective, num_leaves, learning_rate)."""
    def run(frame, folds, rules, grid, exp_dir, *, cols, seed=0, name="", **kw):
        m = [ics.get((p.objective, p.num_leaves, p.learning_rate), 0.0) for p in grid]
        best = max(range(len(m)), key=lambda j: (m[j], -j))
        return pl.DataFrame({"trial": list(range(len(m))), "params": [json.dumps(dataclasses.asdict(p)) for p in grid],
                             "mean_ic": m}).with_columns((pl.col("trial") == best).alias("winner"))
    return run


@pytest.mark.parametrize("grid_ic, stage", [(0.04, "objective"), (0.05, "objective"), (0.06, "grid")])
def test_choose_lgbm_keeps_the_objective_winner_unless_a_grid_point_beats_it(monkeypatch, tmp_path, grid_ic, stage):
    from ltforecast.models import tune
    from ltforecast.pipeline import _choose_lgbm

    mcfg = load_model_config()
    mcfg = dataclasses.replace(mcfg, experiments_dir=tmp_path, lgbm=dataclasses.replace(mcfg.lgbm, tune=True))
    base = mcfg.lgbm.params
    # lambdarank at the base point scores 0.05; the grid points on the larger num_leaves / larger learning
    # rate corner score grid_ic, every other grid point 0.0 (the grid never contains the base point)
    ics = {("regression", base.num_leaves, base.learning_rate): 0.01,
           ("lambdarank", base.num_leaves, base.learning_rate): 0.05,
           ("lambdarank", 2 * base.num_leaves + 1, base.learning_rate * 1.5): grid_ic}
    monkeypatch.setattr(tune, "run", _stub_tune_run(ics))
    catalog = SimpleNamespace(model_columns=lambda: ["a"])
    winner, info = _choose_lgbm(SimpleNamespace(frame=None, folds=None), mcfg, catalog)

    assert info["winner_stage"] == stage
    if stage == "objective":
        assert winner == dataclasses.replace(base, objective="lambdarank")
        assert not info["grid_table"]["winner"].any()
    else:
        assert (winner.num_leaves, winner.learning_rate) == (2 * base.num_leaves + 1, base.learning_rate * 1.5)
        assert info["grid_table"]["winner"].sum() == 1
    assert info["winner"] == dataclasses.asdict(winner)

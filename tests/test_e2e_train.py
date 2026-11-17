"""End to end through the CLI on simulated data: ingest synthetic -> build -> labels -> features ->
baselines --fast -> train --fast -> fit-final --fast (plan sections 5 and 6, DESIGN_56 K).

80 stocks, 2000-2016, seed 0. Development = prediction dates through 2013; 2014 is the gap and 2015
the holdout. A tmp model.yaml moves the first test year to 2010 with a 36-month minimum training
window and 12 inner-validation months, so there are four walk-forward folds (2010-2013), and shrinks
the tree / Monte Carlo settings to the size of the data. The assertions are about the machinery
(columns, ranges, purge and holdout discipline, the report), not about forecasting skill.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest
import yaml
from conftest import run_synthetic_pipeline
from typer.testing import CliRunner

import ltforecast.calibrate as calibrate_mod
import ltforecast.levels as levels_mod
import ltforecast.pipeline as pipeline_mod
from ltforecast.cli import app
from ltforecast.dataset import modeling_frame
from ltforecast.features.catalog import load_catalog
from ltforecast.modelconfig import DEFAULT_MODEL_CONFIG, load_model_config
from ltforecast.models.baselines import baseline_factories
from ltforecast.pipeline import (
    FORECAST_COLS,
    SCORE_COLS,
    VOL_TARGET,
    final_dir,
    fit_final,
    load_final,
    load_inputs,
    oof_dir,
    reports_dir,
    score,
)
from ltforecast.store import Store
from ltforecast.validation.experiments import read_runs, trial_count
from ltforecast.validation.splitter import walk_forward_folds

CATALOG = load_catalog()
START, END = dt.date(2000, 1, 3), dt.date(2016, 12, 30)
DEV_END, HOLDOUT_START, HOLDOUT_END = dt.date(2013, 12, 31), dt.date(2015, 1, 1), dt.date(2015, 12, 31)


def small_model_yaml(path) -> None:
    raw = yaml.safe_load(DEFAULT_MODEL_CONFIG.read_text())
    raw["validation"].update(first_test_year=2010, min_train_months=36, inner_valid_months=12)
    for sec in ("lgbm", "quantile", "hit_classifier"):
        raw[sec]["min_data_in_leaf"] = 50
    raw["calibration"].update(min_rows=200, min_bin_rows=20)
    raw["ensemble"]["importance_max_rows"] = 300
    for sec in ("lgbm", "quantile", "hit_classifier"):
        raw["fast"][sec].update(n_estimators=150, early_stopping_rounds=20)
    raw["fast"]["levels"].update(mc_paths=300)
    path.write_text(yaml.safe_dump(raw))


def _max_date(df, col="date"):
    return df[col].max() if col in df.columns and df.height else None


def _label_end_max(df, target):
    rows = df.filter(pl.col(target).is_not_null()) if target in df.columns else df
    ends = [c for c in rows.columns if c.startswith("label_end_")]
    return max((rows[c].max() for c in ends), default=None)


def instrument(mp, rec):
    """Record what each fitting / calibration step of the pipeline actually receives (not what it
    writes in its metadata): call order is preserved per phase via rec["phase"]."""
    def wrap(mod, name, record):
        orig = getattr(mod, name)

        def wrapped(*a, **kw):
            rec[name].append({"phase": rec["phase"], **record(*a, **kw)})
            return orig(*a, **kw)
        mp.setattr(mod, name, wrapped)

    for k in ("build_hit_table", "fit_on_all", "fit_isotonic", "latest_conformal_offset", "barrier_labels"):
        rec[k] = []
    wrap(pipeline_mod, "build_hit_table",
         lambda prices, sigma, through, *a, **kw: {"through": through, "sigma_max": _max_date(sigma)})
    wrap(pipeline_mod, "fit_on_all", lambda make, frame, rules, *, target="target", **kw: {
        "target": target, "date_max": _max_date(frame.filter(pl.col(target).is_not_null())),
        "label_end_max": _label_end_max(frame, target)})
    wrap(calibrate_mod, "fit_isotonic", lambda oof, labels, **kw: {
        "oof_max": _max_date(oof), "labels_max": _max_date(labels), "labels_end": _label_end_max(labels, "target")})
    wrap(calibrate_mod, "latest_conformal_offset", lambda oof_q, labels, **kw: {
        "oof_max": _max_date(oof_q), "labels_max": _max_date(labels)})
    wrap(levels_mod, "barrier_labels", lambda prices, entries, *a, **kw: {"entries_max": _max_date(entries)})

    orig_wf = pipeline_mod.walk_forward

    def wf(make, frame, folds, rules, **kw):  # keep each vol fold model to check the pre-first-fold sigma
        if kw.get("target") != VOL_TARGET:
            return orig_wf(make, frame, folds, rules, **kw)
        inner = kw.pop("on_fold", None)

        def hook(f, m, shown):
            rec["vol_models"][f.name] = m
            if inner is not None:
                inner(f, m, shown)
        return orig_wf(make, frame, folds, rules, on_fold=hook, **kw)
    rec["vol_models"] = {}
    mp.setattr(pipeline_mod, "walk_forward", wf)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e_train")
    cfg, outputs = run_synthetic_pipeline(
        root, n_stocks=80, start=START, end=END, seed=0,
        dev_end=DEV_END, holdout_start=HOLDOUT_START, holdout_end=HOLDOUT_END,
    )
    config = root / "config" / "universe.yaml"
    small_model_yaml(root / "config" / "model.yaml")
    runner = CliRunner()
    rec = {"phase": None}
    with pytest.MonkeyPatch.context() as mp:
        instrument(mp, rec)
        for cmd in ("baselines", "train", "fit-final"):
            rec["phase"] = cmd
            result = runner.invoke(app, [cmd, "--fast", "--config", str(config)])
            assert result.exit_code == 0, f"ltforecast {cmd} failed:\n{result.output}\n{result.exception!r}"
            outputs[cmd] = result.output
    outputs["calls"], outputs["model_yaml"] = rec, root / "config" / "model.yaml"
    mcfg = load_model_config(root / "config" / "model.yaml", fast=True)
    return cfg, mcfg, Store(cfg.data_dir), outputs


@pytest.fixture(scope="module")
def folds(run):
    cfg, mcfg, store, _ = run
    dev = modeling_frame(store, cfg, CATALOG)
    return walk_forward_folds(dev["date"].unique(), mcfg.validation, cfg.dates.dev_end)


@pytest.fixture(scope="module")
def forecast(run):
    return pl.read_parquet(oof_dir(run[2]) / "forecast.parquet")


def test_four_folds_and_synthetic_warning(run, folds):
    _, _, _, outputs = run
    assert [f.name for f in folds] == ["2010", "2011", "2012", "2013"]
    for cmd in ("baselines", "train", "fit-final"):
        assert "SYNTHETIC DATA" in outputs[cmd]


def test_forecast_columns_and_ranges(forecast):
    assert forecast.columns == list(FORECAST_COLS)
    assert forecast.height > 1000
    assert forecast.select(pl.all().null_count()).sum_horizontal().item() == 0
    for c in ("prob", "rank_pct", "p_tp", "p_sl", "p_none"):
        assert forecast[c].min() >= 0.0 and forecast[c].max() <= 1.0, c
    assert (forecast["q10"] <= forecast["q50"]).all() and (forecast["q50"] <= forecast["q90"]).all()
    assert (forecast["sigma_hat"] > 0).all()
    total = (forecast["p_tp"] + forecast["p_sl"] + forecast["p_none"]).to_numpy()
    np.testing.assert_allclose(total, 1.0, atol=1e-9)
    # levels follow the configured guardrails (TP 10-100%, SL 8-40% away from entry)
    assert forecast["tp_level"].is_between(0.10 - 1e-12, 1.0 + 1e-12).all()
    assert forecast["sl_level"].is_between(-0.40 - 1e-12, -0.08 + 1e-12).all()


def test_oof_rows_only_on_fold_test_dates_and_never_past_dev_end(run, folds, forecast):
    cfg, _, store, _ = run
    test_dates = {d for f in folds for d in f.test_dates}
    fold_of = {d: f.name for f in folds for d in f.test_dates}
    assert set(forecast["date"].unique().to_list()) == test_dates
    assert all(fold_of[d] == f for d, f in forecast.select("date", "fold").unique().iter_rows())
    # every OOF / derived file, the baselines included, holds development dates only
    for path in oof_dir(store).glob("*.parquet"):
        dates = pl.read_parquet(path, columns=["date"])["date"]
        assert dates.max() <= cfg.dates.dev_end, path.name
        if path.stem in {"forecast", "quantile_raw", *baseline_factories(CATALOG)}:
            assert set(dates.unique().to_list()) <= test_dates, path.name
    sigma = pl.read_parquet(oof_dir(store) / "sigma_hat.parquet")
    assert set(sigma.filter(pl.col("oof"))["date"].unique().to_list()) == test_dates


def test_nothing_fit_on_gap_or_holdout_rows(run):
    cfg, _, store, _ = run
    meta = json.loads((final_dir(store) / "metadata.json").read_text())
    assert meta["trained_through"] == cfg.dates.dev_end.isoformat()
    assert meta["fit_date_max"] <= cfg.dates.dev_end.isoformat()
    assert meta["synthetic"] is True and meta["fast"] is True
    for side in final_dir(store).glob("*.pkl.json"):
        m = json.loads(side.read_text())
        assert m["trained_through"] == cfg.dates.dev_end.isoformat(), side.name
        if m.get("fit_date_max"):
            assert m["fit_date_max"] <= cfg.dates.dev_end.isoformat(), side.name
        if m.get("fit_date_min"):
            assert m["fit_date_min"] < cfg.dates.dev_end.isoformat(), side.name


def test_baselines_report(run):
    _, _, store, _ = run
    table = pl.read_csv(reports_dir(store) / "baselines.csv")
    assert table["model"].to_list() == list(baseline_factories(CATALOG))
    b0 = table.filter(pl.col("model") == "b0_no_skill").row(0, named=True)
    assert b0["mean_ic"] == 0.0 and b0["ic_ir"] == 0.0  # a constant prediction has IC 0, not NaN
    assert table.filter(pl.col("model") == "b2_composite")["vs_b2_mean_diff"].item() == 0.0


def test_train_report_lists_all_six_criteria(run):
    _, _, store, outputs = run
    md = (reports_dir(store) / "train_report.md").read_text()
    rep = json.loads((reports_dir(store) / "train_report.json").read_text())
    assert [c["n"] for c in rep["criteria"]] == [1, 2, 3, 4, 5, 6]
    assert {c["status"] for c in rep["criteria"]} <= {"pass", "fail"}  # nothing skipped
    for c in rep["criteria"]:
        assert f"| {c['n']} | {c['title']} | **{c['status'].upper()}** |" in md
    assert "SYNTHETIC DATA" in md
    beats = "YES" if rep["beats_b2"] else "NO"
    assert f"Does the ensemble beat B2 (factor composite)? {beats}." in md
    assert rep["recommendation"] == ("ship ensemble" if rep["beats_b2"] else "ship B2")
    assert f"Recommendation: {rep['recommendation']}." in md
    assert f"recommendation: {rep['recommendation']}" in outputs["train"]
    assert rep["deflated_sharpe"]["n_trials"] == rep["trial_count"] >= 10


def test_experiments_logged_with_trial_counts(run):
    _, mcfg, _, _ = run
    runs = read_runs(mcfg.experiments_dir)
    names = runs["name"].to_list()
    for n in ("baseline/b2_composite", "lgbm_objective/trial0", "lgbm_objective/trial1", "ensemble", "vol",
              "quantile", "exit_pairs", "hit_method"):
        assert n in names, n
    # train re-ran the baselines on the same code and config: counted once, not twice
    b2 = runs.filter(pl.col("name") == "baseline/b2_composite")
    assert b2.height == 2 and b2["n_trials"].to_list() == [1, 0]
    assert runs.filter(pl.col("name") == "exit_pairs")["n_trials"].item() == 29  # 25 pairs + 3 trailing + signal
    assert trial_count(mcfg.experiments_dir) == runs["n_trials"].sum()


def test_final_models_score_any_dates_without_labels(run, forecast):
    cfg, _, store, _ = run
    models = load_final(final_dir(store))
    features = store.scan("features").collect()
    holdout_dates = features.filter(pl.col("date") >= cfg.dates.holdout_start)["date"].unique().sort()[:2]
    # a features frame carrying label columns is fine: score reads only the model columns
    labels = store.scan("labels").select("date", "security_id", "target", "excess_12m").collect()
    out = score(models, features.join(labels, on=["date", "security_id"], how="left"),
                store.scan("features_raw").collect(), store.scan("prices_daily"), store.scan("securities").collect(),
                dates=holdout_dates.to_list())
    assert out.columns == list(SCORE_COLS)
    assert out.height > 0 and set(out["date"].unique().to_list()) == set(holdout_dates.to_list())
    assert (out["trained_through"] == cfg.dates.dev_end).all()
    assert out.select(pl.all().null_count()).sum_horizontal().item() == 0
    assert (out["q10"] <= out["q50"]).all() and (out["q50"] <= out["q90"]).all() and (out["sigma_hat"] > 0).all()
    np.testing.assert_allclose((out["p_tp"] + out["p_sl"] + out["p_none"]).to_numpy(), 1.0, atol=1e-9)
    # scoring a past (dev) date is allowed and returns the same column set as the OOF forecast minus fold
    past = score(models, features, store.scan("features_raw").collect(), store.scan("prices_daily"),
                 store.scan("securities").collect(), dates=[forecast["date"].max()])
    assert past.height == forecast.filter(pl.col("date") == forecast["date"].max()).height


# behaviour, not metadata: what the fitting steps were actually given -------------------------------


def _calls(run, name, phase):
    return [c for c in run[3]["calls"][name] if c["phase"] == phase]


def test_hit_tables_use_residuals_up_to_each_folds_last_train_date(run, folds):
    cfg = run[0]
    train_calls = _calls(run, "build_hit_table", "train")
    assert [c["through"] for c in train_calls] == [f.train_dates[-1] for f in folds]
    assert all(c["sigma_max"] <= cfg.dates.dev_end for c in train_calls)
    final = _calls(run, "build_hit_table", "fit-final")
    assert [c["through"] for c in final] == [cfg.dates.dev_end]


def test_final_fits_and_calibration_never_see_gap_or_holdout_rows(run):
    cfg = run[0]
    fits = _calls(run, "fit_on_all", "fit-final")
    assert {c["target"] for c in fits} >= {"target", VOL_TARGET, "excess_w_12m", "outcome"}
    for c in fits:
        assert c["date_max"] <= cfg.dates.dev_end, c
        if c["label_end_max"] is not None:  # every training label ended before the holdout opened
            assert c["label_end_max"] < cfg.dates.holdout_start, c
    for name in ("fit_isotonic", "latest_conformal_offset"):
        (c,) = _calls(run, name, "fit-final")
        assert c["oof_max"] <= cfg.dates.dev_end and c["labels_max"] <= cfg.dates.dev_end, (name, c)
    (iso,) = _calls(run, "fit_isotonic", "fit-final")
    assert iso["labels_end"] < cfg.dates.holdout_start
    for phase in ("train", "fit-final"):
        for c in _calls(run, "barrier_labels", phase):
            assert c["entries_max"] <= cfg.dates.dev_end, (phase, c)


def test_pre_first_fold_sigma_comes_from_the_first_fold_vol_model(run, folds):
    cfg, mcfg, store, outputs = run
    sigma = pl.read_parquet(oof_dir(store) / "sigma_hat.parquet")
    pre = sigma.filter(~pl.col("oof")).sort("date", "security_id")
    assert pre.height > 0 and pre["date"].max() < folds[0].test_dates[0]
    from ltforecast.models.vol import vol_frame

    inp = load_inputs(store, cfg, mcfg, CATALOG)
    vf = vol_frame(inp.features_raw, inp.frame, inp.prices, inp.securities)
    vf = vf.filter(pl.col("date") < folds[0].test_dates[0]).sort("date", "security_id")
    assert vf.select("date", "security_id").equals(pre.select("date", "security_id"))
    want = outputs["calls"]["vol_models"][folds[0].name].predict(vf)
    np.testing.assert_allclose(pre["sigma_hat"].to_numpy(), want, rtol=1e-12)
    later = outputs["calls"]["vol_models"][folds[-1].name].predict(vf)
    assert not np.allclose(later, want)  # the check can tell the fold models apart


def test_fit_final_refuses_a_train_state_from_a_different_run(run):
    cfg, mcfg, store, _ = run
    full = load_model_config(run[3]["model_yaml"])  # same yaml, not --fast
    with pytest.raises(ValueError, match="fast"):
        fit_final(store, cfg, full, CATALOG)
    state_path = oof_dir(store) / "train_state.json"
    original = state_path.read_text()
    try:
        state = json.loads(original)
        first = sorted(state["data_fingerprint"])[0]
        state["data_fingerprint"][first] = "0:0"
        state_path.write_text(json.dumps(state))
        with pytest.raises(ValueError, match="data changed"):
            fit_final(store, cfg, mcfg, CATALOG)
    finally:
        state_path.write_text(original)


def test_score_of_a_subset_equals_the_full_score_filtered(run):
    cfg, _, store, _ = run
    models = load_final(final_dir(store))
    assert models.min_cross_section >= 10
    features = store.scan("features").collect()
    raw, prices = store.scan("features_raw").collect(), store.scan("prices_daily")
    secs = store.scan("securities").collect()
    day = features.filter(pl.col("date") >= cfg.dates.holdout_start)["date"].min()
    full = score(models, features, raw, prices, secs, dates=[day])
    top = full.sort("pred", descending=True)["security_id"].head(5).to_list()
    sub = score(models, features, raw, prices, secs, dates=[day], security_ids=top)
    assert sub.equals(full.filter(pl.col("security_id").is_in(top)))
    assert sub["rank_pct"].min() > 0.85  # still ranked against the whole universe (alone: 0.1 to 0.9)
    with pytest.raises(ValueError, match="cross-section"):  # one ticker alone is refused
        score(models, features.filter(pl.col("security_id") == top[0]), raw, prices, secs, dates=[day])

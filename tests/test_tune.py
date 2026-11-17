"""Tuning grid and the logged, selection-guarded tuning run."""

import dataclasses
import json

import polars as pl
import pytest
from test_lgbm import COLS, DATES, RULES, TINY, panel

import ltforecast.models.tune as tune
from ltforecast.models.lgbm import LGBMParams
from ltforecast.validation.experiments import read_runs, selection_guard, trial_count
from ltforecast.validation.splitter import walk_forward_folds


def test_default_grid_is_the_plan_ranges():
    g = tune.grid(LGBMParams())
    assert len(g) == 16 and len(set(g)) == 16
    assert {p.num_leaves for p in g} == {15, 31}
    assert {p.min_data_in_leaf for p in g} == {500, 2000}
    assert sorted({p.learning_rate for p in g}) == pytest.approx([0.01, 0.03])
    assert sorted({p.feature_fraction for p in g}) == pytest.approx([0.5, 0.7])
    assert {p.objective for p in g} == {"regression"}
    assert {p.lambda_l2 for p in g} == {10.0}


def test_grid_stays_small_for_any_base():
    for base in (TINY, LGBMParams(objective="lambdarank", min_data_in_leaf=1, feature_fraction=1.0)):
        g = tune.grid(base)
        assert 1 <= len(g) <= 16 and len(set(g)) == len(g)
        assert all(0 < p.feature_fraction <= 1 and p.min_data_in_leaf >= 1 for p in g)
        assert {p.objective for p in g} == {base.objective}
    assert [p.objective for p in tune.objectives(TINY)] == ["regression", "lambdarank"]
    assert 2 + len(tune.grid(LGBMParams())) <= 50


def test_run_logs_every_trial_and_picks_winner_by_selection_guard(tmp_path, monkeypatch):
    df = panel()
    folds = walk_forward_folds(DATES, RULES, DATES[-1])
    calls = []

    def guard(per_fold):
        calls.append(dict(per_fold))
        return selection_guard(per_fold)

    monkeypatch.setattr(tune, "selection_guard", guard)
    weak = dataclasses.replace(TINY, num_leaves=2, max_depth=1, learning_rate=0.001, n_estimators=3,
                               early_stopping_rounds=1)
    candidates = [weak, TINY, dataclasses.replace(TINY, objective="lambdarank")]
    table = tune.run(df, folds, RULES, candidates, tmp_path, cols=COLS)

    assert table["trial"].to_list() == [0, 1, 2]
    assert {"params", "mean_ic", "ic_2005", "ic_2006", "ic_2007", "winner"} <= set(table.columns)
    assert len(calls) == 3 and all(set(c) == {"2005", "2006", "2007"} for c in calls)
    for row, c in zip(table.iter_rows(named=True), calls):
        assert row["mean_ic"] == pytest.approx(sum(c.values()) / 3)
        assert [row[f"ic_{k}"] for k in c] == pytest.approx(list(c.values()))
    assert table["winner"].sum() == 1
    win = table.filter(pl.col("winner"))
    assert win["mean_ic"].item() == table["mean_ic"].max() and win["trial"].item() != 0
    assert tune.best(table, candidates) == candidates[win["trial"].item()]
    assert json.loads(table["params"][2])["objective"] == "lambdarank"

    runs = read_runs(tmp_path)
    assert runs["name"].to_list() == ["lgbm_tune/trial0", "lgbm_tune/trial1", "lgbm_tune/trial2"]
    assert trial_count(tmp_path) == 3
    logged = json.loads(runs["metrics_json"][1])
    assert logged["mean_ic"] == pytest.approx(table["mean_ic"][1])


def test_run_needs_three_folds(tmp_path):
    df = panel()
    folds = walk_forward_folds(DATES, RULES, DATES[-1])[:2]
    with pytest.raises(ValueError, match="at least 3"):
        tune.run(df, folds, RULES, [TINY], tmp_path, cols=COLS)
    with pytest.raises(ValueError, match="empty"):
        tune.run(df, folds, RULES, [], tmp_path, cols=COLS)


def _table(ics):
    best = max(range(len(ics)), key=lambda j: (ics[j], -j))
    return pl.DataFrame({"trial": list(range(len(ics))), "mean_ic": ics}).with_columns(
        (pl.col("trial") == best).alias("winner"))


def test_choose_keeps_objective_winner_when_it_beats_every_grid_point():
    objs = tune.objectives(LGBMParams())
    t_obj = _table([0.050, 0.030])  # regression base wins the objective stage at 0.050
    grid = tune.grid(objs[0])
    t_grid = _table([0.045 + 0.0002 * i for i in range(len(grid))])  # best grid point 0.048 < 0.050
    p, stage, t_grid2 = tune.choose(t_obj, objs, t_grid, grid)
    assert p == objs[0] and stage == "objective"
    assert t_grid2["winner"].sum() == 0  # no grid row is flagged as the shipped config


def test_choose_takes_grid_winner_only_when_strictly_better():
    objs = tune.objectives(LGBMParams())
    t_obj = _table([0.050, 0.030])
    grid = tune.grid(objs[0])
    tie = [0.040] * len(grid)
    tie[3] = 0.050  # ties the objective winner: the earlier-evaluated config keeps it
    p, stage, _ = tune.choose(t_obj, objs, _table(tie), grid)
    assert p == objs[0] and stage == "objective"
    tie[3] = 0.051
    p, stage, t_grid2 = tune.choose(t_obj, objs, _table(tie), grid)
    assert p == grid[3] and stage == "grid"
    assert t_grid2["winner"].to_list() == [i == 3 for i in range(len(grid))]

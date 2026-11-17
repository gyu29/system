"""Small LightGBM tuning grid (plan 6.1: <= 50 trials in all, every one counted).

`objectives(base)` compares regression and lambdarank at the base point (2 trials); `grid(base)` is
then a 2^4 = 16-point grid around the base of the chosen objective, so the whole search is <= 18
LightGBM trials. `run` walks every candidate forward through the purged runner, logs it with
`experiments.log_new(n_trials=1)` as soon as it has been scored (a re-run of the same config on the
same code logs 0), and ranks candidates only by
`experiments.selection_guard` (the mean of the per-fold mean daily IC; never one fold).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ltforecast.modelconfig import ValidationRules
from ltforecast.models.lgbm import LGBM, LGBMParams
from ltforecast.validation.experiments import log_new, selection_guard
from ltforecast.validation.metrics import ic_by_date
from ltforecast.validation.runner import walk_forward
from ltforecast.validation.splitter import Fold

MAX_GRID = 16


def objectives(base: LGBMParams) -> list[LGBMParams]:
    """The two objectives at the base point: [regression, lambdarank]."""
    return [dataclasses.replace(base, objective=o) for o in ("regression", "lambdarank")]


def grid(base: LGBMParams) -> list[LGBMParams]:
    """2 values each of num_leaves, min_data_in_leaf, learning_rate, feature_fraction around `base`.

    With the default base this is exactly the plan's ranges: num_leaves {15, 31}, min_data_in_leaf
    {500, 2000}, learning_rate {0.01, 0.03}, feature_fraction {0.5, 0.7}. Everything else (objective,
    max_depth, bagging, l2, caps) stays at the base. Duplicates are dropped; at most 16 points.
    """
    leaves = (base.num_leaves, 2 * base.num_leaves + 1)
    min_leaf = (max(1, base.min_data_in_leaf // 2), base.min_data_in_leaf * 2)
    lr = (base.learning_rate / 2, base.learning_rate * 1.5)
    ff = (max(0.05, round(base.feature_fraction - 0.1, 6)), min(1.0, round(base.feature_fraction + 0.1, 6)))
    out: list[LGBMParams] = []
    for nl, md, rate, frac in itertools.product(leaves, min_leaf, lr, ff):
        p = dataclasses.replace(base, num_leaves=nl, min_data_in_leaf=md, learning_rate=rate, feature_fraction=frac)
        if p not in out:
            out.append(p)
    assert len(out) <= MAX_GRID
    return out


def _fold_ics(oof: pl.DataFrame, frame: pl.DataFrame, target: str, min_n: int) -> dict[str, float]:
    """Per fold: mean over its test dates of the daily rank IC (NaN if no date qualifies)."""
    scored = oof.join(frame.select("date", "security_id", target), on=["date", "security_id"], how="left")
    out = {}
    for fold in scored["fold"].unique(maintain_order=True).to_list():
        ic = ic_by_date(scored.filter(pl.col("fold") == fold), "pred", target, min_n=min_n)
        out[fold] = float(ic["ic"].mean()) if ic.height else float("nan")
    return out


def run(
    frame: pl.DataFrame,
    folds: Sequence[Fold],
    rules: ValidationRules,
    grid: Sequence[LGBMParams],
    exp_dir: str | Path,
    *,
    cols: Sequence[str],
    seed: int = 0,
    target: str = "target",
    label_end_col: str = "label_end_12m",
    min_n: int = 10,
    name: str = "lgbm_tune",
) -> pl.DataFrame:
    """Walk every candidate forward; one row per trial in grid order.

    Columns: trial (0-based), params (canonical JSON of the LGBMParams), mean_ic (selection_guard of
    the per-fold ICs), ic_<fold> per fold, winner (True on exactly one row: the highest mean_ic, the
    earliest trial on a tie). Each trial is logged to `exp_dir` with n_trials=1 right after scoring.
    `selection_guard` raises if there are fewer than 3 folds or a fold could not be scored.
    """
    if not grid:
        raise ValueError("empty tuning grid")
    cols = list(cols)
    rows = []
    for i, p in enumerate(grid):
        oof = walk_forward(
            lambda p=p: LGBM(cols, p, seed, name=f"{name}_{i}", target=target), frame, folds, rules,
            target=target, label_end_col=label_end_col,
        )
        per_fold = _fold_ics(oof, frame, target, min_n)
        mean_ic = selection_guard(per_fold)
        params = dataclasses.asdict(p)
        log_new(
            exp_dir, f"{name}/trial{i}",
            {"model": "lgbm", "params": params, "seed": seed, "cols": cols, "target": target},
            {"mean_ic": mean_ic, "fold_ic": per_fold},
            n_trials=1,
        )
        rows.append({"trial": i, "params": json.dumps(params, sort_keys=True), "mean_ic": mean_ic}
                    | {f"ic_{k}": v for k, v in per_fold.items()})
    table = pl.DataFrame(rows)
    best = max(range(len(rows)), key=lambda j: (rows[j]["mean_ic"], -j))
    return table.with_columns((pl.col("trial") == best).alias("winner"))


def best(table: pl.DataFrame, grid: Sequence[LGBMParams]) -> LGBMParams:
    """The winning LGBMParams of a `run` table (grid = the list passed to run)."""
    return grid[int(table.filter(pl.col("winner"))["trial"].item())]


def choose(
    t_obj: pl.DataFrame, objs: Sequence[LGBMParams], t_grid: pl.DataFrame, grid: Sequence[LGBMParams],
) -> tuple[LGBMParams, str, pl.DataFrame]:
    """The overall winner across the objective stage and the grid (plan 6.1: highest mean IC).

    `grid(base)` never contains its base, so the objective-stage winner (already scored on the same
    folds and counted as a trial) must stay in the comparison. The grid winner replaces it only if its
    mean_ic is strictly higher (the earlier-evaluated config wins ties). Returns (params, stage, t_grid)
    with stage "objective" or "grid"; when the objective winner is kept, the returned grid table's
    `winner` column is all False so no grid row is reported as the shipped config.
    """
    obj_ic = float(t_obj.filter(pl.col("winner"))["mean_ic"].item())
    grid_ic = float(t_grid.filter(pl.col("winner"))["mean_ic"].item())
    if grid_ic > obj_ic:
        return best(t_grid, grid), "grid", t_grid
    return best(t_obj, objs), "objective", t_grid.with_columns(pl.lit(False).alias("winner"))

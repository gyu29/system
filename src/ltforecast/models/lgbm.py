"""LightGBM point forecast (plan 6.1): regression on the rank label or lambdarank by month.

Conventions
- Inputs are the modeling-frame columns `cols`, passed to LightGBM as a float64 matrix in that fixed
  order; nulls become NaN (trees route missing values themselves). Label columns are refused as inputs.
- Training rows are sorted by (date, security_id) first, so the fit does not depend on the row order
  it is handed, and lambdarank's query groups (one per date) are contiguous.
- Early stopping, for both objectives, is on the mean per-date Spearman IC of the valid set (custom
  feval, higher is better). The per-date ranks use average ranks for ties, and a date on which the
  prediction or the target is constant scores IC 0, as in `validation.metrics`.
- `refit=True`: after early stopping on (train, valid) the model is refit on `full` (the runner's
  refit frame: every labelled row of the training dates, including the purged months between train
  and valid) or, when `full` is None, on train + valid, for `best_iteration` rounds with the same seed;
  without a valid set it trains `n_estimators_no_valid` on train. Early stopping never sees `full`.
  `refit_from(train, valid, full=)` turns an early-stopped refit=False model into that refit=True model
  without repeating the early-stopping pass (used by the ensemble).
- lambdarank: label = decile 0..9 of `target` within the date (ceil(avg rank * 10 / n) - 1),
  label_gain = 0..9, query size <= 10_000 rows (LightGBM's limit).
- `predict` returns the raw booster score; `SeedAverage.predict` returns the mean of the members'
  per-date centered ranks ((avg rank - 0.5) / n - 0.5), which needs the `date` column (not a label).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from ltforecast.dataset import label_columns

OBJECTIVES = ("regression", "lambdarank")
N_RANK_LABELS = 10
MAX_QUERY_ROWS = 10_000  # LightGBM refuses lambdarank queries longer than this


@dataclass(frozen=True)
class LGBMParams:
    objective: str = "regression"        # "regression" on the rank label | "lambdarank"
    num_leaves: int = 15
    max_depth: int = 5
    min_data_in_leaf: int = 1000
    learning_rate: float = 0.02
    feature_fraction: float = 0.6
    bagging_fraction: float = 0.7
    bagging_freq: int = 1
    lambda_l2: float = 10.0
    n_estimators: int = 2000             # cap; early stopping decides
    n_estimators_no_valid: int = 300     # used when no validation set is given
    early_stopping_rounds: int = 100
    refit: bool = True                   # after early stopping, refit on train + valid for best_iteration rounds
    num_threads: int = 0
    # lambdarank only: pairs are formed with the top-k of the current ranking; cost ~ rows x k per round
    lambdarank_truncation_level: int = 100

    def __post_init__(self):
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")


# per-group ranks and IC (vectorized) -------------------------------------------------------------


def _codes(dates: pl.Series | np.ndarray) -> np.ndarray:
    """Dense 0-based group codes in ascending date order."""
    s = dates if isinstance(dates, pl.Series) else pl.Series(dates)
    return (s.rank("dense") - 1).cast(pl.Int64).to_numpy()


def group_avg_rank(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """1-based average ranks of `x` within each group code `g` (ties share their mean rank)."""
    n = len(x)
    if n == 0:
        return np.empty(0)
    order = np.lexsort((x, g))
    xs, gs = x[order], g[order]
    gnew = np.ones(n, dtype=bool)
    gnew[1:] = gs[1:] != gs[:-1]
    new = gnew.copy()
    new[1:] |= xs[1:] != xs[:-1]
    block = np.cumsum(new) - 1
    start = np.flatnonzero(new)
    end = np.append(start[1:], n)
    group_start = np.flatnonzero(gnew)[np.cumsum(gnew) - 1]
    ranks = np.empty(n)
    ranks[order] = (start + end - 1)[block] / 2.0 - group_start + 1.0
    return ranks


def per_date_centered_rank(pred: np.ndarray, dates: pl.Series | np.ndarray) -> np.ndarray:
    """(avg rank - 0.5) / n - 0.5 of `pred` within each date, strictly inside (-0.5, 0.5)."""
    pred = np.asarray(pred, dtype=np.float64)
    g = _codes(dates)
    n = np.bincount(g)[g] if len(g) else np.empty(0)
    return (group_avg_rank(pred, g) - 0.5) / n - 0.5


class _ICScorer:
    """Mean per-date Spearman IC of predictions against a fixed target; the target side is ranked once."""

    def __init__(self, target: np.ndarray, g: np.ndarray, min_n: int = 3):
        self.g = g
        self.size = np.bincount(g).astype(np.float64)
        self.keep = self.size >= min_n
        self.mid = (self.size + 1.0) / 2.0
        self.db = group_avg_rank(target, g) - self.mid[g]
        self.syy = np.bincount(g, self.db * self.db, minlength=len(self.size))

    def __call__(self, pred: np.ndarray) -> float:
        if not self.keep.any():
            return 0.0
        da = group_avg_rank(np.asarray(pred, dtype=np.float64), self.g) - self.mid[self.g]
        m = len(self.size)
        sxx = np.bincount(self.g, da * da, minlength=m)
        sxy = np.bincount(self.g, da * self.db, minlength=m)
        den = np.sqrt(sxx * self.syy)
        ic = np.divide(sxy, den, out=np.zeros(m), where=den > 1e-12)
        return float(ic[self.keep].mean())


def mean_date_ic(pred: np.ndarray, target: np.ndarray, dates: pl.Series | np.ndarray, min_n: int = 3) -> float:
    """Mean over dates (with >= min_n rows) of the Spearman IC of pred vs target; 0.0 if none."""
    return _ICScorer(np.asarray(target, dtype=np.float64), _codes(dates), min_n)(pred)


def rank_labels(target: np.ndarray, g: np.ndarray, n_labels: int = N_RANK_LABELS) -> np.ndarray:
    """Integer bucket 0..n_labels-1 of `target` within each group: ceil(avg rank * n_labels / n) - 1."""
    size = np.bincount(g)[g]
    lab = np.ceil(group_avg_rank(np.asarray(target, dtype=np.float64), g) * n_labels / size) - 1
    return np.clip(lab, 0, n_labels - 1).astype(np.int32)


# the model ---------------------------------------------------------------------------------------


@dataclass
class _Prepared:
    X: np.ndarray
    y: np.ndarray                # target values (the regression label and the feval reference)
    g: np.ndarray                # dense date codes, ascending (rows are sorted by date)
    group_sizes: np.ndarray      # rows per date in order (lambdarank query sizes)


class LGBM:
    """One LightGBM booster on `cols` (implements validation.runner.Model)."""

    def __init__(
        self, cols: Sequence[str], params: LGBMParams = LGBMParams(), seed: int = 0, name: str = "lgbm",
        *, target: str = "target",
    ):
        cols = list(cols)
        if not cols or len(set(cols)) != len(cols):
            raise ValueError("cols must be a non-empty list of distinct column names")
        leaked = sorted(set(label_columns(cols)) | ({target} & set(cols)))
        if leaked:
            raise ValueError(f"label columns cannot be model inputs: {leaked}")
        self.cols = cols
        self.params = params
        self.seed = int(seed)
        self.name = name
        self.target = target
        self.booster: lgb.Booster | None = None
        self.best_iteration: int = 0
        self.valid_ic: float | None = None     # best early-stopping score (None without a valid set)
        self.expected_value: float = 0.0       # TreeSHAP bias term

    # inputs

    def _matrix(self, df: pl.DataFrame) -> np.ndarray:
        missing = [c for c in self.cols if c not in df.columns]
        if missing:
            raise KeyError(f"{self.name}: frame lacks model columns {missing}")
        # nulls (and non-finite values) -> NaN, the missing-value marker for trees
        exprs = [pl.col(c).cast(pl.Float64).fill_nan(None) for c in self.cols]
        return df.select(exprs).to_numpy().astype(np.float64, copy=False)

    def _prepare(self, df: pl.DataFrame) -> _Prepared:
        df = df.filter(pl.col(self.target).is_not_null() & pl.col(self.target).is_finite())
        keys = ["date", "security_id"] if "security_id" in df.columns else ["date"]
        df = df.sort(keys, maintain_order=True)
        g = _codes(df["date"])
        sizes = np.bincount(g) if len(g) else np.empty(0, dtype=np.int64)
        if self.params.objective == "lambdarank" and len(sizes) and sizes.max() > MAX_QUERY_ROWS:
            raise ValueError(f"lambdarank: a date has {int(sizes.max())} rows; LightGBM allows {MAX_QUERY_ROWS}")
        return _Prepared(self._matrix(df), df[self.target].cast(pl.Float64).to_numpy(), g, sizes)

    def _lgb_params(self) -> dict:
        p = self.params
        out = {
            "objective": p.objective, "num_leaves": p.num_leaves, "max_depth": p.max_depth,
            "min_data_in_leaf": p.min_data_in_leaf, "learning_rate": p.learning_rate,
            "feature_fraction": p.feature_fraction, "bagging_fraction": p.bagging_fraction,
            "bagging_freq": p.bagging_freq, "lambda_l2": p.lambda_l2, "num_threads": p.num_threads,
            "seed": self.seed, "deterministic": True, "force_row_wise": True, "verbose": -1, "metric": "None",
            "feature_pre_filter": False,
        }
        if p.objective == "lambdarank":
            out |= {"label_gain": list(range(N_RANK_LABELS)),
                    "lambdarank_truncation_level": p.lambdarank_truncation_level}
        return out

    def _dataset(self, d: _Prepared, reference: lgb.Dataset | None = None) -> lgb.Dataset:
        ranking = self.params.objective == "lambdarank"
        return lgb.Dataset(
            d.X, label=rank_labels(d.y, d.g) if ranking else d.y, group=d.group_sizes if ranking else None,
            feature_name=self.cols, reference=reference, params={"verbose": -1}, free_raw_data=True,
        )

    # fit / predict

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None, *, full: pl.DataFrame | None = None) -> LGBM:
        p, params = self.params, self._lgb_params()
        tr = self._prepare(train)
        if len(tr.y) == 0:
            raise ValueError(f"{self.name}: no labelled training rows")
        va = self._prepare(valid) if valid is not None else None
        if va is None or len(va.y) == 0:
            self.booster = lgb.train(params, self._dataset(tr), num_boost_round=p.n_estimators_no_valid)
            self.best_iteration, self.valid_ic = p.n_estimators_no_valid, None
        else:
            scorer = _ICScorer(va.y, va.g)
            dtrain = self._dataset(tr)
            booster = lgb.train(
                params, dtrain, num_boost_round=p.n_estimators,
                valid_sets=[self._dataset(va, reference=dtrain)], valid_names=["valid"],
                feval=lambda preds, _data: ("ic", scorer(preds), True),
                callbacks=[lgb.early_stopping(p.early_stopping_rounds, first_metric_only=True, verbose=False)],
            )
            self.best_iteration = int(booster.best_iteration) or booster.current_iteration()
            self.valid_ic = float(booster.best_score["valid"]["ic"])
            if p.refit:
                self._train_full(train, valid, full)
            else:
                self.booster = lgb.Booster(model_str=booster.model_to_string(num_iteration=self.best_iteration))
        self.expected_value = float(self.booster.predict(tr.X[:1], pred_contrib=True)[0, -1])
        return self

    def _train_full(self, train: pl.DataFrame, valid: pl.DataFrame, full: pl.DataFrame | None = None) -> None:
        """Booster on `full` (else train + valid) for `best_iteration` rounds (no early stopping)."""
        both = self._prepare(full if full is not None else pl.concat([train, valid], how="diagonal_relaxed"))
        self.booster = lgb.train(self._lgb_params(), self._dataset(both), num_boost_round=self.best_iteration)

    def refit_from(self, train: pl.DataFrame, valid: pl.DataFrame, *, full: pl.DataFrame | None = None) -> LGBM:
        """A new refit=True LGBM built from this early-stopped (refit=False) one, reusing its
        `best_iteration` / `valid_ic` instead of re-running the deterministic early-stopping pass.
        Equals `LGBM(..., refit=True).fit(train, valid, full=full)` for the same frames."""
        if self.booster is None or self.valid_ic is None:
            raise RuntimeError(f"{self.name}: refit_from needs a model fit with a validation set")
        out = LGBM(self.cols, dataclasses.replace(self.params, refit=True), self.seed, self.name,
                   target=self.target)
        out.best_iteration, out.valid_ic = self.best_iteration, self.valid_ic
        out._train_full(train, valid, full)
        out.expected_value = float(out.booster.predict(out._prepare(train).X[:1], pred_contrib=True)[0, -1])
        return out

    def _booster(self) -> lgb.Booster:
        if self.booster is None:
            raise RuntimeError(f"{self.name}: fit before predict")
        return self.booster

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """Raw booster score per row, in the frame's row order."""
        if df.height == 0:
            return np.empty(0)
        return np.asarray(self._booster().predict(self._matrix(df)), dtype=np.float64)

    def contributions(self, df: pl.DataFrame) -> np.ndarray:
        """TreeSHAP contributions (n, len(cols)); rows sum to predict(df) - expected_value."""
        if df.height == 0:
            return np.empty((0, len(self.cols)))
        contrib = np.asarray(self._booster().predict(self._matrix(df), pred_contrib=True), dtype=np.float64)
        self.expected_value = float(contrib[0, -1])
        return contrib[:, :-1]

    def feature_importance(self, kind: str = "gain") -> dict[str, float]:
        return dict(zip(self.cols, self._booster().feature_importance(kind).astype(float).tolist()))


class SeedAverage:
    """K LGBMs that differ only by seed; predict = mean of their per-date centered ranks."""

    def __init__(
        self, cols: Sequence[str], params: LGBMParams = LGBMParams(), seeds: Sequence[int] = (0, 1, 2, 3, 4),
        name: str = "lgbm", *, target: str = "target",
    ):
        seeds = [int(s) for s in seeds]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be a non-empty list of distinct integers")
        self.cols, self.params, self.seeds, self.name, self.target = list(cols), params, seeds, name, target
        self.members: list[LGBM] = [LGBM(cols, params, s, f"{name}_s{s}", target=target) for s in seeds]

    def with_params(self, **changes) -> SeedAverage:
        """An unfitted copy with some LGBMParams fields replaced (e.g. refit=False for the ensemble)."""
        return SeedAverage(self.cols, dataclasses.replace(self.params, **changes), self.seeds, self.name,
                           target=self.target)

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None, *, full: pl.DataFrame | None = None
            ) -> SeedAverage:
        for m in self.members:
            m.fit(train, valid, full=full)
        return self

    def refit_from(self, train: pl.DataFrame, valid: pl.DataFrame, *, full: pl.DataFrame | None = None
                   ) -> SeedAverage:
        """refit=True copy of this early-stopped average: each seed reuses its best_iteration (LGBM.refit_from)."""
        out = self.with_params(refit=True)
        out.members = [m.refit_from(train, valid, full=full) for m in self.members]
        return out

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if df.height == 0:
            return np.empty(0)
        if "date" not in df.columns:
            raise KeyError(f"{self.name}: predict needs the date column to rank within dates")
        dates = df["date"]
        return np.mean([per_date_centered_rank(m.predict(df), dates) for m in self.members], axis=0)

    def contributions(self, df: pl.DataFrame) -> np.ndarray:
        """Mean TreeSHAP across seeds (raw-score scale); `expected_value` = mean bias."""
        out = np.mean([m.contributions(df) for m in self.members], axis=0)
        self.expected_value = float(np.mean([m.expected_value for m in self.members]))
        return out

    @property
    def best_iteration(self) -> list[int]:
        return [m.best_iteration for m in self.members]

"""The blended point forecast (plan 6.1): LightGBM seed average + ridge (B3) + factor composite (B2),
weighted on the inner validation set and shrunk toward equal weights.

Conventions
- A member factory takes one bool, `refit`. `refit=False` must give a model whose valid-set predictions
  are out of sample (fit on train only, valid used at most for early stopping / hyperparameter choice);
  `refit=True` gives the final member, refit on train + valid. A selection-step member that has
  `refit_from(train, valid)` (LGBM / SeedAverage) is turned into the final member with it, so its
  early-stopping pass is not run twice; other members are rebuilt with factory(True).
- Every member's score is turned into a per-date centered rank ((avg rank - 0.5) / n - 0.5, the
  `transform.xs_rank` convention) before blending, so members on different scales are comparable.
- Weights live on the simplex grid {0, step, 2 step, ..., 1}^K summing to 1. The grid point with the
  best mean per-date Spearman IC on the valid set wins (ties: closest to equal weights, then first in
  grid order); the final weights are (1 - shrinkage) * w_best + shrinkage * equal.
- `predict(df)` needs the `date` column (not a label) and returns the weighted mean of the members'
  per-date centered ranks.
"""

from __future__ import annotations
import dataclasses
import functools
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge as _SkRidge
from ltforecast.dataset import label_columns
from ltforecast.features.catalog import Catalog
from ltforecast.models.baselines import BaselineParams, FactorComposite, Ridge
from ltforecast.models.lgbm import LGBMParams, SeedAverage, _codes, _ICScorer, mean_date_ic, per_date_centered_rank
from ltforecast.validation.runner import Model

MAX_GRID_POINTS = 200_000  # guard against an accidental huge simplex (many members, tiny step)


def simplex_grid(k: int, step: float) -> np.ndarray:
    """(G, k) weight vectors with entries in multiples of `step` summing to 1, in lexicographic order
    of the integer compositions (so the first point puts everything on the last member)."""
    if k < 1:
        raise ValueError("simplex_grid: need at least one member")
    n = round(1.0 / step)
    if not 0 < step <= 1 or abs(n * step - 1.0) > 1e-9:
        raise ValueError(f"step must divide 1 exactly (e.g. 0.1, 0.25), got {step}")
    if math.comb(n + k - 1, k - 1) > MAX_GRID_POINTS:
        raise ValueError(f"simplex grid of {k} members at step {step} has more than {MAX_GRID_POINTS} points")
    # stars and bars: choose k - 1 bar positions among n + k - 1 slots
    rows = []
    for bars in itertools.combinations(range(n + k - 1), k - 1):
        edges = (-1, *bars, n + k - 1)
        rows.append([edges[i + 1] - edges[i] - 1 for i in range(k)])
    return np.asarray(rows, dtype=np.float64) / n


class TrainOnlyRidge(Ridge):
    """B3 for the weight-selection step: alpha chosen on valid exactly as `Ridge`, then fit on train
    ONLY, so its valid-set predictions are out of sample (plain `Ridge` refits on train + valid)."""

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None) -> TrainOnlyRidge:
        super().fit(train, valid)  # picks alpha_ (and valid_ic_)
        x, y = self._xy(train.select(*self.cols, self.target))
        if y.size == 0:
            raise ValueError("ridge: no labelled training rows")
        self._model = _SkRidge(alpha=self.alpha_, fit_intercept=True).fit(x, y)
        return self


def _lgbm_member(cols: Sequence[str], params: LGBMParams, seeds: Sequence[int], target: str,
                 refit: bool) -> SeedAverage:
    return SeedAverage(cols, dataclasses.replace(params, refit=refit), seeds, "lgbm", target=target)


def _ridge_member(cols: Sequence[str], alphas: Sequence[float], target: str, ic_min_n: int,
                  refit: bool) -> Ridge:
    cls = Ridge if refit else TrainOnlyRidge
    return cls(cols, alphas, target=target, ic_min_n=ic_min_n, name="ridge")


def _composite_member(catalog: Catalog, refit: bool) -> FactorComposite:
    return FactorComposite(catalog, name="composite")


def default_factories(
    catalog: Catalog,
    *,
    lgbm_cols: Sequence[str] | None = None,
    lgbm_params: LGBMParams = LGBMParams(),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    baseline_params: BaselineParams | None = None,
    target: str = "target",
) -> dict[str, Callable[[bool], Model]]:
    """The plan's members: `lgbm` (SeedAverage over `seeds` on `lgbm_cols`, default
    catalog.model_columns()), `ridge` (B3 on catalog.linear_columns()), `composite` (B2).
    Factories are functools.partial objects (they pickle) taking `refit`."""
    bp = baseline_params or BaselineParams()
    cols = list(lgbm_cols) if lgbm_cols is not None else catalog.model_columns()
    return {
        "lgbm": functools.partial(_lgbm_member, cols, lgbm_params, tuple(seeds), target),
        "ridge": functools.partial(_ridge_member, catalog.linear_columns(), bp.ridge_alphas, target, bp.ic_min_n),
        "composite": functools.partial(_composite_member, catalog),
    }


class Ensemble:
    """Rank-blend of member models (implements validation.runner.Model).

    After fit: `members_` {name: fitted refit=True model}, `weights_` {name: final weight},
    `w_best_` {name: grid weight before shrinkage} (None without a valid set), `valid_ic_` (best grid
    IC, None without a valid set), `member_valid_ic_` {name: valid IC of the member alone}.
    """

    def __init__(
        self,
        factories: Mapping[str, Callable[[bool], Model]],
        step: float = 0.1,
        shrinkage: float = 0.5,
        *,
        target: str = "target",
        ic_min_n: int = 10,
        name: str = "ensemble",
    ):
        if not factories:
            raise ValueError("ensemble: no members")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
        self.factories = dict(factories)
        self.names = list(self.factories)
        self.grid = simplex_grid(len(self.names), step)
        self.step, self.shrinkage = float(step), float(shrinkage)
        self.target, self.ic_min_n, self.name = target, int(ic_min_n), name
        self.members_: dict[str, Model] = {}
        self.weights_: dict[str, float] = {}
        self.w_best_: dict[str, float] | None = None
        self.valid_ic_: float | None = None
        self.member_valid_ic_: dict[str, float] = {}

    def _hide_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        hidden = (set(label_columns(df.columns)) | {self.target}) & set(df.columns)
        return df.drop(sorted(hidden))

    def choose_weights(self, ranks: np.ndarray, target: np.ndarray, dates: pl.Series) -> tuple[np.ndarray, float]:
        """Best simplex-grid weights for member centered ranks `ranks` (n, K) against `target`.
        Returns (w_best, its mean per-date IC). Ties go to the point nearest equal weights."""
        k = ranks.shape[1]
        score = _ICScorer(np.asarray(target, dtype=np.float64), _codes(dates), self.ic_min_n)  # target ranked once
        ics = np.array([score(ranks @ w) for w in self.grid])
        best = ics.max()
        tied = np.flatnonzero(ics >= best - 1e-12)
        dist = ((self.grid[tied] - 1.0 / k) ** 2).sum(axis=1)
        return self.grid[tied[np.argmin(dist)]], float(best)

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None) -> Ensemble:
        k = len(self.names)
        equal = np.full(k, 1.0 / k)
        has_valid = valid is not None and not valid.is_empty()
        selection: dict[str, Model] = {}
        if has_valid:
            v = valid.filter(pl.col(self.target).cast(pl.Float64).is_finite())
            if v.is_empty():
                raise ValueError(f"{self.name}: no labelled validation rows")
            shown, dates = self._hide_labels(v), v["date"]
            y = v[self.target].cast(pl.Float64).to_numpy()
            ranks = np.empty((v.height, k))
            for j, nm in enumerate(self.names):
                m = self.factories[nm](False)
                m.fit(train, valid)
                selection[nm] = m
                ranks[:, j] = per_date_centered_rank(np.asarray(m.predict(shown), dtype=np.float64), dates)
                self.member_valid_ic_[nm] = mean_date_ic(ranks[:, j], y, dates, self.ic_min_n)
            w_best, self.valid_ic_ = self.choose_weights(ranks, y, dates)
            self.w_best_ = dict(zip(self.names, w_best.tolist()))
            w = (1.0 - self.shrinkage) * w_best + self.shrinkage * equal
        else:
            self.w_best_, self.valid_ic_, self.member_valid_ic_ = None, None, {}
            w = equal
        self.weights_ = dict(zip(self.names, w.tolist()))
        self.members_ = {}
        for nm in self.names:
            sel = selection.pop(nm, None)
            if sel is not None and hasattr(sel, "refit_from"):
                # reuse the early-stopped member: refit on train + valid without a second ES pass
                self.members_[nm] = sel.refit_from(train, valid)
            else:
                m = self.factories[nm](True)
                m.fit(train, valid if has_valid else None)
                self.members_[nm] = m
        return self

    def member_ranks(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Per-date centered rank of each member's prediction on `df` (members with weight 0 skipped)."""
        if not self.members_:
            raise RuntimeError(f"{self.name}: fit before predict")
        if "date" not in df.columns:
            raise KeyError(f"{self.name}: predict needs the date column to rank within dates")
        dates = df["date"]
        return {
            nm: per_date_centered_rank(np.asarray(m.predict(df), dtype=np.float64), dates)
            for nm, m in self.members_.items()
            if self.weights_[nm] != 0.0
        }

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if df.height == 0:
            return np.empty(0)
        out = np.zeros(df.height)
        for nm, r in self.member_ranks(df).items():
            out += self.weights_[nm] * r
        return out

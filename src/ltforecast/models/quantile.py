"""Quantile regression for the 12-month forecast range (plan 6.3).

One LightGBM `objective="quantile"` booster per alpha on the winsorized raw excess return
(`excess_w_12m`). Inputs are the model columns plus the trailing log-vol inputs
(`lvol_21, lvol_63, lvol_252`, see `models.vol.vol_inputs`) so the interval width can scale with
volatility. Each booster early-stops on its own valid pinball loss and is then (by default) refit on
train + valid for its best number of rounds. `predict` returns (n, len(alphas)) rearranged (sorted
per row) so the quantiles never cross (Chernozhukov, Fernandez-Val & Galichon 2010). The conformal
correction that targets ~80% coverage lives in `calibrate.walk_forward_conformal`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from ltforecast.validation.metrics import ic_by_date

VOL_COLS = ("lvol_21", "lvol_63", "lvol_252")
TARGET = "excess_w_12m"


@dataclass(frozen=True)
class QuantileParams:
    """Plan 6.1 conservative tree settings; the quantile boosters share them."""

    num_leaves: int = 15
    max_depth: int = 5
    min_data_in_leaf: int = 1000
    learning_rate: float = 0.02
    feature_fraction: float = 0.6
    bagging_fraction: float = 0.7
    bagging_freq: int = 1
    lambda_l2: float = 10.0
    n_estimators: int = 2000  # cap; early stopping decides
    n_estimators_no_valid: int = 300  # used when no validation set is given
    early_stopping_rounds: int = 100
    refit: bool = True  # after early stopping, refit on train + valid for best_iteration rounds
    num_threads: int = 0


def pinball_loss(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    """Mean pinball (quantile) loss: mean(max(alpha * (y - q), (alpha - 1) * (y - q)))."""
    d = np.asarray(y, dtype=np.float64) - np.asarray(q, dtype=np.float64)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1.0) * d)))


def rearrange(q: np.ndarray) -> np.ndarray:
    """Sort each row ascending, so column j is the j-th smallest quantile (non-crossing)."""
    return np.sort(np.asarray(q, dtype=np.float64), axis=1)


class QuantileModel:
    """`validation.runner.Model` producing (n, len(alphas)) quantiles of `target`.

    Columns used: `cols` followed by any `vol_cols` not already in `cols`, in that fixed order,
    as float arrays (nulls -> NaN). Training rows with a null or non-finite target are dropped.
    Attributes after fit: `boosters` (one per alpha), `best_iterations` (rounds used per alpha),
    `valid_pinball` (best valid pinball loss per alpha, empty without a valid set).
    """

    def __init__(
        self,
        cols: Sequence[str],
        alphas: Sequence[float] = (0.1, 0.5, 0.9),
        params: QuantileParams = QuantileParams(),
        seed: int = 0,
        name: str = "quantile",
        target: str = TARGET,
        vol_cols: Sequence[str] = VOL_COLS,
    ):
        alphas = tuple(float(a) for a in alphas)
        if not alphas or list(alphas) != sorted(set(alphas)) or not all(0.0 < a < 1.0 for a in alphas):
            raise ValueError(f"alphas must be distinct, ascending and inside (0, 1), got {alphas}")
        cols = list(dict.fromkeys(cols))
        self.cols = cols + [c for c in vol_cols if c not in cols]
        if target in self.cols:
            raise ValueError(f"target {target!r} cannot also be an input column")
        self.alphas = alphas
        self.params = params
        self.seed = seed
        self.name = name
        self.target = target
        self.boosters: list[lgb.Booster] = []
        self.best_iterations: list[int] = []
        self.valid_pinball: dict[float, float] = {}

    def _lgb_params(self, alpha: float) -> dict:
        p = self.params
        return {
            "objective": "quantile",
            "alpha": alpha,
            "metric": "quantile",  # pinball loss at `alpha`
            "num_leaves": p.num_leaves,
            "max_depth": p.max_depth,
            "min_data_in_leaf": p.min_data_in_leaf,
            "learning_rate": p.learning_rate,
            "feature_fraction": p.feature_fraction,
            "bagging_fraction": p.bagging_fraction,
            "bagging_freq": p.bagging_freq,
            "lambda_l2": p.lambda_l2,
            "num_threads": p.num_threads,
            "verbose": -1,
            "deterministic": True,
            "force_row_wise": True,
            "seed": self.seed,
            "feature_pre_filter": False,
        }

    def _x(self, df: pl.DataFrame) -> np.ndarray:
        missing = [c for c in self.cols if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name}: missing input columns {missing}")
        return df.select(pl.col(c).cast(pl.Float64) for c in self.cols).to_numpy().astype(np.float64)

    def _xy(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        y = pl.col(self.target).cast(pl.Float64)
        df = df.filter(y.is_not_null() & y.is_finite())
        return self._x(df), df[self.target].cast(pl.Float64).to_numpy()

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None) -> QuantileModel:
        x, y = self._xy(train)
        if len(y) == 0:
            raise ValueError(f"{self.name}: no labelled training rows")
        xv, yv = self._xy(valid) if valid is not None else (None, None)
        has_valid = yv is not None and len(yv) > 0
        p = self.params
        self.boosters, self.best_iterations, self.valid_pinball = [], [], {}
        for alpha in self.alphas:
            params = self._lgb_params(alpha)
            if not has_valid:
                rounds = p.n_estimators_no_valid
                booster = lgb.train(params, lgb.Dataset(x, y, free_raw_data=False), num_boost_round=rounds)
            else:
                dtrain = lgb.Dataset(x, y, free_raw_data=False)
                dvalid = lgb.Dataset(xv, yv, reference=dtrain, free_raw_data=False)
                booster = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=p.n_estimators,
                    valid_sets=[dvalid],
                    valid_names=["valid"],
                    callbacks=[lgb.early_stopping(p.early_stopping_rounds, first_metric_only=True, verbose=False)],
                )
                rounds = max(1, int(booster.best_iteration or booster.current_iteration()))
                self.valid_pinball[alpha] = pinball_loss(yv, booster.predict(xv, num_iteration=rounds), alpha)
                if p.refit:
                    xa, ya = np.vstack([x, xv]), np.concatenate([y, yv])
                    booster = lgb.train(params, lgb.Dataset(xa, ya, free_raw_data=False), num_boost_round=rounds)
                # without refit the early-stopped booster is kept and predicts with `rounds`
            self.boosters.append(booster)
            self.best_iterations.append(rounds)
        return self

    def predict_raw(self, df: pl.DataFrame) -> np.ndarray:
        """(n, len(alphas)) booster outputs before rearrangement (may cross)."""
        if not self.boosters:
            raise RuntimeError(f"{self.name}: predict before fit")
        x = self._x(df)
        return np.column_stack(
            [b.predict(x, num_iteration=r) for b, r in zip(self.boosters, self.best_iterations, strict=True)]
        ).astype(np.float64)

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """(n, len(alphas)) quantiles, sorted per row so q[:, 0] <= q[:, 1] <= ... (non-crossing)."""
        return rearrange(self.predict_raw(df))


def width_vol_correlation(q: np.ndarray, sigma_hat: np.ndarray, dates: Sequence | pl.Series, min_n: int = 10) -> float:
    """Spearman correlation of the interval width (last minus first quantile column) with the vol
    forecast, computed per date and averaged over dates with >= `min_n` finite pairs. Plan 6.3 sanity
    rule: this should be clearly positive. NaN when no date qualifies."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 2:
        raise ValueError(f"q must be (n, k >= 2), got shape {q.shape}")
    sigma = np.asarray(sigma_hat, dtype=np.float64)
    d = pl.Series("date", dates)
    if not (len(sigma) == len(d) == q.shape[0]):
        raise ValueError("q, sigma_hat and dates must have the same length")
    df = pl.DataFrame({"date": d, "width": q[:, -1] - q[:, 0], "sigma": sigma})
    ic = ic_by_date(df, "width", "sigma", min_n=min_n)
    return float(ic["ic"].mean()) if ic.height else float("nan")

"""Baselines the main model must beat (plan section 5): B0 no skill, B1 single factors, B2 the factor
composite, B3 ridge regression.

Conventions
- Inputs are the ranked features of the modeling frame, centred at 0 in (-0.5, 0.5). A null (or a
  NaN / inf, which a ranked feature should never hold) counts as missing; a missing value scores
  0.0, the cross-sectional median of a centred rank.
- Every model implements `validation.runner.Model`, is deterministic, pickles, and never reads a
  label column in `predict` (only the feature columns it names are selected).
- Reports: a constant prediction carries no ranking information and has IC 0, not NaN. Compute
  report ICs with `ic_or_zero` (or `metrics.ic_by_date`, which already gives a constant date IC 0).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge as _SkRidge

from ltforecast.features.catalog import Catalog
from ltforecast.validation.metrics import ic_by_date
from ltforecast.validation.runner import Model

DEFAULT_SINGLE_FACTORS = ("mom_12_1", "gross_profitability", "earnings_yield")
DEFAULT_RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
LOW_VOL = ("vol_1y", "idio_vol", "beta_60m")  # the low-vol theme: risk family minus max_ret_1m
# theme -> catalog family it is made of (None: the explicit LOW_VOL list)
THEMES: dict[str, str | None] = {
    "value": "value",
    "quality": "quality",
    "momentum": "momentum",
    "low_vol": None,
    "low_investment": "investment",
}


@dataclass(frozen=True)
class BaselineParams:
    single_factors: tuple[str, ...] = DEFAULT_SINGLE_FACTORS  # one B1 model per name, key b1_<name>
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS
    ic_min_n: int = 10  # stocks a date needs for its IC to count in the ridge alpha choice
    target: str = "target"


def ic_or_zero(pred: object, target: object, dates: object, *, min_n: int = 10) -> float:
    """Mean over dates of the per-date Spearman rank IC of `pred` vs `target`.

    A date on which the prediction (or the target) is constant has IC 0, so a constant prediction
    such as B0 scores exactly 0.0, never NaN. Pairs with a null/non-finite side are left out, and
    dates with fewer than `min_n` complete pairs do not count. NaN only if no date counts."""
    df = pl.DataFrame({
        "date": dates if isinstance(dates, pl.Series) else pl.Series(list(dates)),
        "_p": pl.Series(pred, dtype=pl.Float64, strict=False),
        "_t": pl.Series(target, dtype=pl.Float64, strict=False),
    })
    ic = ic_by_date(df, "_p", "_t", min_n=min_n)["ic"]
    return float(ic.mean()) if ic.len() else math.nan


def _missing(df: pl.DataFrame, cols: Sequence[str], who: str) -> None:
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise ValueError(f"{who}: frame lacks columns {absent}")


def _finite(col: str) -> pl.Expr:
    """The column as Float64 with NaN / inf turned into null."""
    x = pl.col(col).cast(pl.Float64)
    return pl.when(x.is_finite()).then(x)


def _matrix(df: pl.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """(n, len(cols)) float64 in the given column order; missing -> 0.0."""
    return df.select(_finite(c).fill_null(0.0).alias(c) for c in cols).to_numpy().astype(np.float64, copy=False)


class NoSkill:
    """B0: 0.0 for every row (the centred rank of the median, the plan's "0.5 rank"). IC = 0."""

    def __init__(self, name: str = "b0_no_skill"):
        self.name = name

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None) -> NoSkill:
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return np.zeros(df.height, dtype=np.float64)


class SingleFactor:
    """B1: `sign * col`, missing -> 0. No fitting."""

    def __init__(self, col: str, sign: int, name: str | None = None):
        if sign not in (-1, 1):
            raise ValueError(f"sign must be +1 or -1, got {sign!r}")
        self.col, self.sign = col, int(sign)
        self.name = name or f"b1_{col}"

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None) -> SingleFactor:
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        _missing(df, [self.col], self.name)
        return self.sign * _matrix(df, [self.col])[:, 0]


def composite_themes(catalog: Catalog) -> dict[str, tuple[tuple[str, int], ...]]:
    """theme -> ((column, sign), ...) for the B2 composite: plain ranked stock features (not `_sr`)."""
    stock = catalog.names(kind="stock")
    themes = {}
    for theme, family in THEMES.items():
        cols = [c for c in LOW_VOL if c in stock] if family is None else catalog.names(kind="stock", family=family)
        if not cols:
            raise ValueError(f"factor composite: the catalog has no features for theme {theme!r}")
        themes[theme] = tuple((c, catalog.spec(c).sign) for c in cols)
    return themes


class FactorComposite:
    """B2: theme score = mean of the available `sign * rank` of the theme's features; composite = mean
    of the available theme scores; a stock with nothing available scores 0. No fitting."""

    def __init__(self, catalog: Catalog, name: str = "b2_composite"):
        self.name = name
        self.themes = composite_themes(catalog)

    @property
    def columns(self) -> list[str]:
        return [c for members in self.themes.values() for c, _ in members]

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None) -> FactorComposite:
        return self

    def theme_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        """One Float64 column per theme (null when none of its features is available)."""
        _missing(df, self.columns, self.name)
        return df.select(
            pl.mean_horizontal(_finite(c) * s for c, s in members).alias(theme)
            for theme, members in self.themes.items()
        )

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        scores = self.theme_scores(df)
        out = scores.select(pl.mean_horizontal(pl.all()).fill_null(0.0).alias("s"))["s"]
        return out.to_numpy().astype(np.float64, copy=False)


class Ridge:
    """B3: sklearn Ridge(fit_intercept=True) on `cols`, missing -> 0.

    `fit(train, valid)`: one ridge per alpha on `train`, each scored by `ic_or_zero` on `valid`; the
    best mean IC wins (ties go to the larger alpha, i.e. more shrinkage), and the final model is
    refit with that alpha on train + valid. With a single alpha no validation set is needed.
    After fit: `alpha_`, `valid_ic_` {alpha: IC}, `coef_` (in `cols` order), `intercept_`."""

    def __init__(
        self,
        cols: Sequence[str],
        alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
        *,
        target: str = "target",
        ic_min_n: int = 10,
        name: str = "b3_ridge",
    ):
        self.cols = list(cols)
        self.alphas = tuple(float(a) for a in alphas)
        if not self.cols:
            raise ValueError("ridge: no feature columns")
        if not self.alphas or any(not a > 0 or not math.isfinite(a) for a in self.alphas):
            raise ValueError(f"ridge: alphas must be positive and finite, got {self.alphas}")
        self.target, self.ic_min_n, self.name = target, ic_min_n, name
        self.alpha_: float | None = None
        self.valid_ic_: dict[float, float] = {}
        self._model: _SkRidge | None = None

    def _xy(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        _missing(df, [*self.cols, self.target], self.name)
        df = df.filter(pl.col(self.target).cast(pl.Float64).is_finite())
        return _matrix(df, self.cols), df[self.target].cast(pl.Float64).to_numpy()

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None) -> Ridge:
        has_valid = valid is not None and not valid.is_empty()
        if len(self.alphas) == 1 or not has_valid:
            if len(self.alphas) > 1:
                raise ValueError("ridge: choosing among several alphas needs a validation set")
            self.alpha_, self.valid_ic_ = self.alphas[0], {}
        else:
            x, y = self._xy(train)
            _missing(valid, [*self.cols, self.target, "date"], self.name)
            v = valid.filter(pl.col(self.target).cast(pl.Float64).is_finite())
            xv = _matrix(v, self.cols)
            self.valid_ic_ = {}
            for a in self.alphas:
                ic = ic_or_zero(_SkRidge(alpha=a).fit(x, y).predict(xv), v[self.target], v["date"],
                                min_n=self.ic_min_n)
                self.valid_ic_[a] = ic if math.isfinite(ic) else -math.inf
            best = max(self.valid_ic_.values())
            self.alpha_ = max(a for a, ic in self.valid_ic_.items() if ic == best)
        parts = [train] + ([valid] if has_valid else [])
        x, y = self._xy(pl.concat([p.select(*self.cols, self.target) for p in parts], how="vertical_relaxed"))
        if y.size == 0:
            raise ValueError("ridge: no labelled rows to fit on")
        self._model = _SkRidge(alpha=self.alpha_, fit_intercept=True).fit(x, y)
        return self

    @property
    def coef_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("ridge: not fitted")
        return self._model.coef_

    @property
    def intercept_(self) -> float:
        if self._model is None:
            raise RuntimeError("ridge: not fitted")
        return float(self._model.intercept_)

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("ridge: not fitted")
        _missing(df, self.cols, self.name)
        return _matrix(df, self.cols) @ self._model.coef_ + float(self._model.intercept_)


def baseline_factories(catalog: Catalog, params: BaselineParams | None = None) -> dict[str, Callable[[], Model]]:
    """name -> zero-argument factory of a fresh model (for `walk_forward` / `fit_on_all`).
    Keys: b0_no_skill, b1_<factor> per params.single_factors (default mom_12_1, gross_profitability,
    earnings_yield), b2_composite, b3_ridge (on catalog.linear_columns()). Factories pickle."""
    p = params or BaselineParams()
    out: dict[str, Callable[[], Model]] = {"b0_no_skill": NoSkill}
    for col in p.single_factors:
        out[f"b1_{col}"] = functools.partial(SingleFactor, col, catalog.spec(col).sign)
    out["b2_composite"] = functools.partial(FactorComposite, catalog)
    out["b3_ridge"] = functools.partial(
        Ridge, catalog.linear_columns(), p.ridge_alphas, target=p.target, ic_min_n=p.ic_min_n
    )
    return out

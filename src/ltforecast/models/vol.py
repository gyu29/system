"""Volatility model (plan 6.3): forecast 12-month realized volatility, HAR-style.

    lvol_w      log of sqrt(252) x std (ddof=1) of daily log total returns over the trailing w price
                rows of the security ending at the prediction date, inclusive (w = 21 / 63 / 252;
                at least 15 / 40 / 126 returns)
    VolModel    ridge of log(realized_vol_12m) on lvol_21, lvol_63, lvol_252, raw beta_60m, raw
                log_mcap and sector dummies; predict -> sigma_hat (annualized, clipped)

Conventions
- Point-in-time: a window ends at the entry date and uses only price rows dated <= it. An entry gets
  values only if the security has a price row dated exactly on it (nothing is carried forward).
- A daily return is log(adj_close / previous row's adj_close) within the security; null on the first
  row and when either price is missing or not positive. A null return uses up a slot of the window
  without being an observation (same convention as features.risk).
- A window whose std is (numerically) zero has no log vol: null, never -inf.
- The target `log_rvol_12m` = log(realized_vol_12m) from the labels table; it is FUTURE data and is
  hidden from predict by the runner via `target="log_rvol_12m"`.

Scale: one sort of prices_daily by (security_id, date), three rolling stds over the security, then a
semi-join to the entries. No per-row Python.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from ltforecast.dataset import KEYS, LOCKED_PERIODS, PERIODS, HoldoutLockedError, HoldoutToken

TRADING_DAYS = 252
WINDOWS = ((21, 15), (63, 40), (252, 126))  # (rows, min returns)
LVOL_COLS = tuple(f"lvol_{w}" for w, _ in WINDOWS)
RAW_COLS = ("beta_60m", "log_mcap")
TARGET = "log_rvol_12m"
LABEL_END = "label_end_12m"
_MIN_VOL = 1e-6  # annualized; below this a window is treated as constant (rounding noise)
_SID = "security_id"


def _finite(e: pl.Expr) -> pl.Expr:
    return pl.when(e.is_finite()).then(e)


def vol_inputs(prices: pl.DataFrame | pl.LazyFrame, entries: pl.DataFrame) -> pl.DataFrame:
    """One row per entry (security_id, date): security_id, date, lvol_21, lvol_63, lvol_252 (Float64,
    null when missing). Sorted by security_id, date. `prices` needs security_id, date, adj_close."""
    adj = pl.col("adj_close")
    prev = adj.shift(1).over(_SID)
    lr = pl.col("_lr")
    sids = entries.select(_SID).unique()
    computed = (
        prices.lazy()
        .select(_SID, "date", "adj_close")
        .join(sids.lazy(), on=_SID, how="semi")
        .sort(_SID, "date")
        .with_columns(_finite(pl.when((adj > 0) & (prev > 0)).then((adj / prev).log())).alias("_lr"))
        .with_columns(
            *[
                (lr.rolling_std(w, min_samples=k, ddof=1).over(_SID) * np.sqrt(TRADING_DAYS)).alias(f"_v{w}")
                for w, k in WINDOWS
            ]
        )
        .join(entries.lazy().select(_SID, "date"), on=[_SID, "date"], how="semi")
        .select(
            _SID,
            "date",
            *[
                _finite(pl.when(pl.col(f"_v{w}") > _MIN_VOL).then(pl.col(f"_v{w}").log())).alias(f"lvol_{w}")
                for w, _ in WINDOWS
            ],
        )
    )
    return (
        entries.select(_SID, "date")
        .unique()
        .lazy()
        .join(computed, on=[_SID, "date"], how="left")
        .sort(_SID, "date")
        .collect()
    )


def vol_frame(
    features_raw: pl.DataFrame,
    labels: pl.DataFrame,
    prices: pl.DataFrame | pl.LazyFrame,
    securities: pl.DataFrame,
    *,
    periods: Sequence[str] = ("dev",),
    unlock: HoldoutToken | None = None,
) -> pl.DataFrame:
    """Modeling frame for the vol model: features_raw (security_id, date, beta_60m, log_mcap) inner-joined
    to labels whose `period` is in `periods` (dev only unless a HoldoutToken unlocks gap/holdout).

    Columns: date, security_id, sector, lvol_21, lvol_63, lvol_252, beta_60m, log_mcap, log_rvol_12m
    (log of realized_vol_12m; null if missing or not positive), label_end_12m, period. Sorted by date,
    security_id. Walk it forward with `runner.walk_forward(..., target="log_rvol_12m")`.
    """
    unknown = set(periods) - set(PERIODS)
    if unknown or not periods:
        raise ValueError(f"periods must be a non-empty subset of {PERIODS}, got {tuple(periods)}")
    locked = sorted(LOCKED_PERIODS & set(periods))
    if locked and not isinstance(unlock, HoldoutToken):
        raise HoldoutLockedError(f"periods {locked} are locked until the Phase 9 holdout evaluation")

    lab = labels.filter(pl.col("period").is_in(list(periods))).select(
        _SID,
        "date",
        _finite(pl.when(pl.col("realized_vol_12m") > 0).then(pl.col("realized_vol_12m").log())).alias(TARGET),
        LABEL_END,
        "period",
    )
    base = features_raw.select(_SID, "date", *[pl.col(c).cast(pl.Float64) for c in RAW_COLS]).join(
        lab, on=[_SID, "date"], how="inner", validate="1:1"
    )
    sectors = securities.select(_SID, "sector").unique(_SID, keep="first", maintain_order=True)
    out = (
        base.join(vol_inputs(prices, base), on=[_SID, "date"], how="left")
        .join(sectors, on=_SID, how="left")
    )
    return out.select(*KEYS, "sector", *LVOL_COLS, *RAW_COLS, TARGET, LABEL_END, "period").sort(KEYS)


@dataclass(frozen=True)
class VolParams:
    """alpha: ridge penalty per observation on standardized numeric inputs (intercept unpenalized).
    clip: bounds of the annualized sigma_hat.
    refit: when a validation frame is given, valid_r2_ is scored from the train-only fit, then the
    model is re-estimated on train + valid (so the most recent labelled months are not dropped)."""

    alpha: float = 1e-3
    clip_low: float = 0.05
    clip_high: float = 3.0
    numeric: tuple[str, ...] = LVOL_COLS + RAW_COLS
    use_sector: bool = True
    target: str = TARGET
    refit: bool = True


class VolModel:
    """Ridge on log realized vol. Fit learns medians (for missing inputs), standardization, sector
    levels (unseen or null sector -> all dummies 0) and coefficients. Never reads a label in predict."""

    name = "vol_ridge"

    def __init__(self, params: VolParams | None = None):
        self.params = params or VolParams()
        self.medians_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.sectors_: tuple[str, ...] = ()
        self.coef_: np.ndarray | None = None  # on standardized numerics, then raw 0/1 sector dummies
        self.intercept_: float = 0.0
        self.valid_r2_: float | None = None

    # -- design matrix -------------------------------------------------------------------------
    def _numeric(self, df: pl.DataFrame) -> np.ndarray:
        cols = self.params.numeric
        x = df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy().astype(np.float64, copy=True)
        x[~np.isfinite(x)] = np.nan
        return x

    def _dummies(self, df: pl.DataFrame) -> np.ndarray:
        if not self.params.use_sector or not self.sectors_:
            return np.zeros((df.height, 0))
        codes = (
            df.select(pl.col("sector").cast(pl.String).replace_strict(
                list(self.sectors_), list(range(len(self.sectors_))), default=-1, return_dtype=pl.Int64
            ).fill_null(-1))
            .to_series()
            .to_numpy()
        )
        d = np.zeros((df.height, len(self.sectors_)))
        hit = codes >= 0
        d[np.nonzero(hit)[0], codes[hit]] = 1.0
        return d

    def _design(self, df: pl.DataFrame) -> np.ndarray:
        x = self._numeric(df)
        x = np.where(np.isnan(x), self.medians_, x)
        z = (x - self.mean_) / self.scale_
        return np.hstack([z, self._dummies(df)])

    # -- Model protocol ------------------------------------------------------------------------
    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None = None) -> VolModel:
        """Fit on `train`. With a `valid` frame holding finite targets, valid_r2_ is the train-only
        out-of-sample R²; then, if params.refit, the model is re-estimated on train + valid."""
        p = self.params
        self._fit_core(train)
        if valid is not None and valid.height and p.target in valid.columns:
            yv = valid[p.target].cast(pl.Float64).to_numpy()
            if np.isfinite(yv).any():
                self.valid_r2_ = oos_r2(self.predict_log(valid), yv)
                if p.refit:
                    self._fit_core(pl.concat([train, valid], how="diagonal_relaxed"))
        return self

    def _fit_core(self, train: pl.DataFrame) -> None:
        p = self.params
        y_all = train[p.target].cast(pl.Float64).to_numpy()
        keep = np.isfinite(y_all)
        if not keep.any():
            raise ValueError("VolModel.fit: no finite targets")
        tr = train.filter(pl.Series(keep))
        y = y_all[keep]

        x = self._numeric(tr)
        med = np.nanmedian(np.where(np.isnan(x).all(axis=0), 0.0, x), axis=0)
        self.medians_ = np.where(np.isnan(med), 0.0, med)
        x = np.where(np.isnan(x), self.medians_, x)
        self.mean_ = x.mean(axis=0)
        sd = x.std(axis=0)
        self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        if p.use_sector and "sector" in tr.columns:
            self.sectors_ = tuple(sorted(tr["sector"].drop_nulls().cast(pl.String).unique().to_list()))
        else:
            self.sectors_ = ()

        a = np.hstack([(x - self.mean_) / self.scale_, self._dummies(tr)])
        n = len(y)
        am, ym = a.mean(axis=0), y.mean()
        ac = a - am
        gram = ac.T @ ac / n + p.alpha * np.eye(a.shape[1])
        self.coef_ = np.linalg.solve(gram, ac.T @ (y - ym) / n)
        self.intercept_ = float(ym - am @ self.coef_)

    def predict_log(self, df: pl.DataFrame) -> np.ndarray:
        """Log of the clipped sigma_hat, shape (n,)."""
        if self.coef_ is None:
            raise RuntimeError("VolModel is not fitted")
        raw = self.intercept_ + self._design(df) @ self.coef_
        return np.clip(raw, np.log(self.params.clip_low), np.log(self.params.clip_high))

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """sigma_hat: annualized forecast 12m volatility in [clip_low, clip_high], shape (n,)."""
        return np.exp(self.predict_log(df))

    def coefficients(self) -> dict[str, float]:
        """Coefficients in the inputs' own units (numerics per unit of the raw column), plus
        `intercept` and `sector=<level>` dummies."""
        if self.coef_ is None:
            raise RuntimeError("VolModel is not fitted")
        k = len(self.params.numeric)
        b = self.coef_[:k] / self.scale_
        out = {"intercept": float(self.intercept_ - b @ self.mean_)}
        out |= {c: float(v) for c, v in zip(self.params.numeric, b, strict=True)}
        out |= {f"sector={s}": float(v) for s, v in zip(self.sectors_, self.coef_[k:], strict=True)}
        return out


def oos_r2(pred_log: np.ndarray, actual_log: np.ndarray) -> float:
    """1 - SSE / SST (SST around the actuals' own mean) over pairs where both are finite; NaN if
    fewer than 2 pairs or the actuals have no variance."""
    p = np.asarray(pred_log, dtype=np.float64).ravel()
    a = np.asarray(actual_log, dtype=np.float64).ravel()
    if p.shape != a.shape:
        raise ValueError(f"oos_r2: shapes differ {p.shape} vs {a.shape}")
    ok = np.isfinite(p) & np.isfinite(a)
    if ok.sum() < 2:
        return float("nan")
    p, a = p[ok], a[ok]
    sst = float(((a - a.mean()) ** 2).sum())
    if sst <= 0:
        return float("nan")
    return 1.0 - float(((a - p) ** 2).sum()) / sst


def oos_r2_by_fold(
    oof: pl.DataFrame, frame: pl.DataFrame, *, pred_col: str = "pred", target: str = TARGET
) -> pl.DataFrame:
    """Per-fold out-of-sample R² in log space. `oof` is walk_forward output whose `pred_col` is
    sigma_hat; `frame` supplies `target` on (date, security_id). Columns: fold, n, r2."""
    j = oof.join(frame.select(*KEYS, target), on=list(KEYS), how="inner").with_columns(
        _finite(pl.col(pred_col).log()).alias("_p")
    )
    rows = []
    for (fold,), g in j.group_by("fold", maintain_order=True):
        ok = g.filter(pl.col("_p").is_not_null() & pl.col(target).is_finite())
        rows.append({"fold": fold, "n": ok.height, "r2": oos_r2(ok["_p"].to_numpy(), ok[target].to_numpy())})
    return pl.DataFrame(rows, schema={"fold": pl.String, "n": pl.Int64, "r2": pl.Float64}).sort("fold")

"""P(beat the market) and conformal forecast ranges from out-of-fold predictions (plan 6.2, 6.3).

Both maps are fitted walk-forward with the same purge rule: the map applied to fold k's test rows is
fitted only on OOF rows of EARLIER folds whose 12m label window ended strictly before fold k's first
test date (`label_end_12m < first_test`). A row whose outcome was not yet known on that date is never
used, nor is any row of fold k itself or a later fold. The first fold therefore has nothing to fit on:
it gets prob 0.5 / no conformal correction, and a flag saying so.

Inputs
  oof     date, security_id, plus `pred` (isotonic) or `q10, q50, q90` (conformal); one row per key.
          Every date must be a test date of one of `folds` (the runner's output satisfies this).
  labels  date, security_id, `label_end_12m` and the outcome column (`excess_12m` by default).
Rows without an outcome or label end are still scored; they are just never fitted on.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from ltforecast.validation.splitter import Fold

KEYS = ("date", "security_id")
_FOLD = "_fold_pos"


# isotonic probability ------------------------------------------------------------------------------


class IsotonicCalibrator:
    """Monotone map from a score in (0, 1) to P(outcome), clipped to [0, 1] inside and outside the
    fitted range. Thin wrapper on sklearn `IsotonicRegression`; picklable."""

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None
        self.n_obs = 0

    def fit(self, score: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        score = np.asarray(score, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if score.shape != y.shape or score.ndim != 1 or score.size == 0:
            raise ValueError(f"score and y must be equal-length non-empty 1-d arrays, got {score.shape}, {y.shape}")
        if not (np.isfinite(score).all() and np.isfinite(y).all()):
            raise ValueError("score and y must be finite")
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(score, y)
        self.n_obs = int(score.size)
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("IsotonicCalibrator.predict called before fit")
        return np.clip(self._iso.predict(np.asarray(score, dtype=np.float64)), 0.0, 1.0)


def score_pct(pred: str = "pred") -> pl.Expr:
    """Per-date percentile of `pred` in (0, 1): (average rank - 0.5) / count, as for `rank_12m`."""
    return ((pl.col(pred).rank("average") - 0.5) / pl.col(pred).count()).over("date")


def _fold_positions(oof: pl.DataFrame, folds: Sequence[Fold]) -> pl.DataFrame:
    """Adds `_fold_pos` (index into `folds`, which are sorted by first test date) from each row's date.
    Raises if a date belongs to no fold or to two folds."""
    order = sorted(range(len(folds)), key=lambda i: folds[i].test_dates[0])
    if order != list(range(len(folds))):
        raise ValueError("folds must be in order of their first test date")
    pairs = [(day, i) for i, f in enumerate(folds) for day in f.test_dates]
    if len({day for day, _ in pairs}) != len(pairs):
        raise ValueError("a test date belongs to more than one fold")
    mapping = pl.DataFrame(
        {"date": [p[0] for p in pairs], _FOLD: [p[1] for p in pairs]},
        schema={"date": pl.Date, _FOLD: pl.Int64},
    )
    out = oof.join(mapping, on="date", how="left")
    missing = out.filter(pl.col(_FOLD).is_null())
    if missing.height:
        raise ValueError(f"{missing.height} OOF rows have a date in no fold's test dates, e.g. {missing['date'][0]}")
    return out


def _with_labels(oof: pl.DataFrame, labels: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    need = [c for c in cols if c not in labels.columns]
    if need:
        raise ValueError(f"labels is missing columns {need}")
    if oof.select(KEYS).is_duplicated().any():
        raise ValueError("oof has duplicate (date, security_id) rows")
    lab = labels.select(*KEYS, *cols)
    if lab.select(KEYS).is_duplicated().any():
        raise ValueError("labels has duplicate (date, security_id) rows")
    return oof.drop([c for c in cols if c in oof.columns]).join(lab, on=list(KEYS), how="left")


def _usable(k: int, first_test: dt.date, label_end_col: str, value_cols: Sequence[str]) -> pl.Expr:
    """Rows allowed to fit the map for fold position k: earlier fold, label matured, values finite."""
    ok = (pl.col(_FOLD) < k) & (pl.col(label_end_col) < first_test)
    for c in value_cols:
        ok = ok & pl.col(c).is_not_null() & pl.col(c).cast(pl.Float64).is_finite()
    return ok.fill_null(False)


def _isotonic_frame(
    oof: pl.DataFrame, labels: pl.DataFrame, pred: str, outcome: str, label_end_col: str
) -> pl.DataFrame:
    if pred not in oof.columns:
        raise ValueError(f"oof has no column {pred!r}")
    return _with_labels(oof.select(*KEYS, pred), labels, [outcome, label_end_col]).with_columns(
        score_pct(pred).alias("_score"),
        pl.when(pl.col(outcome).is_null()).then(None).otherwise((pl.col(outcome) > 0).cast(pl.Float64)).alias("_y"),
    )


def _enough(n_rows: int, n_dates: int, min_rows: int, min_dates: int) -> bool:
    return n_rows >= max(min_rows, 1) and n_dates >= max(min_dates, 1)


def walk_forward_isotonic(
    oof: pl.DataFrame,
    labels: pl.DataFrame,
    folds: Sequence[Fold],
    *,
    min_rows: int = 2000,
    min_dates: int = 12,
    pred: str = "pred",
    outcome: str = "excess_12m",
    label_end_col: str = "label_end_12m",
) -> pl.DataFrame:
    """Columns: date, security_id, prob (Float64), prob_calibrated (Boolean); sorted by date, security_id.

    score = per-date percentile of `pred`; y = `outcome` > 0. Fold k's rows get the isotonic map fitted
    on the usable rows (earlier folds, `label_end_col` < fold k's first test date, outcome present).
    With fewer than `min_rows` usable rows, or usable rows on fewer than `min_dates` distinct dates
    (one cross-section shares one market state), prob = 0.5, prob_calibrated = False.
    """
    df = _fold_positions(_isotonic_frame(oof, labels, pred, outcome, label_end_col), folds)
    score = df["_score"].to_numpy()
    y = df["_y"].to_numpy()
    pos = df[_FOLD].to_numpy()
    dates = df["date"]
    prob = np.full(df.height, 0.5)
    calibrated = np.zeros(df.height, dtype=bool)
    for k, fold in enumerate(folds):
        rows = pos == k
        if not rows.any():
            continue
        fit = df.select(_usable(k, fold.test_dates[0], label_end_col, ["_y", "_score"])).to_series().to_numpy()
        if not _enough(fit.sum(), dates.filter(pl.Series(fit)).n_unique(), min_rows, min_dates):
            continue
        cal = IsotonicCalibrator().fit(score[fit], y[fit])
        prob[rows] = cal.predict(score[rows])
        calibrated[rows] = True
    return (
        df.select(*KEYS)
        .with_columns(
            pl.Series("prob", prob, dtype=pl.Float64),
            pl.Series("prob_calibrated", calibrated, dtype=pl.Boolean),
        )
        .sort(*KEYS)
    )


def fit_isotonic(
    oof: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    before: dt.date,
    min_rows: int = 2000,
    min_dates: int = 12,
    pred: str = "pred",
    outcome: str = "excess_12m",
    label_end_col: str = "label_end_12m",
) -> IsotonicCalibrator | None:
    """One calibrator on every OOF row whose label ended before `before` (fit-final: pass the first
    date the calibrator will be applied to, e.g. the day after dev_end). None if fewer than `min_rows`
    rows or fewer than `min_dates` distinct dates."""
    df = _isotonic_frame(oof, labels, pred, outcome, label_end_col).with_columns(pl.lit(-1).alias(_FOLD))
    fit = df.filter(_usable(0, before, label_end_col, ["_y", "_score"]))
    if not _enough(fit.height, fit["date"].n_unique(), min_rows, min_dates):
        return None
    return IsotonicCalibrator().fit(fit["_score"].to_numpy(), fit["_y"].to_numpy())


# conformal range -----------------------------------------------------------------------------------


def conformal_offset(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest finite score; inf when that rank exceeds n (too few
    scores; the caller falls back to 0 and flags it). Non-finite scores are ignored."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.asarray(scores, dtype=np.float64)
    s = np.sort(s[np.isfinite(s)])
    n = s.size
    # round first so (9 + 1) * 0.8 = 8.000000000000002 gives rank 8, not 9
    k = math.ceil(round((n + 1) * (1.0 - alpha), 9))
    if n == 0 or k > n:
        return math.inf
    return float(s[max(k, 1) - 1])


def cqr_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Conformalized-quantile-regression scores max(lo - y, y - hi): > 0 outside the interval."""
    return np.maximum(np.asarray(lo, dtype=np.float64) - y, np.asarray(y, dtype=np.float64) - hi)


def _window(df: pl.DataFrame, window_months: int) -> pl.DataFrame:
    """Rows on the most recent `window_months` distinct prediction dates of `df`."""
    if window_months < 1:
        raise ValueError(f"window_months must be >= 1, got {window_months}")
    keep = df["date"].unique().sort().tail(window_months)
    return df.filter(pl.col("date").is_in(keep.implode()))


def _conformal_frame(
    oof_q: pl.DataFrame, labels: pl.DataFrame, cols: Sequence[str], outcome: str, label_end_col: str
) -> pl.DataFrame:
    need = [c for c in cols if c not in oof_q.columns]
    if need:
        raise ValueError(f"oof_q is missing columns {need}")
    df = _with_labels(oof_q.select(*KEYS, *cols), labels, [outcome, label_end_col])
    lo, hi, y = (pl.col(c).cast(pl.Float64) for c in (cols[0], cols[2], outcome))
    return df.with_columns(pl.max_horizontal(lo - y, y - hi).alias("_cqr"))


def _offset_from(
    fit: pl.DataFrame, alpha: float, window_months: int, min_rows: int, min_dates: int
) -> tuple[float, bool]:
    """Offset from the last `window_months` dates of `fit`; (0.0, False) when that window has fewer than
    `min_rows` rows or `min_dates` distinct dates, or too few scores for the rank."""
    w = _window(fit, window_months) if fit.height else fit
    if not _enough(w.height, w["date"].n_unique(), min_rows, min_dates):
        return 0.0, False
    off = conformal_offset(w["_cqr"].to_numpy(), alpha)
    return (off, True) if math.isfinite(off) else (0.0, False)


def walk_forward_conformal(
    oof_q: pl.DataFrame,
    labels: pl.DataFrame,
    folds: Sequence[Fold],
    *,
    alpha: float = 0.2,
    window_months: int = 36,
    min_rows: int = 2000,
    min_dates: int = 12,
    quantile_cols: Sequence[str] = ("q10", "q50", "q90"),
    outcome: str = "excess_12m",
    label_end_col: str = "label_end_12m",
) -> pl.DataFrame:
    """Columns: date, security_id, q10, q50, q90, conformal_offset, conformal_ok; sorted by date, security_id.

    For fold k the offset is `conformal_offset` of the CQR scores (raw lo/hi vs `outcome`) of usable rows
    (earlier folds, label ended before fold k's first test date, all values finite), restricted to the
    most recent `window_months` distinct dates among them. q10 -= offset, q90 += offset, q50 unchanged;
    a negative offset (raw ranges too wide) narrows the range but never past q50. Too few scores, a
    window with fewer than `min_rows` rows or `min_dates` distinct dates, or the first fold: offset 0,
    conformal_ok False. Output columns are always named q10, q50, q90.
    """
    cols = tuple(quantile_cols)
    if len(cols) != 3:
        raise ValueError(f"quantile_cols must name (lo, mid, hi), got {cols}")
    df = _fold_positions(_conformal_frame(oof_q, labels, cols, outcome, label_end_col), folds)
    pos = df[_FOLD].to_numpy()
    offset = np.zeros(df.height)
    ok = np.zeros(df.height, dtype=bool)
    for k, fold in enumerate(folds):
        rows = pos == k
        if not rows.any():
            continue
        fit = df.filter(_usable(k, fold.test_dates[0], label_end_col, [*cols, outcome]))
        offset[rows], ok[rows] = _offset_from(fit, alpha, window_months, min_rows, min_dates)
    lo, mid, hi = (pl.col(c).cast(pl.Float64) for c in cols)
    off = pl.col("conformal_offset")
    return (
        df.select(*KEYS, *cols)
        .with_columns(
            pl.Series("conformal_offset", offset, dtype=pl.Float64),
            pl.Series("conformal_ok", ok, dtype=pl.Boolean),
        )
        .select(
            *KEYS,
            pl.min_horizontal(lo - off, mid).alias("q10"),
            mid.alias("q50"),
            pl.max_horizontal(hi + off, mid).alias("q90"),
            "conformal_offset",
            "conformal_ok",
        )
        .sort(*KEYS)
    )


def latest_conformal_offset(
    oof_q: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    before: dt.date,
    alpha: float = 0.2,
    window_months: int = 36,
    min_rows: int = 2000,
    min_dates: int = 12,
    quantile_cols: Sequence[str] = ("q10", "q50", "q90"),
    outcome: str = "excess_12m",
    label_end_col: str = "label_end_12m",
) -> tuple[float, bool]:
    """(offset, ok) from every OOF row whose label ended before `before`, last `window_months` dates
    (fit-final). (0.0, False) when there are too few scores, rows (`min_rows`) or dates (`min_dates`)."""
    cols = tuple(quantile_cols)
    df = _conformal_frame(oof_q, labels, cols, outcome, label_end_col).with_columns(pl.lit(-1).alias(_FOLD))
    fit = df.filter(_usable(0, before, label_end_col, [*cols, outcome]))
    return _offset_from(fit, alpha, window_months, min_rows, min_dates)

"""Explanations (plan 6.4): additive per-feature attributions of the ensemble score, plain-English top
drivers, and global importance per fold with its stability across folds.

Conventions
- Member attributions are exactly additive on the member's raw scale:
  LightGBM (LGBM / SeedAverage): TreeSHAP (mean across seeds), rows sum to raw score - expected_value;
  ridge (B3): coef * value (missing -> 0), rows sum to prediction - intercept;
  composite (B2): sign * rank / n_available_in_theme / n_available_themes, rows sum to the composite.
- `ensemble_contributions` divides each member's attributions by the member's cross-sectional std of
  its raw prediction on the same date (per date when `df` has a `date` column; a member that is
  constant on a date contributes 0 there), multiplies by `ensemble.weights_` and sums on the union
  of features. It explains the blend of standardized raw scores, a close stand-in for the blend of
  ranks the ensemble actually predicts.
- Percentile of a centered rank r in (-0.5, 0.5): floor((r + 0.5) * 100 + 0.5), i.e. round half up.
- `has_<family>` flags are never shown as drivers.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ltforecast.features.catalog import HAS_PREFIX, SR_SUFFIX, Catalog
from ltforecast.models.baselines import FactorComposite, Ridge

# member attributions --------------------------------------------------------------------------------


@dataclass(frozen=True)
class MemberAttribution:
    cols: list[str]
    values: np.ndarray   # (n, len(cols)); rows sum to raw - base
    raw: np.ndarray      # (n,) the member's raw score
    base: float          # expected value / intercept / 0


def _matrix_zero(df: pl.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """(n, len(cols)) float64; null / NaN / inf -> 0.0."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"frame lacks columns {missing}")
    x = [pl.col(c).cast(pl.Float64) for c in cols]
    return df.select(pl.when(e.is_finite()).then(e).fill_null(0.0).alias(c) for e, c in zip(x, cols)).to_numpy()


def _composite(model: FactorComposite, df: pl.DataFrame) -> MemberAttribution:
    cols, signs, theme_of = [], [], []
    for t, (_, members) in enumerate(model.themes.items()):
        for c, s in members:
            cols.append(c)
            signs.append(float(s))
            theme_of.append(t)
    raw_cols = [pl.col(c).cast(pl.Float64) for c in cols]
    avail = df.select(e.is_finite().fill_null(False).alias(c) for e, c in zip(raw_cols, cols)).to_numpy()
    x = _matrix_zero(df, cols) * np.asarray(signs)
    theme_of = np.asarray(theme_of)
    n_themes = len(model.themes)
    # available features per (row, theme), then available themes per row
    n_in_theme = np.stack([avail[:, theme_of == t].sum(axis=1) for t in range(n_themes)], axis=1)
    n_avail_themes = (n_in_theme > 0).sum(axis=1)
    denom = n_in_theme[:, theme_of] * n_avail_themes[:, None]
    values = np.divide(x, denom, out=np.zeros_like(x), where=avail & (denom > 0))
    return MemberAttribution(cols, values, values.sum(axis=1), 0.0)


def member_contributions(model: object, df: pl.DataFrame) -> MemberAttribution:
    """Additive attributions of one fitted member on `df` (see module conventions).

    Supported: anything with `contributions(df)` + `expected_value` (LGBM, SeedAverage), `Ridge`
    (incl. subclasses), `FactorComposite`. Other members raise TypeError."""
    if isinstance(model, FactorComposite):
        return _composite(model, df)
    if isinstance(model, Ridge):
        values = _matrix_zero(df, model.cols) * np.asarray(model.coef_, dtype=np.float64)
        return MemberAttribution(list(model.cols), values, values.sum(axis=1) + model.intercept_, model.intercept_)
    if hasattr(model, "contributions") and hasattr(model, "cols"):
        values = np.asarray(model.contributions(df), dtype=np.float64)
        base = float(model.expected_value)
        return MemberAttribution(list(model.cols), values, values.sum(axis=1) + base, base)
    raise TypeError(f"no attribution rule for member type {type(model).__name__}")


def _per_date_std(raw: np.ndarray, df: pl.DataFrame) -> np.ndarray:
    """Population std of `raw` within each row's date (whole frame if there is no date column)."""
    if "date" not in df.columns:
        return np.full(len(raw), float(np.std(raw)) if len(raw) else 0.0)
    s = pl.DataFrame({"date": df["date"], "r": raw}).select(pl.col("r").std(ddof=0).over("date"))["r"]
    return s.fill_null(0.0).to_numpy()


def _feature_order(names: Iterable[str], catalog: Catalog) -> list[str]:
    seen = set(names)
    order = [c for c in catalog.model_columns() if c in seen]
    return order + sorted(seen - set(order))


def ensemble_contributions(
    ensemble: object,
    df: pl.DataFrame,
    catalog: Catalog,
    *,
    security_ids: Sequence[int] | None = None,
    min_cross_section: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """(n, n_features) attributions of the blended score and their feature names (catalog model-column
    order, then any other member columns sorted). Uses `ensemble.members_` and `ensemble.weights_`.

    The per-date std is taken over every row of `df`, so `df` must hold the full cross-section of each
    date. With `security_ids`, attributions are computed on all rows and only the rows of those
    securities are returned (in `df` order). A ValueError ("partial cross-section") is raised when a
    date of `df` has fewer than `min_cross_section` rows (pipeline.FinalModels.min_cross_section)."""
    if min_cross_section > 1 and df.height and "date" in df.columns:
        small = df.group_by("date").len().filter(pl.col("len") < min_cross_section)
        if small.height:
            raise ValueError(
                f"ensemble_contributions: partial cross-section on {small['date'].sort().to_list()[:3]} "
                f"(fewer than {min_cross_section} rows); pass the full date and select with security_ids"
            )
    if security_ids is not None:
        out, names = ensemble_contributions(ensemble, df, catalog)
        keep = df["security_id"].is_in(list(security_ids)).to_numpy()
        return out[keep], names
    weights: Mapping[str, float] = ensemble.weights_
    parts = []
    for nm, model in ensemble.members_.items():
        w = float(weights.get(nm, 0.0))
        if w == 0.0:
            continue
        a = member_contributions(model, df)
        sd = _per_date_std(a.raw, df)
        scale = np.divide(w, sd, out=np.zeros_like(sd), where=sd > 1e-12)
        parts.append((a.cols, a.values * scale[:, None]))
    names = _feature_order((c for cols, _ in parts for c in cols), catalog)
    pos = {c: i for i, c in enumerate(names)}
    out = np.zeros((df.height, len(names)))
    for cols, vals in parts:
        idx = np.asarray([pos[c] for c in cols], dtype=np.int64)
        for j, i in enumerate(idx):  # loop over columns (~60), not rows; tolerates repeated names
            out[:, i] += vals[:, j]
    return out, names


# drivers ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Driver:
    feature: str
    label: str
    percentile: int | None
    contribution: float
    text: str


def percentile(rank: float | None) -> int | None:
    """Percentile of a centered rank: round half up of (rank + 0.5) * 100; None when missing."""
    if rank is None or not math.isfinite(rank):
        return None
    return int(math.floor((rank + 0.5) * 100.0 + 0.5))


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _lower_first(text: str) -> str:
    # keep acronyms ("EBITDA/EV") as they are
    return text if len(text) > 1 and text[1].isupper() else text[:1].lower() + text[1:]


def describe(feature: str, value: float | None, catalog: Catalog) -> tuple[str, int | None, str] | None:
    """(label, percentile, text) for one feature, or None for a `has_*` flag (never a driver)."""
    if feature.startswith(HAS_PREFIX):
        return None
    base = feature.removesuffix(SR_SUFFIX)
    try:
        spec = catalog.spec(base)
    except KeyError:
        return feature, None, feature
    if spec.kind == "context":
        return spec.label, None, f"Market context: {_lower_first(spec.label)}"
    label = spec.label + (" vs sector" if feature.endswith(SR_SUFFIX) else "")
    pct = percentile(value)
    return label, pct, f"{label} ({ordinal(pct)} pct)" if pct is not None else f"{label} (missing)"


def top_drivers(
    contrib_row: Sequence[float],
    feature_names: Sequence[str],
    feature_values_row: Sequence[float | None],
    catalog: Catalog,
    k: int = 3,
) -> tuple[list[Driver], list[Driver]]:
    """The k largest positive and k most negative contributions (strictly non-zero, has_* skipped),
    each list ordered by |contribution| descending."""
    c = np.asarray(contrib_row, dtype=np.float64)
    keep = [i for i, f in enumerate(feature_names) if not f.startswith(HAS_PREFIX) and math.isfinite(c[i])]

    def pick(idx: list[int]) -> list[Driver]:
        out = []
        for i in idx[:k]:
            v = feature_values_row[i]
            label, pct, text = describe(feature_names[i], None if v is None else float(v), catalog)
            out.append(Driver(feature_names[i], label, pct, float(c[i]), text))
        return out

    pos = sorted((i for i in keep if c[i] > 0), key=lambda i: (-c[i], i))
    neg = sorted((i for i in keep if c[i] < 0), key=lambda i: (c[i], i))
    return pick(pos), pick(neg)


# global importance -------------------------------------------------------------------------------------


def importance_frame(fold: str, contrib: np.ndarray, names: Sequence[str]) -> pl.DataFrame:
    """One fold's global importance: mean |attribution| per feature. Columns fold, feature, mean_abs."""
    mean_abs = np.abs(contrib).mean(axis=0) if len(contrib) else np.zeros(len(names))
    return pl.DataFrame({"fold": [fold] * len(names), "feature": list(names), "mean_abs": mean_abs},
                        schema={"fold": pl.String, "feature": pl.String, "mean_abs": pl.Float64})


def importance_by_fold(per_fold: Mapping[str, tuple[np.ndarray, Sequence[str]]]) -> pl.DataFrame:
    """{fold: (contributions (n, p), feature names)} -> DataFrame(fold, feature, mean_abs)."""
    frames = [importance_frame(f, c, n) for f, (c, n) in per_fold.items()]
    if not frames:
        return pl.DataFrame(schema={"fold": pl.String, "feature": pl.String, "mean_abs": pl.Float64})
    return pl.concat(frames)


class FoldImportance:
    """`on_fold` hook for `walk_forward`: collects ensemble_contributions on each fold's test frame
    (optionally a seeded random sample of `max_rows` rows). `.table()` -> importance_by_fold."""

    def __init__(self, catalog: Catalog, max_rows: int | None = None, seed: int = 0):
        self.catalog, self.max_rows, self.seed = catalog, max_rows, seed
        self.per_fold: dict[str, tuple[np.ndarray, list[str]]] = {}

    def __call__(self, fold: object, model: object, test_frame: pl.DataFrame) -> None:
        df = test_frame
        if self.max_rows is not None and df.height > self.max_rows:
            df = df.sample(self.max_rows, seed=self.seed)
        self.per_fold[getattr(fold, "name", str(fold))] = ensemble_contributions(model, df, self.catalog)

    def table(self) -> pl.DataFrame:
        return importance_by_fold(self.per_fold)


def importance_stability(df: pl.DataFrame) -> float:
    """Mean pairwise Spearman correlation of the per-feature mean_abs vectors across folds (a feature
    absent from a fold counts 0). 1.0 = identical ordering in every fold. NaN with < 2 folds or when
    no pair is defined (a fold with constant importances)."""
    wide = df.pivot(on="fold", index="feature", values="mean_abs", aggregate_function="first").fill_null(0.0)
    folds = [c for c in wide.columns if c != "feature"]
    if len(folds) < 2 or wide.height < 2:
        return math.nan
    x = wide.select(folds).to_numpy()
    vals = []
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            if np.ptp(x[:, i]) == 0 or np.ptp(x[:, j]) == 0:
                continue
            vals.append(spearmanr(x[:, i], x[:, j]).statistic)
    return float(np.mean(vals)) if vals else math.nan

"""Typed access to config/model.yaml: validation rules, metric settings, pre-registered criteria
(Phase 4) and the model settings of Phases 5-6.

Required sections: validation, metrics, criteria, costs_bps, experiments_dir; each must hold exactly
its fields. Model sections (baselines, lgbm, ensemble, vol, quantile, calibration, levels,
hit_classifier) are optional: a missing key takes the plan-conformant default of the module's own
params dataclass, an unknown key raises. Any other top-level key is a typo and raises.

`fast`: overrides of the model sections for small runs (`--fast`: fewer seeds, no tuning grid, fewer
Monte Carlo paths). They are validated on every load and applied only with `fast=True`; they can never
touch validation, metrics, criteria or costs, so a fast run uses the same folds, purge and thresholds.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ltforecast.config import REPO_ROOT

if TYPE_CHECKING:
    from ltforecast.hitprob import HitClassifierParams
    from ltforecast.models.baselines import BaselineParams
    from ltforecast.models.lgbm import LGBMParams
    from ltforecast.models.quantile import QuantileParams
    from ltforecast.models.vol import VolParams

DEFAULT_MODEL_CONFIG = REPO_ROOT / "config" / "model.yaml"

MODEL_SECTIONS = (
    "baselines", "lgbm", "ensemble", "vol", "quantile", "calibration", "levels", "hit_classifier",
)
# Top-level sections the loader understands. A later phase that adds a section adds its key here
# (and a dataclass); anything else in the file is a typo and raises.
KNOWN_KEYS = frozenset({"validation", "metrics", "criteria", "costs_bps", "experiments_dir", "fast", *MODEL_SECTIONS})


@dataclass(frozen=True)
class ValidationRules:
    first_test_year: int
    test_months: int
    purge_months: int
    embargo_months: int
    min_train_months: int
    inner_valid_months: int


@dataclass(frozen=True)
class MetricRules:
    nw_lags: int
    n_deciles: int
    prob_bins: int


@dataclass(frozen=True)
class Criteria:
    """Plan section 0. Pre-registered: read, never tuned."""

    min_rank_ic: float
    min_ic_tstat: float
    min_positive_spread_years: float
    coverage_range: tuple[float, float]
    calibration_tolerance: float
    tpsl_prob_tolerance: float


@dataclass(frozen=True)
class LGBMSettings:
    """The `lgbm` section: base booster params, the seeds of the ensemble's seed average, and whether
    the 16-point grid runs after the objective comparison (plan 6.1)."""

    params: LGBMParams
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    tune: bool = False


@dataclass(frozen=True)
class EnsembleParams:
    """Plan 6.1 blend: simplex grid step, shrinkage toward equal weights, IC min names per date,
    rows sampled per fold for TreeSHAP importance (None = all)."""

    step: float = 0.1
    shrinkage: float = 0.5
    ic_min_n: int = 10
    importance_max_rows: int | None = 20_000


@dataclass(frozen=True)
class QuantileSettings:
    params: QuantileParams
    alphas: tuple[float, ...] = (0.1, 0.5, 0.9)
    seed: int = 0


@dataclass(frozen=True)
class CalibrationParams:
    """Plan 6.2 / 6.3. min_rows: fewest usable OOF rows for an isotonic map or a conformal offset;
    min_dates: fewest distinct matured prediction dates for either (one month = one market state;
    for conformal they are counted inside the window_months window); alpha: miscoverage of the
    conformal range (0.2 = 10-90%); window_months: recent dates the conformal offset uses; outcome:
    the realized value the range and P(beat market) are judged on; min_bin_rows: a calibration-table
    bin needs this many forecasts before its gap counts toward criteria 5 and 6."""

    min_rows: int = 2000
    min_dates: int = 12
    alpha: float = 0.2
    window_months: int = 36
    outcome: str = "excess_12m"
    min_bin_rows: int = 100


@dataclass(frozen=True)
class LevelsParams:
    """Plan 6.5. Exit simulation (top_fraction, k_trail, signal_threshold) and the Monte Carlo hit
    table (market_return: fixed assumed annual market return, not a forecast; mc_paths; n_buckets
    volatility buckets; theta_nodes / barrier_nodes: the table grid, None = hitprob defaults)."""

    market_return: float = 0.06
    top_fraction: float = 0.20
    k_trail: tuple[float, ...] = (1.0, 1.5, 2.0)
    signal_threshold: float = 0.40
    mc_paths: int = 10_000
    n_buckets: int = 3
    seed: int = 0
    max_residuals_per_bucket: int | None = 2_000_000
    theta_nodes: tuple[float, ...] | None = None
    barrier_nodes: tuple[float, ...] | None = None


@dataclass(frozen=True)
class HitClassifierSettings:
    params: HitClassifierParams
    seed: int = 0


@dataclass(frozen=True)
class ModelConfig:
    validation: ValidationRules
    metrics: MetricRules
    criteria: Criteria
    costs_bps: float
    experiments_dir: Path
    baselines: BaselineParams
    lgbm: LGBMSettings
    ensemble: EnsembleParams
    vol: VolParams
    quantile: QuantileSettings
    calibration: CalibrationParams
    levels: LevelsParams
    hit_classifier: HitClassifierSettings
    raw: dict = field(repr=False, default_factory=dict)
    fast: bool = False


def _section(raw: dict, name: str, cls: type) -> dict:
    """The mapping under `name`, with exactly the fields of `cls` (a missing or misspelled key raises)."""
    if name not in raw:
        raise ValueError(f"model config: missing section {name!r}")
    sec = raw[name]
    want = {f.name for f in dataclasses.fields(cls)}
    if not isinstance(sec, dict) or set(sec) != want:
        got = set(sec) if isinstance(sec, dict) else set()
        raise ValueError(
            f"model config: section {name!r} has unknown keys {sorted(got - want)}, missing {sorted(want - got)}"
        )
    return sec


def _ints(sec: dict, cls: type):
    # no int(): a fractional purge would be truncated silently
    bad = sorted(k for k, v in sec.items() if type(v) is not int)
    if bad:
        raise ValueError(f"model config: {bad} must be whole numbers")
    return cls(**sec)


# optional model sections ---------------------------------------------------------------------------


def _optional(raw: dict, name: str, allowed: set[str]) -> dict:
    """The mapping under `name` ({} when absent); keys outside `allowed` raise."""
    sec = raw.get(name)
    if sec is None:
        return {}
    if not isinstance(sec, dict):
        raise ValueError(f"model config: section {name!r} must be a mapping")
    unknown = sorted(set(sec) - allowed)
    if unknown:
        raise ValueError(f"model config: section {name!r} has unknown keys {unknown}")
    return dict(sec)


def _fields(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def _build(cls: type, values: dict) -> Any:
    """cls(**values) with YAML lists turned into tuples (the params dataclasses are frozen/hashable)
    and a whole-number field given a fraction rejected."""
    types = {f.name: f.type for f in dataclasses.fields(cls)}
    out = {}
    for k, v in values.items():
        if isinstance(v, list):
            v = tuple(v)
        t = str(types.get(k, ""))
        if t == "int" and not (type(v) is int):
            raise ValueError(f"model config: {cls.__name__}.{k} must be a whole number, got {v!r}")
        if t == "float" and type(v) is int:
            v = float(v)
        if t.startswith("tuple[float") and isinstance(v, tuple):
            v = tuple(float(x) for x in v)
        out[k] = v
    try:
        return cls(**out)
    except (TypeError, ValueError) as e:
        raise ValueError(f"model config: {cls.__name__}: {e}") from e


def _model_sections(raw: dict) -> dict[str, Any]:
    # imported here: these modules import the validation runner, which imports this module
    from ltforecast.hitprob import HitClassifierParams
    from ltforecast.models.baselines import BaselineParams
    from ltforecast.models.lgbm import LGBMParams
    from ltforecast.models.quantile import QuantileParams
    from ltforecast.models.vol import VolParams

    lg = _optional(raw, "lgbm", _fields(LGBMParams) | {"seeds", "tune"})
    seeds = tuple(int(s) for s in lg.pop("seeds", (0, 1, 2, 3, 4)))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("model config: lgbm.seeds must be distinct integers")
    tune = lg.pop("tune", False)
    if not isinstance(tune, bool):
        raise ValueError("model config: lgbm.tune must be true or false")

    q = _optional(raw, "quantile", _fields(QuantileParams) | {"alphas", "seed"})
    q_alphas = tuple(float(a) for a in q.pop("alphas", (0.1, 0.5, 0.9)))
    if len(q_alphas) != 3:
        raise ValueError("model config: quantile.alphas must be three levels (lo, mid, hi)")
    q_seed = q.pop("seed", 0)

    h = _optional(raw, "hit_classifier", _fields(HitClassifierParams) | {"seed"})
    h_seed = h.pop("seed", 0)

    cal = _build(CalibrationParams, _optional(raw, "calibration", _fields(CalibrationParams)))
    if not 0.0 < cal.alpha < 1.0 or cal.window_months < 1 or cal.min_rows < 1:
        raise ValueError("model config: calibration needs 0 < alpha < 1, window_months >= 1, min_rows >= 1")
    if not 1 <= cal.min_dates <= cal.window_months:
        raise ValueError("model config: calibration needs 1 <= min_dates <= window_months")
    ens = _build(EnsembleParams, _optional(raw, "ensemble", _fields(EnsembleParams)))
    lev = _build(LevelsParams, _optional(raw, "levels", _fields(LevelsParams)))
    if not 0.0 < lev.top_fraction <= 1.0 or lev.mc_paths < 1 or lev.n_buckets < 1:
        raise ValueError("model config: levels needs 0 < top_fraction <= 1, mc_paths >= 1, n_buckets >= 1")
    return {
        "baselines": _build(BaselineParams, _optional(raw, "baselines", _fields(BaselineParams))),
        "lgbm": LGBMSettings(_build(LGBMParams, lg), seeds, tune),
        "ensemble": ens,
        "vol": _build(VolParams, _optional(raw, "vol", _fields(VolParams))),
        "quantile": QuantileSettings(_build(QuantileParams, q), q_alphas, int(q_seed)),
        "calibration": cal,
        "levels": lev,
        "hit_classifier": HitClassifierSettings(_build(HitClassifierParams, h), int(h_seed)),
    }


def _with_fast(raw: dict) -> dict:
    """`raw` with the `fast` overrides merged into the model sections (key by key)."""
    fast = raw.get("fast") or {}
    if not isinstance(fast, dict):
        raise ValueError("model config: `fast` must be a mapping of section -> overrides")
    bad = sorted(set(fast) - set(MODEL_SECTIONS))
    if bad:
        raise ValueError(f"model config: `fast` may only override model sections {list(MODEL_SECTIONS)}, not {bad}")
    out = dict(raw)
    for name, over in fast.items():
        if not isinstance(over, dict):
            raise ValueError(f"model config: fast.{name} must be a mapping")
        out[name] = {**(raw.get(name) or {}), **over}
    return out


def load_model_config(path: Path | str = DEFAULT_MODEL_CONFIG, *, fast: bool = False) -> ModelConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"model config: {path} is not a mapping")
    unknown = set(raw) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"model config: unknown top-level keys {sorted(unknown)}")
    for key in ("costs_bps", "experiments_dir"):
        if key not in raw:
            raise ValueError(f"model config: missing {key!r}")

    v = _ints(_section(raw, "validation", ValidationRules), ValidationRules)
    if min(v.test_months, v.min_train_months, v.inner_valid_months) < 1 or min(v.purge_months, v.embargo_months) < 0:
        raise ValueError("model config: validation month counts must be positive (purge/embargo >= 0)")
    # every fold must leave something to fit on after the inner validation window and its gap
    if v.min_train_months <= v.inner_valid_months + v.purge_months + v.embargo_months:
        raise ValueError("model config: min_train_months must exceed inner_valid_months + purge + embargo")

    c = _section(raw, "criteria", Criteria)
    lo, hi = (float(x) for x in c["coverage_range"])
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("model config: coverage_range must be [lo, hi] within [0, 1]")

    experiments_dir = Path(raw["experiments_dir"])
    if not experiments_dir.is_absolute():
        experiments_dir = path.parent.parent / experiments_dir

    fast_raw = _with_fast(raw)
    normal = _model_sections(raw)
    sections = _model_sections(fast_raw) if fast else normal  # the fast block is validated either way
    if not fast:
        _model_sections(fast_raw)

    return ModelConfig(
        validation=v,
        metrics=_ints(_section(raw, "metrics", MetricRules), MetricRules),
        criteria=Criteria(**{k: float(x) for k, x in c.items() if k != "coverage_range"}, coverage_range=(lo, hi)),
        costs_bps=float(raw["costs_bps"]),
        experiments_dir=experiments_dir,
        **sections,
        raw=raw,
        fast=fast,
    )

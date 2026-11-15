"""Typed access to config/model.yaml (validation rules, metric settings, pre-registered criteria)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ltforecast.config import REPO_ROOT

DEFAULT_MODEL_CONFIG = REPO_ROOT / "config" / "model.yaml"

# Top-level sections the loader understands. A later phase that adds a section adds its key here
# (and a dataclass); anything else in the file is a typo and raises.
KNOWN_KEYS = frozenset({"validation", "metrics", "criteria", "costs_bps", "experiments_dir"})


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
class ModelConfig:
    validation: ValidationRules
    metrics: MetricRules
    criteria: Criteria
    costs_bps: float
    experiments_dir: Path
    raw: dict = field(repr=False, default_factory=dict)


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


def load_model_config(path: Path | str = DEFAULT_MODEL_CONFIG) -> ModelConfig:
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

    return ModelConfig(
        validation=v,
        metrics=_ints(_section(raw, "metrics", MetricRules), MetricRules),
        criteria=Criteria(**{k: float(x) for k, x in c.items() if k != "coverage_range"}, coverage_range=(lo, hi)),
        costs_bps=float(raw["costs_bps"]),
        experiments_dir=experiments_dir,
        raw=raw,
    )

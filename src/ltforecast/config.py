"""Typed access to config/universe.yaml."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "universe.yaml"


@dataclass(frozen=True)
class Dates:
    history_start: dt.date
    dev_end: dt.date
    holdout_start: dt.date
    holdout_end: dt.date


@dataclass(frozen=True)
class UniverseRules:
    categories: tuple[str, ...]
    min_price: float
    min_market_cap: float
    cpi_series: str
    cpi_base_date: dt.date
    min_adv: float
    adv_window_days: int
    # Sharadar DAILY.marketcap is reported in millions of dollars.
    marketcap_unit: float = 1.0e6


@dataclass(frozen=True)
class PitRules:
    fundamental_lag_trading_days: int
    sf1_dimensions: tuple[str, ...]


@dataclass(frozen=True)
class DelistingRules:
    performance_return: float
    merger_return: float
    performance_actions: tuple[str, ...]
    merger_actions: tuple[str, ...]
    unknown_as: str


@dataclass(frozen=True)
class MacroSeries:
    id: str
    # Revised series (CPI) are taken as first-release vintages; market series are never revised.
    vintage: bool
    # Used when a series has no vintage history: available_date = observation date + lag.
    release_lag_days: int


@dataclass(frozen=True)
class QualityRules:
    max_daily_move_without_action: float
    max_monthly_count_change: float


@dataclass(frozen=True)
class BarrierRules:
    horizon: str
    # None = no barrier on that side
    k_tp: tuple[float | None, ...]
    k_sl: tuple[float | None, ...]
    tp_bounds: tuple[float, float]
    sl_bounds: tuple[float, float]
    sigma_window: int
    sigma_min_obs: int


@dataclass(frozen=True)
class LabelRules:
    horizons: dict[str, int]  # name -> trading days
    primary: str
    winsorize: tuple[float, float]
    vol_min_obs: int
    stale_price_days: int
    barriers: BarrierRules


@dataclass(frozen=True)
class Config:
    data_dir: Path
    dates: Dates
    universe: UniverseRules
    pit: PitRules
    delisting: DelistingRules
    macro: tuple[MacroSeries, ...]
    quality: QualityRules
    labels: LabelRules
    raw: dict = field(repr=False, default_factory=dict)


def _as_date(v) -> dt.date:
    return v if isinstance(v, dt.date) else dt.date.fromisoformat(str(v))


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    data_dir = Path(raw["data_dir"])
    if not data_dir.is_absolute():
        data_dir = path.parent.parent / data_dir

    d, u, p, dl, q = raw["dates"], raw["universe"], raw["pit"], raw["delisting"], raw["quality"]
    dates = Dates(**{k: _as_date(v) for k, v in d.items()})
    if not dates.dev_end < dates.holdout_start <= dates.holdout_end:
        raise ValueError("holdout must start after dev_end")

    lab, bar = raw["labels"], raw["labels"]["barriers"]
    horizons = {str(k): int(v) for k, v in lab["horizons"].items()}
    for name in (lab["primary"], bar["horizon"]):
        if name not in horizons:
            raise ValueError(f"labels horizon {name!r} not in {sorted(horizons)}")

    return Config(
        data_dir=data_dir,
        dates=dates,
        universe=UniverseRules(
            categories=tuple(u["categories"]),
            min_price=float(u["min_price"]),
            min_market_cap=float(u["min_market_cap"]),
            cpi_series=u["cpi_series"],
            cpi_base_date=_as_date(u["cpi_base_date"]),
            min_adv=float(u["min_adv"]),
            adv_window_days=int(u["adv_window_days"]),
            marketcap_unit=float(u.get("marketcap_unit", 1.0e6)),
        ),
        pit=PitRules(
            fundamental_lag_trading_days=int(p["fundamental_lag_trading_days"]),
            sf1_dimensions=tuple(p["sf1_dimensions"]),
        ),
        delisting=DelistingRules(
            performance_return=float(dl["performance_return"]),
            merger_return=float(dl["merger_return"]),
            performance_actions=tuple(dl["performance_actions"]),
            merger_actions=tuple(dl["merger_actions"]),
            unknown_as=dl["unknown_as"],
        ),
        macro=tuple(
            MacroSeries(id=s["id"], vintage=bool(s["vintage"]), release_lag_days=int(s["release_lag_days"]))
            for s in raw["macro"]["series"]
        ),
        quality=QualityRules(
            max_daily_move_without_action=float(q["max_daily_move_without_action"]),
            max_monthly_count_change=float(q["max_monthly_count_change"]),
        ),
        labels=LabelRules(
            horizons=horizons,
            primary=lab["primary"],
            winsorize=(float(lab["winsorize"][0]), float(lab["winsorize"][1])),
            vol_min_obs=int(lab["vol_min_obs"]),
            stale_price_days=int(lab["stale_price_days"]),
            barriers=BarrierRules(
                horizon=bar["horizon"],
                k_tp=tuple(None if k is None else float(k) for k in bar["k_tp"]),
                k_sl=tuple(None if k is None else float(k) for k in bar["k_sl"]),
                tp_bounds=(float(bar["tp_bounds"][0]), float(bar["tp_bounds"][1])),
                sl_bounds=(float(bar["sl_bounds"][0]), float(bar["sl_bounds"][1])),
                sigma_window=int(bar["sigma_window"]),
                sigma_min_obs=int(bar["sigma_min_obs"]),
            ),
        ),
        raw=raw,
    )

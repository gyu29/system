"""The one door to modelling data: features joined to labels, with the holdout locked.

Every dev-time loader goes through `modeling_frame` / `frame_from_tables`. Rows of the locked holdout
(and the gap before it) come back only with a `HoldoutToken`, which Phase 9 (holdout.py) issues once.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import polars as pl

from ltforecast.labels import period

if TYPE_CHECKING:
    from ltforecast.config import Config
    from ltforecast.features.catalog import Catalog
    from ltforecast.store import Store

KEYS = ("date", "security_id")
PERIODS = ("dev", "gap", "holdout", "live")
LOCKED_PERIODS = frozenset({"gap", "holdout"})

# How a label column is recognised (the `labels` table, see labels.build_labels). `excess_` also
# covers `excess_w_*`. The walk-forward runner strips these before a model predicts.
LABEL_PREFIXES = ("fwd_ret_", "bench_ret_", "excess_", "rank_", "realized_vol_", "label_end_")
LABEL_NAMES = ("target", "period")


class HoldoutLockedError(RuntimeError):
    pass


class HoldoutToken:
    """Permission to read gap/holdout rows. Only `HoldoutToken._issue()` builds one, and only
    holdout.py (Phase 9, the one-time evaluation) calls it."""

    __slots__ = ()
    _key = object()

    def __init__(self, key: object = None):
        if key is not HoldoutToken._key:
            raise HoldoutLockedError("a HoldoutToken is issued only by the Phase 9 holdout evaluation")

    @classmethod
    def _issue(cls) -> HoldoutToken:
        return cls(cls._key)


def label_columns(columns: Iterable[str]) -> list[str]:
    """The columns that carry future information (labels and `period`), in their given order."""
    return [c for c in columns if c in LABEL_NAMES or c.startswith(LABEL_PREFIXES)]


def _check_periods(periods: Sequence[str], unlock: HoldoutToken | None) -> None:
    unknown = set(periods) - set(PERIODS)
    if unknown or not periods:
        raise ValueError(f"periods must be a non-empty subset of {PERIODS}, got {tuple(periods)}")
    locked = sorted(LOCKED_PERIODS & set(periods))
    if locked and not isinstance(unlock, HoldoutToken):
        raise HoldoutLockedError(f"periods {locked} are locked until the Phase 9 holdout evaluation")


def frame_from_tables(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    catalog: Catalog,
    *,
    periods: Sequence[str] = ("dev",),
    unlock: HoldoutToken | None = None,
) -> pl.DataFrame:
    """`features` joined to `labels` on (security_id, date), keeping rows whose `period` is in `periods`.

    Columns: date, security_id, sector, catalog.model_columns(), the label columns, period.
    Sorted by date, security_id. `catalog` is anything with a `model_columns()` method.
    Both tables must hold one row per (security_id, date); a duplicate raises.
    """
    _check_periods(periods, unlock)
    model_cols = list(catalog.model_columns())
    missing = [c for c in ("sector", *model_cols) if c not in features.columns]
    if missing:
        raise ValueError(f"features table lacks columns {missing}")

    label_cols = [c for c in labels.columns if c not in KEYS]
    known = set(label_columns(label_cols))
    # the runner hides labels from predict() by name, so a label it cannot name would leak
    unnamed = [c for c in label_cols if c not in known]
    clash = sorted(set(label_columns(model_cols)) | (set(model_cols) & set(label_cols)))
    if unnamed or clash:
        raise ValueError(
            f"label columns not covered by dataset.label_columns: {unnamed}; features named like labels: {clash}"
        )

    out = features.select(*KEYS, "sector", *model_cols).join(
        labels.filter(pl.col("period").is_in(list(periods))),
        on=["security_id", "date"],
        how="inner",
        validate="1:1",
    )
    return out.select(*KEYS, "sector", *model_cols, *[c for c in label_cols if c != "period"], "period").sort(KEYS)


def modeling_frame(
    store: Store,
    cfg: Config,
    catalog: Catalog,
    *,
    periods: Sequence[str] = ("dev",),
    unlock: HoldoutToken | None = None,
) -> pl.DataFrame:
    """`frame_from_tables` on the stored `features` and `labels` tables.

    Locked label rows are never read from disk. The stored `period` must agree with `cfg.dates`: if the
    boundaries moved after the labels were built, a row stamped "dev" could sit inside the holdout.
    """
    _check_periods(periods, unlock)
    labels = store.scan("labels").filter(pl.col("period").is_in(list(periods))).collect()
    moved = labels.filter(pl.col("period") != period(cfg.dates))
    if moved.height:
        raise HoldoutLockedError(
            f"{moved.height} label rows have a `period` that disagrees with the configured dates; rebuild the labels"
        )
    features = store.scan("features").select(*KEYS, "sector", *catalog.model_columns()).collect()
    return frame_from_tables(features, labels, catalog, periods=periods, unlock=unlock)

"""Purged, embargoed walk-forward folds over month-end prediction dates (plan 4.1).

A 12-month label made at month end t is only known ~12 months later, so the training window for a
test block must stop `purge_months + embargo_months` month ends before the block's first date.
Dates are month-end TRADING days (2009-02-27, 2010-05-28), so "k months before" is counted by
position in the sorted list of distinct dates, never by calendar arithmetic.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import polars as pl

from ltforecast.modelconfig import ValidationRules


class LeakageError(AssertionError):
    pass


@dataclass(frozen=True)
class Fold:
    name: str  # "2010" (the test year)
    train_dates: tuple[dt.date, ...]
    test_dates: tuple[dt.date, ...]


def _block_name(block: int, rules: ValidationRules) -> str:
    first_month = block * rules.test_months
    year = rules.first_test_year + first_month // 12
    return str(year) if rules.test_months == 12 else f"{year}-{first_month % 12 + 1:02d}"


def walk_forward_folds(dates: Iterable[dt.date], rules: ValidationRules, end: dt.date) -> list[Fold]:
    """Expanding-window folds. `dates` = the distinct month-end prediction dates; dates after `end`
    (pass cfg.dates.dev_end) are dropped, so nothing later can reach any fold.

    Test blocks are `test_months` calendar months long, starting in January of `first_test_year`; the
    last one stops at `end` and may be partial. Train dates = every date before the block's first date
    except the last `purge_months + embargo_months` of them. Folds with fewer than `min_train_months`
    train dates are skipped.
    """
    ds = sorted({d for d in dates if d is not None and d <= end})
    gap = rules.purge_months + rules.embargo_months
    blocks: dict[int, list[int]] = {}
    for i, d in enumerate(ds):
        month = (d.year - rules.first_test_year) * 12 + d.month - 1
        if month >= 0:
            blocks.setdefault(month // rules.test_months, []).append(i)

    folds = []
    for block, idx in blocks.items():
        n_train = idx[0] - gap
        if n_train < rules.min_train_months:
            continue
        folds.append(Fold(_block_name(block, rules), tuple(ds[:n_train]), tuple(ds[i] for i in idx)))
    return folds


def inner_split(
    train_dates: Sequence[dt.date], rules: ValidationRules
) -> tuple[tuple[dt.date, ...], tuple[dt.date, ...]]:
    """(inner_train, inner_valid): inner_valid = the last `inner_valid_months` train dates; inner_train
    stops a further `purge_months + embargo_months` dates before them."""
    ds = sorted(train_dates)
    n_train = len(ds) - rules.inner_valid_months - rules.purge_months - rules.embargo_months
    if n_train < 1:
        raise ValueError(f"{len(ds)} train dates leave nothing to fit on after the inner validation split")
    return tuple(ds[:n_train]), tuple(ds[len(ds) - rules.inner_valid_months :])


def assert_purged(
    train_df: pl.DataFrame,
    first_test_date: dt.date,
    label_end_col: str = "label_end_12m",
    *,
    target: str = "target",
) -> None:
    """Raise LeakageError if any training row's label window ends on/after `first_test_date`.

    A null label end proves nothing, so it is a violation whenever the row has a target to train on.
    A row with neither (the window ran past the data) carries no label and is dropped before fitting,
    so it is allowed. Without a `target` column every null label end is a violation.
    """
    end = pl.col(label_end_col)
    labelled = pl.col(target).is_not_null() if target in train_df.columns else pl.lit(True)
    bad = train_df.filter((end >= first_test_date) | (end.is_null() & labelled))
    if bad.height:
        raise LeakageError(
            f"{bad.height} training rows have {label_end_col} on/after the first test date "
            f"{first_test_date} (or no label end at all):\n{bad.head(5)}"
        )

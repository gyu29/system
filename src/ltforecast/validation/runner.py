"""Walk-forward driver: fit on the purged past, predict the test block, collect out-of-fold scores."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import polars as pl

from ltforecast.dataset import label_columns
from ltforecast.modelconfig import ValidationRules
from ltforecast.validation.splitter import Fold, assert_purged, inner_split


class Model(Protocol):
    name: str

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None) -> Model: ...

    def predict(self, df: pl.DataFrame) -> np.ndarray: ...  # one score per row, higher = better


def walk_forward(
    make_model: Callable[[], Model],
    frame: pl.DataFrame,
    folds: Sequence[Fold],
    rules: ValidationRules,
    *,
    target: str = "target",
    label_end_col: str = "label_end_12m",
) -> pl.DataFrame:
    """Out-of-fold predictions. Columns: date, security_id, fold, pred; sorted by date, security_id.

    Per fold a fresh model is fit on (inner_train, inner_valid) from `inner_split`, both without
    null-target rows; the purged months between the two are not passed to the model. `assert_purged`
    is checked for train vs the first test date and for inner_train vs the first inner_valid date.
    Every test row is scored, labelled or not, from a frame with all label columns (and `target`)
    removed, so a model cannot read a test label.
    """
    hidden = sorted(set(label_columns(frame.columns)) | {target})
    labelled = pl.col(target).is_not_null()
    # narrow copy for the outer purge check, so the full training frame is not materialized twice
    ends = frame.filter(labelled).select("date", target, label_end_col)

    out = []
    for fold in folds:
        in_train = pl.col("date").is_in(list(fold.train_dates))
        assert_purged(ends.filter(in_train), fold.test_dates[0], label_end_col, target=target)

        train_dates, valid_dates = inner_split(fold.train_dates, rules)
        train = frame.filter(pl.col("date").is_in(list(train_dates)) & labelled)
        valid = frame.filter(pl.col("date").is_in(list(valid_dates)) & labelled)
        if train.is_empty() or valid.is_empty():
            raise ValueError(f"fold {fold.name}: no labelled rows to train or validate on")
        assert_purged(train, valid_dates[0], label_end_col, target=target)

        test = frame.filter(pl.col("date").is_in(list(fold.test_dates)))
        if test.is_empty():
            raise ValueError(f"fold {fold.name}: no rows on its test dates")
        model = make_model()
        model.fit(train, valid)
        pred = np.asarray(model.predict(test.drop(hidden)), dtype=np.float64)
        if pred.shape != (test.height,):
            raise ValueError(f"fold {fold.name}: predict returned shape {pred.shape} for {test.height} rows")
        if not np.isfinite(pred).all():
            raise ValueError(f"fold {fold.name}: predict returned {int((~np.isfinite(pred)).sum())} non-finite values")
        out.append(
            test.select("date", "security_id").with_columns(
                pl.lit(fold.name).alias("fold"), pl.Series("pred", pred, dtype=pl.Float64)
            )
        )
    if not out:
        return pl.DataFrame(
            schema={"date": pl.Date, "security_id": pl.Int64, "fold": pl.String, "pred": pl.Float64}
        )
    return pl.concat(out).sort("date", "security_id")

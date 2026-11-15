"""Leakage tests for features (plan 3.3) and the univariate IC report.

- `assert_sources_not_after`: no feature row was built from an input dated after its prediction date.
- `shift_test`: use last month's feature value in place of this month's. An honest signal loses a
  little, or moves by noise; a feature whose IC clearly IMPROVES (grows stronger in the signal's own
  direction, positive or negative) when it is made one month staler is misaligned with the labels,
  i.e. something leaks.
- `shuffle_labels`: labels permuted within each date; anything fitted or scored on them should show
  an IC of about zero.

The functions take a modelling frame (`dataset.modeling_frame`: ranked features joined to labels).
Callers pass development rows only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ltforecast.dataset import label_columns
from ltforecast.pit.asof import assert_no_lookahead
from ltforecast.validation.metrics import ic_by_date, ic_summary, paired_ic_test

SRC_PREFIX = "src_date_"
KEYS = ("date", "security_id")


def assert_sources_not_after(raw: pl.DataFrame, date_col: str = "date") -> None:
    """Raise LookaheadError if any `src_date_*` column of `raw` is after the row's `date_col`.
    A frame without such columns raises ValueError: the check would prove nothing."""
    src = [c for c in raw.columns if c.startswith(SRC_PREFIX)]
    if not src:
        raise ValueError(f"no {SRC_PREFIX}* columns to check")
    for c in src:
        assert_no_lookahead(raw, date_col, c)


def univariate_ic(
    frame: pl.DataFrame, features: Sequence[str], target: str = "target", *, lags: int = 12, min_n: int = 10
) -> pl.DataFrame:
    """Rank IC of each feature against `target`, per date, summarized.

    Columns: feature, mean_ic, std_ic, ic_ir, nw_tstat, n_months, pct_positive (`metrics.ic_summary`),
    one row per feature in the given order. Dates with fewer than `min_n` complete pairs are skipped;
    a feature with no usable date has n_months 0 and NaN statistics. Date-level (context) features are
    constant within a date and have no cross-sectional IC: do not pass them.
    """
    rows = []
    for f in features:
        ic = ic_by_date(frame.select("date", f, target), f, target, min_n)
        rows.append({"feature": f, **ic_summary(ic, lags)})
    schema = {"feature": pl.String, "mean_ic": pl.Float64, "std_ic": pl.Float64, "ic_ir": pl.Float64,
              "nw_tstat": pl.Float64, "n_months": pl.Int64, "pct_positive": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def shift_test(
    frame: pl.DataFrame,
    features: Sequence[str],
    target: str = "target",
    *,
    lags: int = 12,
    min_n: int = 10,
    min_gain: float = 0.02,
    min_tstat: float = 3.0,
) -> pl.DataFrame:
    """Per feature, the IC of its current value against the IC of its value one month earlier.

    The shifted value at date t is the same security's value at the previous date of the frame. Both
    ICs are computed on the rows where both values exist, so the comparison is paired.
    Columns: feature, ic, ic_shifted, diff (= ic_shifted - ic, mean over the common months), nw_tstat
    (Newey-West t-stat of the monthly differences), orientation, gain, gain_tstat, n_months, flagged.
    "Better" means a stronger IC in the signal's own direction, so the test gives the same verdict for
    a feature and its negation. `orientation` = sign(ic + ic_shifted), i.e. the sign of the larger of
    the two ICs in absolute value (+1 if that is 0 or NaN); it is taken from the data, not from the
    catalog sign, so a leak whose current IC is about 0 is oriented by its stale IC. `gain` =
    orientation * diff and `gain_tstat` = orientation * nw_tstat: positive when the stale value is
    stronger, negative for the normal decay of an honest signal of either sign.
    `flagged` = the stale value is clearly better: gain >= min_gain and gain_tstat >= min_tstat. Both are
    needed. The two ICs are almost perfectly correlated month by month, so a harmless gain of 0.005
    (e.g. a price ratio that carries one month less of the latest return) is easily "significant";
    and with few months a large gain can be luck. The CSV keeps every number for inspection.
    """
    features = list(features)
    dates = frame.select("date").unique().sort("date")
    following = dates.with_columns(pl.col("date").shift(-1).alias("_next")).drop_nulls()
    shifted = (
        frame.select(*KEYS, *features)
        .join(following, on="date")
        .select(pl.col("_next").alias("date"), "security_id", *[pl.col(f).alias(f"{f}__shifted") for f in features])
    )
    both = frame.select(*KEYS, target, *features).join(shifted, on=list(KEYS), how="inner")

    rows = []
    for f in features:
        s = f"{f}__shifted"
        pair = both.select("date", f, s, target).drop_nulls()
        ic, ic_s = ic_by_date(pair, f, target, min_n), ic_by_date(pair, s, target, min_n)
        test = paired_ic_test(ic_s, ic, lags)
        mean_ic = float(ic["ic"].mean()) if ic.height else float("nan")
        mean_ic_s = float(ic_s["ic"].mean()) if ic_s.height else float("nan")
        total = mean_ic + mean_ic_s
        orientation = -1 if np.isfinite(total) and total < 0 else 1
        diff, t = float(test["mean_diff"]), float(test["nw_tstat"])
        gain, gain_t = orientation * diff, orientation * t
        rows.append({
            "feature": f,
            "ic": mean_ic,
            "ic_shifted": mean_ic_s,
            "diff": diff,
            "nw_tstat": t,
            "orientation": orientation,
            "gain": gain,
            "gain_tstat": gain_t,
            "n_months": int(test["n_months"]),
            "flagged": bool(gain >= min_gain and np.isfinite(gain_t) and gain_t >= min_tstat),
        })
    schema = {"feature": pl.String, "ic": pl.Float64, "ic_shifted": pl.Float64, "diff": pl.Float64,
              "nw_tstat": pl.Float64, "orientation": pl.Int64, "gain": pl.Float64, "gain_tstat": pl.Float64,
              "n_months": pl.Int64, "flagged": pl.Boolean}
    return pl.DataFrame(rows, schema=schema)


def shuffle_labels(frame: pl.DataFrame, seed: int = 0, cols: Sequence[str] | None = None) -> pl.DataFrame:
    """`frame` with its labels permuted across the stocks of each date (one permutation per date,
    applied to all of `cols` together, so a row's target and returns still belong to one stock).

    `cols` defaults to every label column except `period` and the `label_end_*` dates, which describe
    the row's own window and are needed for purging. Row order and all other columns are unchanged.
    """
    if cols is None:
        cols = [c for c in label_columns(frame.columns) if c != "period" and not c.startswith("label_end_")]
    cols = list(cols)
    key = pl.Series("_shuffle_key", np.random.default_rng(seed).random(frame.height))
    return (
        frame.with_columns(key)
        .with_columns(pl.col(cols).sort_by("_shuffle_key").over("date"))
        .drop("_shuffle_key")
    )

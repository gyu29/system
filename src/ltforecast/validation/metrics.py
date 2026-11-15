"""Forecast-quality metrics (plan 4.2) and the multiple-testing haircut (plan 4.3).

Conventions
- Every cross-sectional statistic is computed per prediction date first and then aggregated across
  dates, so a month counts once however many stocks it holds. Frames need a `date` column.
- Rank IC is the Spearman correlation: Pearson correlation of average ranks (ties share a rank).
  A date on which either side is constant carries no ranking information and gets IC 0.
- Newey-West uses 1/n autocovariances and Bartlett weights w_j = 1 - j / (lags + 1); with lags = 0 the
  standard error is sqrt(population variance / n). A series is taken to be consecutive months.
- Buckets ("deciles") come from average ranks, bucket = ceil(rank * n / count), so tied predictions
  land in the same bucket and the result does not depend on row order.
- Rows with a null or non-finite input are left out of the statistic that needs it.
- Everything runs eagerly: the default lazy engine adds floats in a run-dependent order, which moves
  the last bit of a mean from one run to the next. The same input must give the same report.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm

from ltforecast.quality import CheckResult

REGIMES = ("bull", "bear", "high_vol")
SIZE_TERCILES = ("small", "mid", "large")
_EPS = 1e-12  # slack for comparing a statistic to a threshold it should be allowed to equal


# rank ic -------------------------------------------------------------------------------------------


def _finite_pairs(df: pl.DataFrame, keys: Sequence[str], a: str, b: str) -> pl.DataFrame:
    """Rows where both columns are finite, as Float64 columns `_a` and `_b` next to `keys`."""
    return df.select(
        *keys, pl.col(a).cast(pl.Float64).alias("_a"), pl.col(b).cast(pl.Float64).alias("_b")
    ).filter(pl.col("_a").is_finite() & pl.col("_b").is_finite())


def _rank_ic(df: pl.DataFrame, pred: str, target: str, keys: Sequence[str], min_n: int) -> pl.DataFrame:
    """Spearman correlation of pred and target within each `keys` group. Columns: *keys, ic, n."""
    constant = (pl.col("_a").min() == pl.col("_a").max()) | (pl.col("_b").min() == pl.col("_b").max())
    return (
        _finite_pairs(df, keys, pred, target)
        .with_columns(pl.col("_a").rank("average").over(keys), pl.col("_b").rank("average").over(keys))
        .group_by(keys)
        .agg(
            pl.when(constant).then(0.0).otherwise(pl.corr("_a", "_b")).alias("ic"),
            pl.len().cast(pl.Int64).alias("n"),
        )
        .filter(pl.col("n") >= min_n)
        .sort(keys)
    )


def ic_by_date(df: pl.DataFrame, pred: str, target: str, min_n: int = 10) -> pl.DataFrame:
    """Rank IC per prediction date. Columns: date, ic, n (pairs used). Sorted by date.
    Dates with fewer than `min_n` complete pairs are dropped."""
    return _rank_ic(df, pred, target, ["date"], min_n)


def _values(x: Any) -> np.ndarray:
    """A float array without nulls/NaN. A frame contributes its `ic` column in date order."""
    if isinstance(x, pl.DataFrame):
        x = (x.sort("date") if "date" in x.columns else x)["ic"]
    a = x.cast(pl.Float64).to_numpy() if isinstance(x, pl.Series) else np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def newey_west_se(x: Any, lags: int) -> float:
    """Newey-West standard error of the mean of a time series.

    se = sqrt(S / n), S = g_0 + 2 * sum_{j=1..lags} (1 - j / (lags + 1)) * g_j, with
    g_j = (1 / n) * sum_t (x_t - mean)(x_{t-j} - mean). lags = 0 gives sqrt(population variance / n).
    NaN when there are fewer than two observations."""
    a = _values(x)
    n = a.size
    if n < 2:
        return math.nan
    d = a - a.mean()
    s = float(d @ d) / n
    for j in range(1, min(lags, n - 1) + 1):
        s += 2.0 * (1.0 - j / (lags + 1)) * float(d[j:] @ d[:-j]) / n
    return math.sqrt(max(s, 0.0) / n)


def newey_west_tstat(x: Any, lags: int) -> float:
    """mean / newey_west_se. A series with no variation gives 0 when its mean is 0, else +/-inf."""
    a = _values(x)
    if a.size < 2:
        return math.nan
    mean, se = float(a.mean()), newey_west_se(a, lags)
    if se == 0.0:
        return 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    return mean / se


def ic_summary(ic: Any, lags: int = 12) -> dict[str, float | int]:
    """Summary of a monthly IC series (the frame from `ic_by_date`, or a series/array in date order).

    mean_ic, std_ic (sample, ddof = 1), ic_ir = mean / std (per month, not annualized),
    nw_tstat, n_months, pct_positive (share of months with IC > 0)."""
    a = _values(ic)
    n = int(a.size)
    if n == 0:
        return {"mean_ic": math.nan, "std_ic": math.nan, "ic_ir": math.nan, "nw_tstat": math.nan,
                "n_months": 0, "pct_positive": math.nan}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else math.nan
    return {
        "mean_ic": mean,
        "std_ic": std,
        "ic_ir": mean / std if std > 0 else math.nan,
        "nw_tstat": newey_west_tstat(a, lags),
        "n_months": n,
        "pct_positive": float((a > 0).mean()),
    }


def ic_breakdown(
    df: pl.DataFrame, pred: str, target: str, by: str, *, lags: int = 12, min_n: int = 10
) -> pl.DataFrame:
    """Rank IC within each (`by` group, date), summarized per group. Columns: <by>, mean_ic, nw_tstat,
    n_months. Rows with a null group are left out. When `by` is a date-level label (a market regime),
    a group's months are not consecutive and the t-stat treats them as if they were."""
    ics = _rank_ic(df.filter(pl.col(by).is_not_null()), pred, target, [by, "date"], min_n)
    rows = []
    for (key,), group in ics.group_by(by, maintain_order=True):
        s = ic_summary(group, lags)
        rows.append({by: key, "mean_ic": s["mean_ic"], "nw_tstat": s["nw_tstat"], "n_months": s["n_months"]})
    schema = {by: ics.schema[by], "mean_ic": pl.Float64, "nw_tstat": pl.Float64, "n_months": pl.Int64}
    return pl.DataFrame(rows, schema=schema)


def paired_ic_test(ic_a: pl.DataFrame, ic_b: pl.DataFrame, lags: int = 12) -> dict[str, float | int]:
    """Is model A's IC higher than model B's? Newey-West t-test of the mean of (ic_a - ic_b) over the
    dates both have. Returns mean_diff, nw_tstat, n_months."""
    diff = (
        ic_a.select("date", "ic")
        .join(ic_b.select("date", pl.col("ic").alias("ic_b")), on="date", how="inner")
        .sort("date")
        .select((pl.col("ic") - pl.col("ic_b")).alias("ic"))
    )
    a = _values(diff["ic"])
    return {
        "mean_diff": float(a.mean()) if a.size else math.nan,
        "nw_tstat": newey_west_tstat(a, lags),
        "n_months": int(a.size),
    }


# deciles and spreads -------------------------------------------------------------------------------


def _bucket(col: str, n: int, by: str = "date") -> pl.Expr:
    """1..n by average rank within `by`: ceil(rank * n / count)."""
    rank, count = pl.col(col).rank("average").over(by), pl.col(col).count().over(by)
    return (rank * n / count).ceil().clip(1, n).cast(pl.Int32)


def decile_returns_by_date(df: pl.DataFrame, pred: str, ret: str, n: int = 10) -> pl.DataFrame:
    """Mean `ret` of each prediction bucket on each date. Columns: date, decile (1 = lowest
    predictions), mean_ret, n. Dates with fewer than `n` complete rows cannot fill the buckets and
    are dropped."""
    return (
        _finite_pairs(df, ["date"], pred, ret)
        .filter(pl.len().over("date") >= n)
        .with_columns(_bucket("_a", n).alias("decile"))
        .group_by("date", "decile")
        .agg(pl.col("_b").mean().alias("mean_ret"), pl.len().cast(pl.Int64).alias("n"))
        .sort("date", "decile")
    )


def decile_table(df: pl.DataFrame, pred: str, ret: str, n: int = 10) -> pl.DataFrame:
    """Mean return per predicted decile: deciles are formed per date and the per-date means are then
    averaged across dates (each date weighs the same). Columns: decile, mean_ret, n (stock-months)."""
    return (
        decile_returns_by_date(df, pred, ret, n)
        .group_by("decile")
        .agg(pl.col("mean_ret").mean(), pl.col("n").sum())
        .sort("decile")
    )


def monotonicity(table: pl.DataFrame) -> float:
    """Spearman correlation of decile number and mean_ret: +1 = returns rise with every decile."""
    if table.height < 2:
        return math.nan
    rho = table.select(pl.corr(pl.col("decile").rank("average"), pl.col("mean_ret").rank("average"))).item()
    return math.nan if rho is None else float(rho)


def spread_by_date(df: pl.DataFrame, pred: str, ret: str, n: int = 10) -> pl.DataFrame:
    """Top-minus-bottom mean `ret` per date. Columns: date, spread.

    Tied predictions share a bucket, so a large tie at either end (about 20% of a date's names or
    more, e.g. one tree leaf or a discrete score) leaves bucket `n` or bucket 1 empty. The spread is
    therefore taken between the highest and the lowest bucket that is populated on the date. Only a
    date with a single populated bucket (every prediction tied), or with fewer than `n` complete rows,
    has no spread and is dropped; `spread_coverage` counts those dates."""
    return (
        decile_returns_by_date(df, pred, ret, n)
        .group_by("date")
        .agg(
            (
                pl.col("mean_ret").sort_by("decile").last() - pl.col("mean_ret").sort_by("decile").first()
            ).alias("spread"),
            pl.col("decile").n_unique().alias("_buckets"),
        )
        .filter(pl.col("_buckets") >= 2)
        .select("date", "spread")
        .sort("date")
    )


def spread_coverage(df: pl.DataFrame, pred: str, ret: str, n: int = 10) -> dict[str, float | int]:
    """How much of the data the spread rests on. n_spread_dates_dropped = dates with at least one
    complete (pred, ret) pair but no spread (fewer than `n` such rows, or every prediction tied);
    tied_pred_share = share of the complete rows on dates with >= `n` of them whose prediction equals
    another prediction on the same date (NaN without any such row)."""
    pairs = _finite_pairs(df, ["date"], pred, ret)
    kept = spread_by_date(df, pred, ret, n).height
    rows = pairs.filter(pl.len().over("date") >= n)
    tied = rows.select((pl.len().over("date", "_a") > 1).mean()).item() if rows.height else None
    return {
        "n_spread_dates_dropped": int(pairs["date"].n_unique()) - kept,
        "tied_pred_share": math.nan if tied is None else float(tied),
    }


def spread_by_year(df: pl.DataFrame, pred: str, ret: str, n: int = 10) -> pl.DataFrame:
    """Mean monthly top-minus-bottom spread per calendar year of the PREDICTION date (a December
    forecast belongs to its own year although its return is earned in the next).
    Columns: year, spread, n_months."""
    return (
        spread_by_date(df, pred, ret, n)
        .group_by(pl.col("date").dt.year().cast(pl.Int64).alias("year"))
        .agg(pl.col("spread").mean(), pl.len().cast(pl.Int64).alias("n_months"))
        .sort("year")
    )


def pct_positive_years(spreads: pl.DataFrame) -> float:
    """Share of years with a strictly positive spread. NaN without any year."""
    s = spreads["spread"].drop_nulls()
    return float((s > 0).mean()) if s.len() else math.nan


# calibration and coverage --------------------------------------------------------------------------


def calibration_table(df: pl.DataFrame, prob: str, outcome: str, bins: int = 10) -> pl.DataFrame:
    """Forecast probability against realized frequency in `bins` equal-width probability bins
    [k / bins, (k + 1) / bins); the last bin includes 1. `outcome` is boolean or 0/1.
    Columns: bin_lo, bin_hi, mean_prob, realized, n. Empty bins are omitted."""
    return (
        _finite_pairs(df, [], prob, outcome)
        .with_columns((pl.col("_a") * bins).floor().clip(0, bins - 1).cast(pl.Int64).alias("bin"))
        .group_by("bin")
        .agg(
            pl.col("_a").mean().alias("mean_prob"),
            pl.col("_b").mean().alias("realized"),
            pl.len().cast(pl.Int64).alias("n"),
        )
        .sort("bin")
        .select(
            (pl.col("bin") / bins).alias("bin_lo"),
            ((pl.col("bin") + 1) / bins).alias("bin_hi"),
            "mean_prob",
            "realized",
            "n",
        )
    )


def max_calibration_gap(table: pl.DataFrame, min_n: int = 1) -> float:
    """Largest |mean_prob - realized| over bins holding at least `min_n` forecasts. NaN if none do."""
    gap = table.filter(pl.col("n") >= min_n).select((pl.col("mean_prob") - pl.col("realized")).abs().max()).item()
    return math.nan if gap is None else float(gap)


def interval_coverage(df: pl.DataFrame, lo: str, hi: str, actual: str) -> float:
    """Share of rows with lo <= actual <= hi, among rows where all three are present. NaN if none."""
    cols = [pl.col(c).cast(pl.Float64) for c in (lo, hi, actual)]
    rows = df.select(cols).drop_nulls().filter(pl.all_horizontal(pl.all().is_finite()))
    if rows.is_empty():
        return math.nan
    return float(rows.select(pl.col(actual).is_between(pl.col(lo), pl.col(hi)).mean()).item())


# breakdown keys ------------------------------------------------------------------------------------


def size_terciles(df: pl.DataFrame, mcap_col: str = "market_cap") -> pl.DataFrame:
    """Adds `size_tercile` (small | mid | large): thirds of `mcap_col` within each date. The market
    cap must be the one known at the prediction date. Null where the market cap is null."""
    k = _bucket(mcap_col, 3)
    return df.with_columns(
        pl.when(k == 1).then(pl.lit("small")).when(k == 2).then(pl.lit("mid")).when(k == 3).then(pl.lit("large"))
        .alias("size_tercile")
    )


def market_regimes(
    benchmark: pl.DataFrame,
    dates: Iterable[dt.date],
    *,
    vol_window: int = 63,
    ret_window: int = 252,
    min_history: int = 252,
) -> pl.DataFrame:
    """The market regime at each date, from benchmark rows on or before it only. Columns: date, regime.

    high_vol  the trailing `vol_window`-day std of bench_ret is above the 2/3 quantile (linear
              interpolation) of all such daily values up to and including that day, once at least
              `min_history` of them exist;
    bull/bear otherwise, by the sign of the trailing `ret_window`-day benchmark return (> 0 = bull).
    Null while there are fewer than `ret_window` earlier days, or when the benchmark has no row within
    7 calendar days before the date. `benchmark` needs date and bench_ret; leading days before the
    index has any member (n_members == 0) are ignored if that column is present.
    """
    b = benchmark.sort("date")
    if "n_members" in b.columns:
        b = b.filter(pl.col("n_members").cum_max() > 0)
    level = (1 + pl.col("bench_ret").fill_null(0.0)).cum_prod()
    b = b.select(
        pl.col("date").alias("bench_date"),
        pl.col("bench_ret").fill_null(0.0).rolling_std(vol_window, min_samples=vol_window).alias("vol"),
        (level / level.shift(ret_window) - 1).alias("ret"),
    ).with_row_index("i")
    at = (
        pl.DataFrame({"date": pl.Series("date", list(dates), dtype=pl.Date)})
        .unique()
        .sort("date")
        .join_asof(b, left_on="date", right_on="bench_date", strategy="backward", tolerance="7d")
    )
    # the threshold at day i sees vols[: i + 1] only, so later benchmark rows cannot change it
    vols = b["vol"].to_numpy()
    high = []
    for i in at["i"].to_list():
        hist = vols[: i + 1][np.isfinite(vols[: i + 1])] if i is not None else np.empty(0)
        is_high = hist.size >= min_history and np.isfinite(vols[i]) and vols[i] > np.quantile(hist, 2 / 3)
        high.append(bool(is_high))
    return at.with_columns(pl.Series("high", high, dtype=pl.Boolean)).select(
        "date",
        pl.when(pl.col("ret").is_null())
        .then(None)
        .when(pl.col("high"))
        .then(pl.lit("high_vol"))
        .when(pl.col("ret") > 0)
        .then(pl.lit("bull"))
        .otherwise(pl.lit("bear"))
        .alias("regime"),
    )


# multiple-testing haircut --------------------------------------------------------------------------


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum Sharpe ratio among `n_trials` unskilled strategies whose estimated Sharpe ratios
    have cross-trial variance `var_sr` (Bailey & Lopez de Prado 2014, eq. for SR0):
    sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e))), g = Euler-Mascheroni constant.
    One trial is no selection: 0."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    g = float(np.euler_gamma)
    z = (1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(var_sr) * float(z)


def deflated_sharpe(
    sr: float, n_obs: int, n_trials: int, var_sr: float, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """Deflated Sharpe ratio: the probability that the true Sharpe ratio is above the expected maximum
    of `n_trials` unskilled trials, Phi((SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2)).

    `sr` and `var_sr` are per period (not annualized), `n_obs` = T return observations, `skew` and
    `kurt` (non-excess; 3 = normal) describe the returns."""
    sr0 = expected_max_sharpe(n_trials, var_sr)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if n_obs < 2 or denom <= 0:
        return math.nan
    return float(norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)))


# report --------------------------------------------------------------------------------------------


def evaluate(
    preds: pl.DataFrame,
    *,
    pred: str = "pred",
    target: str = "target",
    ret: str = "excess_12m",
    lags: int = 12,
    n_deciles: int = 10,
    min_n: int = 10,
) -> dict[str, Any]:
    """The standard report for out-of-fold predictions joined with their labels
    (columns date, security_id, `pred`, `target`, `ret`).

    Keys: ic (the `ic_summary` dict), ic_by_date, deciles, monotonicity, spread_by_year,
    pct_positive_years, mean_spread (mean of the monthly spreads), n_spread_dates_dropped and
    tied_pred_share (`spread_coverage`), n_obs (rows with pred and target), n_dates. Frames stay Polars frames; `to_jsonable` turns the whole report into JSON-ready objects."""
    ics = ic_by_date(preds, pred, target, min_n)
    deciles = decile_table(preds, pred, ret, n_deciles)
    monthly = spread_by_date(preds, pred, ret, n_deciles)
    by_year = (
        monthly.group_by(pl.col("date").dt.year().cast(pl.Int64).alias("year"))
        .agg(pl.col("spread").mean(), pl.len().cast(pl.Int64).alias("n_months"))
        .sort("year")
    )
    return {
        "ic": ic_summary(ics, lags),
        "ic_by_date": ics,
        "deciles": deciles,
        "monotonicity": monotonicity(deciles),
        "spread_by_year": by_year,
        "pct_positive_years": pct_positive_years(by_year),
        "mean_spread": float(monthly["spread"].mean()) if monthly.height else math.nan,
        **spread_coverage(preds, pred, ret, n_deciles),
        "n_obs": int(ics["n"].sum()),
        "n_dates": ics.height,
    }


def to_jsonable(obj: Any) -> Any:
    """Recursively convert a report to plain JSON-ready objects: frames -> lists of row dicts,
    dates -> ISO strings, numpy scalars -> Python numbers, NaN/inf -> None, dataclasses -> dicts."""
    if isinstance(obj, pl.DataFrame):
        return [to_jsonable(row) for row in obj.to_dicts()]
    if isinstance(obj, pl.Series):
        return [to_jsonable(v) for v in obj.to_list()]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k if isinstance(k, str) else str(to_jsonable(k)): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    return obj


def check_criteria(
    report: dict[str, Any],
    criteria: Any,
    *,
    vs_baseline: dict[str, Any] | None = None,
    coverage: float | None = None,
    calibration_gap: float | None = None,
) -> list[CheckResult]:
    """The pre-registered success criteria 1-5 of plan section 0 as pass | fail | skip rows.

    report           from `evaluate` (criteria 1 and 2);
    vs_baseline      `paired_ic_test(model, baseline)` for criterion 3: the IC gain must be positive with
                     a Newey-West t-stat >= criteria.min_ic_tstat;
    coverage         `interval_coverage` of the 10-90% range, for criterion 4;
    calibration_gap  `max_calibration_gap` of P(beat market), for criterion 5.
    A criterion whose input is not supplied is "skip"; a NaN statistic fails.
    `criteria` needs min_rank_ic, min_ic_tstat, min_positive_spread_years, coverage_range,
    calibration_tolerance (modelconfig.Criteria)."""
    ic = report["ic"]
    ic_ok = ic["mean_ic"] >= criteria.min_rank_ic - _EPS and ic["nw_tstat"] >= criteria.min_ic_tstat - _EPS
    years = report["spread_by_year"].height
    pct = report["pct_positive_years"]
    out = [
        _check(
            "rank_ic",
            ic_ok,
            f"mean rank IC {ic['mean_ic']:.4f} (need >= {criteria.min_rank_ic}), Newey-West t "
            f"{ic['nw_tstat']:.2f} (need >= {criteria.min_ic_tstat}), {ic['n_months']} months",
        ),
        _check(
            "spread_positive_years",
            pct >= criteria.min_positive_spread_years - _EPS,
            f"top-minus-bottom spread positive in {pct:.0%} of {years} years "
            f"(need >= {criteria.min_positive_spread_years:.0%})",
        ),
    ]
    if vs_baseline is None:
        out.append(CheckResult("beats_baseline", "skip", "no paired baseline comparison supplied"))
    else:
        diff, t = vs_baseline["mean_diff"], vs_baseline["nw_tstat"]
        out.append(_check(
            "beats_baseline",
            diff > 0 and t >= criteria.min_ic_tstat - _EPS,
            f"IC gain over baseline {diff:+.4f}, Newey-West t {t:.2f} (need >= {criteria.min_ic_tstat}), "
            f"{vs_baseline['n_months']} months",
        ))
    if coverage is None:
        out.append(CheckResult("interval_coverage", "skip", "no interval coverage supplied"))
    else:
        lo, hi = criteria.coverage_range
        out.append(_check(
            "interval_coverage",
            lo - _EPS <= coverage <= hi + _EPS,
            f"10-90% range holds {coverage:.1%} of outcomes (need {lo:.0%}-{hi:.0%})",
        ))
    if calibration_gap is None:
        out.append(CheckResult("calibration", "skip", "no calibration gap supplied"))
    else:
        out.append(_check(
            "calibration",
            calibration_gap <= criteria.calibration_tolerance + _EPS,
            f"largest gap between forecast and realized P(beat market) {calibration_gap:.1%} "
            f"(need <= {criteria.calibration_tolerance:.0%})",
        ))
    return out


def _check(name: str, ok: bool, detail: str) -> CheckResult:
    return CheckResult(name, "pass" if ok else "fail", detail)

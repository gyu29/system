"""Exit-rule simulation and the choice of the TP/SL multipliers (plan 6.5, "Choosing the multipliers").

Conventions
- A position = one (security_id, entry date) from `entries`. Its path is the total-return price ratio
  vs the entry close on trading days 1..h after entry (`Paths.close`, `Paths.open`, shape (n, h)),
  NaN on days without a price and after the data ends. A delisting inside the window is one extra day
  that opens and closes at last close * (1 + delisting return), exactly as `levels._price_paths` builds it.
- Exit rules return `(exit_day, exit_ret)`: exit_day in 1..h (trading days after entry, int64) and
  exit_ret = fill / entry - 1 (float64, gross of costs). Crossings are checked on closes; if the day
  opened beyond the level the fill is that open (the gap rule of `levels`), else the level itself.
  No exit inside the window: the last available close in the window (as `levels.barrier_labels`).
- Portfolio: each entry date is a sleeve of equal-weight positions held up to h days; a position's
  value follows its path until its exit and then sits in cash at 0% until the sleeve's horizon. A
  one-way `cost` is charged at entry (so it lowers the sleeve's first-day return) and at exit. The
  daily portfolio return is the mean of the daily returns of the sleeves active that day.
- `max_drawdown` is reported as a return (<= 0); "no worse than hold" means `max_dd >= hold max_dd`.

Scale: paths are float32 (n, h) matrices (80k positions x 252 days = 80 MB each); rules are vectorized
over positions and `select_pair` loops over the 5 + 5 barrier levels and the pairs only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ltforecast.config import BarrierRules
from ltforecast.levels import _first_hit, _price_paths, level_ratio
from ltforecast.validation.experiments import log_run, selection_guard

TRADING_DAYS_PER_YEAR = 252
SIGNAL_THRESHOLD = 0.40     # plan 6.5: sell when the model rank drops below the 40th percentile
TOP_FRACTION = 0.20         # plan 6.5: buy the top quintile each month
DEFAULT_K_TRAIL = (1.0, 1.5, 2.0)
_CHUNK = 20_000             # rows per block in the sleeve accumulation (bounds peak memory)

Exit = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Paths:
    """Price paths of n positions over h trading days after entry.

    entries: the kept entry rows in path order (sorted by date, security_id), with their own columns
    plus `d` (calendar index of the entry date, Int64). Entries off the calendar or without a full
    h-day window inside the calendar are dropped (as in `levels.barrier_labels`).
    close, open: (n, h) total-return ratios vs the entry close, NaN where there is no price.
    """

    entries: pl.DataFrame
    close: np.ndarray
    open: np.ndarray
    horizon: int
    calendar: pl.Series

    def __len__(self) -> int:
        return self.close.shape[0]


@dataclass(frozen=True)
class ExitSelection:
    """Result of `select_pair`.

    k_tp, k_sl: the chosen level pair (None = no barrier on that side). With verdict "risk_reference"
    it is the best pair to SHOW as reference levels, not an exit rule.
    table: k_tp, k_sl, fold, sharpe, max_dd for every grid pair (hold = null/null) and fold.
    summary: k_tp, k_sl, mean_sharpe, mean_max_dd, eligible (mean max_dd no worse than hold).
    hold: {"sharpe": mean net Sharpe, "max_dd": mean max drawdown} of plain holding.
    variants: variant (hold | trailing | signal), param (k_trail, the signal threshold, null for
    hold), fold, sharpe, max_dd. Reported only; never selectable as the level pair.
    n_trials: configurations looked at (grid pairs + variants other than hold).
    """

    k_tp: float | None
    k_sl: float | None
    table: pl.DataFrame
    summary: pl.DataFrame
    hold: dict[str, float]
    verdict: str
    variants: pl.DataFrame
    n_trials: int


# --------------------------------------------------------------------------------------------- paths


def position_paths(
    prices: pl.DataFrame,
    entries: pl.DataFrame,
    delist_rets: pl.DataFrame,
    calendar: pl.Series,
    horizon: int,
    dtype: type = np.float32,
) -> Paths:
    """Paths for `entries` (security_id, date, any extra columns) from `prices` (security_id, date,
    open, close, adj_close) and `delist_rets` (security_id, date, delist_return).

    One flat NaN-padded array holds every security's dense daily path (first..last price day plus
    h + 1 padding days), so all windows are gathered in one fancy-index without a per-row loop.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    cal = calendar.to_frame("date").with_row_index("d").with_columns(pl.col("d").cast(pl.Int64))
    px = _price_paths(prices, delist_rets, cal).with_columns(pl.col("d").cast(pl.Int64))

    seg = (
        px.group_by("security_id")
        .agg(pl.col("d").min().alias("first"), pl.col("d").max().alias("last"))
        .sort("security_id")
        .with_columns((pl.col("last") - pl.col("first") + horizon + 2).alias("len"))
        .with_columns((pl.col("len").cum_sum() - pl.col("len")).alias("offset"))
    )
    size = int(seg["len"].sum()) if seg.height else 0
    flat_close = np.full(size, np.nan)
    flat_open = np.full(size, np.nan)
    pos = px.join(seg, on="security_id").select(pl.col("offset") + pl.col("d") - pl.col("first"), "adj_close", "adj_open")
    at = pos.to_series(0).to_numpy()
    flat_close[at] = pos["adj_close"].to_numpy()
    flat_open[at] = pos["adj_open"].to_numpy()

    ent = (
        entries.with_row_index("_row")
        .join(cal, on="date")
        .filter(pl.col("d") + horizon < cal.height)
        .sort("date", "security_id", "_row")
        .drop("_row")
    )
    # an entry outside its security's price range has no entry close: its path stays all-NaN
    loc = (
        ent.select("security_id", "d")
        .join(seg, on="security_id", how="left", maintain_order="left")
        .select(
            ((pl.col("d") >= pl.col("first")) & (pl.col("d") <= pl.col("last"))).fill_null(False).alias("inside"),
            (pl.col("offset") + pl.col("d") - pl.col("first")).fill_null(0).alias("base"),
        )
    )
    n = ent.height
    inside = loc["inside"].to_numpy()
    base = loc["base"].to_numpy().astype(np.int64)

    close = np.full((n, horizon), np.nan, dtype=dtype)
    open_ = np.full((n, horizon), np.nan, dtype=dtype)
    rows = np.flatnonzero(inside)
    steps = np.arange(1, horizon + 1)
    for s in range(0, len(rows), _CHUNK):
        r = rows[s : s + _CHUNK]
        b = base[r]
        entry = flat_close[b][:, None]
        idx = b[:, None] + steps
        close[r] = flat_close[idx] / entry
        open_[r] = flat_open[idx] / entry
    return Paths(entries=ent, close=close, open=open_, horizon=horizon, calendar=calendar)


# --------------------------------------------------------------------------------------------- exit rules


def _last_valid(close: np.ndarray) -> Exit:
    """Index of the last non-NaN close in each row (h - 1 when none) and that close (NaN when none)."""
    n, h = close.shape
    valid = ~np.isnan(close)
    last = h - 1 - valid[:, ::-1].argmax(axis=1)
    ret = np.where(valid.any(axis=1), close[np.arange(n), last].astype(np.float64), np.nan)
    return last, ret


def _combine(tp: Exit, sl: Exit, time: Exit) -> Exit:
    """Earliest of the two barrier hits (index, fill ratio); neither -> the time exit. As barrier_labels."""
    (tp_i, tp_fill), (sl_i, sl_fill), (last, time_ret) = tp, sl, time
    is_tp, is_sl = tp_i < sl_i, sl_i < tp_i
    day = np.where(is_tp, tp_i, np.where(is_sl, sl_i, last)) + 1
    ret = np.where(is_tp, tp_fill, np.where(is_sl, sl_fill, time_ret)) - 1
    return day.astype(np.int64), ret.astype(np.float64)


def _levels(paths: Paths, ratio, side: str) -> Exit:
    """First close beyond `ratio` (scalar or (n,), None = no barrier) and its fill, via levels._first_hit."""
    n = len(paths)
    if ratio is None:
        return np.full(n, paths.horizon), np.full(n, np.nan)
    lv = np.broadcast_to(np.asarray(ratio, dtype=np.float64), (n,))
    return _first_hit(paths.close, paths.open, lv, side)


def hold(paths: Paths) -> Exit:
    """Plain hold: exit at the last close inside the window (the horizon day unless the data ends)."""
    last, ret = _last_valid(paths.close)
    return (last + 1).astype(np.int64), ret - 1


def fixed(paths: Paths, tp_ratio=None, sl_ratio=None) -> Exit:
    """Fixed take-profit / stop-loss at `tp_ratio` / `sl_ratio` x entry (scalars or (n,) arrays from
    `levels.level_ratio`; None = no barrier). Same outcomes as `levels.barrier_labels`."""
    return _combine(_levels(paths, tp_ratio, "tp"), _levels(paths, sl_ratio, "sl"), _last_valid(paths.close))


def trailing(paths: Paths, trail_dist) -> Exit:
    """Trailing stop: on day j the stop is (1 - dist) x the highest close from entry through day j - 1;
    checked on closes, filled at the open when the day gapped below the stop. `trail_dist` is a
    fraction in (0, 1), scalar or (n,) (see `trail_distance`)."""
    n, h = paths.close.shape
    dist = np.broadcast_to(np.asarray(trail_dist, dtype=np.float64), (n,))
    run = np.empty_like(paths.close)
    run[:, 0] = 1.0
    if h > 1:
        np.fmax.accumulate(paths.close[:, :-1], axis=1, out=run[:, 1:])
        np.fmax(run[:, 1:], 1.0, out=run[:, 1:])
    stop = run * (1 - dist)[:, None].astype(run.dtype)
    hit = paths.close <= stop
    idx = np.where(hit.any(axis=1), hit.argmax(axis=1), h)
    r, c = np.arange(n), np.minimum(idx, h - 1)
    o, s = paths.open[r, c].astype(np.float64), stop[r, c].astype(np.float64)
    fill = np.where(o <= s, o, s)
    return _combine((np.full(n, h), np.full(n, np.nan)), (idx, fill), _last_valid(paths.close))


def trail_distance(sigma: np.ndarray, k_trail: float, bounds: tuple[float, float]) -> np.ndarray:
    """Trailing-stop distance 1 - exp(-k_trail x sigma), kept inside the SL bounds (plan 6.5)."""
    return 1 - level_ratio(np.asarray(sigma, dtype=np.float64), k_trail, bounds, "sl")


def signal(paths: Paths, exit_day_hint: np.ndarray) -> Exit:
    """Signal exit at the close of day `exit_day_hint` (1..h; h = no signal), the first monthly re-score
    where the model's percentile fell below the threshold (see `signal_exit_days`). No price that day:
    the next available close. Data ending earlier: the hold exit."""
    n, h = paths.close.shape
    hint = np.clip(np.asarray(exit_day_hint, dtype=np.int64), 1, h)
    hold_day, hold_ret = hold(paths)
    ok = ~np.isnan(paths.close) & (np.arange(h)[None, :] >= (hint - 1)[:, None])
    has = ok.any(axis=1)
    first = ok.argmax(axis=1)
    ret = paths.close[np.arange(n), first].astype(np.float64) - 1
    use_hold = ~has | (hold_day <= hint)
    return np.where(use_hold, hold_day, first + 1).astype(np.int64), np.where(use_hold, hold_ret, ret)


def signal_exit_days(
    paths: Paths, scores: pl.DataFrame, threshold: float = SIGNAL_THRESHOLD, col: str = "rank_pct"
) -> np.ndarray:
    """For each position, the day index (1..h) of the first re-score strictly after entry where
    `scores[col]` < threshold for that security; h when there is none inside the window.
    `scores` = (security_id, date, col), the out-of-fold percentiles known at each re-score close."""
    h = paths.horizon
    cal = paths.calendar.to_frame("date").with_row_index("d_s").with_columns(pl.col("d_s").cast(pl.Int64))
    low = (
        scores.filter(pl.col(col) < threshold)
        .join(cal, on="date")
        .select(pl.col("security_id").cast(pl.Int64), "d_s")
        .unique()
        .sort("d_s")
    )
    pos = (
        paths.entries.select(pl.col("security_id").cast(pl.Int64), "d")
        .with_row_index("_i")
        .with_columns((pl.col("d") + 1).alias("_key"))
        .sort("_key")
    )
    # both sides are sorted on the asof key above; polars cannot verify that within `by` groups
    j = pos.join_asof(
        low, left_on="_key", right_on="d_s", by="security_id", strategy="forward", check_sortedness=False
    ).sort("_i")
    step = (j["d_s"] - j["d"]).to_numpy()
    step = np.where(np.isnan(step.astype(np.float64)), h, step.astype(np.float64))
    return np.minimum(step, h).astype(np.int64)


# --------------------------------------------------------------------------------------------- portfolio


def _sleeves(paths: Paths, positions, exit_day: np.ndarray, exit_ret: np.ndarray, cost: float):
    """(entry calendar index per sleeve, (k, h) daily sleeve returns). Positions without a finite
    exit return (no entry price) are left out; a sleeve with none is dropped."""
    n, h = paths.close.shape
    idx = np.arange(n) if positions is None else np.asarray(positions)
    if idx.dtype == bool:
        idx = np.flatnonzero(idx)
    idx = idx[np.isfinite(exit_ret[idx])]
    d_all = paths.entries["d"].to_numpy()
    idx = idx[np.argsort(d_all[idx], kind="stable")]
    d0 = d_all[idx]
    days, sleeve, counts = np.unique(d0, return_inverse=True, return_counts=True)
    total = np.zeros((len(days), h + 1))
    keep = 1.0 - cost
    cols = np.arange(h + 1, dtype=np.int32)
    for s in range(0, len(idx), _CHUNK):
        r = idx[s : s + _CHUNK]
        m = len(r)
        g = np.empty((m, h + 1), dtype=paths.close.dtype)
        g[:, 0] = 1.0
        g[:, 1:] = paths.close[r]
        # forward-fill missing closes (an untraded day keeps the last value)
        last = np.where(np.isnan(g), 0, cols[None, :]).astype(np.int32)
        np.maximum.accumulate(last, axis=1, out=last)
        g = np.take_along_axis(g, last, axis=1)
        fill = ((1 + exit_ret[r]) * keep).astype(g.dtype)
        v = np.where(cols[None, :] >= exit_day[r][:, None], fill[:, None], g)
        sl = sleeve[s : s + _CHUNK]
        starts = np.flatnonzero(np.r_[True, sl[1:] != sl[:-1]])
        total[sl[starts]] += np.add.reduceat(v.astype(np.float64), starts, axis=0)
    value = total * keep / counts[:, None]
    value[:, 0] = 1.0  # capital before the entry cost
    return days, value[:, 1:] / value[:, :-1] - 1


def _aggregate(days: np.ndarray, rets: np.ndarray, calendar: pl.Series) -> pl.DataFrame:
    if len(days) == 0:
        return pl.DataFrame(schema={"date": pl.Date, "ret": pl.Float64})
    h = rets.shape[1]
    t = (days[:, None] + np.arange(1, h + 1)).ravel()
    s = np.bincount(t, weights=rets.ravel())
    c = np.bincount(t)
    on = np.flatnonzero(c)
    return pl.DataFrame({"date": calendar.gather(on), "ret": s[on] / c[on]})


def portfolio_returns(
    positions, paths: Paths, exit_day: np.ndarray, exit_ret: np.ndarray, cost: float
) -> pl.DataFrame:
    """Daily returns (date, ret) of the overlapping equal-weight monthly sleeves of `positions` (row
    indices or a boolean mask into `paths`; None = all) under one exit rule, net of a one-way `cost`
    at entry and exit. Only days with at least one active sleeve are returned."""
    days, rets = _sleeves(paths, positions, exit_day, exit_ret, cost)
    return _aggregate(days, rets, paths.calendar)


def _ret_array(ret) -> np.ndarray:
    if isinstance(ret, pl.DataFrame):
        ret = ret["ret"]
    return np.asarray(ret.to_numpy() if isinstance(ret, pl.Series) else ret, dtype=np.float64)


def net_sharpe(ret) -> float:
    """Annualized Sharpe of daily net returns: mean / std (ddof 1) x sqrt(252); NaN if undefined."""
    r = _ret_array(ret)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else float("nan")


def max_drawdown(ret) -> float:
    """Largest peak-to-trough fall of cumulative wealth (starting at 1), as a return <= 0."""
    r = _ret_array(ret)
    if len(r) == 0:
        return 0.0
    w = np.concatenate([[1.0], np.cumprod(1 + r)])
    return float((w / np.maximum.accumulate(w) - 1).min())


# --------------------------------------------------------------------------------------------- selection


def exit_entries(
    oof: pl.DataFrame, sigma: pl.DataFrame, top_frac: float = TOP_FRACTION, pred: str = "pred"
) -> pl.DataFrame:
    """The top `top_frac` of names by `pred` on each date (floor(n x frac), at least 1; ties broken by
    security_id), joined with `sigma` (security_id, date, sigma); rows without sigma are dropped.
    Columns: security_id, date, fold, pred, sigma."""
    o = oof.filter(pl.col(pred).is_not_null() & pl.col(pred).is_finite()).sort("date", "security_id")
    k = (pl.len().over("date") * top_frac + 1e-9).floor().clip(lower_bound=1)
    top = o.filter((-pl.col(pred)).rank("ordinal").over("date") <= k)
    return (
        top.select("security_id", "date", "fold", pl.col(pred).alias("pred"))
        .join(sigma.select("security_id", "date", "sigma"), on=["security_id", "date"], how="inner")
        .filter(pl.col("sigma").is_not_null() & pl.col("sigma").is_finite())
        .sort("date", "security_id")
    )


def _by_fold(days: np.ndarray, rets: np.ndarray, day_fold: dict[int, str], calendar: pl.Series, folds):
    out = {}
    sleeve_fold = np.array([day_fold[int(x)] for x in days], dtype=object)  # one entry per month
    for f in folds:
        sel = sleeve_fold == f
        r = _aggregate(days[sel], rets[sel], calendar)["ret"]
        out[f] = (net_sharpe(r), max_drawdown(r))
    return out


def select_pair(
    paths: Paths,
    rules: BarrierRules,
    *,
    cost: float,
    k_trail: Sequence[float] = DEFAULT_K_TRAIL,
    signal_days: np.ndarray | None = None,
    signal_threshold: float = SIGNAL_THRESHOLD,
    exp_dir: str | Path | None = None,
) -> ExitSelection:
    """Choose one global (k_tp, k_sl) pair on dev walk-forward folds (plan 6.5).

    `paths.entries` needs `fold` and `sigma` (from `exit_entries`); per fold, the portfolio holds the
    positions entered on that fold's test dates. The pair with the best MEAN net Sharpe across folds
    (`selection_guard`) is chosen among non-hold pairs whose mean max drawdown is no worse than hold's
    (or among all non-hold pairs when none qualifies, as reference levels). verdict = "exit_orders"
    when it beats hold on mean net Sharpe and is no worse on mean max drawdown, else
    "risk_reference". Trailing stops (k_trail, distance from `trail_distance`) and the signal exit
    (`signal_days`, from `signal_exit_days`) are evaluated and reported as variants only. With
    `exp_dir` the trials are logged to the experiment log.
    """
    ent = paths.entries
    for c in ("fold", "sigma"):
        if c not in ent.columns:
            raise ValueError(f"paths.entries needs a {c!r} column (build entries with exit_entries)")
    sigma = ent["sigma"].to_numpy().astype(np.float64)
    if not np.isfinite(sigma).all():
        raise ValueError("sigma must be finite for every position")
    folds = list(dict.fromkeys(ent["fold"].to_list()))
    day_fold = dict(zip(ent["d"].to_list(), ent["fold"].to_list()))
    time = _last_valid(paths.close)

    def evaluate(exit_: Exit) -> dict:
        days, rets = _sleeves(paths, None, exit_[0], exit_[1], cost)
        return _by_fold(days, rets, day_fold, paths.calendar, folds)

    tp = {k: _levels(paths, level_ratio(sigma, k, rules.tp_bounds, "tp"), "tp") for k in rules.k_tp}
    sl = {k: _levels(paths, level_ratio(sigma, k, rules.sl_bounds, "sl"), "sl") for k in rules.k_sl}
    pairs = [(a, b) for a in rules.k_tp for b in rules.k_sl]
    none_tp, none_sl = _levels(paths, None, "tp"), _levels(paths, None, "sl")
    hold_res = evaluate(_combine(none_tp, none_sl, time))

    rows = []
    for a, b in dict.fromkeys([(None, None), *pairs]):
        res = hold_res if (a, b) == (None, None) else evaluate(_combine(tp.get(a, none_tp), sl.get(b, none_sl), time))
        rows += [{"k_tp": a, "k_sl": b, "fold": f, "sharpe": s, "max_dd": m} for f, (s, m) in res.items()]
    schema = {"k_tp": pl.Float64, "k_sl": pl.Float64, "fold": pl.String, "sharpe": pl.Float64, "max_dd": pl.Float64}
    table = pl.DataFrame(rows, schema=schema)

    def means(df: pl.DataFrame) -> tuple[float, float]:
        return (
            selection_guard(dict(zip(df["fold"], df["sharpe"]))),
            selection_guard(dict(zip(df["fold"], df["max_dd"]))),
        )

    hold_sharpe, hold_dd = means(table.filter(pl.col("k_tp").is_null() & pl.col("k_sl").is_null()))
    summary_rows = []
    for (a, b), g in table.group_by("k_tp", "k_sl", maintain_order=True):
        s, m = means(g)
        summary_rows.append({"k_tp": a, "k_sl": b, "mean_sharpe": s, "mean_max_dd": m, "eligible": m >= hold_dd})
    summary = pl.DataFrame(
        summary_rows,
        schema={"k_tp": pl.Float64, "k_sl": pl.Float64, "mean_sharpe": pl.Float64, "mean_max_dd": pl.Float64,
                "eligible": pl.Boolean},
    )

    cands = [r for r in summary_rows if not (r["k_tp"] is None and r["k_sl"] is None)]
    pool = [r for r in cands if r["eligible"]] or cands
    if pool:
        best = max(pool, key=lambda r: r["mean_sharpe"])  # first in grid order on ties
        k_tp, k_sl = best["k_tp"], best["k_sl"]
        beats = best["mean_sharpe"] > hold_sharpe and best["mean_max_dd"] >= hold_dd
    else:
        k_tp = k_sl = None
        beats = False
    verdict = "exit_orders" if beats else "risk_reference"

    var_rows = [{"variant": "hold", "param": None, "fold": f, "sharpe": s, "max_dd": m} for f, (s, m) in hold_res.items()]
    for k in k_trail:
        res = evaluate(trailing(paths, trail_distance(sigma, k, rules.sl_bounds)))
        var_rows += [{"variant": "trailing", "param": k, "fold": f, "sharpe": s, "max_dd": m} for f, (s, m) in res.items()]
    if signal_days is not None:
        res = evaluate(signal(paths, signal_days))
        var_rows += [
            {"variant": "signal", "param": signal_threshold, "fold": f, "sharpe": s, "max_dd": m} for f, (s, m) in res.items()
        ]
    variants = pl.DataFrame(var_rows, schema={"variant": pl.String, "param": pl.Float64, "fold": pl.String,
                                              "sharpe": pl.Float64, "max_dd": pl.Float64})
    n_trials = len(dict.fromkeys(pairs)) + len(k_trail) + (signal_days is not None)

    sel = ExitSelection(
        k_tp=k_tp, k_sl=k_sl, table=table, summary=summary, hold={"sharpe": hold_sharpe, "max_dd": hold_dd},
        verdict=verdict, variants=variants, n_trials=n_trials,
    )
    if exp_dir is not None:
        log_run(
            exp_dir, "exit_pairs",
            {"k_tp": list(rules.k_tp), "k_sl": list(rules.k_sl), "tp_bounds": list(rules.tp_bounds),
             "sl_bounds": list(rules.sl_bounds), "k_trail": list(k_trail), "signal": signal_days is not None,
             "signal_threshold": signal_threshold, "cost": cost, "horizon": paths.horizon},
            {"k_tp": k_tp, "k_sl": k_sl, "verdict": verdict, "hold": sel.hold, "summary": summary.to_dicts()},
            n_trials=n_trials,
        )
    return sel

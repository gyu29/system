"""Take-profit / stop-loss levels and triple-barrier labels (plan 2.7 and 6.5).

Levels scale with volatility: TP = entry * exp(k_tp * sigma), SL = entry * exp(-k_sl * sigma), with
the distance from entry kept inside configured bounds. Barrier labels walk the daily total-return
path after entry and record which level the close crosses first within the horizon:

- Crossings are checked on daily closes only.
- Fill: if the day opened beyond the level (an overnight gap), the fill is that open: worse than the
  stop, or better than the target. Otherwise the level was crossed during the day and fills at it.
- A delisting inside the window is a final day that opens and closes at
  last close * (1 + delisting return), so a -30% delisting gaps through most stops.
- No crossing: exit at the close on the horizon day (or at the delisting value).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ltforecast.config import BarrierRules

TRADING_DAYS_PER_YEAR = 252

OUT_SCHEMA = {
    "security_id": pl.Int64, "date": pl.Date, "sigma": pl.Float64, "k_tp": pl.Float64, "k_sl": pl.Float64,
    "tp_level": pl.Float64, "sl_level": pl.Float64, "outcome": pl.String, "exit_day": pl.Int64,
    "exit_date": pl.Date, "exit_ret": pl.Float64,
}


def level_ratio(sigma: np.ndarray, k: float | None, bounds: tuple[float, float], side: str) -> np.ndarray | None:
    """Barrier price / entry price for each sigma; None when k is None (no barrier on that side)."""
    if k is None:
        return None
    lo, hi = bounds
    if side == "tp":
        return 1 + np.clip(np.expm1(k * sigma), lo, hi)
    if side == "sl":
        return 1 - np.clip(-np.expm1(-k * sigma), lo, hi)
    raise ValueError(f"side must be 'tp' or 'sl', not {side!r}")


def trailing_sigma(prices: pl.DataFrame, window: int, min_obs: int) -> pl.DataFrame:
    """Annualized vol of daily log returns over the trailing `window` rows, known at each close.
    Columns: security_id, date, sigma."""
    lr = (pl.col("ret") + 1).log()
    return (
        prices.select("security_id", "date", "ret")
        .sort("security_id", "date")
        .with_columns(
            (lr.rolling_std(window_size=window, min_samples=min_obs).over("security_id") * TRADING_DAYS_PER_YEAR**0.5)
            .alias("sigma")
        )
        .select("security_id", "date", "sigma")
    )


def _first_hit(close: np.ndarray, open_: np.ndarray, level: np.ndarray | None, side: str):
    """Index of the first close beyond `level` (horizon if never) and the fill ratio there."""
    m, h = close.shape
    if level is None:
        return np.full(m, h), np.full(m, np.nan)
    hit = close >= level[:, None] if side == "tp" else close <= level[:, None]
    idx = np.where(hit.any(axis=1), hit.argmax(axis=1), h)
    o = open_[np.arange(m), np.minimum(idx, h - 1)]
    gapped = o >= level if side == "tp" else o <= level
    return idx, np.where(gapped, o, level)


def _price_paths(prices: pl.DataFrame, delist_rets: pl.DataFrame, cal: pl.DataFrame) -> pl.DataFrame:
    """security_id, d (calendar index), adj_close, adj_open, with a final row on each delisting day."""
    cols = ["security_id", "d", "adj_close", "adj_open"]
    px = prices.join(cal, on="date").select(
        "security_id",
        "d",
        "adj_close",
        # SEP open is split- but not dividend-adjusted; scale it onto the total-return path
        (pl.col("open") * pl.col("adj_close") / pl.col("close")).alias("adj_open"),
    )
    last_close = px.sort("d").group_by("security_id").agg(pl.col("adj_close").last().alias("last_close"))
    delist_day = (
        delist_rets.join(last_close, on="security_id")
        .join(cal, on="date")
        .with_columns((pl.col("last_close") * (1 + pl.col("delist_return"))).alias("adj_close"))
        .with_columns(pl.col("adj_close").alias("adj_open"))
        .select(cols)
    )
    return pl.concat([px.select(cols), delist_day]).sort("security_id", "d")


def barrier_labels(
    prices: pl.DataFrame,
    entries: pl.DataFrame,
    delist_rets: pl.DataFrame,
    calendar: pl.Series,
    horizon: int,
    rules: BarrierRules,
) -> pl.DataFrame:
    """One row per (entry, k_tp, k_sl) for entries (security_id, date, sigma) with a complete window.

    Columns: security_id, date, sigma, k_tp, k_sl, tp_level, sl_level (returns from entry; null = no
    barrier), outcome (tp | sl | time), exit_day (trading days after entry), exit_date, exit_ret.
    """
    cal = calendar.to_frame("date").with_row_index("d")
    ent = (
        entries.drop_nulls("sigma")
        .join(cal, on="date")
        .filter(pl.col("d") + horizon < cal.height)
        .sort("security_id", "d")
    )
    ent_by_sid = ent.partition_by("security_id", as_dict=True)

    out: dict[str, list[np.ndarray]] = {k: [] for k in OUT_SCHEMA if k not in ("date", "exit_date")} | {"d": []}
    for (sid,), g in _price_paths(prices, delist_rets, cal).partition_by("security_id", as_dict=True).items():
        e = ent_by_sid.get((sid,))
        if e is None:
            continue
        # dense daily path over the calendar, NaN on days without a price, padded for the last window
        d = g["d"].to_numpy()
        first = d[0]
        close = np.full(d[-1] - first + 2 + horizon, np.nan)
        open_ = close.copy()
        close[d - first] = g["adj_close"].to_numpy()
        open_[d - first] = g["adj_open"].to_numpy()

        ei = e["d"].to_numpy() - first
        sigma = e["sigma"].to_numpy()
        entry = close[ei][:, None]
        rc = np.lib.stride_tricks.sliding_window_view(close, horizon)[ei + 1] / entry
        ro = np.lib.stride_tricks.sliding_window_view(open_, horizon)[ei + 1] / entry
        m = len(ei)

        valid = ~np.isnan(rc)
        last = horizon - 1 - valid[:, ::-1].argmax(axis=1)
        time_ret = np.where(valid.any(axis=1), rc[np.arange(m), last], np.nan)

        tp, sl = {}, {}
        for k in rules.k_tp:
            lv = level_ratio(sigma, k, rules.tp_bounds, "tp")
            tp[k] = (lv, *_first_hit(rc, ro, lv, "tp"))
        for k in rules.k_sl:
            lv = level_ratio(sigma, k, rules.sl_bounds, "sl")
            sl[k] = (lv, *_first_hit(rc, ro, lv, "sl"))

        nan = np.full(m, np.nan)
        for k_tp, (tp_lv, tp_i, tp_fill) in tp.items():
            for k_sl, (sl_lv, sl_i, sl_fill) in sl.items():
                is_tp, is_sl = tp_i < sl_i, sl_i < tp_i
                out["security_id"].append(np.full(m, sid))
                out["d"].append(e["d"].to_numpy())
                out["sigma"].append(sigma)
                out["k_tp"].append(nan if k_tp is None else np.full(m, k_tp))
                out["k_sl"].append(nan if k_sl is None else np.full(m, k_sl))
                out["tp_level"].append(nan if tp_lv is None else tp_lv - 1)
                out["sl_level"].append(nan if sl_lv is None else sl_lv - 1)
                out["outcome"].append(np.where(is_tp, "tp", np.where(is_sl, "sl", "time")))
                out["exit_day"].append(np.where(is_tp, tp_i, np.where(is_sl, sl_i, last)) + 1)
                out["exit_ret"].append(np.where(is_tp, tp_fill, np.where(is_sl, sl_fill, time_ret)) - 1)

    if not out["d"]:
        return pl.DataFrame(schema=OUT_SCHEMA)
    floats = ["sigma", "k_tp", "k_sl", "tp_level", "sl_level", "exit_ret"]
    df = pl.DataFrame({k: np.concatenate(v) for k, v in out.items()}).with_columns(
        pl.col(floats).fill_nan(None),
        pl.col("security_id").cast(OUT_SCHEMA["security_id"]),
        pl.col("exit_day").cast(pl.Int64),
        pl.col("d").cast(pl.UInt32),
    )
    return (
        df.with_columns((pl.col("d") + pl.col("exit_day")).cast(pl.UInt32).alias("d_exit"))
        .join(cal, on="d")
        .join(cal.rename({"d": "d_exit", "date": "exit_date"}), on="d_exit")
        .select(list(OUT_SCHEMA))
        .sort("date", "security_id", "k_tp", "k_sl", nulls_last=True)
    )

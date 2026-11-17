"""Hit probabilities for the TP/SL levels (plan 6.5, "Hit probabilities").

For each position: P(TP first), P(SL first), P(neither within the horizon) and E[return | neither],
estimated two ways and compared on out-of-fold multiclass Brier score.

1. Monte Carlo. The log price follows X_t = sum_{s<=t} (mu_d + sigma_d * z_s) on trading days
   t = 1..H, with mu_d = mu_annual / 252, sigma_d = sigma_annual / sqrt(252) and shocks z bootstrapped
   (iid, with replacement) from the pooled standardized residuals of the stock's volatility bucket,
   re-standardized to mean 0 and variance 1 (so the drift comes only from mu and the scale only from
   sigma; the residuals contribute the shape: fat tails, skew). As in `levels.barrier_labels` the
   barriers are checked on daily closes: TP is hit when X_t >= log(tp_ratio), SL when
   X_t <= log(sl_ratio); the first one wins, neither = time exit at day H.
   mu_annual = q50 log excess + log(1 + market_return) (`annual_drift`), the market return being a
   fixed config value (~6%/yr, not a forecast).

   `simulate` does this directly for ONE stock. `HitTable` solves the SCALED problem once per bucket:
   dividing by sigma_d, a hit happens when W(t) + theta * t leaves (-b, a), with theta = mu_d / sigma_d,
   a = log(tp) / sigma_d, b = -log(sl) / sigma_d. Internally the table uses horizon units
   (Theta = theta * sqrt(H), A = a / sqrt(H), B = b / sqrt(H); with H = 252 these are mu / sigma,
   log(tp) / sigma and -log(sl) / sigma), tabulated on a grid and interpolated TRILINEARLY:
   linear in Theta, and linear in u = A / (1 + A) (likewise B) so that "no barrier" (A = inf) is the
   grid node u = 1 and is looked up exactly. Queries outside the grid are clamped to its edge
   (|Theta| > 5, i.e. |mu| > 5 sigma, or a barrier closer than 0.02 horizon sigmas).
   All grid cells of one Theta are evaluated on the SAME simulated paths (first-passage times of
   the running max / min against every barrier node), and the same shocks are reused across Theta
   (common random numbers), so the table is smooth and cheap to build (~1-2 s per bucket).

   E[ret | neither] depends on sigma as well as on (Theta, A, B), so the table stores the first two
   conditional moments (m, v) of the scaled log return Y_H = X_H / sigma_h given neither barrier was
   hit, and returns E[ret | neither] ~= exp(sigma_h * m + sigma_h^2 * v / 2) - 1 (second-order
   cumulant expansion; Y_H is confined to (-B, A) on that event, so the error is small).
   A cell with no "neither" paths stores m = v = 0 (it only matters with weight p_none ~ 0).

2. Direct classifier. `HitClassifier`: LightGBM multiclass (tp | sl | time) on the barrier labels of
   the chosen pair, features = model columns + sigma_hat; implements `validation.runner.Model` with
   three outputs (use `walk_forward(..., target="outcome", outputs=PROB_COLS)`).

Volatility buckets: terciles (by default) of the sigma values dated <= `through`; bucket 0 is the
lowest-volatility one (`vol_bucket_edges`, `assign_buckets`). Point in time: `standardized_residuals`
uses returns dated <= `through` and, for each return, the latest sigma dated strictly before it.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import polars as pl

from ltforecast.dataset import label_columns

TRADING_DAYS_PER_YEAR = 252
CLASSES = ("tp", "sl", "time")              # classifier class order = probability column order
PROB_COLS = ("p_tp", "p_sl", "p_none")
METHODS = ("mc", "classifier", "average")   # tie order in choose_method
KEYS = ("date", "security_id")

DEFAULT_THETA_NODES = tuple(float(x) for x in np.round(np.arange(-5.0, 5.0 + 1e-9, 0.25), 2))
DEFAULT_BARRIER_NODES = (
    0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75,
    2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, float("inf"),
)


@dataclass(frozen=True)
class HitProb:
    """Hit probabilities (floats for one position, arrays for many). p_tp + p_sl + p_none = 1.
    mean_ret_none = E[exit return | neither barrier hit] as a simple return (NaN if p_none = 0)."""

    p_tp: float | np.ndarray
    p_sl: float | np.ndarray
    p_none: float | np.ndarray
    mean_ret_none: float | np.ndarray

    def as_array(self) -> np.ndarray:
        """(n, 3) array in PROB_COLS order (n = 1 for scalars)."""
        return np.column_stack([np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in
                                (self.p_tp, self.p_sl, self.p_none)])


# drift, buckets and residuals ---------------------------------------------------------------------


def annual_drift(q50_log_excess: np.ndarray | float, market_return: float = 0.06) -> np.ndarray | float:
    """mu_annual = q50 log excess return + log(1 + market_return) (annual log drift of the price)."""
    return np.asarray(q50_log_excess, dtype=np.float64) + np.log1p(market_return)


def vol_bucket_edges(sigma: pl.DataFrame, through: dt.date, n_buckets: int = 3) -> np.ndarray:
    """The n_buckets - 1 quantile cut points of the finite, positive `sigma` values dated <= through."""
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")
    s = (
        sigma.filter((pl.col("date") <= through) & pl.col("sigma").is_finite() & (pl.col("sigma") > 0))
        ["sigma"].cast(pl.Float64).to_numpy()
    )
    if s.size == 0:
        raise ValueError(f"no sigma values dated <= {through}")
    return np.quantile(s, np.arange(1, n_buckets) / n_buckets)


def assign_buckets(sigma: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bucket index 0..len(edges) per sigma (0 = lowest vol); -1 where sigma is not finite."""
    s = np.asarray(sigma, dtype=np.float64)
    b = np.searchsorted(np.asarray(edges, dtype=np.float64), s, side="right")
    return np.where(np.isfinite(s), b, -1).astype(np.int64)


def standardized_residuals(
    prices: pl.DataFrame,
    sigma: pl.DataFrame,
    through: dt.date,
    n_buckets: int = 3,
    *,
    max_staleness_days: int = 45,
    clip: float = 15.0,
    max_per_bucket: int | None = None,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Daily log returns / (sigma / sqrt(252)), pooled by volatility bucket, from data <= `through`.

    prices: security_id, date, ret (daily total return). sigma: security_id, date, sigma (annualized;
    daily or monthly rows). Each return dated d <= through is scaled by that stock's latest sigma dated
    strictly before d (and at most `max_staleness_days` older); returns without one are dropped, as are
    null returns and ret <= -1. The bucket is that sigma's bucket under `vol_bucket_edges(sigma, through)`.
    Residuals are clipped to +-clip (bad ticks) and returned raw (not demeaned); the simulators
    re-standardize them. With `max_per_bucket`, a bucket larger than that is subsampled (fixed seed).
    Rows are taken in (date, security_id) order, so the output is deterministic.
    """
    edges = vol_bucket_edges(sigma, through, n_buckets)
    r = (
        prices.filter((pl.col("date") <= through) & pl.col("ret").is_not_null() & (pl.col("ret") > -1))
        .select("security_id", "date", pl.col("ret").cast(pl.Float64).log1p().alias("lr"))
        .sort("date", "security_id")
    )
    s = (
        sigma.filter((pl.col("date") <= through) & pl.col("sigma").is_finite() & (pl.col("sigma") > 0))
        .select("security_id", pl.col("date").alias("sigma_date"), pl.col("sigma").cast(pl.Float64))
        .sort("sigma_date", "security_id")
    )
    j = (
        r.join_asof(
            s, left_on="date", right_on="sigma_date", by="security_id", strategy="backward",
            allow_exact_matches=False, tolerance=f"{int(max_staleness_days)}d", check_sortedness=False,
        )
        .drop_nulls("sigma")
        .sort("date", "security_id")
    )
    sig = j["sigma"].to_numpy()
    z = np.clip(j["lr"].to_numpy() / (sig / np.sqrt(TRADING_DAYS_PER_YEAR)), -clip, clip)
    bucket = assign_buckets(sig, edges)
    rng = np.random.default_rng(seed)
    out: dict[int, np.ndarray] = {}
    for b in range(n_buckets):
        zb = z[bucket == b]
        if max_per_bucket is not None and zb.size > max_per_bucket:
            zb = zb[np.sort(rng.choice(zb.size, size=max_per_bucket, replace=False))]
        out[b] = zb
    return out


def _standardize(shocks: np.ndarray) -> np.ndarray:
    """Finite shocks rescaled to mean 0 and variance 1 (population moments)."""
    z = np.asarray(shocks, dtype=np.float64).ravel()
    z = z[np.isfinite(z)]
    sd = z.std() if z.size else 0.0
    if z.size < 2 or sd <= 0:
        raise ValueError("need at least two distinct finite shocks")
    return (z - z.mean()) / sd


# direct simulation --------------------------------------------------------------------------------


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True per row, or the row length if none."""
    h = mask.shape[1]
    return np.where(mask.any(axis=1), mask.argmax(axis=1), h)


def _barrier_log(ratio: float | None, side: str) -> float:
    if ratio is None or not np.isfinite(ratio):
        return np.inf if side == "tp" else -np.inf
    if ratio <= 0 or (side == "tp" and ratio <= 1) or (side == "sl" and ratio >= 1):
        raise ValueError(f"{side} ratio {ratio} is on the wrong side of the entry price")
    return float(np.log(ratio))


def simulate(
    mu_annual: float,
    sigma_annual: float,
    tp_ratio: float | None,
    sl_ratio: float | None,
    shocks: np.ndarray,
    n_paths: int = 10_000,
    horizon: int = TRADING_DAYS_PER_YEAR,
    seed: int = 0,
) -> HitProb:
    """Direct Monte Carlo for one stock (see module docstring). tp_ratio / sl_ratio are barrier price /
    entry price (e.g. 1.25 and 0.85); None = no barrier on that side (p = 0 there)."""
    if sigma_annual <= 0:
        raise ValueError("sigma_annual must be > 0")
    up, dn = _barrier_log(tp_ratio, "tp"), _barrier_log(sl_ratio, "sl")
    z = _standardize(shocks)
    rng = np.random.default_rng(seed)
    step = mu_annual / TRADING_DAYS_PER_YEAR + sigma_annual / np.sqrt(TRADING_DAYS_PER_YEAR) * rng.choice(
        z, size=(n_paths, horizon)
    )
    x = np.cumsum(step, axis=1)
    ti, si = _first_true(x >= up), _first_true(x <= dn)
    is_tp, is_sl = ti < si, si < ti
    none = ~(is_tp | is_sl)
    mean_none = float(np.expm1(x[none, -1]).mean()) if none.any() else float("nan")
    return HitProb(float(is_tp.mean()), float(is_sl.mean()), float(none.mean()), mean_none)


# the table ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HitTableParams:
    """Grid and simulation settings. theta_nodes: drift over the horizon in horizon sigmas (ascending);
    barrier_nodes: barrier distance in horizon sigmas (ascending, > 0, may end with inf = no barrier)."""

    n_paths: int = 10_000
    horizon: int = TRADING_DAYS_PER_YEAR
    seed: int = 0
    theta_nodes: tuple[float, ...] = DEFAULT_THETA_NODES
    barrier_nodes: tuple[float, ...] = DEFAULT_BARRIER_NODES

    def __post_init__(self):
        t, b = np.asarray(self.theta_nodes, float), np.asarray(self.barrier_nodes, float)
        if t.size < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("theta_nodes must be >= 2 strictly ascending values")
        if b.size < 2 or np.any(np.diff(b) <= 0) or b[0] <= 0 or np.isinf(b[:-1]).any():
            raise ValueError("barrier_nodes must be >= 2 strictly ascending positive values (inf only last)")
        if self.n_paths < 1 or self.horizon < 1:
            raise ValueError("n_paths and horizon must be >= 1")


def _first_passage(runmax: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """(N, L) index of the first t with runmax[i, t] >= levels[l] (H if never); runmax rows are
    nondecreasing. One global searchsorted: each row is clipped into a band and offset so the
    flattened matrix is sorted."""
    n, h = runmax.shape
    levels = np.asarray(levels, dtype=np.float64)
    if levels.size == 0:
        return np.empty((n, 0), dtype=np.int64)
    lo, hi = float(levels.min()) - 1.0, float(levels.max()) + 1.0
    span = hi - lo + 1.0
    off = np.arange(n, dtype=np.float64)[:, None] * span
    flat = (np.clip(runmax, lo, hi) + off).ravel()
    idx = np.searchsorted(flat, (levels[None, :] + off).ravel(), side="left").reshape(n, levels.size)
    return idx - np.arange(n, dtype=np.int64)[:, None] * h


def _tabulate(z: np.ndarray, p: HitTableParams, rng: np.random.Generator) -> np.ndarray:
    """(n_theta, n_barrier, n_barrier, 5) array of p_tp, p_sl, p_none, m, v for one bucket."""
    n, h = p.n_paths, p.horizon
    nodes = np.asarray(p.barrier_nodes, dtype=np.float64)
    fin = nodes[np.isfinite(nodes)]
    nb = nodes.size
    s = np.cumsum(rng.choice(z, size=(n, h)), axis=1) / np.sqrt(h)   # unit variance over the horizon
    t = np.arange(1, h + 1, dtype=np.float64) / h
    never = np.full((n, nb - fin.size), h, dtype=np.int64)
    out = np.zeros((len(p.theta_nodes), nb, nb, 5))
    for i, theta in enumerate(p.theta_nodes):
        y = s + theta * t
        up = np.hstack([_first_passage(np.maximum.accumulate(y, axis=1), fin), never])    # (n, nb)
        dn = np.hstack([_first_passage(np.maximum.accumulate(-y, axis=1), fin), never])
        tp = (up[:, :, None] < dn[:, None, :]).sum(axis=0)
        sl = (dn[:, None, :] < up[:, :, None]).sum(axis=0)
        nu, nd = (up == h).astype(np.float64), (dn == h).astype(np.float64)
        yh = y[:, -1][:, None]
        cnt = nu.T @ nd
        s1, s2 = (nu * yh).T @ nd, (nu * yh * yh).T @ nd
        m = np.divide(s1, cnt, out=np.zeros_like(s1), where=cnt > 0)
        v = np.divide(s2, cnt, out=np.zeros_like(s2), where=cnt > 0) - m * m
        out[i] = np.stack([tp / n, sl / n, cnt / n, m, np.maximum(v, 0.0)], axis=-1)
    return out


def _u(x: np.ndarray) -> np.ndarray:
    """Barrier axis coordinate A / (1 + A), with inf -> 1."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(np.isinf(x), 1.0, x / (1.0 + x))


def _axis(nodes: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower node index and weight on the upper node for each query, clamped to the node range."""
    q = np.clip(q, nodes[0], nodes[-1])
    i = np.clip(np.searchsorted(nodes, q, side="right") - 1, 0, nodes.size - 2)
    return i, (q - nodes[i]) / (nodes[i + 1] - nodes[i])


@dataclass
class HitTable:
    """Tabulated scaled hit problem per volatility bucket (see module docstring).

    tables[bucket]: (n_theta, n_barrier, n_barrier, 5) = p_tp, p_sl, p_none, m, v at
    (theta_nodes[i], barrier_nodes[j] as A, barrier_nodes[k] as B). `edges`: the bucket cut points
    the residuals were pooled with (optional; used by `score` when given sigma instead of buckets).
    """

    params: HitTableParams
    tables: dict[int, np.ndarray]
    edges: np.ndarray | None = None
    _u_nodes: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._u_nodes = _u(np.asarray(self.params.barrier_nodes, dtype=np.float64))

    @classmethod
    def build(
        cls,
        shocks_by_bucket: dict[int, np.ndarray],
        params: HitTableParams = HitTableParams(),
        edges: np.ndarray | None = None,
    ) -> HitTable:
        """Simulate every bucket's table. Bucket b uses seed (params.seed, b), so a table does not
        depend on which other buckets are built."""
        tables = {}
        for b in sorted(shocks_by_bucket):
            rng = np.random.default_rng([params.seed, int(b)])
            tables[int(b)] = _tabulate(_standardize(shocks_by_bucket[b]), params, rng)
        return cls(params, tables, None if edges is None else np.asarray(edges, dtype=np.float64))

    # interpolation

    def _interp(self, theta_h, a_h, b_h, bucket) -> np.ndarray:
        """(n, 5) trilinear interpolation in horizon units; bucket per row."""
        th, ah, bh = (np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in (theta_h, a_h, b_h))
        n = max(th.size, ah.size, bh.size)
        th, ah, bh = (np.broadcast_to(x, (n,)) for x in (th, ah, bh))
        bk = np.broadcast_to(np.atleast_1d(np.asarray(bucket, dtype=np.int64)), (n,))
        if np.isnan(th).any() or np.isnan(ah).any() or np.isnan(bh).any():
            raise ValueError("theta / barrier inputs contain NaN (use inf for no barrier)")
        if (ah <= 0).any() or (bh <= 0).any():
            raise ValueError("barrier distances must be > 0")
        unknown = sorted(set(np.unique(bk).tolist()) - set(self.tables))
        if unknown:
            raise KeyError(f"no table for buckets {unknown}; built: {sorted(self.tables)}")
        i, wi = _axis(np.asarray(self.params.theta_nodes, dtype=np.float64), th)
        j, wj = _axis(self._u_nodes, _u(ah))
        k, wk = _axis(self._u_nodes, _u(bh))
        out = np.empty((n, 5))
        for b in np.unique(bk):
            r = bk == b
            t = self.tables[int(b)]
            ii, jj, kk = i[r], j[r], k[r]
            a0, a1, c0, c1, d0, d1 = 1 - wi[r], wi[r], 1 - wj[r], wj[r], 1 - wk[r], wk[r]
            acc = np.zeros((r.sum(), 5))
            for di, w1 in ((0, a0), (1, a1)):
                for dj, w2 in ((0, c0), (1, c1)):
                    for dk, w3 in ((0, d0), (1, d1)):
                        acc += (w1 * w2 * w3)[:, None] * t[ii + di, jj + dj, kk + dk]
            out[r] = acc
        return out

    def _result(self, vals: np.ndarray, sigma_h: np.ndarray | None) -> HitProb:
        p = np.clip(vals[:, :3], 0.0, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        if sigma_h is None:
            mean_none = np.full(len(vals), np.nan)
        else:
            mean_none = np.expm1(sigma_h * vals[:, 3] + 0.5 * sigma_h**2 * vals[:, 4])
            mean_none = np.where(p[:, 2] > 0, mean_none, np.nan)
        return HitProb(p[:, 0], p[:, 1], p[:, 2], mean_none)

    def lookup(self, theta, a, b, bucket, sigma_daily=None) -> HitProb:
        """Scaled problem in DAILY units: theta = mu_d / sigma_d, a = log(tp) / sigma_d,
        b = -log(sl) / sigma_d (inf or None = no barrier). Arrays broadcast. mean_ret_none is
        computed only when `sigma_daily` is given (else NaN)."""
        rh = np.sqrt(self.params.horizon)
        a = np.inf if a is None else np.asarray(a, dtype=np.float64)
        b = np.inf if b is None else np.asarray(b, dtype=np.float64)
        vals = self._interp(np.asarray(theta, dtype=np.float64) * rh, a / rh, b / rh, bucket)
        sh = None if sigma_daily is None else np.broadcast_to(
            np.asarray(sigma_daily, dtype=np.float64) * rh, (len(vals),))
        return self._result(vals, sh)

    def score(
        self,
        mu_annual: np.ndarray,
        sigma_annual: np.ndarray,
        tp_ratio: np.ndarray | None,
        sl_ratio: np.ndarray | None,
        bucket: np.ndarray | None = None,
    ) -> HitProb:
        """Vectorized hit probabilities for many positions in the units of `simulate`.
        tp_ratio / sl_ratio: barrier / entry price per row; None or NaN = no barrier. `bucket` per row,
        or None to derive it from sigma_annual with `self.edges`."""
        mu = np.atleast_1d(np.asarray(mu_annual, dtype=np.float64))
        sig = np.atleast_1d(np.asarray(sigma_annual, dtype=np.float64))
        n = max(mu.size, sig.size)
        if (sig <= 0).any() or not np.isfinite(sig).all():
            raise ValueError("sigma_annual must be finite and > 0")
        if bucket is None:
            if self.edges is None:
                raise ValueError("pass bucket, or build the table with edges")
            bucket = assign_buckets(sig, self.edges)
        sigma_h = np.broadcast_to(sig * np.sqrt(self.params.horizon / TRADING_DAYS_PER_YEAR), (n,))
        theta_h = mu * (self.params.horizon / TRADING_DAYS_PER_YEAR) / sigma_h

        def dist(ratio, sign):
            if ratio is None:
                return np.full(n, np.inf)
            r = np.broadcast_to(np.asarray(ratio, dtype=np.float64), (n,))
            with np.errstate(divide="ignore", invalid="ignore"):
                d = sign * np.log(r) / sigma_h
            return np.where(np.isnan(r), np.inf, d)

        vals = self._interp(theta_h, dist(tp_ratio, 1.0), dist(sl_ratio, -1.0), bucket)
        return self._result(vals, sigma_h)


# direct classifier --------------------------------------------------------------------------------


@dataclass(frozen=True)
class HitClassifierParams:
    """Plan 6.1 conservative tree settings, multiclass."""

    num_leaves: int = 15
    max_depth: int = 5
    min_data_in_leaf: int = 1000
    learning_rate: float = 0.02
    feature_fraction: float = 0.6
    bagging_fraction: float = 0.7
    bagging_freq: int = 1
    lambda_l2: float = 10.0
    n_estimators: int = 2000             # cap; early stopping on valid multi_logloss decides
    n_estimators_no_valid: int = 300
    early_stopping_rounds: int = 100
    refit: bool = True                   # refit on train + valid for best_iteration rounds
    num_threads: int = 0


class HitClassifier:
    """LightGBM multiclass (tp | sl | time) on the barrier outcome of the chosen pair.

    Inputs: `cols` then `sigma_col` (if not already in cols), float64, nulls -> NaN. Training rows are
    those whose `outcome_col` is one of CLASSES, sorted by (date, security_id). `predict_proba(df)` /
    `predict(df)` -> (n, 3) in CLASSES order (= PROB_COLS), rows summing to 1. Implements
    `validation.runner.Model`; pass `extra_label_cols` for any barrier-label columns in the frame.
    """

    def __init__(
        self, cols: Sequence[str], params: HitClassifierParams = HitClassifierParams(), seed: int = 0,
        name: str = "hit_clf", *, outcome_col: str = "outcome", sigma_col: str = "sigma_hat",
    ):
        cols = list(dict.fromkeys(cols))
        self.cols = cols + ([sigma_col] if sigma_col not in cols else [])
        leaked = sorted(set(label_columns(self.cols)) | ({outcome_col} & set(self.cols)))
        if leaked:
            raise ValueError(f"label columns cannot be model inputs: {leaked}")
        self.params, self.seed, self.name, self.outcome_col = params, int(seed), name, outcome_col
        self.booster: lgb.Booster | None = None
        self.best_iteration = 0

    def _matrix(self, df: pl.DataFrame) -> np.ndarray:
        missing = [c for c in self.cols if c not in df.columns]
        if missing:
            raise KeyError(f"{self.name}: frame lacks model columns {missing}")
        return df.select([pl.col(c).cast(pl.Float64).fill_nan(None) for c in self.cols]).to_numpy().astype(
            np.float64, copy=False)

    def _xy(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = df.filter(pl.col(self.outcome_col).is_in(list(CLASSES)))
        keys = [k for k in KEYS if k in df.columns]
        if keys:
            df = df.sort(keys, maintain_order=True)
        y = df[self.outcome_col].replace_strict(list(CLASSES), list(range(len(CLASSES))),
                                                return_dtype=pl.Int32).to_numpy()
        return self._matrix(df), y

    def _lgb_params(self) -> dict:
        p = self.params
        return {
            "objective": "multiclass", "num_class": len(CLASSES), "num_leaves": p.num_leaves,
            "max_depth": p.max_depth, "min_data_in_leaf": p.min_data_in_leaf, "learning_rate": p.learning_rate,
            "feature_fraction": p.feature_fraction, "bagging_fraction": p.bagging_fraction,
            "bagging_freq": p.bagging_freq, "lambda_l2": p.lambda_l2, "num_threads": p.num_threads,
            "seed": self.seed, "deterministic": True, "force_row_wise": True, "verbose": -1,
            "metric": "multi_logloss", "feature_pre_filter": False,
        }

    def _ds(self, X, y, reference=None) -> lgb.Dataset:
        return lgb.Dataset(X, label=y, feature_name=self.cols, reference=reference, params={"verbose": -1})

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None, *, full: pl.DataFrame | None = None
            ) -> HitClassifier:
        """Early stopping on valid; with params.refit the booster is refit for best_iteration rounds on
        `full` (the runner's refit frame) when given, else on train + valid."""
        p, params = self.params, self._lgb_params()
        X, y = self._xy(train)
        if len(y) == 0:
            raise ValueError(f"{self.name}: no labelled training rows")
        Xv, yv = self._xy(valid) if valid is not None else (None, np.empty(0))
        if len(yv) == 0:
            self.booster = lgb.train(params, self._ds(X, y), num_boost_round=p.n_estimators_no_valid)
            self.best_iteration = p.n_estimators_no_valid
            return self
        dtrain = self._ds(X, y)
        booster = lgb.train(
            params, dtrain, num_boost_round=p.n_estimators,
            valid_sets=[self._ds(Xv, yv, reference=dtrain)], valid_names=["valid"],
            callbacks=[lgb.early_stopping(p.early_stopping_rounds, first_metric_only=True, verbose=False)],
        )
        self.best_iteration = int(booster.best_iteration) or booster.current_iteration()
        if p.refit:
            Xa, ya = self._xy(full) if full is not None else (np.vstack([X, Xv]), np.concatenate([y, yv]))
            self.booster = lgb.train(params, self._ds(Xa, ya), num_boost_round=self.best_iteration)
        else:
            self.booster = lgb.Booster(model_str=booster.model_to_string(num_iteration=self.best_iteration))
        return self

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError(f"{self.name}: fit before predict")
        if df.height == 0:
            return np.empty((0, len(CLASSES)))
        p = np.asarray(self.booster.predict(self._matrix(df)), dtype=np.float64).reshape(df.height, len(CLASSES))
        return p / p.sum(axis=1, keepdims=True)

    predict = predict_proba


# choosing and using the probabilities -------------------------------------------------------------


def multiclass_brier(p: np.ndarray, outcome: Sequence[str] | np.ndarray) -> float:
    """Mean over rows of sum_k (p_k - 1[outcome = k])^2, classes in CLASSES order."""
    p = np.asarray(p, dtype=np.float64)
    o = np.asarray(outcome)
    y = np.column_stack([o == c for c in CLASSES]).astype(np.float64)
    return float(((p - y) ** 2).sum(axis=1).mean()) if len(p) else float("nan")


def combine(mc: np.ndarray, clf: np.ndarray, method: str) -> np.ndarray:
    """The (n, 3) probabilities of a method: mc, classifier, or their average."""
    if method == "mc":
        return np.asarray(mc, dtype=np.float64)
    if method == "classifier":
        return np.asarray(clf, dtype=np.float64)
    if method == "average":
        return 0.5 * (np.asarray(mc, dtype=np.float64) + np.asarray(clf, dtype=np.float64))
    raise ValueError(f"method must be one of {METHODS}, not {method!r}")


def choose_method(
    oof_mc: pl.DataFrame, oof_clf: pl.DataFrame, outcomes: pl.DataFrame
) -> tuple[str, pl.DataFrame]:
    """Pick "mc" | "classifier" | "average" by the lowest multiclass Brier score on OOF rows.

    oof_mc, oof_clf: date, security_id, p_tp, p_sl, p_none. outcomes: date, security_id, outcome
    (tp | sl | time). Scored on the rows present in all three with a known outcome. Ties go to the
    earlier entry of METHODS. Table: method, brier, n (sorted as METHODS).
    """
    cols = list(PROB_COLS)
    o = outcomes.select(*KEYS, "outcome").filter(pl.col("outcome").is_in(list(CLASSES)))
    j = (
        o.join(oof_mc.select(*KEYS, *cols), on=list(KEYS), how="inner")
        .join(oof_clf.select(*KEYS, *cols), on=list(KEYS), how="inner", suffix="_clf")
        .sort(list(KEYS))
    )
    if j.is_empty():
        raise ValueError("choose_method: no OOF rows with an outcome in common")
    mc, clf = j.select(cols).to_numpy(), j.select([f"{c}_clf" for c in cols]).to_numpy()
    out = j["outcome"].to_numpy()
    scores = [multiclass_brier(combine(mc, clf, m), out) for m in METHODS]
    best = METHODS[int(np.argmin(scores))]
    table = pl.DataFrame({"method": list(METHODS), "brier": scores, "n": [j.height] * len(METHODS)})
    return best, table


def _probs(p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(p, HitProb):
        return tuple(np.asarray(x, dtype=np.float64) for x in (p.p_tp, p.p_sl, p.p_none))
    a = np.asarray(p, dtype=np.float64)
    return a[..., 0], a[..., 1], a[..., 2]


def _term(prob: np.ndarray, ret) -> np.ndarray:
    """prob * ret, with 0 where prob == 0 (a missing barrier's NaN return does not poison the sum)."""
    with np.errstate(invalid="ignore"):
        return np.where(prob > 0, prob * np.asarray(ret, dtype=np.float64), 0.0)


def expected_value(p, tp_ret, sl_ret, mean_ret_none, cost: float = 0.002):
    """p_tp * tp_ret + p_sl * sl_ret + p_none * mean_ret_none - 2 * cost (one-way cost, in and out).
    p: HitProb or (..., 3) array in PROB_COLS order; sl_ret is negative (e.g. -0.12). A term whose
    probability is 0 contributes 0 even if its return is NaN (no barrier on that side)."""
    pt, ps, pn = _probs(p)
    ev = _term(pt, tp_ret) + _term(ps, sl_ret) + _term(pn, mean_ret_none) - 2.0 * cost
    return float(ev) if np.ndim(ev) == 0 else ev


def reward_risk(tp_ret, sl_ret):
    """tp_ret / |sl_ret|; NaN when either barrier is missing (NaN) or sl_ret is 0."""
    t, s = np.asarray(tp_ret, dtype=np.float64), np.abs(np.asarray(sl_ret, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where(s > 0, t / s, np.nan)
    return float(rr) if np.ndim(rr) == 0 else rr


# criterion-6 calibration -------------------------------------------------------------------------


def hit_calibration(
    hits: pl.DataFrame,
    k_tp: float | None,
    k_sl: float | None,
    bins: int = 10,
    min_bin_rows: int = 1,
) -> dict:
    """Calibration of the hit probabilities for plan criterion 6, over the classes the chosen pair has.

    hits: p_tp, p_sl, p_none, outcome (in CLASSES; other rows are dropped).
    "tp" is tested only when k_tp is not None, "sl" only when k_sl is not None, "time" (p_none) whenever
    the pair has at least one barrier. A class whose probability column is identically 0 is untested
    (a 0-vs-0 comparison certifies nothing). The hold pair (no barriers) is not applicable.

    Returns {"applicable": bool, "tested": [classes in CLASSES order], "tables": {class: calibration
    table}, "gaps": {class: max gap}, "gap": max over tested classes (NaN if none has a bin with
    >= min_bin_rows rows), "worst": class giving "gap" (first in CLASSES order on ties) or None}."""
    from ltforecast.validation import metrics as M

    out = {"applicable": k_tp is not None or k_sl is not None, "tested": [], "tables": {}, "gaps": {},
           "gap": math.nan, "worst": None}
    if not out["applicable"]:
        return out
    has = {"tp": k_tp is not None, "sl": k_sl is not None, "time": True}
    rows = hits.filter(pl.col("outcome").is_in(list(CLASSES)))
    for cls, col in zip(CLASSES, PROB_COLS):
        if not has[cls]:
            continue
        p = rows[col].cast(pl.Float64)
        if not (p.is_not_null() & (p != 0.0)).any():
            continue
        df = rows.select(pl.col(col), (pl.col("outcome") == cls).cast(pl.Float64).alias("_hit"))
        table = M.calibration_table(df, col, "_hit", bins)
        out["tested"].append(cls)
        out["tables"][cls] = table
        out["gaps"][cls] = M.max_calibration_gap(table, min_bin_rows)
    finite = [(g, c) for c, g in out["gaps"].items() if math.isfinite(g)]
    if finite:
        best = max(g for g, _ in finite)
        out["gap"] = best
        out["worst"] = next(c for g, c in finite if g == best)
    return out

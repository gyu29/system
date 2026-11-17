"""Phases 5-6 orchestration (DESIGN_56 section K): walk-forward training and its report, the final
fit on all development rows, and scoring of any prediction dates.

train(store, cfg, mcfg, catalog) -> TrainResult. Development rows only; every step walks forward on
the SAME folds (`splitter.walk_forward_folds` on the dev dates, purge + embargo asserted by the runner):
  1. baselines B0-B3                          -> models/oof/<name>.parquet, reports/baselines.csv
  2. LightGBM regression vs lambdarank at the base params; optional 16-point grid (lgbm.tune)
  3. ensemble (seed-averaged LightGBM + ridge + composite) -> pred; rank_pct = per-date percentile
  4. prob = walk-forward isotonic map of rank_pct (earlier folds, matured labels only)
  5. vol model -> sigma_hat
  6. quantile model + walk-forward conformal correction -> q10, q50, q90
  7. exit simulation on the top quintile -> one global (k_tp, k_sl) pair and its verdict; then the
     barrier labels of that pair REBUILT with sigma = sigma_hat (`levels.barrier_labels`)
  8. hit probabilities: Monte Carlo table (built per fold from residuals up to the fold's last train
     date) and the multiclass classifier, compared on OOF Brier score -> p_tp, p_sl, p_none
  9. evaluation: all six pre-registered criteria (`metrics.check_criteria`), paired IC test vs B2,
     deflated Sharpe with the logged trial count -> reports/train_report.{json,md}
Outputs: models/oof/forecast.parquet with FORECAST_COLS, plus the OOF pieces fit_final needs.

sigma for the barrier labels: OOF sigma_hat on test dates. Rows dated before the first test year
(training rows of the hit classifier only) take the first fold's vol model, which was fit on labels
that ended before that fold's test year, so no classifier fold sees a label from its own test year.

fit_final(...) -> models/final/: every model refit on all dev rows (`runner.fit_on_all`), the isotonic
map on all dev OOF rows, the latest conformal offset, the chosen pair, the hit table and classifier,
metadata.json. load_final(dir) -> FinalModels; score(models, ...) -> SCORE_COLS for any dates, never
reading a label. Each score row carries `trained_through`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import json
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ltforecast.dataset import KEYS, label_columns, modeling_frame
from ltforecast.validation import metrics as M
from ltforecast.validation.experiments import config_hash, git_state, log_new, trial_count
from ltforecast.validation.runner import fit_on_all, walk_forward
from ltforecast.validation.splitter import Fold, walk_forward_folds

if TYPE_CHECKING:
    from ltforecast.config import BarrierRules, Config
    from ltforecast.features.catalog import Catalog
    from ltforecast.modelconfig import ModelConfig
    from ltforecast.store import Store

FORECAST_COLS = (
    "date", "security_id", "fold", "pred", "rank_pct", "prob", "prob_calibrated", "q10", "q50", "q90",
    "sigma_hat", "p_tp", "p_sl", "p_none", "tp_level", "sl_level",
)
SCORE_COLS = tuple(c for c in FORECAST_COLS if c != "fold") + ("trained_through",)
Q_COLS = ("q10", "q50", "q90")
PROB_COLS = ("p_tp", "p_sl", "p_none")
VOL_TARGET = "log_rvol_12m"
Q_TARGET = "excess_w_12m"
B2 = "b2_composite"
# plan section 0 numbering of the check_criteria rows
CRITERIA_ORDER = {
    "rank_ic": (1, "Mean rank IC >= 0.04 with Newey-West t >= 2"),
    "spread_positive_years": (2, "Top-minus-bottom decile spread positive in >= 65% of years"),
    "beats_baseline": (3, "Beats the factor-composite baseline (B2), paired IC test"),
    "interval_coverage": (4, "10-90% range holds 75-85% of outcomes"),
    "calibration": (5, "P(beats market) calibrated within 5 points"),
    "tpsl_rules": (6, "TP/SL rules match or beat holding; P(TP first) calibrated within 5 points"),
}


# paths and provenance ------------------------------------------------------------------------------


def oof_dir(store: Store) -> Path:
    return store.root / "models" / "oof"


def final_dir(store: Store) -> Path:
    return store.root / "models" / "final"


def reports_dir(store: Store) -> Path:
    return store.root / "reports"


def provenance(store: Store) -> dict:
    """data/core/PROVENANCE.json written by `build`; {"synthetic": None} when it is missing."""
    path = store.root / "core" / "PROVENANCE.json"
    return json.loads(path.read_text()) if path.exists() else {"synthetic": None}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(M.to_jsonable(obj), indent=2, sort_keys=True, allow_nan=False) + "\n")


# inputs --------------------------------------------------------------------------------------------


@dataclass
class Inputs:
    """Everything a run reads, dev rows only (dates <= `through`)."""

    frame: pl.DataFrame          # dataset.modeling_frame: features + labels
    folds: list[Fold]
    features_raw: pl.DataFrame
    prices: pl.DataFrame         # security_id, date, open, close, adj_close, ret (all dates: label windows)
    securities: pl.DataFrame
    calendar: pl.Series
    delist: pl.DataFrame
    through: dt.date

    @property
    def labels(self) -> pl.DataFrame:
        return self.frame.select(*KEYS, *label_columns(self.frame.columns))


def load_inputs(store: Store, cfg: Config, mcfg: ModelConfig, catalog: Catalog, *, through: dt.date | None = None
                ) -> Inputs:
    from ltforecast.labels import delisting_returns

    through = cfg.dates.dev_end if through is None else through
    if through > cfg.dates.dev_end:
        raise ValueError(f"through {through} is after dev_end {cfg.dates.dev_end}: holdout rows stay locked")
    frame = modeling_frame(store, cfg, catalog, periods=("dev",)).filter(pl.col("date") <= through)
    if frame.is_empty():
        raise ValueError("no development rows: run `ltforecast labels` and `ltforecast features` first")
    folds = walk_forward_folds(frame["date"].unique(), mcfg.validation, through)
    raw = (
        store.scan("features_raw")
        .select(*KEYS, "beta_60m", "log_mcap")
        .join(frame.select(*KEYS).lazy(), on=list(KEYS), how="semi")
        .collect()
    )
    prices = store.scan("prices_daily").select("security_id", "date", "open", "close", "adj_close", "ret").collect()
    securities = store.scan("securities").collect()
    calendar = store.scan("benchmark_daily").select("date").collect().to_series().sort()
    return Inputs(
        frame=frame, folds=folds, features_raw=raw, prices=prices, securities=securities, calendar=calendar,
        delist=delisting_returns(securities, cfg.delisting, calendar), through=through,
    )


def _need_folds(folds: Sequence[Fold]) -> None:
    if len(folds) < 3:
        raise ValueError(
            f"{len(folds)} walk-forward folds; selection needs at least 3 (check validation.first_test_year "
            "and dates.dev_end)"
        )


def _join(left: pl.DataFrame, right: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    return left.join(right.select(*KEYS, *cols), on=list(KEYS), how="left")


def rank_pct(pred: str = "pred") -> pl.Expr:
    """Per-date percentile of the prediction in (0, 1): (average rank - 0.5) / n."""
    return ((pl.col(pred).rank("average") - 0.5) / pl.col(pred).count()).over("date")


# step 1: baselines ---------------------------------------------------------------------------------


def _ic_row(name: str, report: dict) -> dict:
    s = report["ic"]
    ic_ir = s["ic_ir"]
    if not math.isfinite(ic_ir) and s["mean_ic"] == 0.0:  # constant prediction (B0): IC 0, not NaN
        ic_ir = 0.0
    return {
        "model": name, "mean_ic": s["mean_ic"], "nw_tstat": s["nw_tstat"], "ic_ir": ic_ir,
        "pct_positive_months": s["pct_positive"], "pct_positive_years": report["pct_positive_years"],
        "mean_spread": report["mean_spread"], "monotonicity": report["monotonicity"],
        "n_months": s["n_months"], "n_obs": report["n_obs"],
    }


def _evaluate(oof: pl.DataFrame, labels: pl.DataFrame, mcfg: ModelConfig, pred: str = "pred") -> dict:
    ev = _join(oof, labels, ["target", mcfg.calibration.outcome])
    return M.evaluate(ev, pred=pred, ret=mcfg.calibration.outcome, lags=mcfg.metrics.nw_lags,
                      n_deciles=mcfg.metrics.n_deciles)


def run_baselines(store: Store, cfg: Config, mcfg: ModelConfig, catalog: Catalog, *, inputs: Inputs | None = None
                  ) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """B0-B3 walked forward. Writes models/oof/<name>.parquet and reports/baselines.csv; logs one trial
    per baseline. Returns (table, {name: oof}). The table's vs_b2_* columns are the paired IC test."""
    from ltforecast.models.baselines import baseline_factories

    inp = inputs or load_inputs(store, cfg, mcfg, catalog)
    _need_folds(inp.folds)
    labels = inp.labels
    out_dir = oof_dir(store)
    out_dir.mkdir(parents=True, exist_ok=True)
    oofs, rows, ics = {}, [], {}
    for name, make in baseline_factories(catalog, mcfg.baselines).items():
        oof = walk_forward(make, inp.frame, inp.folds, mcfg.validation)
        oof.write_parquet(out_dir / f"{name}.parquet")
        report = _evaluate(oof, labels, mcfg)
        oofs[name], ics[name] = oof, report["ic_by_date"]
        row = _ic_row(name, report)
        rows.append(row)
        log_new(mcfg.experiments_dir, f"baseline/{name}",
                {"model": name, "params": dataclasses.asdict(mcfg.baselines), "folds": [f.name for f in inp.folds]},
                {k: v for k, v in row.items() if k != "model"}, n_trials=1)
    for row in rows:
        t = M.paired_ic_test(ics[row["model"]], ics[B2], mcfg.metrics.nw_lags)
        row["vs_b2_mean_diff"], row["vs_b2_tstat"] = t["mean_diff"], t["nw_tstat"]
    table = pl.DataFrame(rows)
    reports_dir(store).mkdir(parents=True, exist_ok=True)
    table.write_csv(reports_dir(store) / "baselines.csv")
    return table, oofs


# steps 2-3: point forecast -------------------------------------------------------------------------


def ensemble_factory(catalog: Catalog, mcfg: ModelConfig, lgbm_params) -> functools.partial:
    """A picklable zero-argument factory of the plan 6.1 ensemble with `lgbm_params`."""
    from ltforecast.models.ensemble import Ensemble, default_factories

    members = default_factories(
        catalog, lgbm_params=lgbm_params, seeds=mcfg.lgbm.seeds, baseline_params=mcfg.baselines,
    )
    e = mcfg.ensemble
    return functools.partial(Ensemble, members, e.step, e.shrinkage, ic_min_n=e.ic_min_n)


def _choose_lgbm(inp: Inputs, mcfg: ModelConfig, catalog: Catalog) -> tuple[Any, dict]:
    from ltforecast.models import tune

    cols, exp, rules = catalog.model_columns(), mcfg.experiments_dir, mcfg.validation
    seed = mcfg.lgbm.seeds[0]
    objs = tune.objectives(mcfg.lgbm.params)
    t_obj = tune.run(inp.frame, inp.folds, rules, objs, exp, cols=cols, seed=seed, name="lgbm_objective")
    winner = tune.best(t_obj, objs)
    info = {"objective_table": t_obj, "grid_table": None}
    if mcfg.lgbm.tune:
        grid = tune.grid(winner)
        t_grid = tune.run(inp.frame, inp.folds, rules, grid, exp, cols=cols, seed=seed, name="lgbm_grid")
        winner = tune.best(t_grid, grid)
        info["grid_table"] = t_grid
    info["winner"] = dataclasses.asdict(winner)
    return winner, info


# step 8 helpers (shared by train and score) --------------------------------------------------------


def level_ratios(sigma: np.ndarray, k_tp: float | None, k_sl: float | None, rules: BarrierRules
                 ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(TP price / entry, SL price / entry) per row from levels.level_ratio; None = no barrier."""
    from ltforecast.levels import level_ratio

    return level_ratio(sigma, k_tp, rules.tp_bounds, "tp"), level_ratio(sigma, k_sl, rules.sl_bounds, "sl")


def _level_cols(df: pl.DataFrame, k_tp, k_sl, rules: BarrierRules) -> pl.DataFrame:
    tp, sl = level_ratios(df["sigma_hat"].to_numpy(), k_tp, k_sl, rules)
    n = df.height
    return df.with_columns(
        pl.Series("tp_level", np.full(n, np.nan) if tp is None else tp - 1, dtype=pl.Float64).fill_nan(None),
        pl.Series("sl_level", np.full(n, np.nan) if sl is None else sl - 1, dtype=pl.Float64).fill_nan(None),
    )


def table_params(mcfg: ModelConfig, horizon: int):
    from ltforecast.hitprob import HitTableParams

    lv = mcfg.levels
    kw = {"n_paths": lv.mc_paths, "horizon": horizon, "seed": lv.seed}
    if lv.theta_nodes is not None:
        kw["theta_nodes"] = tuple(lv.theta_nodes)
    if lv.barrier_nodes is not None:
        kw["barrier_nodes"] = tuple(lv.barrier_nodes)
    return HitTableParams(**kw)


def month_end_sigma(prices: pl.DataFrame, keys: pl.DataFrame, rules: BarrierRules) -> pl.DataFrame:
    """Trailing realized vol (levels.trailing_sigma) on the (security_id, date) rows of `keys`:
    the point-in-time scale used to standardize residuals and to cut the volatility buckets."""
    from ltforecast.levels import trailing_sigma

    sig = trailing_sigma(prices, rules.sigma_window, rules.sigma_min_obs)
    return sig.join(keys.select("security_id", "date"), on=["security_id", "date"], how="semi").drop_nulls("sigma")


def build_hit_table(prices: pl.DataFrame, sigma: pl.DataFrame, through: dt.date, mcfg: ModelConfig, horizon: int):
    """Monte Carlo hit table from daily residuals dated <= `through`, buckets cut on sigma <= `through`.
    A bucket with too few residuals borrows the pooled residuals."""
    from ltforecast.hitprob import HitTable, standardized_residuals, vol_bucket_edges

    lv = mcfg.levels
    resid = standardized_residuals(
        prices, sigma, through, lv.n_buckets, max_per_bucket=lv.max_residuals_per_bucket, seed=lv.seed,
    )
    pooled = np.concatenate([z for z in resid.values() if z.size]) if resid else np.empty(0)
    if pooled.size < 2 or np.ptp(pooled) == 0:
        raise ValueError(f"no usable daily residuals dated <= {through} for the hit table")
    shocks = {b: (z if z.size >= 2 and np.ptp(z) > 0 else pooled) for b, z in resid.items()}
    edges = vol_bucket_edges(sigma, through, lv.n_buckets)
    return HitTable.build(shocks, table_params(mcfg, horizon), edges=edges)


def mc_probs(table, q50: np.ndarray, sigma_hat: np.ndarray, tp_ratio, sl_ratio, market_return: float) -> np.ndarray:
    """(n, 3) Monte Carlo p_tp, p_sl, p_none from the hit table (drift = q50 + log(1 + market_return))."""
    from ltforecast.hitprob import annual_drift

    if len(q50) == 0:
        return np.empty((0, 3))
    p = table.score(annual_drift(q50, market_return), sigma_hat, tp_ratio, sl_ratio)
    return p.as_array()


# train ---------------------------------------------------------------------------------------------


@dataclass
class TrainResult:
    forecast: pl.DataFrame
    report: dict
    checks: list
    recommendation: str
    paths: dict[str, Path] = field(default_factory=dict)


def _sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2 or x.std(ddof=1) == 0:
        return math.nan
    return float(x.mean() / x.std(ddof=1))


def horizon_months(outcome: str) -> int:
    """Months spanned by one outcome window, from the `_<n>m` suffix of its name (1 without one)."""
    m = re.search(r"_(\d+)m$", outcome)
    return int(m.group(1)) if m else 1


def spread_deflated_sharpe(spread: np.ndarray, n_trials: int, var_sr: float | None, horizon: int) -> dict:
    """Deflated Sharpe of a monthly spread series whose observations are `horizon`-month returns.

    Consecutive windows overlap by horizon - 1 months, so the sample holds about T / horizon independent
    observations: n_eff = clip(T / horizon, 2, T) replaces T in the sqrt(T - 1) scaling and in the
    1 / n fallback for `var_sr` (used when `var_sr` is None). The Sharpe itself stays the full-sample
    monthly one (every month used, no dependence on the starting month)."""
    from scipy.stats import kurtosis, skew

    finite = np.asarray(spread, dtype=float)
    finite = finite[np.isfinite(finite)]
    sr = _sharpe(finite)
    n_months = int(finite.size)
    n_eff = float(min(n_months, max(n_months / max(horizon, 1), 2.0))) if n_months else 0.0
    if var_sr is None:
        var_sr, source = 1.0 / max(n_eff, 1.0), "1 / n_eff (null variance of a Sharpe estimate)"
    else:
        source = "cross-strategy variance of this run's spread Sharpes"
    dsr = M.deflated_sharpe(
        sr, n_eff, n_trials, var_sr,
        float(skew(finite)) if n_months > 2 else 0.0,
        float(kurtosis(finite, fisher=False)) if n_months > 3 else 3.0,
    ) if math.isfinite(sr) else math.nan
    return {"spread_sharpe_monthly": sr, "n_months": n_months, "horizon_months": horizon, "n_eff": n_eff,
            "n_trials": n_trials, "var_sr": var_sr, "var_sr_source": source, "dsr": dsr}


# train-state provenance ----------------------------------------------------------------------------


def effective_config_hash(mcfg: ModelConfig) -> str:
    """Hash of the model.yaml AND the fast flag: `config_hash(mcfg.raw)` alone is the same for a --fast
    and a full run, because `raw` is the un-merged yaml."""
    return config_hash({"raw": mcfg.raw, "fast": mcfg.fast})


def data_fingerprint(frame: pl.DataFrame) -> dict[str, str]:
    """{ISO date: "<rows>:<checksum>"} over every column of `frame`, independent of row order. Per date,
    so a refit through an earlier date can be checked against the train run. Uses polars row hashes:
    stable for one polars version (re-run train after upgrading polars)."""
    h = frame.select("date").with_columns(pl.Series("_h", frame.hash_rows(seed=0).to_numpy() >> np.uint64(20)))
    agg = h.group_by("date").agg(pl.len().alias("n"), pl.col("_h").sum().alias("s")).sort("date")
    return {d.isoformat(): f"{n}:{s}" for d, n, s in agg.iter_rows()}


def check_train_state(state: dict, mcfg: ModelConfig, through: dt.date) -> None:
    """Raise unless train_state.json came from a train run with this model config and fast flag that
    reached `through`. (The data check is `data_fingerprint`, after the inputs are loaded.)"""
    missing = [k for k in ("effective_config_hash", "fast", "dev_end") if k not in state]
    if missing:
        raise ValueError(f"train_state.json lacks {missing} (written by an older train): re-run `ltforecast train`")
    if bool(state["fast"]) != mcfg.fast:
        raise ValueError(
            f"train_state.json is from a {'--fast' if state['fast'] else 'full'} train run but fit-final runs "
            f"{'--fast' if mcfg.fast else 'full'}: re-run train with the same fast setting"
        )
    if state["effective_config_hash"] != effective_config_hash(mcfg):
        raise ValueError("train_state.json was written under a different model config: re-run `ltforecast train`")
    if through > dt.date.fromisoformat(str(state["dev_end"])):
        raise ValueError(f"through {through} is after the train run's last date {state['dev_end']}: re-run train")


def train(store: Store, cfg: Config, mcfg: ModelConfig, catalog: Catalog) -> TrainResult:
    """Walk-forward training and the train report (module docstring, steps 1-9). Dev rows only."""
    from ltforecast.calibrate import walk_forward_conformal, walk_forward_isotonic
    from ltforecast.exits import exit_entries, position_paths, select_pair, signal_exit_days
    from ltforecast.explain import FoldImportance, importance_stability
    from ltforecast.hitprob import HitClassifier, choose_method, combine
    from ltforecast.levels import barrier_labels
    from ltforecast.models.quantile import QuantileModel, width_vol_correlation
    from ltforecast.models.vol import LVOL_COLS, VolModel, oos_r2_by_fold, vol_frame

    started = time.time()
    inp = load_inputs(store, cfg, mcfg, catalog)
    folds, rules, exp = inp.folds, mcfg.validation, mcfg.experiments_dir
    _need_folds(folds)
    labels, cols, cal = inp.labels, catalog.model_columns(), mcfg.calibration
    barriers = cfg.labels.barriers
    horizon = cfg.labels.horizons[barriers.horizon]
    fold_names = [f.name for f in folds]
    out_dir = oof_dir(store)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. baselines
    base_table, base_oof = run_baselines(store, cfg, mcfg, catalog, inputs=inp)

    # 2. LightGBM objective (+ optional grid)
    winner, lgbm_info = _choose_lgbm(inp, mcfg, catalog)

    # 3. ensemble -> pred, rank_pct
    importance = FoldImportance(catalog, max_rows=mcfg.ensemble.importance_max_rows)
    weights: dict[str, dict] = {}

    def on_ens_fold(fold, model, shown):
        weights[fold.name] = {"weights": model.weights_, "member_valid_ic": model.member_valid_ic_,
                              "valid_ic": model.valid_ic_}
        importance(fold, model, shown)

    oof = walk_forward(ensemble_factory(catalog, mcfg, winner), inp.frame, folds, rules, on_fold=on_ens_fold)
    oof = oof.with_columns(rank_pct("pred").alias("rank_pct"))
    ens_report = _evaluate(oof, labels, mcfg)
    log_new(exp, "ensemble",
            {"lgbm": dataclasses.asdict(winner), "seeds": list(mcfg.lgbm.seeds),
             "ensemble": dataclasses.asdict(mcfg.ensemble), "folds": fold_names},
            _ic_row("ensemble", ens_report), n_trials=1)

    # 4. P(beat market)
    prob = walk_forward_isotonic(oof, labels, folds, min_rows=cal.min_rows, min_dates=cal.min_dates,
                                 outcome=cal.outcome)

    # 5. volatility -> sigma_hat
    vf = vol_frame(inp.features_raw, inp.frame, inp.prices, inp.securities)
    vol_models: dict[str, Any] = {}
    oof_vol = walk_forward(
        functools.partial(VolModel, mcfg.vol), vf, folds, rules, target=VOL_TARGET,
        on_fold=lambda f, m, _: vol_models.__setitem__(f.name, m),
    )
    vol_r2 = oos_r2_by_fold(oof_vol, vf)
    sigma_oof = oof_vol.select(*KEYS, pl.col("pred").alias("sigma_hat"))
    pre = vf.filter(pl.col("date") < folds[0].test_dates[0])
    sigma_all = pl.concat([
        pre.select(*KEYS).with_columns(
            pl.Series("sigma_hat", vol_models[folds[0].name].predict(pre), dtype=pl.Float64), pl.lit(False).alias("oof")
        ),
        sigma_oof.with_columns(pl.lit(True).alias("oof")),
    ]).sort(*KEYS)
    log_new(exp, "vol", {"params": dataclasses.asdict(mcfg.vol), "folds": fold_names},
            {"oos_r2": dict(zip(vol_r2["fold"], vol_r2["r2"]))}, n_trials=1)

    # 6. quantiles + conformal
    qframe = inp.frame.join(vf.select(*KEYS, *LVOL_COLS), on=list(KEYS), how="left")
    q = mcfg.quantile
    oof_q = walk_forward(
        functools.partial(QuantileModel, cols, q.alphas, q.params, q.seed), qframe, folds, rules,
        target=Q_TARGET, outputs=Q_COLS,
    )
    conf = walk_forward_conformal(oof_q, labels, folds, alpha=cal.alpha, window_months=cal.window_months,
                                  min_rows=cal.min_rows, min_dates=cal.min_dates, outcome=cal.outcome)
    qs = _join(conf, sigma_oof, ["sigma_hat"])
    wvc = width_vol_correlation(qs.select(Q_COLS).to_numpy(), qs["sigma_hat"].to_numpy(), qs["date"])
    raw_cov = M.interval_coverage(_join(oof_q, labels, [cal.outcome]), "q10", "q90", cal.outcome)
    log_new(exp, "quantile", {"params": dataclasses.asdict(q.params), "alphas": list(q.alphas),
                              "calibration": dataclasses.asdict(cal), "folds": fold_names},
            {"raw_coverage": raw_cov, "width_vol_corr": wvc}, n_trials=1)

    # 7. exit rules -> (k_tp, k_sl), then barrier labels of that pair with sigma = sigma_hat
    cost = mcfg.costs_bps / 1e4
    lv = mcfg.levels
    ent = exit_entries(oof, sigma_oof.rename({"sigma_hat": "sigma"}), lv.top_fraction)
    paths = position_paths(inp.prices, ent, inp.delist, inp.calendar, horizon)
    sdays = signal_exit_days(paths, oof.select("security_id", "date", "rank_pct"), lv.signal_threshold)
    sel = select_pair(paths, barriers, cost=cost, k_trail=lv.k_trail, signal_days=sdays,
                      signal_threshold=lv.signal_threshold)
    n_positions = len(paths)
    del paths
    log_new(exp, "exit_pairs",
            {"k_tp": list(barriers.k_tp), "k_sl": list(barriers.k_sl), "tp_bounds": list(barriers.tp_bounds),
             "sl_bounds": list(barriers.sl_bounds), "k_trail": list(lv.k_trail), "signal": True,
             "signal_threshold": lv.signal_threshold, "cost": cost, "top_fraction": lv.top_fraction,
             "folds": fold_names},
            {"k_tp": sel.k_tp, "k_sl": sel.k_sl, "verdict": sel.verdict, "hold": sel.hold,
             "summary": sel.summary.to_dicts()},
            n_trials=sel.n_trials)
    chosen = dataclasses.replace(barriers, k_tp=(sel.k_tp,), k_sl=(sel.k_sl,))
    bar = barrier_labels(
        inp.prices, sigma_all.select("security_id", "date", pl.col("sigma_hat").alias("sigma")),
        inp.delist, inp.calendar, horizon, chosen,
    ).select(*KEYS, "outcome", "exit_ret", "exit_day")
    bar.write_parquet(out_dir / "barrier_labels.parquet")

    # 8. hit probabilities: Monte Carlo (per-fold table) and classifier, compared on OOF Brier
    base = _join(_join(oof.select(*KEYS, "fold"), sigma_oof, ["sigma_hat"]), conf, ["q50"])
    base = _level_cols(base, sel.k_tp, sel.k_sl, barriers)
    tsig = month_end_sigma(inp.prices, inp.frame, barriers)
    mc = np.empty((base.height, 3))
    fold_col = base["fold"].to_numpy()
    for f in folds:
        rows = np.flatnonzero(fold_col == f.name)
        if rows.size == 0:
            continue
        part = base[rows]
        table = build_hit_table(inp.prices, tsig, f.train_dates[-1], mcfg, horizon)
        tp, sl = level_ratios(part["sigma_hat"].to_numpy(), sel.k_tp, sel.k_sl, barriers)
        mc[rows] = mc_probs(table, part["q50"].to_numpy(), part["sigma_hat"].to_numpy(), tp, sl, lv.market_return)
    mc_df = base.select(*KEYS).with_columns(*(pl.Series(c, mc[:, j]) for j, c in enumerate(PROB_COLS)))

    hframe = _join(_join(inp.frame, sigma_all, ["sigma_hat"]), bar, ["outcome"])
    hc = mcfg.hit_classifier
    oof_clf = walk_forward(
        functools.partial(HitClassifier, cols, hc.params, hc.seed), hframe, folds, rules,
        target="outcome", outputs=PROB_COLS,
    )
    try:
        method, brier = choose_method(mc_df, oof_clf, bar)
    except ValueError:  # no OOF row has a complete barrier window
        method, brier = "mc", pl.DataFrame(schema={"method": pl.String, "brier": pl.Float64, "n": pl.Int64})
    log_new(exp, "hit_method", {"pair": [sel.k_tp, sel.k_sl], "classifier": dataclasses.asdict(hc.params),
                                "mc_paths": lv.mc_paths, "market_return": lv.market_return, "folds": fold_names},
            {"method": method, "brier": brier.to_dicts()}, n_trials=len(brier) or 1)
    clf_al = mc_df.select(*KEYS).join(oof_clf, on=list(KEYS), how="left")
    probs = combine(mc, clf_al.select(PROB_COLS).to_numpy(), method)

    # forecast
    forecast = (
        base.select(*KEYS, "fold", "sigma_hat", "tp_level", "sl_level")
        .join(oof.select(*KEYS, "pred", "rank_pct"), on=list(KEYS), how="left")
        .join(prob, on=list(KEYS), how="left")
        .join(conf.select(*KEYS, *Q_COLS), on=list(KEYS), how="left")
        .with_columns(*(pl.Series(c, probs[:, j]) for j, c in enumerate(PROB_COLS)))
        .select(FORECAST_COLS)
        .sort(*KEYS)
    )

    # 9. evaluation
    conf_dates = conf.group_by("date").agg(pl.col("conformal_offset").first(), pl.col("conformal_ok").first()).sort("date")
    report, checks, rec = evaluate_forecast(
        forecast, labels, bar, base_oof, base_table, mcfg, sel=sel, conformal=conf_dates, extras={
            "lgbm": lgbm_info, "ensemble_weights": weights, "vol_r2": vol_r2, "width_vol_corr": wvc,
            "raw_coverage": raw_cov, "conformal_offsets": conf_dates,
            "importance": importance.table(), "hit_method": method, "hit_brier": brier,
            "n_exit_positions": n_positions,
        },
    )
    report["importance_stability"] = importance_stability(report["extras"]["importance"]) \
        if report["extras"]["importance"].height else math.nan
    report["provenance"] = provenance(store)
    report["folds"] = [{"name": f.name, "first_test": f.test_dates[0], "last_test": f.test_dates[-1],
                        "last_train": f.train_dates[-1]} for f in folds]
    report["fast"] = mcfg.fast
    report["runtime_s"] = time.time() - started

    # outputs
    paths = {
        "forecast": out_dir / "forecast.parquet",
        "quantile_raw": out_dir / "quantile_raw.parquet",
        "sigma_hat": out_dir / "sigma_hat.parquet",
        "state": out_dir / "train_state.json",
        "report_json": reports_dir(store) / "train_report.json",
        "report_md": reports_dir(store) / "train_report.md",
        "importance": reports_dir(store) / "importance_by_fold.csv",
    }
    forecast.write_parquet(paths["forecast"])
    oof_q.write_parquet(paths["quantile_raw"])
    sigma_all.write_parquet(paths["sigma_hat"])
    _write_json(paths["state"], {
        "lgbm_params": dataclasses.asdict(winner), "k_tp": sel.k_tp, "k_sl": sel.k_sl, "verdict": sel.verdict,
        "hit_method": method, "recommendation": rec, "folds": fold_names, "dev_end": inp.through,
        "model_config_hash": config_hash(mcfg.raw), "fast": mcfg.fast,
        "effective_config_hash": effective_config_hash(mcfg), "data_fingerprint": data_fingerprint(inp.frame),
    })
    report["extras"]["importance"].write_csv(paths["importance"])
    _write_json(paths["report_json"], report)
    paths["report_md"].write_text(report_markdown(report))
    return TrainResult(forecast=forecast, report=report, checks=checks, recommendation=rec, paths=paths)


# evaluation and report -----------------------------------------------------------------------------


def evaluate_forecast(
    forecast: pl.DataFrame,
    labels: pl.DataFrame,
    barriers: pl.DataFrame,
    baseline_oof: dict[str, pl.DataFrame],
    baseline_table: pl.DataFrame,
    mcfg: ModelConfig,
    *,
    sel,
    conformal: pl.DataFrame | None = None,
    extras: dict | None = None,
) -> tuple[dict, list, str]:
    """Step 9: (report, check_criteria rows, recommendation). All six plan-0 criteria are evaluated;
    recommendation = "ship ensemble" only when criterion 3 (beats B2) passes, else "ship B2".

    conformal: optional (date, conformal_ok) rows. Criterion 4 is then judged on the dates whose
    conformal offset was fitted (conformal_ok); the all-rows coverage is reported beside it. When no
    date is conformal_ok (or `conformal` is None), criterion 4 uses all rows."""
    from ltforecast.hitprob import hit_calibration

    cal, lags, crit = mcfg.calibration, mcfg.metrics.nw_lags, mcfg.criteria
    report = _evaluate(forecast, labels, mcfg)
    ic_b2 = _evaluate(baseline_oof[B2], labels, mcfg)["ic_by_date"]
    paired = M.paired_ic_test(report["ic_by_date"], ic_b2, lags)

    ev = _join(forecast, labels, [cal.outcome])
    coverage_all = M.interval_coverage(ev, "q10", "q90", cal.outcome)
    ok_dates = [] if conformal is None else conformal.filter(pl.col("conformal_ok"))["date"].to_list()
    if ok_dates:
        coverage = M.interval_coverage(ev.filter(pl.col("date").is_in(ok_dates)), "q10", "q90", cal.outcome)
        coverage_rows = "conformal_ok"
    else:
        coverage, coverage_rows = coverage_all, "all"
    calibrated = ev.filter(pl.col("prob_calibrated")).with_columns(
        pl.when(pl.col(cal.outcome).is_null()).then(None).otherwise((pl.col(cal.outcome) > 0).cast(pl.Float64))
        .alias("_beat")
    )
    prob_table = M.calibration_table(calibrated, "prob", "_beat", mcfg.metrics.prob_bins)
    prob_gap = M.max_calibration_gap(prob_table, cal.min_bin_rows)
    every = ev.with_columns(
        pl.when(pl.col(cal.outcome).is_null()).then(None).otherwise((pl.col(cal.outcome) > 0).cast(pl.Float64))
        .alias("_beat")
    )
    prob_table_all = M.calibration_table(every, "prob", "_beat", mcfg.metrics.prob_bins)

    hits = _join(forecast, barriers, ["outcome"])
    hc = hit_calibration(hits, sel.k_tp, sel.k_sl, mcfg.metrics.prob_bins, cal.min_bin_rows)
    hold_dd = sel.hold["max_dd"]
    chosen = sel.summary.filter(
        pl.col("k_tp").eq_missing(sel.k_tp) & pl.col("k_sl").eq_missing(sel.k_sl)
    ).to_dicts()
    chosen = chosen[0] if chosen else {"mean_sharpe": math.nan, "mean_max_dd": math.nan}
    tpsl = {
        "beats_hold": sel.verdict == "exit_orders",
        "p_gap": hc["gap"],
        "p_gap_class": hc["worst"],
        "p_tested": hc["tested"],
        "p_applicable": hc["applicable"],
        "detail": (f"pair k_tp={sel.k_tp}, k_sl={sel.k_sl}: mean net Sharpe {chosen['mean_sharpe']:.2f} vs hold "
                   f"{sel.hold['sharpe']:.2f}, mean max drawdown {chosen['mean_max_dd']:.1%} vs hold {hold_dd:.1%}"),
    }
    checks = M.check_criteria(report, crit, vs_baseline=paired, coverage=coverage, calibration_gap=prob_gap,
                              tpsl=tpsl)

    # deflated Sharpe of the monthly top-minus-bottom spread, haircut by every logged trial; the spreads
    # are overlapping `horizon`-month returns, so the sample size is the number of independent windows
    n_dec, outcome = mcfg.metrics.n_deciles, cal.outcome
    spread = M.spread_by_date(ev, "pred", outcome, n_dec)["spread"].to_numpy()
    sr = _sharpe(spread)
    srs = [sr] + [
        _sharpe(M.spread_by_date(_join(o, labels, [outcome]), "pred", outcome, n_dec)["spread"].to_numpy())
        for name, o in baseline_oof.items() if name != "b0_no_skill"
    ]
    srs = [s for s in srs if math.isfinite(s)]
    n_trials = max(trial_count(mcfg.experiments_dir), 1)
    dsr = spread_deflated_sharpe(spread, n_trials, float(np.var(srs, ddof=1)) if len(srs) >= 2 else None,
                                 horizon_months(outcome))

    beats_b2 = next(c for c in checks if c.name == "beats_baseline").status == "pass"
    rec = "ship ensemble" if beats_b2 else "ship B2"
    report = {
        "evaluation": {k: v for k, v in report.items() if k != "ic_by_date"},
        "ic_by_date": report["ic_by_date"],
        "vs_b2": paired,
        "beats_b2": beats_b2,
        "recommendation": rec,
        "criteria": [{"n": CRITERIA_ORDER[c.name][0], "name": c.name, "title": CRITERIA_ORDER[c.name][1],
                      "status": c.status, "detail": c.detail} for c in checks],
        "all_criteria_pass": all(c.status == "pass" for c in checks),
        "coverage": coverage,
        "coverage_rows": coverage_rows,
        "coverage_all_rows": coverage_all,
        "prob_calibration": prob_table,
        "prob_calibration_gap": prob_gap,
        "prob_calibration_all_rows": prob_table_all,
        "prob_calibration_gap_all_rows": M.max_calibration_gap(prob_table_all, cal.min_bin_rows),
        "hit_calibration": {"tables": hc["tables"], "gaps": hc["gaps"], "gap": hc["gap"], "worst": hc["worst"],
                            "tested": hc["tested"], "applicable": hc["applicable"]},
        "tp_calibration": hc["tables"].get("tp", pl.DataFrame()),
        "tp_calibration_gap": hc["gaps"].get("tp", math.nan),
        "exits": {"k_tp": sel.k_tp, "k_sl": sel.k_sl, "verdict": sel.verdict, "hold": sel.hold,
                  "summary": sel.summary, "variants": sel.variants, "n_trials": sel.n_trials},
        "deflated_sharpe": dsr,
        "trial_count": n_trials,
        "baselines": baseline_table,
        "extras": extras or {},
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": git_state()[0],
    }
    return report, checks, rec


def _fmt(x: Any, spec: str = ".4f") -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int) and not spec.startswith(("+", ".")):
        return format(x, spec)
    if isinstance(x, int):
        return str(x)
    return format(x, spec) if isinstance(x, float) else str(x)


def _md_table(df: pl.DataFrame, specs: dict[str, str] | None = None) -> str:
    specs = specs or {}
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.width]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(_fmt(v, specs.get(c, ".4f")) for c, v in zip(df.columns, row)) + " |")
    return "\n".join(lines)


def _hit_calibration_line(r: dict) -> str:
    hc = r.get("hit_calibration")
    if hc is None:  # reports written before hit_calibration existed
        return f"largest P(TP first) calibration gap {_fmt(r.get('tp_calibration_gap'), '.3f')}."
    if not hc.get("applicable", True):
        return "hit-probability calibration not applicable (hold pair, no barriers)."
    per = ", ".join(f"{c} {_fmt(g, '.3f')}" for c, g in hc["gaps"].items()) or "none tested"
    return f"largest hit-probability calibration gap {_fmt(hc['gap'], '.3f')} ({hc['worst'] or 'n/a'}; per class: {per})."


def report_markdown(r: dict) -> str:
    """The human-readable train report. States the B2 comparison and the recommendation plainly and
    stamps SYNTHETIC DATA when the core tables were built from simulated data."""
    ex = r["extras"]
    out = ["# Train report (walk-forward, development period)", ""]
    synthetic = (r.get("provenance") or {}).get("synthetic")
    if synthetic:
        out += ["> **SYNTHETIC DATA.** These results come from simulated data with a planted signal. "
                "They test the machinery and say nothing about real markets.", ""]
    elif synthetic is None:
        out += ["> Data provenance unknown (no data/core/PROVENANCE.json).", ""]
    folds = r.get("folds", [])
    span = f"{folds[0]['name']}-{folds[-1]['name']}" if folds else "n/a"
    out += [
        f"Created {r['created_utc']}, git {r['git_commit'][:12]}, {len(folds)} folds ({span})"
        + (", --fast run" if r.get("fast") else "") + ".", "",
        "## Verdict", "",
        f"- **Does the ensemble beat B2 (factor composite)? {'YES' if r['beats_b2'] else 'NO'}.** "
        f"Paired IC test: mean IC gain {_fmt(r['vs_b2']['mean_diff'], '+.4f')}, Newey-West t "
        f"{_fmt(r['vs_b2']['nw_tstat'], '.2f')} over {r['vs_b2']['n_months']} months.",
        f"- **Recommendation: {r['recommendation']}.**"
        + ("" if r["beats_b2"] else " Plan 5: if the ML model cannot beat B2, ship B2."),
        f"- Pre-registered criteria: {sum(c['status'] == 'pass' for c in r['criteria'])} of {len(r['criteria'])} pass"
        + (" (all pass)." if r["all_criteria_pass"] else "."),
        f"- Trials logged so far: {r['trial_count']}; deflated Sharpe of the decile spread "
        f"{_fmt(r['deflated_sharpe']['dsr'], '.3f')}.", "",
        "## Pre-registered success criteria (plan section 0)", "",
        "| # | criterion | status | detail |", "|---|---|---|---|",
    ]
    for c in r["criteria"]:
        out.append(f"| {c['n']} | {c['title']} | **{c['status'].upper()}** | {c['detail']} |")
    ev = r["evaluation"]
    out += ["", "## Point forecast", "",
            f"Ensemble: mean rank IC {_fmt(ev['ic']['mean_ic'])}, NW t {_fmt(ev['ic']['nw_tstat'], '.2f')}, "
            f"spread positive in {_fmt(ev['pct_positive_years'], '.0%')} of years, monotonicity "
            f"{_fmt(ev['monotonicity'], '.2f')}, {ev['n_obs']} stock-months.", "",
            "Baselines (same folds):", "", _md_table(r["baselines"]), ""]
    lg = ex.get("lgbm") or {}
    if lg.get("objective_table") is not None:
        out += ["LightGBM objective comparison (single seed, mean of per-fold IC):", "",
                _md_table(lg["objective_table"].drop("params")), ""]
    if lg.get("grid_table") is not None:
        out += [f"Tuning grid: {lg['grid_table'].height} trials.", ""]
    w = ex.get("ensemble_weights") or {}
    if w:
        rows = [{"fold": f, **{k: v for k, v in d["weights"].items()}} for f, d in w.items()]
        out += ["Ensemble weights per fold (after shrinkage):", "", _md_table(pl.DataFrame(rows), {"fold": "s"}), ""]
    out += [f"Importance stability across folds (mean pairwise Spearman): "
            f"{_fmt(r.get('importance_stability'), '.2f')}.", ""]

    out += ["## P(beats market) calibration (rows with a fitted isotonic map)", "",
            f"Largest gap in bins with >= the configured minimum rows: {_fmt(r['prob_calibration_gap'], '.3f')}.", "",
            _md_table(r["prob_calibration"]) if r["prob_calibration"].height else "(no calibrated rows)", ""]
    if r.get("prob_calibration_all_rows") is not None:
        out += [f"All rows (uncalibrated folds forecast 0.5): largest gap "
                f"{_fmt(r.get('prob_calibration_gap_all_rows'), '.3f')}.", ""]
    out += ["## Forecast range", "",
            f"Coverage of the conformalized 10-90% range: {_fmt(r['coverage'], '.1%')} "
            f"(on {'dates with a fitted conformal offset' if r.get('coverage_rows') == 'conformal_ok' else 'all rows'}; "
            f"all rows {_fmt(r.get('coverage_all_rows', r['coverage']), '.1%')}; "
            f"raw quantiles {_fmt(ex.get('raw_coverage'), '.1%')}). Width vs sigma_hat Spearman per date: "
            f"{_fmt(ex.get('width_vol_corr'), '.2f')}.", ""]
    if ex.get("vol_r2") is not None:
        out += ["Volatility model, out-of-sample R2 (log space) per fold:", "", _md_table(ex["vol_r2"], {"fold": "s", "n": "d"}), ""]
    x = r["exits"]
    out += ["## TP/SL levels", "",
            f"Chosen pair k_tp={x['k_tp']}, k_sl={x['k_sl']}; verdict **{x['verdict']}** "
            + ("(levels are exit orders)." if x["verdict"] == "exit_orders"
               else "(levels are shown as risk reference levels, not exit orders)."),
            f"Plain hold: mean net Sharpe {_fmt(x['hold']['sharpe'], '.2f')}, mean max drawdown "
            f"{_fmt(x['hold']['max_dd'], '.1%')}. {x['n_trials']} configurations counted.", "",
            _md_table(x["summary"].sort("mean_sharpe", descending=True, nulls_last=True).head(10)), "",
            "Exit variants (reported, not selectable):", "",
            _md_table(x["variants"].group_by("variant", "param", maintain_order=True)
                      .agg(pl.col("sharpe").mean().alias("mean_sharpe"), pl.col("max_dd").mean().alias("mean_max_dd")),
                      {"variant": "s"}), "",
            f"Hit probabilities: method **{ex.get('hit_method')}** (lowest multiclass Brier on OOF rows); "
            + _hit_calibration_line(r), ""]
    if ex.get("hit_brier") is not None and ex["hit_brier"].height:
        out += [_md_table(ex["hit_brier"], {"method": "s", "n": "d"}), ""]
    d = r["deflated_sharpe"]
    out += ["## Multiple testing", "",
            f"Trials in the experiment log: {d['n_trials']}. Monthly spread Sharpe {_fmt(d['spread_sharpe_monthly'], '.3f')} "
            f"over {d['n_months']} months of overlapping {d['horizon_months']}-month returns, i.e. "
            f"{_fmt(d['n_eff'], '.1f')} independent observations (n_eff = months / horizon). Deflated Sharpe "
            f"{_fmt(d['dsr'], '.3f')} on n_eff (variance of trial Sharpes: {d['var_sr_source']}).", ""]
    return "\n".join(out)


# fit_final / load_final / score --------------------------------------------------------------------


@dataclass
class FinalModels:
    ensemble: Any
    vol: Any
    quantile: Any
    isotonic: Any | None
    conformal_offset: float
    conformal_ok: bool
    k_tp: float | None
    k_sl: float | None
    verdict: str
    barriers: BarrierRules
    hit_table: Any
    hit_classifier: Any
    method: str
    market_return: float
    trained_through: dt.date
    model_columns: list[str]
    meta: dict = field(default_factory=dict)
    min_cross_section: int = 1   # score refuses a date with fewer rows (a partial universe)


FINAL_FILES = ("ensemble", "vol", "quantile", "isotonic", "hit_table", "hit_classifier")


def _fit_meta(frame: pl.DataFrame, target: str, extra: dict | None = None) -> dict:
    rows = frame.filter(pl.col(target).is_not_null())
    return {"fit_rows": rows.height, "fit_date_min": rows["date"].min(), "fit_date_max": rows["date"].max(),
            "target": target, **(extra or {})}


def min_cross_section(frame: pl.DataFrame, mcfg: ModelConfig) -> int:
    """Fewest rows a scored date may hold: half the smallest dev cross-section (room for universe drift),
    at least ensemble.ic_min_n. Ranks, prob and drivers are relative to the rows of a date, so a single
    ticker, a watchlist or a sector must not be scored on its own."""
    smallest = int(frame.group_by("date").len()["len"].min())
    return max(mcfg.ensemble.ic_min_n, smallest // 2)


def fit_final(store: Store, cfg: Config, mcfg: ModelConfig, catalog: Catalog, *, through: dt.date | None = None
              ) -> Path:
    """Refit everything on all dev rows dated <= `through` (default dev_end) -> models/final/.
    Needs the train outputs (winner params, chosen pair and method, OOF predictions)."""
    from ltforecast.calibrate import fit_isotonic, latest_conformal_offset
    from ltforecast.hitprob import HitClassifier
    from ltforecast.levels import barrier_labels
    from ltforecast.models.io import save_model
    from ltforecast.models.lgbm import LGBMParams
    from ltforecast.models.quantile import QuantileModel
    from ltforecast.models.vol import LVOL_COLS, VolModel, vol_frame

    odir = oof_dir(store)
    state_path = odir / "train_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"{state_path} missing: run `ltforecast train` first")
    state = json.loads(state_path.read_text())
    check_train_state(state, mcfg, cfg.dates.dev_end if through is None else through)
    inp = load_inputs(store, cfg, mcfg, catalog, through=through)
    through = inp.through
    fp, stored = data_fingerprint(inp.frame), state.get("data_fingerprint") or {}
    changed = [d for d, v in fp.items() if stored.get(d) != v]
    if changed:
        raise ValueError(f"development data changed since train on {len(changed)} dates (first {changed[0]}): "
                         "re-run `ltforecast train`")
    rules, cols, cal = mcfg.validation, catalog.model_columns(), mcfg.calibration
    barriers = cfg.labels.barriers
    horizon = cfg.labels.horizons[barriers.horizon]
    labels = inp.labels
    k_tp, k_sl = state["k_tp"], state["k_sl"]
    out = final_dir(store)
    out.mkdir(parents=True, exist_ok=True)
    common = {"trained_through": through, "columns": cols}

    winner = LGBMParams(**state["lgbm_params"])
    ens = fit_on_all(ensemble_factory(catalog, mcfg, winner), inp.frame, rules)
    save_model(ens, out / "ensemble.pkl", common | _fit_meta(inp.frame, "target", {
        "params": winner, "seeds": mcfg.lgbm.seeds, "weights": ens.weights_}))

    vf = vol_frame(inp.features_raw, inp.frame, inp.prices, inp.securities)
    vol = fit_on_all(functools.partial(VolModel, mcfg.vol), vf, rules, target=VOL_TARGET)
    save_model(vol, out / "vol.pkl", common | _fit_meta(vf, VOL_TARGET, {"params": mcfg.vol}))

    qframe = inp.frame.join(vf.select(*KEYS, *LVOL_COLS), on=list(KEYS), how="left")
    q = mcfg.quantile
    qm = fit_on_all(functools.partial(QuantileModel, cols, q.alphas, q.params, q.seed), qframe, rules, target=Q_TARGET)
    save_model(qm, out / "quantile.pkl", common | _fit_meta(qframe, Q_TARGET, {"params": q.params, "alphas": q.alphas}))

    # calibration maps from the walk-forward OOF rows dated <= through (all of their labels have ended)
    end = dt.date(9999, 12, 31)
    forecast = pl.read_parquet(odir / "forecast.parquet").filter(pl.col("date") <= through)
    iso = fit_isotonic(forecast, labels, before=end, min_rows=cal.min_rows, min_dates=cal.min_dates,
                       outcome=cal.outcome)
    save_model(iso, out / "isotonic.pkl", common | {"calibrated": iso is not None, "fit_rows": 0 if iso is None else iso.n_obs,
                                                     "fit_date_max": forecast["date"].max()})
    q_raw = pl.read_parquet(odir / "quantile_raw.parquet").filter(pl.col("date") <= through)
    offset, ok = latest_conformal_offset(q_raw, labels, before=end, alpha=cal.alpha,
                                         window_months=cal.window_months, min_rows=cal.min_rows,
                                         min_dates=cal.min_dates, outcome=cal.outcome)

    # hit probabilities: the table from residuals <= through; the classifier on the barrier labels of
    # the chosen pair, levels from the final vol model's sigma_hat on every dev row
    tsig = month_end_sigma(inp.prices, inp.frame, barriers)
    table = build_hit_table(inp.prices, tsig, through, mcfg, horizon)
    save_model(table, out / "hit_table.pkl", common | {"residuals_through": through, "n_paths": mcfg.levels.mc_paths})
    sig = vf.select(*KEYS).with_columns(pl.Series("sigma_hat", vol.predict(vf), dtype=pl.Float64))
    chosen = dataclasses.replace(barriers, k_tp=(k_tp,), k_sl=(k_sl,))
    bar = barrier_labels(inp.prices, sig.select("security_id", "date", pl.col("sigma_hat").alias("sigma")),
                         inp.delist, inp.calendar, horizon, chosen)
    hframe = _join(_join(inp.frame, sig, ["sigma_hat"]), bar, ["outcome"])
    hc = mcfg.hit_classifier
    clf = fit_on_all(functools.partial(HitClassifier, cols, hc.params, hc.seed), hframe, rules, target="outcome")
    save_model(clf, out / "hit_classifier.pkl", common | _fit_meta(hframe, "outcome", {"params": hc.params}))

    prov = provenance(store)
    meta = {
        "git_commit": git_state()[0],
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "trained_through": through,
        "dev_end": cfg.dates.dev_end,
        "feature_columns": cols,
        "model_config_hash": config_hash(mcfg.raw),
        "effective_config_hash": effective_config_hash(mcfg),
        "fast": mcfg.fast,
        "effective": {"lgbm_params": winner, "seeds": mcfg.lgbm.seeds, "vol": mcfg.vol, "quantile": mcfg.quantile,
                      "hit_classifier": mcfg.hit_classifier, "mc_paths": mcfg.levels.mc_paths},
        "min_cross_section": min_cross_section(inp.frame, mcfg),
        "trial_count": trial_count(mcfg.experiments_dir),
        "synthetic": prov.get("synthetic"),
        "lgbm_params": winner,
        "k_tp": k_tp, "k_sl": k_sl, "verdict": state["verdict"],
        "barriers": {"tp_bounds": barriers.tp_bounds, "sl_bounds": barriers.sl_bounds, "horizon": barriers.horizon},
        "hit_method": state["hit_method"],
        "market_return": mcfg.levels.market_return,
        "conformal_offset": offset, "conformal_ok": ok,
        "recommendation": state.get("recommendation"),
        "fit_date_max": max(inp.frame["date"].max(), forecast["date"].max()),
    }
    _write_json(out / "metadata.json", meta)
    return out


def load_final(path: Path | str) -> FinalModels:
    """The models/final/ directory written by fit_final (pickles verified against their sidecars)."""
    from ltforecast.config import BarrierRules
    from ltforecast.models.io import load_model

    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    objs = {name: load_model(path / f"{name}.pkl")[0] for name in FINAL_FILES}
    b = meta["barriers"]
    barriers = BarrierRules(
        horizon=b["horizon"], k_tp=(meta["k_tp"],), k_sl=(meta["k_sl"],), tp_bounds=tuple(b["tp_bounds"]),
        sl_bounds=tuple(b["sl_bounds"]), sigma_window=0, sigma_min_obs=0,
    )
    return FinalModels(
        ensemble=objs["ensemble"], vol=objs["vol"], quantile=objs["quantile"], isotonic=objs["isotonic"],
        conformal_offset=float(meta["conformal_offset"]), conformal_ok=bool(meta["conformal_ok"]),
        k_tp=meta["k_tp"], k_sl=meta["k_sl"], verdict=meta["verdict"], barriers=barriers,
        hit_table=objs["hit_table"], hit_classifier=objs["hit_classifier"], method=meta["hit_method"],
        market_return=float(meta["market_return"]), trained_through=dt.date.fromisoformat(meta["trained_through"]),
        model_columns=list(meta["feature_columns"]), meta=meta,
        min_cross_section=int(meta.get("min_cross_section", 1)),
    )


def score(
    models: FinalModels,
    features: pl.DataFrame,
    features_raw: pl.DataFrame,
    prices: pl.DataFrame | pl.LazyFrame,
    securities: pl.DataFrame,
    *,
    dates: Sequence[dt.date] | None = None,
    security_ids: Sequence[int] | None = None,
) -> pl.DataFrame:
    """SCORE_COLS for every (date, security_id) of `features` (optionally only `dates` and only
    `security_ids`).

    `features` must hold the COMPLETE universe cross-section of every scored date: pred, rank_pct and
    prob are ranks within a date. They are computed on all rows of the date, and only then is the output
    cut to `security_ids` (a ticker report passes the full month and `security_ids=[id]`). A date with
    fewer than `models.min_cross_section` rows raises instead of returning ranks against a partial set.

    Reads only inputs known at each prediction date: model columns of `features`, beta_60m / log_mcap
    of `features_raw`, adj_close of `prices` dated <= the date (trailing vol), and sectors. Label
    columns are never read (a `features` frame that carries them is cut to the model columns first).
    Scoring dates <= trained_through is allowed (reports); each row records trained_through.
    """
    from ltforecast.hitprob import combine
    from ltforecast.models.vol import LVOL_COLS, RAW_COLS, vol_inputs

    cols = models.model_columns
    x = features.select(*KEYS, *cols)
    if dates is not None:
        x = x.filter(pl.col("date").is_in(list(dates)))
    x = x.sort(*KEYS)
    empty = pl.DataFrame(schema={c: pl.Float64 for c in SCORE_COLS} | {
        "date": pl.Date, "security_id": pl.Int64, "prob_calibrated": pl.Boolean, "trained_through": pl.Date})
    if x.is_empty():
        return empty
    sizes = x.group_by("date").len().filter(pl.col("len") < models.min_cross_section).sort("date")
    if sizes.height:
        d, n = sizes.row(0)
        raise ValueError(
            f"{sizes.height} scored dates hold a partial cross-section (first {d}: {n} rows < "
            f"{models.min_cross_section}): pass the full universe of each date and cut with security_ids="
        )

    # cross-sectional outputs on every row of each date
    pred = models.ensemble.predict(x)
    out = x.select(*KEYS).with_columns(pl.Series("pred", pred, dtype=pl.Float64)) \
        .with_columns(rank_pct("pred").alias("rank_pct"))
    if models.isotonic is not None:
        prob, ok = models.isotonic.predict(out["rank_pct"].to_numpy()), True
    else:
        prob, ok = np.full(out.height, 0.5), False
    out = out.with_columns(pl.Series("prob", prob, dtype=pl.Float64), pl.lit(ok).alias("prob_calibrated"))
    if security_ids is not None:
        keep = pl.col("security_id").is_in(list(security_ids))
        x, out = x.filter(keep), out.filter(keep)
        if x.is_empty():
            return empty
    keys = x.select(*KEYS)

    sectors = securities.select("security_id", "sector").unique("security_id", keep="first", maintain_order=True)
    vin = (
        keys.join(features_raw.select(*KEYS, *[pl.col(c).cast(pl.Float64) for c in RAW_COLS]), on=list(KEYS), how="left")
        .join(vol_inputs(prices, keys), on=list(KEYS), how="left")
        .join(sectors, on="security_id", how="left")
    )
    sigma = models.vol.predict(vin)
    out = out.with_columns(pl.Series("sigma_hat", sigma, dtype=pl.Float64))

    qx = x.join(vin.select(*KEYS, *LVOL_COLS), on=list(KEYS), how="left")
    qr = models.quantile.predict(qx)
    off = models.conformal_offset
    q10, q50, q90 = np.minimum(qr[:, 0] - off, qr[:, 1]), qr[:, 1], np.maximum(qr[:, 2] + off, qr[:, 1])
    out = out.with_columns(pl.Series("q10", q10), pl.Series("q50", q50), pl.Series("q90", q90))
    out = _level_cols(out, models.k_tp, models.k_sl, models.barriers)

    tp, sl = level_ratios(sigma, models.k_tp, models.k_sl, models.barriers)
    mc = mc_probs(models.hit_table, q50, sigma, tp, sl, models.market_return)
    clf = models.hit_classifier.predict_proba(x.with_columns(pl.Series("sigma_hat", sigma, dtype=pl.Float64)))
    p = combine(mc, clf, models.method)
    out = out.with_columns(*(pl.Series(c, p[:, j], dtype=pl.Float64) for j, c in enumerate(PROB_COLS)))
    return out.with_columns(pl.lit(models.trained_through, dtype=pl.Date).alias("trained_through")).select(SCORE_COLS)

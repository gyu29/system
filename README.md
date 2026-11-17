## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
export NASDAQ_DATA_LINK_API_KEY=...   # Sharadar Core US Equities bundle
export FRED_API_KEY=...               # free, https://fred.stlouisfed.org/docs/api/api_key.html
.venv/bin/ltforecast ingest sharadar  # or --from-dir ~/Downloads for hand-downloaded SHARADAR_*.zip
.venv/bin/ltforecast ingest fred
.venv/bin/ltforecast build            # data/core/<table>/year=YYYY/*.parquet
.venv/bin/ltforecast check            # exits 1 on any FAIL; writes data/reports/{coverage,spot_check}.csv
.venv/bin/ltforecast labels           # data/core/{labels,benchmark_daily,barrier_labels}
.venv/bin/ltforecast features         # data/core/{features_raw,features}; data/reports/feature_{ic,shift_test}.csv
.venv/bin/ltforecast baselines        # data/models/oof/<baseline>.parquet; data/reports/baselines.csv
.venv/bin/ltforecast train            # data/models/oof/forecast.parquet; data/reports/train_report.{md,json}
.venv/bin/ltforecast fit-final        # data/models/final/ (every model refit on all development rows)
.venv/bin/pytest
```

Without vendor keys, `.venv/bin/ltforecast ingest synthetic [--n-stocks 300 --start 2000-01-03 --end 2025-12-31 --seed 0]`
replaces the two downloads with simulated vendor-shaped data that carries a modest planted signal
(parameters are recorded in `data/raw/SYNTHETIC.json`); every later command runs unchanged.
Real and synthetic raw data never mix: `ingest synthetic` refuses to overwrite vendor raw files unless
given `--force`, and `ingest sharadar` / `ingest fred` refuse to write over synthetic raw data unless
given `--replace-synthetic` (which deletes all synthetic raw files first). `build` records where the
core tables came from in `data/core/PROVENANCE.json` (`{"synthetic": true, ...}` plus a printed
SYNTHETIC DATA warning, or `{"synthetic": false}`).

Config: `config/universe.yaml` (universe filters, holdout dates, PIT, delisting and label rules),
`config/features.yaml` (the feature catalog), `config/model.yaml` (validation rules, the
pre-registered success criteria and the Phase 5-6 model settings). `baselines`, `train` and `fit-final`
read the `model.yaml` next to `--config` (or `--model-config`); `--fast` applies the file's `fast`
block (fewer seeds, no tuning grid, fewer Monte Carlo paths) and never changes folds, purge or criteria.

## Labels (`labels.py`, `levels.py`)

One row per in-universe (stock, month end); horizons 3m/6m/12m = 63/126/252 trading days.

- `fwd_ret_h`: total return from `closeadj`. A delisting inside the window earns the assumed delisting
  return on the next trading day; the proceeds then sit in the benchmark until the window ends.
- `bench_ret_h`: a daily cap-weighted index of the universe. Members are the stocks in the universe at
  the previous month end, weighted by their previous-day market cap (`benchmark_daily`).
- `excess_h = log(1+fwd) − log(1+bench)`. `excess_w_h` is winsorized 1/99% per date.
  `rank_h` is the per-date percentile; `target = rank_12m` is the training label.
- `label_end_h` is the last day of each window, which Phase 4 uses to purge overlapping training rows.
  `period` marks each row dev / gap / holdout / live.
- Null labels: a window that runs past the data, or a stock whose prices stop without a delisting.
- `barrier_labels`: 25 TP/SL pairs per entry. Closes are checked daily. A gap through a level fills at
  the open; otherwise the fill is at the level. **σ is trailing 12m realized vol for now**, and becomes
  the forecast vol in Phase 6, when these labels get rebuilt.

## Features (`features/`)

One row per in-universe (stock, month end). `config/features.yaml` is the catalog: 36 features in ten
families, each with its definition, source fields, lookback, availability lag and citation.

- `features_raw`: the unranked values plus one `src_date_<family>` per family, the latest date of any
  input behind that row. The build fails if a source date is after the prediction date.
- `features`: what the models read. Stock features are ranked across stocks on each date into
  (−0.5, 0.5); value and quality also get a within-sector rank (`<name>_sr`). Missing stays null, with a
  `has_<family>` flag; linear models impute 0, the median rank. Ranks are never sign-flipped: the
  expected sign is in the catalog. The four context features are date-level and left unranked.
- Fundamentals are the latest fiscal period whose filing was usable at t (filing date + 1 trading
  day), as the market saw it then, and go null once the filing is older than 550 days. Price features
  use data through the month-end close only. Share counts and EPS are put on one split basis.
- `ltforecast features` also writes two development-period reports: the univariate rank IC of every
  feature (`feature_ic.csv`) and the shift test (`feature_shift_test.csv`), which flags a feature whose
  IC clearly improves when its value is made one month staler, in whichever direction the feature
  predicts (`orientation`, from the data; `gain` and `gain_tstat` are the oriented improvement and its
  Newey-West t-stat), so a feature and its negation get the same verdict. `leakage.shuffle_labels` permutes the
  labels within each date for the random-label test.

## Validation (`dataset.py`, `validation/`)

- `dataset.modeling_frame` joins features to labels and returns development rows only. Gap and holdout
  rows need a `HoldoutToken`, which nothing before the Phase 9 evaluation can issue.
- `splitter.walk_forward_folds`: expanding window, one test year per fold from 2010, with 12 purged
  months plus a 1-month embargo between the last training date and the test year. Inner validation is
  the last 24 training months, purged the same way. `assert_purged` checks every training label ends
  before the first test date.
- `runner.walk_forward` fits a fresh model per fold and scores the test rows from a frame with every
  label column removed.
- `metrics.py`: monthly rank IC with a Newey-West t-stat (12 lags), decile table and top-minus-bottom
  spread by year, calibration, interval coverage, size and regime breakdowns, deflated Sharpe, and
  `check_criteria` against the thresholds in `config/model.yaml`. Those thresholds are pre-registered
  and are never tuned.
- `experiments.log_run` appends every run to `experiments/runs.csv` (config hash, git commit, metrics,
  trial count); the trial count feeds the deflated Sharpe.

## Baselines (`models/baselines.py`)

- B0 predicts 0 for everyone (IC exactly 0); B1 is one signed factor (momentum 12-1, gross
  profitability, earnings yield); B2 averages the signed ranks of five themes (value, quality,
  momentum, low vol, low investment) with nothing fitted; B3 is a ridge on every ranked feature with
  alpha chosen by inner-validation IC.
- `ltforecast baselines` walks all six through the same purged folds as the main model. B2 is the bar:
  if the main model cannot beat it, B2 ships.

## Main model and TP/SL levels (`pipeline.py`, `models/`, `calibrate.py`, `exits.py`, `hitprob.py`, `explain.py`)

`ltforecast train` runs every step on development rows only, on the same walk-forward folds:

- Point forecast: LightGBM regression vs lambdarank at the base settings (optional 16-point grid), then
  an ensemble of 5 LightGBM seeds + B3 + B2, weighted on inner validation and shrunk halfway to equal
  weights. `rank_pct` is the per-date percentile.
- `prob` = P(12m excess > 0) from an isotonic map fit on earlier folds whose labels had ended.
- `sigma_hat`: a HAR-style ridge of log 12m realized vol. `q10/q50/q90`: LightGBM quantiles plus a
  conformal offset from the last 36 matured months, so the 10-90% range targets 80% coverage.
- TP/SL: the top quintile is simulated each month with 20 bps costs and gap fills; one global
  (k_tp, k_sl) pair is chosen by mean net Sharpe across folds subject to drawdown no worse than holding.
  If it does not beat holding, the levels are reported as risk reference levels. Trailing-stop and
  signal exits are reported alongside. The barrier labels are rebuilt with sigma = `sigma_hat`.
- `p_tp/p_sl/p_none`: a Monte Carlo table (bootstrapped residuals per volatility bucket, drift from
  q50 + an assumed 6% market return) and a multiclass LightGBM, whichever has the lower OOF Brier score
  (or their average).
- `data/reports/train_report.md` checks all six pre-registered criteria, says whether the ensemble
  beats B2 (paired Newey-West IC test) and what to ship, and gives the deflated Sharpe with the trial
  count from `experiments/runs.csv`. A re-run of the same config on the same code logs 0 trials.
  Results on synthetic data are stamped SYNTHETIC DATA.

`ltforecast fit-final` refits everything on all development rows into `data/models/final/` (pickles with
JSON sidecars, `metadata.json` with git commit, `trained_through`, config hash, trial count). In code,
`pipeline.load_final` + `pipeline.score` produce the forecast columns for any prediction dates without
reading labels; each row records `trained_through`.

## Point-in-time rules enforced in code

- Fundamentals: SF1 `AR*` dimensions only, first filing per period wins, usable from
  `datekey` + 1 trading day (`pit/asof.py`; `asof_join` asserts no row is used before it was public).
- Universe: unadjusted price > $5, market cap > $300M in `cpi_base_date` dollars deflated by the
  CPI *as released* on each date, 63-day ADV > $1M, Sharadar common-stock categories. Delisted names included.
- Macro: CPI from ALFRED first releases; observations older than ALFRED's vintage history fall back to
  observation date + `release_lag_days` and are flagged `first_release = false`.

## Vendor assumptions to verify on the first real ingest

These are encoded in code/config but were only tested against synthetic data:

1. `DAILY.marketcap` is in **millions** (`universe.marketcap_unit`) and is computed from filings known on that date.
2. `SEP.volume` is split-adjusted like `SEP.close`, so `close × volume` is true dollar volume.
   (`check` → `split_consistency` verifies the price side.)
3. `ACTIONS` uses the same tickers as `TICKERS(table=SEP)`, with delisting codes
   `bankruptcyliquidation`, `regulatorydelisting`, `voluntarydelisting`, `acquisitionby`, `mergerfrom`, `delisted`.
4. Sector/industry come from `TICKERS`, which is current-only — not historical as the plan asks.
5. SF1 AR* `sharesbas`, `shareswa`, `eps`, `epsdil` and `dps` are as first reported: on the share basis
   of every split with ex-date before the filing date (`datekey`), and not restated for later splits.
   `features.investment.split_adjusted_context` (also used by the Piotroski score in `features/quality.py`)
   and `features.earnings.split_adjusted_eps_context` depend on this. SF1 `sharefactor` is not a
   split-basis marker. (`check` → `share_basis` verifies it: a raw share ratio of about 1 across a split
   means the counts are restated, and the split division in those helpers must be removed.)

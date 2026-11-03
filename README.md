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
.venv/bin/pytest
```

Config (universe filters, holdout dates, PIT, delisting and label rules): `config/universe.yaml`.

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

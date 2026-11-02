## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
export NASDAQ_DATA_LINK_API_KEY=...   # Sharadar Core US Equities bundle
export FRED_API_KEY=...               # free, https://fred.stlouisfed.org/docs/api/api_key.html
.venv/bin/ltforecast ingest sharadar  # or --from-dir ~/Downloads for hand-downloaded SHARADAR_*.zip
.venv/bin/ltforecast ingest fred
.venv/bin/ltforecast build            # data/core/<table>/year=YYYY/*.parquet
.venv/bin/ltforecast check            # exits 1 on any FAIL; writes data/reports/{coverage,spot_check}.csv
.venv/bin/pytest
```

Config (universe filters, holdout dates, PIT and delisting rules): `config/universe.yaml`.

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

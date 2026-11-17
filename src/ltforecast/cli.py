"""ltforecast command line: ingest -> build -> check -> labels -> features -> baselines -> train -> fit-final."""

from __future__ import annotations

from pathlib import Path

import json

import polars as pl
import typer

from ltforecast.config import DEFAULT_CONFIG, load_config
from ltforecast.ingest import fred, sharadar, synthetic
from ltforecast.pit import tables
from ltforecast.quality import universe_counts
from ltforecast.store import Store
from ltforecast.universe import build_universe

app = typer.Typer(no_args_is_help=True, help="Long-horizon stock return forecasting")
ingest_app = typer.Typer(no_args_is_help=True, help="Download vendor data into data/raw")
app.add_typer(ingest_app, name="ingest")

ConfigOpt = typer.Option(DEFAULT_CONFIG, "--config", help="Path to universe.yaml")
ModelConfigOpt = typer.Option(
    None, "--model-config", help="Path to model.yaml (default: model.yaml next to --config, else config/model.yaml)"
)
FastOpt = typer.Option(False, "--fast", help="Small run: fewer seeds, no tuning grid, fewer Monte Carlo paths")
ReplaceSyntheticOpt = typer.Option(
    False, "--replace-synthetic", help="Delete the synthetic raw data in data/raw before writing vendor data"
)


def _guard_vendor_write(store: Store, replace_synthetic: bool) -> None:
    """Refuse a vendor write over synthetic raw data (a mixed raw set), unless asked to replace it."""
    try:
        synthetic.guard_vendor_write(store, replace_synthetic)
    except RuntimeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1) from e


@ingest_app.command("sharadar")
def ingest_sharadar(
    table: list[str] = typer.Option(list(sharadar.TABLES), "--table", help="Tables to download"),
    from_dir: Path | None = typer.Option(None, help="Load hand-downloaded SHARADAR_<TABLE>*.zip/csv files instead"),
    replace_synthetic: bool = ReplaceSyntheticOpt,
    config: Path = ConfigOpt,
) -> None:
    store = Store(load_config(config).data_dir)
    _guard_vendor_write(store, replace_synthetic)
    for t in table:
        if from_dir:
            matches = sorted(from_dir.glob(f"SHARADAR_{t}*"))
            if not matches:
                raise typer.BadParameter(f"no SHARADAR_{t}* file in {from_dir}")
            df = sharadar.load_export(matches[-1], t)
        else:
            typer.echo(f"requesting {t} export ...")
            df = sharadar.download_table(t)
        path = store.write_raw("sharadar", t, df)
        typer.echo(f"{t}: {df.height:,} rows -> {path}")


@ingest_app.command("fred")
def ingest_fred(replace_synthetic: bool = ReplaceSyntheticOpt, config: Path = ConfigOpt) -> None:
    cfg = load_config(config)
    store = Store(cfg.data_dir)
    _guard_vendor_write(store, replace_synthetic)
    # Two years before history_start, so trailing macro features exist from day one.
    start = cfg.dates.history_start.replace(year=cfg.dates.history_start.year - 2)
    frames = []
    for s in cfg.macro:
        df = fred.fetch_series(s, start)
        no_vintage = df.filter(~pl.col("first_release")).height
        typer.echo(f"{s.id}: {df.height:,} obs ({no_vintage} without a first-release vintage)")
        frames.append(df)
    store.write_raw("fred", "OBSERVATIONS", tables.build_macro(frames))


@ingest_app.command("synthetic")
def ingest_synthetic(
    n_stocks: int = typer.Option(300, help="Number of simulated securities"),
    start: str = typer.Option("2000-01-03", help="First price date (YYYY-MM-DD)"),
    end: str = typer.Option("2025-12-31", help="Last price date (YYYY-MM-DD)"),
    seed: int = typer.Option(0, help="Random seed; the data is deterministic given the seed"),
    force: bool = typer.Option(False, "--force", help="Overwrite vendor raw data already in data/raw"),
    config: Path = ConfigOpt,
) -> None:
    """Write simulated vendor-shaped data (with a planted signal) into data/raw, in place of the downloads."""
    import datetime as dt

    store = Store(load_config(config).data_dir)
    frames = synthetic.generate(n_stocks, dt.date.fromisoformat(start), dt.date.fromisoformat(end), seed)
    params = {"n_stocks": n_stocks, "start": start, "end": end, "seed": seed}
    try:
        marker = synthetic.write_raw(store, frames, params, force=force)
    except FileExistsError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1) from e
    for name, df in frames.items():
        typer.echo(f"{name}: {df.height:,} rows")
    typer.echo(f"synthetic data, not vendor data: parameters in {marker}")


@app.command()
def build(config: Path = ConfigOpt) -> None:
    """Build core point-in-time tables and the monthly universe from data/raw."""
    cfg = load_config(config)
    store = Store(cfg.data_dir)

    actions = store.scan_raw("sharadar", "ACTIONS").collect()
    securities = tables.build_securities(store.scan_raw("sharadar", "TICKERS").collect(), actions, cfg.delisting)
    store.write("securities", securities)
    typer.echo(f"securities: {securities.height:,} ({securities['delisted_date'].is_not_null().sum():,} delisted)")

    prices = tables.build_prices(store.scan_raw("sharadar", "SEP"), actions, securities)
    store.write("prices_daily", prices)
    typer.echo(f"prices_daily: {prices.height:,}")

    fund = tables.build_fundamentals(store.scan_raw("sharadar", "SF1"), securities, cfg.pit.sf1_dimensions)
    store.write("fundamentals_pit", fund)
    typer.echo(f"fundamentals_pit: {fund.height:,}")

    mcap = tables.build_marketcap(store.scan_raw("sharadar", "DAILY"), securities, cfg.universe.marketcap_unit)
    store.write("marketcap_daily", mcap)

    macro = store.scan_raw("fred", "OBSERVATIONS").collect()
    store.write("macro_pit", macro)

    universe = build_universe(prices, securities, mcap, macro, cfg.universe, cfg.dates.history_start)
    store.write("universe", universe)
    n = universe_counts(universe)["n"]
    typer.echo(f"universe: {n.len()} months, median {n.median():.0f} stocks/month")

    # provenance of the core tables: later phases (holdout, report, production) read it
    provenance = {"synthetic": False}
    if synthetic.raw_is_synthetic(store):
        provenance = {"synthetic": True, **json.loads(synthetic.marker_path(store).read_text())}
        typer.echo("WARNING: SYNTHETIC DATA. The core tables are built from simulated data, not vendor data.")
    path = store.root / "core" / "PROVENANCE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2) + "\n")


@app.command()
def labels(config: Path = ConfigOpt) -> None:
    """Build forward-return and triple-barrier labels from the core tables (plan section 2)."""
    from ltforecast.labels import build_labels, delisting_returns
    from ltforecast.levels import barrier_labels, trailing_sigma

    cfg = load_config(config)
    store = Store(cfg.data_dir)
    rules = cfg.labels
    prices = store.scan("prices_daily").collect()
    securities = store.scan("securities").collect()

    lab, bench = build_labels(
        prices,
        store.scan("universe").collect(),
        store.scan("marketcap_daily").collect(),
        securities,
        rules,
        cfg.delisting,
        cfg.dates,
    )
    store.write("benchmark_daily", bench)
    store.write("labels", lab)
    for name in rules.horizons:
        n = lab[f"fwd_ret_{name}"].is_not_null().sum()
        typer.echo(f"labels {name}: {n:,} of {lab.height:,} rows")

    b = rules.barriers
    calendar = bench["date"]
    entries = (
        lab.filter(pl.col(f"fwd_ret_{b.horizon}").is_not_null())
        .select("security_id", "date")
        .join(trailing_sigma(prices, b.sigma_window, b.sigma_min_obs), on=["security_id", "date"], how="left")
    )
    bar = barrier_labels(
        prices,
        entries,
        delisting_returns(securities, cfg.delisting, calendar),
        calendar,
        rules.horizons[b.horizon],
        b,
    )
    store.write("barrier_labels", bar)
    typer.echo(f"barrier_labels: {bar.height:,} rows ({len(b.k_tp) * len(b.k_sl)} TP/SL pairs)")


@app.command()
def features(config: Path = ConfigOpt) -> None:
    """Build features_raw and features (plan section 3), then the univariate IC and shift-test reports."""
    from ltforecast.dataset import modeling_frame
    from ltforecast.features.build import SF1_METRICS, build_features
    from ltforecast.features.catalog import load_catalog
    from ltforecast.features.leakage import shift_test, univariate_ic
    from ltforecast.features.panel import make_context

    cfg = load_config(config)
    store = Store(cfg.data_dir)
    catalog = load_catalog()
    raw, feats = build_features(make_context(store, cfg, metrics=SF1_METRICS), catalog)
    store.write("features_raw", raw)
    store.write("features", feats)
    typer.echo(f"features: {feats.height:,} rows, {len(catalog.names())} features, {len(catalog.model_columns())} model columns")

    # Reports use development rows only: modeling_frame never reads gap or holdout labels.
    frame = modeling_frame(store, cfg, catalog, periods=("dev",))
    cols = catalog.linear_columns()
    signs = pl.DataFrame({"feature": catalog.names(), "sign": [catalog.spec(n).sign for n in catalog.names()]})
    ic = univariate_ic(frame, cols).join(signs, on="feature", how="left")
    shift = shift_test(frame, cols)
    reports = store.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ic.write_csv(reports / "feature_ic.csv")
    shift.write_csv(reports / "feature_shift_test.csv")
    flagged = shift.filter(pl.col("flagged"))["feature"].to_list()
    typer.echo(f"univariate IC and shift test on {frame.height:,} dev rows -> {reports}")
    typer.echo(f"shift test: {', '.join(flagged) + ' FLAGGED (IC improves when lagged a month)' if flagged else 'no feature flagged'}")


@app.command()
def check(config: Path = ConfigOpt) -> None:
    """Run data-quality tests. Exits non-zero if any check fails."""
    from ltforecast.quality import run_all

    cfg = load_config(config)
    store = Store(cfg.data_dir)
    results = run_all(store, cfg)
    for r in results:
        typer.echo(f"[{r.status.upper():4}] {r.name}: {r.detail}")
    typer.echo(f"coverage and spot-check reports in {store.root / 'reports'}")
    if any(r.status == "fail" for r in results):
        raise typer.Exit(1)


def _model_config(config: Path, model_config: Path | None, fast: bool):
    """model.yaml: --model-config, else the one next to universe.yaml, else the repo default."""
    from ltforecast.modelconfig import DEFAULT_MODEL_CONFIG, load_model_config

    if model_config is None:
        beside = Path(config).parent / "model.yaml"
        model_config = beside if beside.exists() else DEFAULT_MODEL_CONFIG
    return load_model_config(model_config, fast=fast)


def _synthetic_warning(store: Store) -> None:
    from ltforecast.pipeline import provenance

    if provenance(store).get("synthetic"):
        typer.echo("WARNING: SYNTHETIC DATA. Results say nothing about real markets.")


@app.command()
def baselines(fast: bool = FastOpt, config: Path = ConfigOpt, model_config: Path | None = ModelConfigOpt) -> None:
    """Walk the plan-5 baselines (B0-B3) forward; writes data/models/oof/<name>.parquet and
    data/reports/baselines.csv."""
    from ltforecast.features.catalog import load_catalog
    from ltforecast.pipeline import reports_dir, run_baselines

    cfg = load_config(config)
    store = Store(cfg.data_dir)
    table, _ = run_baselines(store, cfg, _model_config(config, model_config, fast), load_catalog())
    _synthetic_warning(store)
    for row in table.iter_rows(named=True):
        typer.echo(f"{row['model']:<26} mean IC {row['mean_ic']:+.4f}  NW t {row['nw_tstat']:+.2f}")
    typer.echo(f"baselines -> {reports_dir(store) / 'baselines.csv'}")


@app.command()
def train(fast: bool = FastOpt, config: Path = ConfigOpt, model_config: Path | None = ModelConfigOpt) -> None:
    """Walk-forward training of the plan-6 models on development rows; writes
    data/models/oof/forecast.parquet and data/reports/train_report.{json,md}."""
    from ltforecast.features.catalog import load_catalog
    from ltforecast.pipeline import train as run_train

    cfg = load_config(config)
    store = Store(cfg.data_dir)
    result = run_train(store, cfg, _model_config(config, model_config, fast), load_catalog())
    _synthetic_warning(store)
    for c in result.report["criteria"]:
        typer.echo(f"[{c['status'].upper():4}] {c['n']}. {c['name']}: {c['detail']}")
    typer.echo(f"ensemble beats B2: {'yes' if result.report['beats_b2'] else 'no'}; "
               f"recommendation: {result.recommendation}; trials logged: {result.report['trial_count']}")
    typer.echo(f"forecast -> {result.paths['forecast']}; report -> {result.paths['report_md']}")


@app.command("fit-final")
def fit_final(fast: bool = FastOpt, config: Path = ConfigOpt, model_config: Path | None = ModelConfigOpt) -> None:
    """Refit every model on all development rows (after `train`); writes data/models/final/."""
    from ltforecast.features.catalog import load_catalog
    from ltforecast.pipeline import fit_final as run_fit_final

    cfg = load_config(config)
    store = Store(cfg.data_dir)
    out = run_fit_final(store, cfg, _model_config(config, model_config, fast), load_catalog())
    _synthetic_warning(store)
    typer.echo(f"final models (trained through {cfg.dates.dev_end}) -> {out}")


if __name__ == "__main__":
    app()

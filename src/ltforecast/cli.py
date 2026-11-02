"""ltforecast command line: ingest -> build -> check."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

from ltforecast.config import DEFAULT_CONFIG, load_config
from ltforecast.ingest import fred, sharadar
from ltforecast.pit import tables
from ltforecast.quality import universe_counts
from ltforecast.store import Store
from ltforecast.universe import build_universe

app = typer.Typer(no_args_is_help=True, help="Long-horizon stock return forecasting")
ingest_app = typer.Typer(no_args_is_help=True, help="Download vendor data into data/raw")
app.add_typer(ingest_app, name="ingest")

ConfigOpt = typer.Option(DEFAULT_CONFIG, "--config", help="Path to universe.yaml")


@ingest_app.command("sharadar")
def ingest_sharadar(
    table: list[str] = typer.Option(list(sharadar.TABLES), "--table", help="Tables to download"),
    from_dir: Path | None = typer.Option(None, help="Load hand-downloaded SHARADAR_<TABLE>*.zip/csv files instead"),
    config: Path = ConfigOpt,
) -> None:
    store = Store(load_config(config).data_dir)
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
def ingest_fred(config: Path = ConfigOpt) -> None:
    cfg = load_config(config)
    # Two years before history_start, so trailing macro features exist from day one.
    start = cfg.dates.history_start.replace(year=cfg.dates.history_start.year - 2)
    frames = []
    for s in cfg.macro:
        df = fred.fetch_series(s, start)
        no_vintage = df.filter(~pl.col("first_release")).height
        typer.echo(f"{s.id}: {df.height:,} obs ({no_vintage} without a first-release vintage)")
        frames.append(df)
    Store(cfg.data_dir).write_raw("fred", "OBSERVATIONS", tables.build_macro(frames))


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


if __name__ == "__main__":
    app()

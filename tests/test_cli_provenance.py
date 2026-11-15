"""CLI wiring of the synthetic-data guards (ingest/synthetic.py): --force, --replace-synthetic, and
the provenance record `build` writes to data/core/PROVENANCE.json."""

from __future__ import annotations

import datetime as dt
import json

import polars as pl
import pytest
from conftest import run_synthetic_pipeline
from typer.testing import CliRunner

from ltforecast.cli import app
from ltforecast.ingest import synthetic
from ltforecast.store import Store

START, END = dt.date(2015, 1, 2), dt.date(2017, 12, 29)
PARAMS = {"n_stocks": 20, "start": START.isoformat(), "end": END.isoformat(), "seed": 3}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("prov")
    cfg, outputs = run_synthetic_pipeline(
        root, n_stocks=20, start=START, end=END, seed=3, dev_end=dt.date(2016, 12, 31),
        holdout_start=dt.date(2017, 3, 31), holdout_end=dt.date(2017, 12, 29), steps=("build",),
    )
    return root / "config" / "universe.yaml", Store(cfg.data_dir), outputs


def invoke(config, *args):
    return CliRunner().invoke(app, [*args, "--config", str(config)])


def test_build_records_synthetic_provenance_and_warns(built):
    _, store, outputs = built
    assert json.loads((store.root / "core" / "PROVENANCE.json").read_text()) == {"synthetic": True, **PARAMS}
    assert "SYNTHETIC DATA" in outputs["build"]
    assert json.loads(synthetic.marker_path(store).read_text()) == PARAMS


def test_vendor_ingest_refuses_to_write_over_synthetic_data(built):
    config, store, _ = built
    before = store.raw_path("sharadar", "SEP").read_bytes()
    result = invoke(config, "ingest", "fred")   # refused before any download is attempted
    assert result.exit_code == 1
    assert "--replace-synthetic" in result.output
    assert store.raw_path("sharadar", "SEP").read_bytes() == before
    assert synthetic.raw_is_synthetic(store)


def test_replace_synthetic_clears_the_raw_set_and_force_overrides_the_vendor_guard(tmp_path):
    root = tmp_path / "r"
    cfg, _ = run_synthetic_pipeline(
        root, n_stocks=20, start=START, end=END, seed=3, dev_end=dt.date(2016, 12, 31),
        holdout_start=dt.date(2017, 3, 31), holdout_end=dt.date(2017, 12, 29), steps=(),
    )
    config, store = root / "config" / "universe.yaml", Store(cfg.data_dir)
    frames = {t: pl.read_parquet(store.raw_path("sharadar", t)) for t in ("TICKERS",)}
    empty = tmp_path / "downloads"
    empty.mkdir()
    # the guard runs first and clears the synthetic set; the load then fails on the empty directory
    result = invoke(config, "ingest", "sharadar", "--table", "TICKERS", "--from-dir", str(empty), "--replace-synthetic")
    assert result.exit_code != 0
    assert not synthetic.raw_is_synthetic(store)
    assert not store.raw_path("sharadar", "SEP").exists()
    # a vendor file now present: synthetic ingest refuses without --force, and succeeds with it
    store.write_raw("sharadar", "TICKERS", frames["TICKERS"])
    args = ["ingest", "synthetic", "--n-stocks", "20", "--start", START.isoformat(), "--end", END.isoformat(),
            "--seed", "3"]
    refused = invoke(config, *args)
    assert refused.exit_code == 1
    assert not synthetic.raw_is_synthetic(store)
    assert invoke(config, *args, "--force").exit_code == 0
    assert json.loads(synthetic.marker_path(store).read_text()) == PARAMS

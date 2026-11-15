"""build_raw / build_features on simulated data: shape, catalog agreement and point-in-time behaviour."""

from __future__ import annotations

import dataclasses
import datetime as dt
from types import SimpleNamespace

import polars as pl
import pytest
from conftest import run_synthetic_pipeline

from ltforecast.features import build, value
from ltforecast.features.catalog import load_catalog
from ltforecast.features.leakage import SRC_PREFIX
from ltforecast.features.panel import make_context
from ltforecast.pit.asof import LookaheadError
from ltforecast.store import Store

CATALOG = load_catalog()
CUT = dt.date(2004, 6, 30)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    # `labels` is needed for benchmark_daily; `features` is called directly below
    cfg, _ = run_synthetic_pipeline(
        tmp_path_factory.mktemp("build"), n_stocks=60, start=dt.date(2000, 1, 3), end=dt.date(2006, 12, 29),
        dev_end=dt.date(2004, 12, 31), holdout_start=dt.date(2006, 1, 1), holdout_end=dt.date(2006, 12, 31),
        steps=("build", "labels"),
    )
    return cfg, Store(cfg.data_dir)


@pytest.fixture(scope="module")
def ctx(run):
    cfg, store = run
    return make_context(store, cfg)


@pytest.fixture(scope="module")
def built(ctx):
    return build.build_features(ctx, CATALOG)


def test_raw_has_one_row_per_entry_and_the_catalog_columns(ctx, built):
    raw, _ = built
    assert raw.height == ctx.entries.height > 1000
    assert raw.select("security_id", "date").sort("security_id", "date").equals(ctx.entries)
    assert raw.equals(raw.sort("date", "security_id"))
    src = [c for c in raw.columns if c.startswith(SRC_PREFIX)]
    assert raw.columns == ["date", "security_id", "sector", *CATALOG.names(), *src]
    # one source-date column per family, each of type Date
    assert sorted(src) == sorted(f"{SRC_PREFIX}{f}" for f in {CATALOG.spec(n).family for n in CATALOG.names()})
    assert all(raw.schema[c] == pl.Float64 for c in CATALOG.names())
    assert all(raw.schema[c] == pl.Date for c in src)
    values = raw.select(CATALOG.names())
    assert values.select(pl.all().is_nan().any()).row(0) == (False,) * values.width
    assert values.select(pl.all().is_infinite().any()).row(0) == (False,) * values.width


def test_features_are_the_transform_of_raw(built):
    raw, feats = built
    assert feats.columns == ["date", "security_id", "sector", *CATALOG.model_columns()]
    assert feats.select("date", "security_id").equals(raw.select("date", "security_id"))
    # a rank is null exactly where the raw value is
    for name in CATALOG.names(kind="stock"):
        assert feats[name].is_null().equals(raw[name].is_null()), name


def test_every_family_has_a_source_date_wherever_it_has_a_value(built):
    raw, _ = built
    for family in {CATALOG.spec(n).family for n in CATALOG.names()}:
        any_value = pl.any_horizontal(pl.col(CATALOG.names(family=family)).is_not_null())
        missing = raw.filter(any_value & pl.col(f"{SRC_PREFIX}{family}").is_null())
        assert missing.is_empty(), f"{family}: {missing.height} rows with a value and no source date"


def test_sf1_metrics_list_gives_the_same_features(run, built):
    cfg, store = run
    raw, feats = build.build_features(make_context(store, cfg, metrics=build.SF1_METRICS), CATALOG)
    assert raw.equals(built[0])
    assert feats.equals(built[1])


def test_nothing_dated_after_t_changes_the_rows_up_to_t(ctx, built):
    """Rebuild with every input cut at CUT: rows dated <= CUT (raw values, source dates and ranks)
    must be identical to the ones built with the whole history."""
    upto = pl.col("date") <= CUT
    cut = dataclasses.replace(
        ctx,
        entries=ctx.entries.filter(upto),
        universe=ctx.universe.filter(upto),
        prices=ctx.prices.filter(upto),
        marketcap=ctx.marketcap.filter(upto),
        benchmark=ctx.benchmark.filter(upto),
        fundamentals=ctx.fundamentals.filter(pl.col("available_date") <= CUT),
        macro=ctx.macro.filter(pl.col("available_date") <= CUT),
        calendar=ctx.calendar.filter(ctx.calendar <= CUT),
    )
    raw_cut, feats_cut = build.build_features(cut, CATALOG)
    assert raw_cut.height > 500
    assert raw_cut.equals(built[0].filter(upto))
    assert feats_cut.equals(built[1].filter(upto))


def test_catalog_and_modules_must_agree(ctx):
    names = CATALOG.names()
    fewer = SimpleNamespace(names=lambda *a, **k: names[1:])
    with pytest.raises(ValueError, match=f"not in the catalog.*{names[0]}"):
        build.build_raw(ctx, fewer)
    more = SimpleNamespace(names=lambda *a, **k: [*names, "analyst_revisions"])
    with pytest.raises(ValueError, match="not computed.*analyst_revisions"):
        build.build_raw(ctx, more)


def test_a_source_dated_after_the_prediction_date_fails_the_build(ctx, monkeypatch):
    honest = value.compute

    def leaky(c):
        return honest(c).with_columns(pl.col("date").dt.offset_by("1d").alias("src_date_value"))

    monkeypatch.setattr(value, "compute", leaky)
    with pytest.raises(LookaheadError):
        build.build_raw(ctx, CATALOG)

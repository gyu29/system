"""End to end on simulated data, through the CLI: ingest synthetic -> build -> labels -> features,
then the modelling frame, the walk-forward runner and the metrics (plan sections 3 and 4).

200 stocks x 12 years with seed 0. Development = prediction dates through 2008, 2009 is the gap and
2010 the holdout, so every period exists. The planted signal is modest (rank IC of a few points per
characteristic), so the sign checks below hold for this pinned seed, not for every seed.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest
from conftest import run_synthetic_pipeline

from ltforecast.dataset import HoldoutLockedError, frame_from_tables, modeling_frame
from ltforecast.features.catalog import load_catalog
from ltforecast.features.leakage import assert_sources_not_after, shuffle_labels, univariate_ic
from ltforecast.modelconfig import ValidationRules
from ltforecast.store import CORE_TABLES, Store
from ltforecast.validation.metrics import evaluate
from ltforecast.validation.runner import walk_forward
from ltforecast.validation.splitter import walk_forward_folds

CATALOG = load_catalog()
START, END = dt.date(2000, 1, 3), dt.date(2011, 12, 30)
DEV_END, HOLDOUT_START, HOLDOUT_END = dt.date(2008, 12, 31), dt.date(2010, 1, 1), dt.date(2010, 12, 31)
WARM_UP_END = dt.date(2005, 1, 1)  # five years in: every lookback (60m beta, 5 fiscal years) can be full
# The model.yaml rules need 2000-2009 before the first test year; here the first test year is 2006.
RULES = ValidationRules(
    first_test_year=2006, test_months=12, purge_months=12, embargo_months=1, min_train_months=36,
    inner_valid_months=12,
)
# The characteristics ingest/synthetic.py plants (LOADINGS), with the sign of their drift.
PLANTED = {"gross_profitability": 1, "book_to_price": 1, "mom_12_1": 1, "asset_growth": -1}


class PlantedMix:
    """The simplest possible model: no fitting, the signed sum of the planted characteristics' ranks."""

    name = "planted_mix"

    def fit(self, train, valid, *, full=None):
        assert train["target"].null_count() == 0 and valid["target"].null_count() == 0
        assert full is not None and full["target"].null_count() == 0
        return self

    def predict(self, df):
        assert "target" not in df.columns and "excess_12m" not in df.columns
        return df.select(pl.sum_horizontal([s * pl.col(c).fill_null(0.0) for c, s in PLANTED.items()])).to_series().to_numpy()


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cfg, outputs = run_synthetic_pipeline(
        tmp_path_factory.mktemp("e2e"), n_stocks=200, start=START, end=END, seed=0,
        dev_end=DEV_END, holdout_start=HOLDOUT_START, holdout_end=HOLDOUT_END,
    )
    return cfg, Store(cfg.data_dir), outputs


@pytest.fixture(scope="module")
def tables(run):
    _, store, _ = run
    return {t: store.scan(t).collect() for t in ("universe", "labels", "features_raw", "features")}


@pytest.fixture(scope="module")
def dev(run):
    cfg, store, _ = run
    return modeling_frame(store, cfg, CATALOG, periods=("dev",))


def test_every_table_and_report_is_written(run):
    cfg, store, outputs = run
    for table in CORE_TABLES:
        assert any(store.core_path(table).glob("year=*/data.parquet")) or (store.core_path(table) / "data.parquet").exists(), table
    marker = json.loads((store.root / "raw" / "SYNTHETIC.json").read_text())
    assert marker == {"n_stocks": 200, "start": START.isoformat(), "end": END.isoformat(), "seed": 0}
    assert "synthetic data, not vendor data" in outputs["ingest"]
    summary = outputs["features"]
    assert f"{len(CATALOG.names())} features" in summary and "dev rows" in summary
    assert "no feature flagged" in summary

    ic = pl.read_csv(store.root / "reports" / "feature_ic.csv")
    shift = pl.read_csv(store.root / "reports" / "feature_shift_test.csv")
    assert ic["feature"].to_list() == shift["feature"].to_list() == CATALOG.linear_columns()
    assert ic["n_months"].max() <= 108  # development months only: 2000-2008
    assert not shift["flagged"].any()


def test_features_cover_exactly_the_universe(tables):
    members = tables["universe"].filter(pl.col("in_universe")).select("date", "security_id").sort("date", "security_id")
    feats, raw = tables["features"], tables["features_raw"]
    assert members.height > 8000
    assert feats.columns == ["date", "security_id", "sector", *CATALOG.model_columns()]
    assert feats.sort("date", "security_id").select("date", "security_id").equals(members)
    assert raw.sort("date", "security_id").select("date", "security_id").equals(members)
    assert feats.select("date", "security_id").is_unique().all()


def test_feature_values_are_ranks_flags_or_finite_context(tables):
    feats = tables["features"]
    for col in CATALOG.linear_columns():
        lo, hi = feats[col].min(), feats[col].max()
        assert -0.5 < lo and hi < 0.5, (col, lo, hi)
    for col in CATALOG.flag_names():
        assert set(feats[col].unique().to_list()) <= {0, 1}, col
    for name, frame in (("features", feats), ("features_raw", tables["features_raw"])):
        floats = frame.select(pl.selectors.float())
        assert floats.width >= len(CATALOG.names())
        bad = [c for c in floats.columns if floats[c].is_nan().any() or floats[c].is_infinite().any()]
        assert not bad, f"{name}: NaN or inf in {bad}"


def test_no_source_date_is_after_its_prediction_date(tables):
    assert_sources_not_after(tables["features_raw"])


def test_coverage_after_the_warm_up_years(tables):
    raw = tables["features_raw"].filter(pl.col("date") >= WARM_UP_END)
    names = CATALOG.names()
    coverage = pl.DataFrame({
        "feature": names,
        "family": [CATALOG.spec(n).family for n in names],
        "coverage": [1.0 - raw[n].null_count() / raw.height for n in names],
    })
    with pl.Config(tbl_rows=-1):
        table = str(coverage)
    # Long lookbacks stay partly empty for names listed during the sample: four fiscal years
    # (eps_growth_stability, op_margin_stability), 36 months of prices (rev_36_12). Measured: 0.90-0.96.
    floor = {"eps_growth_stability": 0.80, "op_margin_stability": 0.80, "rev_36_12": 0.80}
    low = coverage.filter(pl.col("coverage") < pl.col("feature").replace_strict(floor, default=0.85, return_dtype=pl.Float64))
    assert low.is_empty(), f"implausibly low coverage for {low['feature'].to_list()}:\n{table}"
    # and no feature is filled in for every row by construction: the first years must have gaps
    early = tables["features_raw"].filter(pl.col("date") < dt.date(2001, 1, 1))
    assert early["beta_60m"].null_count() == early.height, table


def test_dev_frame_has_no_locked_rows_and_the_holdout_needs_a_token(run, tables, dev):
    cfg, store, _ = run
    labels = tables["labels"]
    counts = dict(labels.group_by("period").len().iter_rows())
    assert set(counts) == {"dev", "gap", "holdout", "live"} and min(counts.values()) > 500

    assert dev["period"].unique().to_list() == ["dev"]
    assert dev["date"].max() <= DEV_END
    assert dev.height == counts["dev"]
    from_tables = frame_from_tables(tables["features"], labels, CATALOG, periods=("dev",))
    assert from_tables.equals(dev)

    for periods in (("holdout",), ("gap",), ("dev", "holdout")):
        with pytest.raises(HoldoutLockedError):
            frame_from_tables(tables["features"], labels, CATALOG, periods=periods)
        with pytest.raises(HoldoutLockedError):
            modeling_frame(store, cfg, CATALOG, periods=periods)


def test_walk_forward_on_the_dev_frame(dev):
    folds = walk_forward_folds(dev["date"].unique(), RULES, DEV_END)
    assert [f.name for f in folds] == ["2006", "2007", "2008"]
    dates = sorted(dev["date"].unique().to_list())
    for fold in folds:
        # 13 month ends (12 purge + 1 embargo) lie between the last train date and the first test date
        between = [d for d in dates if fold.train_dates[-1] < d < fold.test_dates[0]]
        assert len(between) == 13
        train = dev.filter(pl.col("date").is_in(list(fold.train_dates)) & pl.col("target").is_not_null())
        assert train["label_end_12m"].max() < fold.test_dates[0]
        assert max(fold.test_dates) <= DEV_END

    preds = walk_forward(PlantedMix, dev, folds, RULES)
    test_rows = dev.filter(pl.col("date").dt.year() >= 2006)
    assert preds.height == test_rows.height
    assert preds["fold"].unique().sort().to_list() == ["2006", "2007", "2008"]

    report = evaluate(preds.join(dev.select("date", "security_id", "target", "excess_12m"), on=["date", "security_id"]))
    assert report["n_dates"] == 36 and report["ic"]["n_months"] == 36
    assert np.isfinite(report["ic"]["mean_ic"]) and np.isfinite(report["ic"]["nw_tstat"])
    assert report["deciles"].height == 10
    assert report["ic"]["mean_ic"] > 0.0, report["ic"]


def test_planted_signals_show_up_and_shuffled_labels_do_not(dev):
    """Plan section 3 exit criteria: profitability and momentum have a positive IC; with labels
    shuffled within each date every IC is about zero."""
    mix = pl.sum_horizontal([s * pl.col(c).fill_null(0.0) for c, s in PLANTED.items()]).alias("planted_mix")
    frame = dev.with_columns(mix)
    cols = [*PLANTED, "planted_mix"]
    ic = dict(univariate_ic(frame, cols).select("feature", "mean_ic").iter_rows())
    for col, sign in PLANTED.items():
        assert sign * ic[col] > 0.0, ic
    assert ic["planted_mix"] > 0.04, ic

    shuffled = shuffle_labels(frame, seed=1)
    assert shuffled["label_end_12m"].equals(frame["label_end_12m"])
    ic_shuffled = dict(univariate_ic(shuffled, cols).select("feature", "mean_ic").iter_rows())
    # 107 months x about 107 stocks: the standard error of a mean IC of noise is roughly 0.01
    assert all(abs(v) < 0.04 for v in ic_shuffled.values()), ic_shuffled
    assert abs(ic_shuffled["planted_mix"]) < ic["planted_mix"] / 2

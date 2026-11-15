"""config/model.yaml loads into typed rules; typos and impossible settings raise."""

import datetime as dt
from pathlib import Path

import pytest
import yaml

from ltforecast.config import REPO_ROOT
from ltforecast.modelconfig import (
    DEFAULT_MODEL_CONFIG,
    Criteria,
    MetricRules,
    ValidationRules,
    load_model_config,
)


def write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config" / "model.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(DEFAULT_MODEL_CONFIG.read_text())


def test_default_config_values():
    mc = load_model_config()
    assert mc.validation == ValidationRules(
        first_test_year=2010, test_months=12, purge_months=12, embargo_months=1,
        min_train_months=60, inner_valid_months=24,
    )
    assert mc.metrics == MetricRules(nw_lags=12, n_deciles=10, prob_bins=10)
    assert mc.costs_bps == 20.0
    assert mc.experiments_dir == REPO_ROOT / "experiments"
    assert mc.raw["validation"]["purge_months"] == 12


def test_criteria_are_the_preregistered_numbers():
    # plan section 0; if this test has to change, the criteria were tuned
    assert load_model_config().criteria == Criteria(
        min_rank_ic=0.04,
        min_ic_tstat=2.0,
        min_positive_spread_years=0.65,
        coverage_range=(0.75, 0.85),
        calibration_tolerance=0.05,
        tpsl_prob_tolerance=0.05,
    )


def test_rules_are_frozen():
    mc = load_model_config()
    with pytest.raises(AttributeError):
        mc.criteria.min_rank_ic = 0.01
    with pytest.raises(AttributeError):
        mc.validation.purge_months = 0


def test_purge_covers_the_label_horizon():
    from ltforecast.config import load_config

    cfg, v = load_config(), load_model_config().validation
    assert v.purge_months * 21 >= cfg.labels.horizons[cfg.labels.primary]
    # the first test year starts after the history does, with room for min_train_months
    assert dt.date(v.first_test_year, 1, 1) > cfg.dates.history_start


def test_experiments_dir_relative_to_the_config_files_repo(tmp_path, raw):
    assert load_model_config(write(tmp_path, raw)).experiments_dir == tmp_path / "experiments"
    raw["experiments_dir"] = str(tmp_path / "elsewhere")
    assert load_model_config(write(tmp_path, raw)).experiments_dir == tmp_path / "elsewhere"


def test_unknown_top_level_key_raises(tmp_path, raw):
    raw["validaton"] = raw["validation"]
    with pytest.raises(ValueError, match="validaton"):
        load_model_config(write(tmp_path, raw))


def test_misspelled_or_missing_section_key_raises(tmp_path, raw):
    raw["validation"]["purge_month"] = raw["validation"].pop("purge_months")
    with pytest.raises(ValueError, match="purge_month"):
        load_model_config(write(tmp_path, raw))


@pytest.mark.parametrize("missing", ["criteria", "metrics", "costs_bps", "experiments_dir"])
def test_missing_section_raises(tmp_path, raw, missing):
    del raw[missing]
    with pytest.raises(ValueError, match=missing):
        load_model_config(write(tmp_path, raw))


def test_training_window_too_short_for_the_inner_split_raises(tmp_path, raw):
    # 24 validation months + 12 purge + 1 embargo = 37: nothing would be left to fit on
    raw["validation"]["min_train_months"] = 37
    with pytest.raises(ValueError, match="min_train_months"):
        load_model_config(write(tmp_path, raw))
    raw["validation"]["min_train_months"] = 38
    assert load_model_config(write(tmp_path, raw)).validation.min_train_months == 38


def test_bad_values_raise(tmp_path, raw):
    raw["criteria"]["coverage_range"] = [0.85, 0.75]
    with pytest.raises(ValueError, match="coverage_range"):
        load_model_config(write(tmp_path, raw))
    raw["criteria"]["coverage_range"] = [0.75, 0.85]
    raw["validation"]["purge_months"] = -1
    with pytest.raises(ValueError, match="purge"):
        load_model_config(write(tmp_path, raw))


def test_fractional_month_count_raises(tmp_path, raw):
    # int(12.5) would quietly shorten the purge to 12
    raw["validation"]["purge_months"] = 12.5
    with pytest.raises(ValueError, match="purge_months"):
        load_model_config(write(tmp_path, raw))


def test_empty_file_raises(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("# nothing here\n")
    with pytest.raises(ValueError, match="not a mapping"):
        load_model_config(path)

"""Leakage checks on small hand-made frames: source dates, univariate IC, shift test, label shuffle."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from ltforecast.features.leakage import assert_sources_not_after, shift_test, shuffle_labels, univariate_ic
from ltforecast.pit.asof import LookaheadError

D = dt.date


def month_ends(n: int) -> list[dt.date]:
    return [D(2010 + m // 12, m % 12 + 1, 28) for m in range(n)]


def panel(n_dates: int = 36, n_stocks: int = 40, seed: int = 0) -> pl.DataFrame:
    """date x security grid with an independent standard-normal `target` per row, and the label
    columns the modelling frame carries."""
    rng = np.random.default_rng(seed)
    dates = month_ends(n_dates)
    return pl.DataFrame({
        "date": np.repeat(np.array(dates, dtype="datetime64[D]"), n_stocks),
        "security_id": np.tile(np.arange(1, n_stocks + 1), n_dates),
        "target": rng.standard_normal(n_dates * n_stocks),
    }).with_columns(
        pl.col("date").cast(pl.Date),
        (pl.col("target") * 0.1).alias("excess_12m"),
        pl.col("date").dt.offset_by("1y").alias("label_end_12m"),
        pl.lit("dev").alias("period"),
    )


# --- assert_sources_not_after ---

def raw_rows(src_value, src_risk):
    return pl.DataFrame({
        "date": [D(2020, 3, 31), D(2020, 4, 30)],
        "security_id": [1, 1],
        "book_to_price": [0.5, 0.6],
        "src_date_value": src_value,
        "src_date_risk": src_risk,
    }, schema_overrides={"src_date_value": pl.Date, "src_date_risk": pl.Date})


def test_sources_on_or_before_the_date_pass_and_nulls_are_ignored():
    assert_sources_not_after(raw_rows([D(2020, 2, 14), None], [D(2020, 3, 31), D(2020, 4, 30)]))


@pytest.mark.parametrize("col", ["value", "risk"])
def test_a_source_one_day_after_the_date_raises(col):
    ok = [D(2020, 2, 14), D(2020, 2, 14)]
    late = [D(2020, 2, 14), D(2020, 5, 1)]
    frame = raw_rows(late, ok) if col == "value" else raw_rows(ok, late)
    with pytest.raises(LookaheadError, match=f"src_date_{col}"):
        assert_sources_not_after(frame)


def test_a_frame_without_source_columns_is_refused():
    with pytest.raises(ValueError, match="src_date_"):
        assert_sources_not_after(raw_rows([None, None], [None, None]).drop("src_date_value", "src_date_risk"))


# --- univariate_ic ---

def test_univariate_ic_of_exact_inverse_and_unrelated_features():
    f = panel().with_columns(
        pl.col("target").alias("same"),
        (-pl.col("target")).alias("inverse"),
        pl.Series("noise", np.random.default_rng(9).standard_normal(36 * 40)),
        pl.lit(None, pl.Float64).alias("empty"),
    )
    out = univariate_ic(f, ["same", "inverse", "noise", "empty"])
    assert out.columns == ["feature", "mean_ic", "std_ic", "ic_ir", "nw_tstat", "n_months", "pct_positive"]
    assert out["feature"].to_list() == ["same", "inverse", "noise", "empty"]
    same, inverse, noise, empty = out.to_dicts()
    assert same["mean_ic"] == pytest.approx(1.0) and same["n_months"] == 36 and same["pct_positive"] == 1.0
    assert inverse["mean_ic"] == pytest.approx(-1.0) and inverse["pct_positive"] == 0.0
    # 36 dates x 40 stocks: the standard error of a mean IC of pure noise is about 1/sqrt(1440) = 0.026
    assert abs(noise["mean_ic"]) < 0.08
    assert empty["n_months"] == 0 and np.isnan(empty["mean_ic"])


def test_univariate_ic_skips_dates_with_too_few_pairs():
    f = panel(n_dates=3, n_stocks=12).with_columns(
        pl.when(pl.col("date") == month_ends(3)[1]).then(None).otherwise(pl.col("target")).alias("x")
    )
    assert univariate_ic(f, ["x"])["n_months"].to_list() == [2]
    assert univariate_ic(f, ["x"], min_n=13)["n_months"].to_list() == [0]


# --- shift_test ---

def with_leaky_and_honest(f: pl.DataFrame) -> pl.DataFrame:
    """`leaky`(t) = the stock's target at t+1: built one month too late, so last month's value lines
    up exactly with this month's label. `honest`(t) = target(t) plus noise."""
    noise = pl.Series("_n", np.random.default_rng(3).standard_normal(f.height))
    return f.sort("security_id", "date").with_columns(
        pl.col("target").shift(-1).over("security_id").alias("leaky"),
        (pl.col("target") + noise).alias("honest"),
    ).sort("date", "security_id")


def test_shift_test_flags_a_feature_that_improves_when_lagged():
    out = shift_test(with_leaky_and_honest(panel()), ["leaky", "honest"])
    assert out.columns == ["feature", "ic", "ic_shifted", "diff", "nw_tstat", "orientation", "gain",
                           "gain_tstat", "n_months", "flagged"]
    leaky, honest = out.to_dicts()
    # shifted leaky(t) = leaky(t-1) = target(t): IC exactly 1; both need t-1 and t+1, so 34 of 36 months
    assert leaky["ic_shifted"] == pytest.approx(1.0) and leaky["n_months"] == 34
    assert abs(leaky["ic"]) < 0.1
    assert leaky["diff"] == pytest.approx(leaky["ic_shifted"] - leaky["ic"])
    assert leaky["flagged"] is True
    # honest: about 0.7 now, about 0 a month stale (targets are independent across months)
    assert honest["ic"] > 0.6 and abs(honest["ic_shifted"]) < 0.1 and honest["n_months"] == 35
    assert honest["diff"] < -0.5 and honest["flagged"] is False


def test_shift_test_uses_the_previous_date_of_the_same_security():
    # two dates, ten stocks: the shifted value on date 2 is the stock's own value on date 1
    d1, d2 = month_ends(2)
    ids = list(range(1, 11))
    f = pl.DataFrame({
        "date": [d1] * 10 + [d2] * 10,
        "security_id": ids * 2,
        "x": [float(i) for i in ids] + [float(-i) for i in ids],
        "target": [0.0] * 10 + [float(i) for i in ids],
    }, schema_overrides={"date": pl.Date})
    row = shift_test(f, ["x"]).to_dicts()[0]
    assert (row["ic"], row["ic_shifted"], row["diff"], row["n_months"]) == (-1.0, 1.0, 2.0, 1)
    assert row["flagged"] is False  # one month proves nothing: no t-stat


def test_shift_test_needs_both_a_large_gain_and_a_tstat():
    f = with_leaky_and_honest(panel())
    assert shift_test(f, ["leaky"], min_gain=1.5)["flagged"].to_list() == [False]
    assert shift_test(f, ["leaky"], min_tstat=1e9)["flagged"].to_list() == [False]


# --- shuffle_labels ---

def test_shuffle_permutes_labels_within_each_date_and_nothing_else():
    f = panel().with_columns(pl.col("target").alias("feature"))
    s = shuffle_labels(f, seed=1)
    assert s.columns == f.columns and s.height == f.height
    for col in ("date", "security_id", "feature", "label_end_12m", "period"):
        assert s[col].equals(f[col]), col
    assert not s["target"].equals(f["target"])
    # the same values on each date, in another order
    per_date = lambda df: df.sort("date", "target").select("date", "target", "excess_12m")  # noqa: E731
    assert per_date(s).equals(per_date(f))
    # one permutation for all label columns: a row's return still belongs to its target
    assert (s["excess_12m"] - s["target"] * 0.1).abs().max() < 1e-12


def test_shuffle_is_seeded_and_kills_the_ic():
    f = panel().with_columns(pl.col("target").alias("feature"))
    assert shuffle_labels(f, seed=1).equals(shuffle_labels(f, seed=1))
    assert not shuffle_labels(f, seed=1)["target"].equals(shuffle_labels(f, seed=2)["target"])
    assert univariate_ic(f, ["feature"])["mean_ic"][0] == pytest.approx(1.0)
    assert abs(univariate_ic(shuffle_labels(f, seed=1), ["feature"])["mean_ic"][0]) < 0.08


def test_shuffle_only_the_named_columns():
    f = panel()
    s = shuffle_labels(f, seed=1, cols=["target"])
    assert s["excess_12m"].equals(f["excess_12m"]) and not s["target"].equals(f["target"])


# --- shift_test: orientation (a sign -1 feature must be tested like its negation) ---

def test_shift_test_result_does_not_change_when_a_feature_is_negated():
    f = with_leaky_and_honest(panel()).with_columns(
        (-pl.col("leaky")).alias("neg_leaky"), (-pl.col("honest")).alias("neg_honest")
    )
    leaky, honest, neg_leaky, neg_honest = shift_test(f, ["leaky", "honest", "neg_leaky", "neg_honest"]).to_dicts()
    # neg_leaky: IC about 0 now, exactly -1 a month stale: a blatant misalignment, oriented by the stale IC
    assert neg_leaky["ic_shifted"] == pytest.approx(-1.0) and neg_leaky["orientation"] == -1
    assert neg_leaky["diff"] == pytest.approx(-leaky["diff"])
    assert neg_leaky["gain"] == pytest.approx(leaky["gain"]) and neg_leaky["gain"] > 0.9
    assert neg_leaky["gain_tstat"] == pytest.approx(leaky["gain_tstat"])
    assert neg_leaky["flagged"] is True and leaky["flagged"] is True
    assert leaky["orientation"] == 1 and leaky["gain"] == pytest.approx(leaky["diff"])
    assert honest["flagged"] is False and neg_honest["flagged"] is False
    assert neg_honest["gain"] == pytest.approx(honest["gain"]) and honest["gain"] < -0.5


def test_shift_test_does_not_flag_the_normal_decay_of_a_negative_signal():
    # target = a_i + e_t (a persistent stock part and a monthly part, each N(0,1));
    # x(t) = -(target(t) + noise): Pearson IC about -2/sqrt(6) = -0.82 now and -1/sqrt(6) = -0.41 a month
    # stale. The stale value is WEAKER: the signed diff is about +0.41, the oriented gain about -0.41.
    n_dates, n_stocks = 36, 40
    rng = np.random.default_rng(5)
    a = np.tile(rng.standard_normal(n_stocks), n_dates)
    f = panel(n_dates, n_stocks).with_columns(
        pl.Series("target", a + rng.standard_normal(n_dates * n_stocks)),
    ).with_columns((-(pl.col("target") + pl.Series(rng.standard_normal(n_dates * n_stocks)))).alias("x"))
    row = shift_test(f, ["x"]).to_dicts()[0]
    assert row["ic"] < -0.6 and -0.6 < row["ic_shifted"] < -0.2
    assert row["diff"] > 0.2 and row["orientation"] == -1
    assert row["gain"] == pytest.approx(-row["diff"]) and row["gain_tstat"] == pytest.approx(-row["nw_tstat"])
    assert row["flagged"] is False

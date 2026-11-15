"""Hand-computed checks of the validation metrics (plan 4.2, 4.3).

Every expectation below is a literal number or explicit arithmetic worked out in the comment next to
it; scipy is used only as a second, independent implementation.
"""

import datetime as dt
import json
import math
import statistics
import time
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from conftest import business_days
from scipy.stats import norm, spearmanr

from ltforecast.quality import CheckResult
from ltforecast.validation import metrics as m

d = dt.date
D1, D2, D3, D4, D5 = d(2020, 1, 31), d(2020, 2, 28), d(2020, 3, 31), d(2020, 4, 30), d(2020, 5, 29)


def xs(date: dt.date, pred, target, name: str = "target") -> pl.DataFrame:
    """One date's cross-section."""
    n = len(pred)
    return pl.DataFrame(
        {"date": [date] * n, "security_id": list(range(n)), "pred": pred, name: target},
        schema={"date": pl.Date, "security_id": pl.Int64, "pred": pl.Float64, name: pl.Float64},
    )


# rank ic -------------------------------------------------------------------------------------------


def test_spearman_with_ties_is_pearson_of_average_ranks():
    # pred   1    2    2    3    4    5     ranks 1, 2.5, 2.5, 4, 5, 6
    # target 10   30   20   20   50   40    ranks 1, 4, 2.5, 2.5, 6, 5
    # deviations from the mean rank 3.5:
    #   pred   -2.5  -1   -1    0.5  1.5  2.5    sum of squares 6.25 + 1 + 1 + 0.25 + 2.25 + 6.25 = 17
    #   target -2.5   0.5 -1   -1    2.5  1.5    sum of squares 17
    #   cross products 6.25 - 0.5 + 1 - 0.5 + 3.75 + 3.75 = 13.75
    pred, target = [1.0, 2, 2, 3, 4, 5], [10.0, 30, 20, 20, 50, 40]
    out = m.ic_by_date(xs(D1, pred, target), "pred", "target", min_n=5)
    assert out.columns == ["date", "ic", "n"]
    assert out.rows() == [(D1, pytest.approx(13.75 / 17), 6)]
    assert out["ic"][0] == pytest.approx(spearmanr(pred, target).statistic)
    # the textbook shortcut 1 - 6 sum(d^2) / (n (n^2 - 1)) is wrong with ties: d^2 = 0 + 2.25 + 0 + 2.25 + 1 + 1
    assert out["ic"][0] != pytest.approx(1 - 6 * 6.5 / (6 * 35))


def test_ic_by_date_per_date_nulls_min_n_and_constant_side():
    df = pl.concat(
        [
            # no ties, ranks 1..5 vs 2, 1, 4, 3, 5: d^2 = 1 + 1 + 1 + 1 + 0, rho = 1 - 6 * 4 / (5 * 24) = 0.8
            xs(D1, [1.0, 2, 3, 4, 5], [0.2, 0.1, 0.4, 0.3, 0.5]),
            # exactly inverse on the five complete pairs; the null target and the NaN prediction are not pairs
            xs(D2, [1.0, 2, 3, 4, 5, 6, math.nan], [5.0, 4, 3, 2, 1, None, 0]),
            # four pairs < min_n: dropped
            xs(D3, [1.0, 2, 3, 4], [1.0, 2, 3, 4]),
            # every prediction equal: no ranking, IC 0 (not NaN)
            xs(D4, [0.0] * 5, [1.0, 2, 3, 4, 5]),
        ]
    ).sample(fraction=1.0, shuffle=True, seed=1)
    out = m.ic_by_date(df, "pred", "target", min_n=5)
    assert out.rows() == [(D1, pytest.approx(0.8), 5), (D2, pytest.approx(-1.0), 5), (D4, 0.0, 5)]
    assert m.ic_by_date(df, "pred", "target").is_empty()  # default min_n = 10


def test_ic_by_date_is_vectorized_at_real_scale():
    # ~900k rows x 250 dates, the size of the real out-of-fold frame
    rng = np.random.default_rng(0)
    n_dates, n_stocks = 250, 3600
    pred = rng.normal(size=n_dates * n_stocks)
    target = 0.05 * pred + rng.normal(size=pred.size)
    df = pl.DataFrame({"date": np.repeat(np.arange(n_dates), n_stocks), "pred": pred, "target": target})
    start = time.perf_counter()
    out = m.ic_by_date(df, "pred", "target")
    elapsed = time.perf_counter() - start
    assert out.height == n_dates and out["n"].unique().to_list() == [n_stocks]
    for i in (0, 123, 249):
        sl = slice(i * n_stocks, (i + 1) * n_stocks)
        assert out["ic"][i] == pytest.approx(spearmanr(pred[sl], target[sl]).statistic, abs=1e-12)
    assert elapsed < 5.0, f"ic_by_date took {elapsed:.1f}s on 900k rows"


# newey-west ----------------------------------------------------------------------------------------

X = [2.0, 4, 1, 5, 3, 9]
# mean 24 / 6 = 4, deviations -2, 0, -3, 1, -1, 5; autocovariances with the 1/n normalization:
G0 = (4 + 0 + 9 + 1 + 1 + 25) / 6  # 40/6, the population variance
G1 = (0 * -2 + -3 * 0 + 1 * -3 + -1 * 1 + 5 * -1) / 6  # -9/6
G2 = (-3 * -2 + 1 * 0 + -1 * -3 + 5 * 1) / 6  # 14/6
G3 = (1 * -2 + -1 * 0 + 5 * -3) / 6  # -17/6
G4 = (-1 * -2 + 5 * 0) / 6  # 2/6
G5 = (5 * -2) / 6  # -10/6


def test_newey_west_lags_zero_is_the_ordinary_standard_error():
    # 1/n variance throughout: se = sqrt(population variance / n) = sqrt((40/6) / 6) = sqrt(10) / 3
    assert m.newey_west_se(X, 0) == pytest.approx(math.sqrt(10) / 3)
    assert m.newey_west_se(X, 0) == pytest.approx(statistics.pstdev(X) / math.sqrt(6))
    assert m.newey_west_tstat(X, 0) == pytest.approx(4 / (math.sqrt(10) / 3))


def test_newey_west_bartlett_weights_hand_computed():
    # lags = 1: w1 = 1 - 1/2;  S = 40/6 + 2 * (1/2) * (-9/6) = 31/6;  se = sqrt(31/36)
    assert m.newey_west_se(X, 1) == pytest.approx(math.sqrt((G0 + 2 * (1 - 1 / 2) * G1) / 6))
    assert m.newey_west_se(X, 1) == pytest.approx(math.sqrt(31 / 36))
    # lags = 2: w1 = 2/3, w2 = 1/3;  S = 40/6 + 2 * (2/3 * -9/6 + 1/3 * 14/6) = 56/9;  se = sqrt(28/27)
    s2 = G0 + 2 * ((1 - 1 / 3) * G1 + (1 - 2 / 3) * G2)
    assert s2 == pytest.approx(56 / 9)
    assert m.newey_west_se(X, 2) == pytest.approx(math.sqrt(s2 / 6))
    assert m.newey_west_se(X, 2) == pytest.approx(math.sqrt(28 / 27))
    assert m.newey_west_tstat(X, 2) == pytest.approx(4 / math.sqrt(28 / 27))
    # more lags than the series has: the weights still come from `lags`, autocovariances stop at n - 1
    # S = (40 + 2 * (10 * -9 + 9 * 14 + 8 * -17 + 7 * 2 + 6 * -10) / 11) / 6 = 74/33
    s10 = G0 + 2 * sum((1 - j / 11) * g for j, g in enumerate([G1, G2, G3, G4, G5], start=1))
    assert s10 == pytest.approx(74 / 33)
    assert m.newey_west_se(X, 10) == pytest.approx(math.sqrt(74 / 33 / 6))


def test_newey_west_inputs_and_degenerate_series():
    assert m.newey_west_se(pl.Series(X), 1) == pytest.approx(math.sqrt(31 / 36))
    assert m.newey_west_se(np.array(X), 1) == pytest.approx(math.sqrt(31 / 36))
    # an IC frame is read in date order whatever its row order (the lag-1 term depends on the order)
    dates = [d(2020, k, 1) for k in range(1, 7)]
    frame = pl.DataFrame({"date": dates, "ic": X}).sample(fraction=1.0, shuffle=True, seed=3)
    assert frame["ic"].to_list() != X
    assert m.newey_west_se(frame, 1) == pytest.approx(math.sqrt(31 / 36))
    assert math.isnan(m.newey_west_se([1.0], 3)) and math.isnan(m.newey_west_tstat([], 3))
    assert m.newey_west_tstat([0.0, 0.0, 0.0], 2) == 0.0  # a no-skill model: IC 0 every month
    assert m.newey_west_tstat([0.5, 0.5, 0.5], 2) == math.inf


def test_overlapping_series_has_a_smaller_tstat_with_12_lags():
    # a 12-month moving sum of iid shocks, like monthly ICs of overlapping 12-month labels
    shocks = np.random.default_rng(0).normal(0.3, 1.0, 300 + 11)
    x = np.convolve(shocks, np.ones(12), mode="valid")
    assert x.size == 300
    se0, se12 = m.newey_west_se(x, 0), m.newey_west_se(x, 12)
    t0, t12 = m.newey_west_tstat(x, 0), m.newey_west_tstat(x, 12)
    assert se12 > 2 * se0
    assert 0 < t12 < t0 / 2
    # iid shocks themselves: the correction is small either way
    assert m.newey_west_se(shocks, 12) == pytest.approx(m.newey_west_se(shocks, 0), rel=0.35)


def test_ic_summary_hand_computed():
    # mean 0.1; deviations 0, -0.15, 0.1, 0.05; sum of squares 0.035
    # sample std sqrt(0.035 / 3); population variance 0.035 / 4, se = sqrt(0.035 / 4 / 4)
    s = m.ic_summary([0.1, -0.05, 0.2, 0.15], lags=0)
    assert s["mean_ic"] == pytest.approx(0.1)
    assert s["std_ic"] == pytest.approx(math.sqrt(0.035 / 3))
    assert s["ic_ir"] == pytest.approx(0.1 / math.sqrt(0.035 / 3))
    assert s["nw_tstat"] == pytest.approx(0.1 / math.sqrt(0.035 / 16))
    assert s["n_months"] == 4 and s["pct_positive"] == 0.75
    frame = pl.DataFrame({"date": [D1, D2, D3, D4], "ic": [0.1, -0.05, 0.2, 0.15], "n": [50] * 4})
    assert m.ic_summary(frame, lags=0) == s
    empty = m.ic_summary(frame.head(0))
    assert empty["n_months"] == 0 and math.isnan(empty["mean_ic"])


def test_paired_ic_test_uses_common_dates():
    a = pl.DataFrame({"date": [D1, D2, D3, D4], "ic": [0.10, 0.06, 0.08, 0.50]})
    b = pl.DataFrame({"date": [D3, D2, D1, D5], "ic": [0.02, 0.05, 0.06, -0.90]})
    # common dates D1, D2, D3: differences 0.04, 0.01, 0.06; mean 0.11 / 3
    # deviations 1/300, -8/300, 7/300; population variance (1 + 64 + 49) / 90000 / 3
    out = m.paired_ic_test(a, b, lags=0)
    se = math.sqrt(114 / 90000 / 3 / 3)
    assert out["n_months"] == 3
    assert out["mean_diff"] == pytest.approx(0.11 / 3)
    assert out["nw_tstat"] == pytest.approx(0.11 / 3 / se)
    flipped = m.paired_ic_test(b, a, lags=0)
    assert flipped["mean_diff"] == pytest.approx(-0.11 / 3) and flipped["nw_tstat"] == pytest.approx(-0.11 / 3 / se)


def test_ic_breakdown_by_group():
    rows = []
    for date in (D1, D2, D3):
        for i in range(5):
            rows.append((date, "tech", float(i), float(i)))  # ranked exactly right: IC 1
            rows.append((date, "energy", float(i), float(-i)))  # exactly wrong: IC -1
            rows.append((date, None, float(i), float(i)))  # no sector: left out
    # utilities only on D1, ranks 1..5 vs 2, 1, 4, 3, 5: IC 0.8
    rows += [(D1, "utilities", float(i), t) for i, t in enumerate([0.2, 0.1, 0.4, 0.3, 0.5])]
    df = pl.DataFrame(rows, schema=["date", "sector", "pred", "target"], orient="row")
    out = m.ic_breakdown(df, "pred", "target", "sector", lags=0, min_n=5).sort("sector")
    assert out.columns == ["sector", "mean_ic", "nw_tstat", "n_months"]
    assert out["sector"].to_list() == ["energy", "tech", "utilities"]
    assert out["mean_ic"].to_list() == pytest.approx([-1.0, 1.0, 0.8])
    assert out["n_months"].to_list() == [3, 3, 1]
    assert out["nw_tstat"][0] == -math.inf and out["nw_tstat"][1] == math.inf  # no variation across months
    # pooled across sectors the two cancel: the breakdown is what shows them
    assert m.ic_by_date(df.filter(pl.col("sector").is_in(["tech", "energy"])), "pred", "target")["ic"].to_list() == \
        pytest.approx([0.0] * 3)


# deciles and spreads -------------------------------------------------------------------------------


def two_dates_twenty_stocks() -> pl.DataFrame:
    # date A: predictions 1..20, return 0.01 * i            -> decile k holds i = 2k - 1, 2k: mean 0.01 * (2k - 0.5)
    # date B: predictions 101..120, return 0.5 - 0.02 * i   -> decile k: mean 0.5 - 0.02 * (2k - 0.5)
    i = list(range(1, 21))
    a = xs(D1, [float(v) for v in i], [0.01 * v for v in i], "excess_12m")
    b = xs(D2, [100.0 + v for v in i], [0.5 - 0.02 * v for v in i], "excess_12m")
    return pl.concat([a, b]).sample(fraction=1.0, shuffle=True, seed=7)


def test_decile_table_forms_deciles_per_date_then_averages_dates():
    table = m.decile_table(two_dates_twenty_stocks(), "pred", "excess_12m", n=10)
    assert table.columns == ["decile", "mean_ret", "n"]
    assert table["decile"].to_list() == list(range(1, 11))
    assert table["n"].to_list() == [4] * 10  # 2 stocks per decile per date
    # mean of the two per-date means: (0.01 * (2k - 0.5) + 0.5 - 0.02 * (2k - 0.5)) / 2 = 0.25 - 0.005 * (2k - 0.5)
    assert table["mean_ret"].to_list() == pytest.approx([0.25 - 0.005 * (2 * k - 0.5) for k in range(1, 11)])
    assert table["mean_ret"][0] == pytest.approx(0.2425) and table["mean_ret"][9] == pytest.approx(0.1525)
    # pooling the 40 rows would put date A in deciles 1-5 and date B in 6-10:
    # decile 1 = mean(0.01..0.04) = 0.025, decile 10 = 0.5 - 0.02 * 18.5 = 0.13
    assert abs(table["mean_ret"][0] - 0.025) > 0.2 and abs(table["mean_ret"][9] - 0.13) > 0.02
    assert m.monotonicity(table) == pytest.approx(-1.0)


def test_decile_table_weighs_dates_equally_not_stocks():
    # two buckets. date A, 2 stocks: returns 0.0 | 0.1.  date B, 6 stocks: returns 0.2 x3 | 0.6 x3
    df = pl.concat(
        [
            xs(D1, [1.0, 2], [0.0, 0.1], "ret"),
            xs(D2, [1.0, 2, 3, 4, 5, 6], [0.2, 0.2, 0.2, 0.6, 0.6, 0.6], "ret"),
        ]
    )
    table = m.decile_table(df, "pred", "ret", n=2)
    # bucket 1: (0.0 + 0.2) / 2 = 0.1 (stock-weighted would be 0.6 / 4 = 0.15); bucket 2: (0.1 + 0.6) / 2
    assert table.rows() == [(1, pytest.approx(0.1), 4), (2, pytest.approx(0.35), 4)]
    per_date = m.decile_returns_by_date(df, "pred", "ret", n=2)
    assert per_date.rows() == [
        (D1, 1, pytest.approx(0.0), 1), (D1, 2, pytest.approx(0.1), 1),
        (D2, 1, pytest.approx(0.2), 3), (D2, 2, pytest.approx(0.6), 3),
    ]


def test_buckets_handle_ties_nulls_and_thin_dates():
    df = pl.concat(
        [
            # four tied predictions share average rank 2.5 -> ceil(2.5 * 2 / 4) = bucket 2; none in bucket 1
            xs(D1, [1.0, 1, 1, 1], [0.1, 0.2, 0.3, 0.4], "ret"),
            # the null return is not ranked: 4 rows remain, buckets {1, 2} | {3, 4}
            xs(D2, [1.0, 2, 3, 4, 5], [0.0, 0.2, 0.4, 0.8, None], "ret"),
            # one row cannot fill two buckets: dropped
            xs(D3, [1.0], [9.0], "ret"),
        ]
    )
    per_date = m.decile_returns_by_date(df, "pred", "ret", n=2)
    assert per_date.rows() == [
        (D1, 2, pytest.approx(0.25), 4), (D2, 1, pytest.approx(0.1), 2), (D2, 2, pytest.approx(0.6), 2),
    ]
    # D1 has every prediction tied (a single bucket), so no spread
    assert m.spread_by_date(df, "pred", "ret", n=2).rows() == [(D2, pytest.approx(0.5))]


def tied_end(date: dt.date, top: bool) -> pl.DataFrame:
    """100 stocks, ret = 0.01 * i, the top (or bottom) 25 predictions tied: one tree leaf at an end."""
    i = np.arange(100, dtype=float)
    pred = np.where(i >= 75, 75.0, i) if top else np.where(i < 25, 24.0, i)
    return xs(date, pred.tolist(), (0.01 * i).tolist(), "ret")


def test_spread_survives_a_tie_that_empties_an_end_decile():
    # top tie: ranks 1..75 fill deciles 1-7 (10 each) and 8 (ranks 71-75); the 25 tied share average
    # rank 88 -> ceil(8.8) = decile 9, so decile 10 is empty. Spread = highest populated (9: i 75..99,
    # mean 0.87) minus lowest (1: i 0..9, mean 0.045) = 0.825
    top = tied_end(D1, top=True)
    assert m.decile_returns_by_date(top, "pred", "ret")["decile"].max() == 9
    assert m.spread_by_date(top, "pred", "ret").rows() == [(D1, pytest.approx(0.825))]
    # bottom tie: average rank 13 -> ceil(1.3) = decile 2, decile 1 empty. Spread = decile 10
    # (i 90..99, mean 0.945) minus decile 2 (i 0..24, mean 0.12) = 0.825
    bottom = tied_end(D2, top=False)
    assert m.decile_returns_by_date(bottom, "pred", "ret")["decile"].min() == 2
    assert m.spread_by_date(bottom, "pred", "ret").rows() == [(D2, pytest.approx(0.825))]


def test_evaluate_counts_dropped_spread_dates_and_tie_share():
    every_tied = xs(D3, [1.0] * 20, [0.01 * i for i in range(20)], "ret")
    df = pl.concat([tied_end(D1, top=True), tied_end(D2, top=False), every_tied]).with_columns(
        pl.col("ret").alias("target")
    )
    report = m.evaluate(df, ret="ret", lags=0)
    # D1 and D2 keep a spread of 0.825 each; D3 (all 20 tied) has no spread and is counted, not hidden
    assert report["spread_by_year"].rows() == [(2020, pytest.approx(0.825), 2)]
    assert report["pct_positive_years"] == 1.0
    assert report["n_spread_dates_dropped"] == 1
    # tied predictions: 25 + 25 + 20 of 220 evaluated rows
    assert report["tied_pred_share"] == pytest.approx(70 / 220)
    # a near-perfect model with a tied top group passes criterion 2
    assert statuses(m.check_criteria(report, CRITERIA))["spread_positive_years"] == "pass"


def test_monotonicity_is_spearman_of_decile_and_return():
    # ranks 1, 3, 2, 4, 5 against 1..5: d^2 = 0 + 1 + 1 + 0 + 0, rho = 1 - 6 * 2 / (5 * 24) = 0.9
    table = pl.DataFrame({"decile": [1, 2, 3, 4, 5], "mean_ret": [0.1, 0.3, 0.2, 0.4, 0.5], "n": [9] * 5})
    assert m.monotonicity(table) == pytest.approx(0.9)
    assert math.isnan(m.monotonicity(table.head(1)))


def test_spread_by_date_two_dates_twenty_stocks():
    # date A: 0.195 - 0.015 = 0.18; date B: (0.5 - 0.39) - (0.5 - 0.03) = -0.36
    out = m.spread_by_date(two_dates_twenty_stocks(), "pred", "excess_12m", n=10)
    assert out.rows() == [(D1, pytest.approx(0.18)), (D2, pytest.approx(-0.36))]


def test_spread_by_year_uses_the_prediction_date_year():
    pred = [1.0, 2, 3, 4]
    df = pl.concat(
        [
            xs(d(2019, 12, 31), pred, [0.0, 0.0, 0.1, 0.1], "ret"),  # top - bottom = +0.1, earned during 2020
            xs(d(2020, 1, 31), pred, [0.2, 0.2, 0.0, 0.0], "ret"),  # -0.2
            xs(d(2020, 2, 28), pred, [0.0, 0.0, 0.1, 0.1], "ret"),  # +0.1
        ]
    )
    out = m.spread_by_year(df, "pred", "ret", n=2)
    assert out.columns == ["year", "spread", "n_months"]
    # 2019: the December forecast alone; 2020: (-0.2 + 0.1) / 2.
    # had December 2019 been filed under the year its label ends, 2020 would be (0.1 - 0.2 + 0.1) / 3 = 0.
    assert out.rows() == [(2019, pytest.approx(0.1), 1), (2020, pytest.approx(-0.05), 2)]
    assert m.pct_positive_years(out) == 0.5


def test_pct_positive_years_counts_strictly_positive():
    spreads = pl.DataFrame({"year": [2010, 2011, 2012, 2013], "spread": [0.1, 0.0, -0.2, 0.3]})
    assert m.pct_positive_years(spreads) == 0.5
    assert math.isnan(m.pct_positive_years(spreads.head(0)))


# calibration and coverage --------------------------------------------------------------------------

PROBS = [0.05, 0.15, 0.15, 0.55, 0.6, 0.65, 0.95, 1.0, None, 0.3]
OUTCOMES = [0, 0, 1, 1, 1, 0, 1, 1, 1, None]  # the last two rows are incomplete and not counted


def test_calibration_table_bins_and_max_gap():
    df = pl.DataFrame({"prob": PROBS, "beat": OUTCOMES})
    table = m.calibration_table(df, "prob", "beat", bins=10)
    assert table.columns == ["bin_lo", "bin_hi", "mean_prob", "realized", "n"]
    # [0, 0.1): 0.05 -> 0/1     [0.1, 0.2): 0.15, 0.15 -> 1/2     [0.5, 0.6): 0.55 -> 1/1
    # [0.6, 0.7): 0.6 (lower edge belongs here), 0.65 -> 1/2     [0.9, 1]: 0.95, 1.0 -> 2/2
    assert table["bin_lo"].to_list() == pytest.approx([0.0, 0.1, 0.5, 0.6, 0.9])
    assert table["bin_hi"].to_list() == pytest.approx([0.1, 0.2, 0.6, 0.7, 1.0])
    assert table["mean_prob"].to_list() == pytest.approx([0.05, 0.15, 0.55, 0.625, 0.975])
    assert table["realized"].to_list() == pytest.approx([0.0, 0.5, 1.0, 0.5, 1.0])
    assert table["n"].to_list() == [1, 2, 1, 2, 2]
    # gaps 0.05, 0.35, 0.45, 0.125, 0.025
    assert m.max_calibration_gap(table) == pytest.approx(0.45)
    assert m.max_calibration_gap(table, min_n=2) == pytest.approx(0.35)  # the 0.45 bin holds one forecast
    assert math.isnan(m.max_calibration_gap(table, min_n=3))
    as_bool = df.with_columns(pl.col("beat").cast(pl.Boolean))
    assert m.calibration_table(as_bool, "prob", "beat", bins=10).equals(table)


def test_calibration_table_other_bin_count():
    # two bins: [0, 0.5) holds 0.05, 0.15, 0.15 (1 of 3 beat); [0.5, 1] holds the other five (4 of 5)
    table = m.calibration_table(pl.DataFrame({"prob": PROBS, "beat": OUTCOMES}), "prob", "beat", bins=2)
    assert table["n"].to_list() == [3, 5]
    assert table["mean_prob"].to_list() == pytest.approx([0.35 / 3, (0.55 + 0.6 + 0.65 + 0.95 + 1.0) / 5])
    assert table["realized"].to_list() == pytest.approx([1 / 3, 4 / 5])


def test_interval_coverage_excludes_nulls():
    df = pl.DataFrame(
        {
            "q10": [-1.0, -1.0, -1.0, None, -1.0, 0.0, -1.0],
            "q90": [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, None],
            "actual": [0.0, 2.0, 1.0, 0.0, None, -0.2, 0.0],
        }
    )
    # rows 1-3 and 6 are complete: inside, above, on the upper bound (counts as inside), below -> 2 / 4
    assert m.interval_coverage(df, "q10", "q90", "actual") == 0.5
    # counting the three incomplete rows as misses would give 2 / 7
    assert math.isnan(m.interval_coverage(df.filter(pl.col("q10").is_null()), "q10", "q90", "actual"))


# breakdown keys ------------------------------------------------------------------------------------


def test_size_terciles_are_formed_per_date():
    df = pl.DataFrame(
        {
            "date": [D1] * 6 + [D2] * 7,
            "security_id": list(range(6)) + list(range(7)),
            "market_cap": [10.0, 20, 30, 40, 50, 60] + [6000.0, 5000, 4000, 3000, 2000, 1000, None],
        }
    )
    out = m.size_terciles(df, "market_cap")
    assert out.columns == ["date", "security_id", "market_cap", "size_tercile"]
    # pooled thirds would call every D1 stock small and every D2 stock large
    assert out["size_tercile"].to_list() == (
        ["small", "small", "mid", "mid", "large", "large"] + ["large", "large", "mid", "mid", "small", "small", None]
    )


# daily benchmark returns in units of 0.001; day-over-day |change| = 4, 5, 3, 30, 2, 1, 50, 60, 6
BENCH_RET = [10, 14, 9, 12, -18, -20, -21, 29, 89, 83]
BENCH_DAYS = business_days(d(2020, 1, 1), d(2020, 1, 14)).to_list()  # Jan 1, 2, 3, 6, 7, 8, 9, 10, 13, 14


def small_benchmark() -> pl.DataFrame:
    return pl.DataFrame({"date": BENCH_DAYS, "bench_ret": [r / 1000 for r in BENCH_RET]})


def test_market_regimes_hand_computed():
    # windows: vol = std of the last 2 returns = |change| / sqrt(2); return over the last 3 days;
    # high_vol once 3 vols exist and today's is above the 2/3 quantile of all vols so far.
    # day  |change|  vols so far (sorted)        2/3 quantile   high?   3-day return        regime
    #  0-2                                                               (no 3 earlier days)  null
    #  3      3      3 4 5                       4.33           no      +14 +9 +12  > 0      bull
    #  4     30      3 4 5 30                    5              yes                           high_vol
    #  5      2      2 3 4 5 30                  4.67           no      +12 -18 -20 < 0      bear
    #  6      1      1 2 3 4 5 30                4.33           no      all negative         bear
    #  7     50      1 2 3 4 5 30 50             5              yes                           high_vol
    #  8     60      1 2 3 4 5 30 50 60          21.7           yes                           high_vol
    #  9      6      1 2 3 4 5 6 30 50 60        14             no      +29 +89 +83 > 0      bull
    kw = {"vol_window": 2, "ret_window": 3, "min_history": 3}
    expected = [None, None, None, "bull", "high_vol", "bear", "bear", "high_vol", "high_vol", "bull"]
    out = m.market_regimes(small_benchmark(), BENCH_DAYS, **kw)
    assert out.columns == ["date", "regime"]
    assert out["date"].to_list() == BENCH_DAYS
    assert out["regime"].to_list() == expected
    assert set(expected) - {None} == set(m.REGIMES)

    # day 4 with a 5-vol minimum history: only 4 vols exist, so it falls back to the sign of
    # 1.009 * 1.012 * 0.982 - 1 = +0.0027 -> bull. Day 5 has its 5 vols and is unchanged.
    longer = m.market_regimes(small_benchmark(), BENCH_DAYS, vol_window=2, ret_window=3, min_history=5)
    assert longer["regime"].to_list() == [None, None, None, "bull", "bull", "bear", "bear", "high_vol", "high_vol", "bull"]

    # non-trading dates take the last trading day on or BEFORE them; nothing within 7 days -> null.
    # Sunday Jan 5 reads Friday Jan 3 (null), not Monday Jan 6 (bull).
    asked = [d(2020, 1, 18), d(2020, 1, 5), d(2020, 1, 12), d(2020, 2, 20), d(2020, 1, 12)]
    odd = m.market_regimes(small_benchmark(), asked, **kw)
    assert odd.rows() == [
        (d(2020, 1, 5), None), (d(2020, 1, 12), "high_vol"), (d(2020, 1, 18), "bull"), (d(2020, 2, 20), None),
    ]

    # days before the index has any member are not history
    padded = pl.concat(
        [
            pl.DataFrame({"date": [d(2019, 12, 27), d(2019, 12, 30), d(2019, 12, 31)], "bench_ret": [0.0] * 3,
                          "n_members": [0] * 3}),
            small_benchmark().with_columns(pl.lit(5, dtype=pl.Int64).alias("n_members")),
        ]
    )
    assert m.market_regimes(padded.drop("n_members"), BENCH_DAYS, **kw)["regime"].to_list() != expected
    assert m.market_regimes(padded, BENCH_DAYS, **kw)["regime"].to_list() == expected


def test_market_regimes_small_case_ignores_future_rows():
    kw = {"vol_window": 2, "ret_window": 3, "min_history": 3}
    full = m.market_regimes(small_benchmark(), BENCH_DAYS, **kw)
    for k in range(1, 11):
        upto = m.market_regimes(small_benchmark().head(k), BENCH_DAYS[:k], **kw)
        assert upto.equals(full.head(k)), f"regime of the first {k} days changed when later days were added"


def test_market_regimes_default_windows_use_only_the_past():
    # calm rising market, a crash, a calm falling market, a wild stretch, calm again
    rng = np.random.default_rng(5)
    days = business_days(d(2015, 1, 1), d(2019, 12, 31))
    seg = days.len() // 5
    drift = np.repeat([0.0008, -0.003, -0.0006, 0.0005, 0.0008], [seg, seg, seg, seg, days.len() - 4 * seg])
    scale = np.repeat([0.006, 0.02, 0.006, 0.025, 0.005], [seg, seg, seg, seg, days.len() - 4 * seg])
    bench = pl.DataFrame({"date": days, "bench_ret": drift + scale * rng.standard_normal(days.len())})
    dates = bench.group_by_dynamic("date", every="1mo").agg(pl.col("date").last().alias("d"))["d"].to_list()

    full = m.market_regimes(bench, dates)
    assert full["date"].to_list() == dates
    assert set(full["regime"].drop_nulls().unique()) == set(m.REGIMES)
    # 2015 has 261 weekdays: Nov 30 is the 238th (237 earlier days, no 252-day return yet), Dec 31 the 261st
    assert full["regime"].head(11).null_count() == 11
    assert full["regime"].tail(-11).null_count() == 0

    # 1. each date's regime is the same when the benchmark stops at that date
    for cut in dates[14::6]:
        upto = m.market_regimes(bench.filter(pl.col("date") <= cut), [x for x in dates if x <= cut])
        assert upto.equals(full.filter(pl.col("date") <= cut)), f"regimes before {cut} depend on later rows"

    # 2. appending a violent future does not rewrite the past
    future = business_days(d(2020, 1, 1), d(2021, 12, 31))
    shock = pl.DataFrame({"date": future, "bench_ret": 0.08 * rng.standard_normal(future.len()) - 0.01})
    extended = m.market_regimes(pl.concat([bench, shock]), dates + [future[-1]])
    assert extended.head(len(dates)).equals(full)
    assert extended["regime"][-1] == "high_vol"


# deflated sharpe -----------------------------------------------------------------------------------

EULER = 0.5772156649015329


def test_expected_max_sharpe_formula():
    # N = 10, V = 1: (1 - g) * Phi^-1(0.9) + g * Phi^-1(1 - 1 / (10 e))
    by_hand = (1 - EULER) * norm.ppf(0.9) + EULER * norm.ppf(1 - 1 / (10 * math.e))
    assert by_hand == pytest.approx(1.5745, abs=5e-4)
    assert m.expected_max_sharpe(10, 1.0) == pytest.approx(by_hand, rel=1e-12)
    # the approximation is close to the exact expected maximum of 10 standard normals, 1.5388
    assert m.expected_max_sharpe(10, 1.0) == pytest.approx(1.5388, abs=0.05)
    # scales with the standard deviation of the trial Sharpe ratios, grows with the number of trials
    assert m.expected_max_sharpe(10, 4.0) == pytest.approx(2 * by_hand)
    values = [m.expected_max_sharpe(n, 1.0) for n in (1, 2, 10, 100, 1000)]
    assert values[0] == 0.0 and values == sorted(values) and len(set(values)) == 5
    with pytest.raises(ValueError):
        m.expected_max_sharpe(0, 1.0)


def test_deflated_sharpe_reproduces_bailey_lopez_de_prado_2014():
    # the paper's numerical example: annualized SR 2.5 from T = 1250 daily returns (250 a year), the best
    # of N = 100 trials whose annualized Sharpe ratios have variance 1/2, skewness -3, kurtosis 10.
    # The paper reports SR0 = 0.1132 (per day) and DSR = 0.9004.
    sr = 2.5 / math.sqrt(250)
    var_sr = 0.5 / 250
    assert m.expected_max_sharpe(100, var_sr) == pytest.approx(0.1132, abs=5e-5)
    assert m.deflated_sharpe(sr, 1250, 100, var_sr, skew=-3, kurt=10) == pytest.approx(0.9004, abs=5e-5)

    # the same numbers recomputed here with scipy
    sr0 = math.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1 / 100) + EULER * norm.ppf(1 - 1 / (100 * math.e)))
    dsr = norm.cdf((sr - sr0) * math.sqrt(1250 - 1) / math.sqrt(1 - (-3) * sr + (10 - 1) / 4 * sr**2))
    assert sr0 == pytest.approx(0.113172, abs=1e-6)
    assert dsr == pytest.approx(0.900397, abs=1e-6)
    assert m.deflated_sharpe(sr, 1250, 100, var_sr, skew=-3, kurt=10) == pytest.approx(dsr, rel=1e-12)


def test_deflated_sharpe_single_trial_and_monotonicity():
    # one trial, normal returns: SR0 = 0 and DSR = Phi(SR * sqrt(T - 1) / sqrt(1 + SR^2 / 2))
    # SR 0.1, T = 101: Phi(0.1 * 10 / sqrt(1.005)) = Phi(0.99751) = 0.84074
    assert m.deflated_sharpe(0.1, 101, 1, 0.01) == pytest.approx(norm.cdf(1 / math.sqrt(1.005)), rel=1e-12)
    assert m.deflated_sharpe(0.1, 101, 1, 0.01) == pytest.approx(0.84074, abs=1e-5)
    # the Sharpe ratio that equals the expected maximum is a coin flip
    assert m.deflated_sharpe(m.expected_max_sharpe(50, 0.01), 500, 50, 0.01) == pytest.approx(0.5)

    sr, t, v = 2.5 / math.sqrt(250), 1250, 0.5 / 250
    by_trials = [m.deflated_sharpe(sr, t, n, v, -3, 10) for n in (1, 10, 100, 1000, 10000)]
    assert by_trials == sorted(by_trials, reverse=True) and len(set(by_trials)) == 5  # more trials, lower DSR
    assert by_trials[0] > 0.99 and by_trials[-1] < 0.5
    assert m.deflated_sharpe(sr, t, 100, 4 * v, -3, 10) < m.deflated_sharpe(sr, t, 100, v, -3, 10)  # more dispersed trials
    assert m.deflated_sharpe(sr, 2 * t, 100, v, -3, 10) > m.deflated_sharpe(sr, t, 100, v, -3, 10)  # longer track record
    assert m.deflated_sharpe(sr, t, 100, v, -3, 10) < m.deflated_sharpe(sr, t, 100, v)  # negative skew, fat tails
    assert math.isnan(m.deflated_sharpe(sr, 1, 100, v))


# report and criteria -------------------------------------------------------------------------------


def oof_frame() -> pl.DataFrame:
    """24 month ends x 20 stocks. pred = stock number i (0..19) every month.
    2019: excess = 0.01 * i in all 12 months (ranked exactly right).
    2020: right in Jan-Apr, exactly inverted (excess = -0.01 * i) in May-Dec.
    One extra stock per month has a prediction but no label yet."""
    month_ends = pl.date_range(d(2019, 1, 1), d(2020, 12, 1), "1mo", eager=True).dt.month_end().to_list()
    rows = []
    for date in month_ends:
        sign = 1.0 if (date.year == 2019 or date.month <= 4) else -1.0
        for i in range(20):
            rows.append((date, i, float(i), (i + 0.5) / 20 - 0.5 if sign > 0 else 0.5 - (i + 0.5) / 20, sign * 0.01 * i))
        rows.append((date, 99, 5.0, None, None))
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "security_id": pl.Int64, "pred": pl.Float64, "target": pl.Float64,
                "excess_12m": pl.Float64},
        orient="row",
    )


def test_evaluate_report_hand_computed():
    report = m.evaluate(oof_frame(), lags=0)
    # IC is +1 in 16 months and -1 in 8: mean 1/3, population variance 1 - 1/9, se = sqrt(8/9 / 24) = sqrt(1/27)
    ic = report["ic"]
    assert ic["mean_ic"] == pytest.approx(1 / 3)
    assert ic["nw_tstat"] == pytest.approx(math.sqrt(3))
    assert ic["n_months"] == 24 and ic["pct_positive"] == pytest.approx(16 / 24)
    assert report["ic_by_date"].columns == ["date", "ic", "n"]
    assert report["ic_by_date"]["n"].to_list() == [20] * 24  # the unlabeled stock is not a pair
    assert report["n_obs"] == 480 and report["n_dates"] == 24
    # decile k holds stocks 2k - 2 and 2k - 1: +-0.01 * (2k - 1.5); averaged over dates: (16 - 8) / 24 of it
    assert report["deciles"]["mean_ret"].to_list() == pytest.approx([0.01 * (2 * k - 1.5) / 3 for k in range(1, 11)])
    assert report["deciles"]["n"].to_list() == [48] * 10
    assert report["monotonicity"] == pytest.approx(1.0)
    # monthly spread = +-(0.185 - 0.005) = +-0.18; 2019: 0.18; 2020: (4 - 8) / 12 * 0.18 = -0.06
    assert report["spread_by_year"].rows() == [(2019, pytest.approx(0.18), 12), (2020, pytest.approx(-0.06), 12)]
    assert report["pct_positive_years"] == 0.5
    assert report["mean_spread"] == pytest.approx(0.18 * (16 - 8) / 24)
    # the default 12 lags see the 8-month losing streak as persistent: a smaller t-stat
    assert 0 < m.evaluate(oof_frame())["ic"]["nw_tstat"] < math.sqrt(3)


def test_evaluate_matches_its_parts_and_takes_column_names():
    df = oof_frame().rename({"pred": "score", "target": "rank_12m", "excess_12m": "excess_6m"})
    report = m.evaluate(df, pred="score", target="rank_12m", ret="excess_6m", lags=3, n_deciles=5)
    assert report["ic_by_date"].equals(m.ic_by_date(df, "score", "rank_12m"))
    assert report["ic"] == m.ic_summary(report["ic_by_date"], 3)
    assert report["deciles"].equals(m.decile_table(df, "score", "excess_6m", 5))
    assert report["spread_by_year"].equals(m.spread_by_year(df, "score", "excess_6m", 5))


def test_metrics_are_bit_reproducible():
    rng = np.random.default_rng(2)
    n_dates, n_stocks = 60, 500
    pred = rng.normal(size=n_dates * n_stocks)
    df = pl.DataFrame(
        {
            "date": np.repeat(np.arange(n_dates), n_stocks),
            "pred": pred,
            "target": 0.1 * pred + rng.normal(size=pred.size),
            "prob": rng.uniform(size=pred.size),
        }
    ).with_columns((pl.col("target") > 0).alias("beat"))
    for _ in range(4):
        assert m.ic_by_date(df, "pred", "target").equals(m.ic_by_date(df, "pred", "target"))
        assert m.decile_table(df, "pred", "target").equals(m.decile_table(df, "pred", "target"))
        assert m.spread_by_date(df, "pred", "target").equals(m.spread_by_date(df, "pred", "target"))
        assert m.calibration_table(df, "prob", "beat").equals(m.calibration_table(df, "prob", "beat"))


def test_to_jsonable_report_round_trips_through_json():
    report = m.evaluate(oof_frame(), lags=0)
    plain = m.to_jsonable(report)
    back = json.loads(json.dumps(plain, allow_nan=False))
    assert back == plain
    assert set(back) == set(report)
    assert back["ic"]["n_months"] == 24
    assert back["ic_by_date"][0] == {"date": "2019-01-31", "ic": pytest.approx(1.0), "n": 20}
    assert back["spread_by_year"][1] == {"year": 2020, "spread": pytest.approx(-0.06), "n_months": 12}
    assert len(back["deciles"]) == 10 and back["deciles"][0]["decile"] == 1
    # the frames also go straight to CSV
    assert report["deciles"].write_csv().splitlines()[0] == "decile,mean_ret,n"


def test_to_jsonable_scalars():
    obj = {
        "nan": math.nan, "inf": -math.inf, "np": np.float64(0.25), "int": np.int64(3), "date": d(2020, 1, 31),
        "arr": np.array([1.0, math.nan]), "tuple": (1, "a"), "series": pl.Series([1, None]),
        "check": CheckResult("rank_ic", "pass", "ok"), 2020: True, "none": None,
    }
    assert m.to_jsonable(obj) == {
        "nan": None, "inf": None, "np": 0.25, "int": 3, "date": "2020-01-31", "arr": [1.0, None],
        "tuple": [1, "a"], "series": [1, None],
        "check": {"name": "rank_ic", "status": "pass", "detail": "ok"}, "2020": True, "none": None,
    }
    json.dumps(m.to_jsonable(obj), allow_nan=False)


CRITERIA = SimpleNamespace(
    min_rank_ic=0.04,
    min_ic_tstat=2.0,
    min_positive_spread_years=0.65,
    coverage_range=(0.75, 0.85),
    calibration_tolerance=0.05,
    tpsl_prob_tolerance=0.05,
)


def fake_report(mean_ic: float, tstat: float, positive_years: int, years: int = 20) -> dict:
    spreads = pl.DataFrame(
        {"year": list(range(2000, 2000 + years)), "spread": [0.05] * positive_years + [-0.05] * (years - positive_years)}
    )
    return {
        "ic": {"mean_ic": mean_ic, "nw_tstat": tstat, "n_months": 12 * years},
        "spread_by_year": spreads,
        "pct_positive_years": m.pct_positive_years(spreads),
    }


def statuses(results: list[CheckResult]) -> dict[str, str]:
    return {r.name: r.status for r in results}


def test_check_criteria_one_and_two():
    out = m.check_criteria(fake_report(0.05, 2.5, 15), CRITERIA)
    assert all(isinstance(r, CheckResult) for r in out)
    assert [r.name for r in out] == ["rank_ic", "spread_positive_years", "beats_baseline", "interval_coverage", "calibration"]
    assert statuses(out) == {
        "rank_ic": "pass", "spread_positive_years": "pass",
        "beats_baseline": "skip", "interval_coverage": "skip", "calibration": "skip",
    }
    assert "0.0500" in out[0].detail and "2.50" in out[0].detail and "75%" in out[1].detail

    # exactly at the thresholds passes: IC 0.04, t 2.0, 13 of 20 years = 65%
    assert statuses(m.check_criteria(fake_report(0.04, 2.0, 13), CRITERIA))["rank_ic"] == "pass"
    assert statuses(m.check_criteria(fake_report(0.04, 2.0, 13), CRITERIA))["spread_positive_years"] == "pass"
    # criterion 1 needs BOTH the level and the t-stat
    assert statuses(m.check_criteria(fake_report(0.039, 5.0, 15), CRITERIA))["rank_ic"] == "fail"
    assert statuses(m.check_criteria(fake_report(0.08, 1.99, 15), CRITERIA))["rank_ic"] == "fail"
    assert statuses(m.check_criteria(fake_report(-0.08, -5.0, 15), CRITERIA))["rank_ic"] == "fail"
    # 12 of 20 years = 60% < 65%
    assert statuses(m.check_criteria(fake_report(0.05, 2.5, 12), CRITERIA))["spread_positive_years"] == "fail"
    # an undefined statistic fails rather than passes or skips
    nan_report = fake_report(math.nan, math.nan, 0, years=0)
    assert statuses(m.check_criteria(nan_report, CRITERIA))["rank_ic"] == "fail"
    assert statuses(m.check_criteria(nan_report, CRITERIA))["spread_positive_years"] == "fail"


def test_check_criteria_three_to_five():
    report = fake_report(0.05, 2.5, 15)

    def run(**kw) -> dict[str, str]:
        return statuses(m.check_criteria(report, CRITERIA, **kw))

    # 3: IC gain over the baseline, positive with a Newey-West t >= 2
    assert run(vs_baseline={"mean_diff": 0.010, "nw_tstat": 2.4, "n_months": 240})["beats_baseline"] == "pass"
    assert run(vs_baseline={"mean_diff": 0.010, "nw_tstat": 1.2, "n_months": 240})["beats_baseline"] == "fail"
    assert run(vs_baseline={"mean_diff": -0.010, "nw_tstat": -2.4, "n_months": 240})["beats_baseline"] == "fail"
    # 4: the 10-90% range should hold 75-85% of outcomes (bounds included)
    assert [run(coverage=c)["interval_coverage"] for c in (0.74, 0.75, 0.80, 0.85, 0.86, math.nan)] == \
        ["fail", "pass", "pass", "pass", "fail", "fail"]
    # 5: the largest calibration gap may be at most 5 points
    assert [run(calibration_gap=g)["calibration"] for g in (0.0, 0.05, 0.051, math.nan)] == \
        ["pass", "pass", "fail", "fail"]
    # supplying one input leaves the others skipped and criteria 1-2 untouched
    assert run(coverage=0.8) == {
        "rank_ic": "pass", "spread_positive_years": "pass",
        "beats_baseline": "skip", "interval_coverage": "pass", "calibration": "skip",
    }


def test_check_criteria_end_to_end_on_an_evaluated_frame():
    oof = oof_frame()
    report = m.evaluate(oof, lags=0)
    # mean IC 1/3 clears 0.04 but t = sqrt(3) = 1.73 < 2; spread positive in 1 of 2 years < 65%
    out = statuses(m.check_criteria(report, CRITERIA))
    assert out["rank_ic"] == "fail" and out["spread_positive_years"] == "fail"

    # a baseline that predicts the opposite has IC -1/+1 where the model has +1/-1:
    # difference +2 in 16 months, -2 in 8; mean 2/3
    baseline = oof.with_columns((-pl.col("pred")).alias("pred"))
    paired = m.paired_ic_test(report["ic_by_date"], m.ic_by_date(baseline, "pred", "target"), lags=0)
    assert paired["mean_diff"] == pytest.approx(2 / 3) and paired["nw_tstat"] == pytest.approx(math.sqrt(3))
    assert statuses(m.check_criteria(report, CRITERIA, vs_baseline=paired))["beats_baseline"] == "fail"

    # a real Criteria dataclass works the same way, when the config module is importable
    modelconfig = pytest.importorskip("ltforecast.modelconfig")
    real = modelconfig.Criteria(**vars(CRITERIA))
    assert statuses(m.check_criteria(report, real)) == out

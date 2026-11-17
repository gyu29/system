"""Walk-forward isotonic probability and conformal ranges: the purge rule, with literal dates.

Three folds, test years 2010, 2011, 2012 (three month ends each). Four securities per date with
pred = security_id, so the per-date score is 0.125, 0.375, 0.625, 0.875 for securities 1..4.
"""

import datetime as dt
import math
import pickle

import numpy as np
import polars as pl
import pytest

from ltforecast.calibrate import (
    IsotonicCalibrator,
    conformal_offset,
    cqr_scores,
    fit_isotonic,
    latest_conformal_offset,
    walk_forward_conformal,
    walk_forward_isotonic,
)
from ltforecast.validation.splitter import Fold

d = dt.date
A_DATES = (d(2010, 1, 29), d(2010, 6, 30), d(2010, 12, 31))
B_DATES = (d(2011, 1, 31), d(2011, 6, 30), d(2011, 12, 30))
C_DATES = (d(2012, 1, 31), d(2012, 6, 29), d(2012, 12, 31))
FOLDS = [Fold("2010", (), A_DATES), Fold("2011", (), B_DATES), Fold("2012", (), C_DATES)]
SECS = (1, 2, 3, 4)
UP = {1: -0.1, 2: -0.1, 3: 0.1, 4: 0.1}  # high score -> beat the market
DOWN = {s: -v for s, v in UP.items()}  # the opposite relation


def oof_pred(dates) -> pl.DataFrame:
    return pl.DataFrame(
        {"date": [t for t in dates for _ in SECS], "security_id": [s for _ in dates for s in SECS],
         "pred": [float(s) for _ in dates for s in SECS]},
        schema={"date": pl.Date, "security_id": pl.Int64, "pred": pl.Float64},
    )


def labels_for(spec) -> pl.DataFrame:
    """spec: list of (date, label_end, {sid: excess})."""
    rows = [(t, s, ex[s], end) for t, end, ex in spec for s in SECS]
    return pl.DataFrame(
        rows, schema={"date": pl.Date, "security_id": pl.Int64, "excess_12m": pl.Float64, "label_end_12m": pl.Date},
        orient="row",
    )


def probs_on(out: pl.DataFrame, day: dt.date) -> list[float]:
    return out.filter(pl.col("date") == day).sort("security_id")["prob"].to_list()


# -- isotonic ----------------------------------------------------------------------------------------


def base_spec():
    """A: matured before 2011 (UP). B: label ends 2012-01-31 .. 2012-12-31, so NOT matured on C's first
    test date 2012-01-31 (the first is equal, i.e. still not before it), with the opposite relation.
    C: adversarial early label ends (DOWN) that must never be used, C being the fold itself."""
    return (
        [(t, d(2010, 12, 30), UP) for t in A_DATES]
        + [(B_DATES[0], d(2012, 1, 31), DOWN), (B_DATES[1], d(2012, 6, 29), DOWN), (B_DATES[2], d(2012, 12, 31), DOWN)]
        + [(t, d(2009, 1, 30), DOWN) for t in C_DATES]
    )


def test_immature_rows_of_earlier_folds_are_excluded():
    oof = oof_pred(A_DATES + B_DATES + C_DATES)
    out = walk_forward_isotonic(oof, labels_for(base_spec()), FOLDS, min_rows=4, min_dates=1)
    assert out.columns == ["date", "security_id", "prob", "prob_calibrated"]
    assert out.schema["prob"] == pl.Float64 and out.schema["prob_calibrated"] == pl.Boolean
    # fold 2010 has no earlier fold
    a = out.filter(pl.col("date").is_in(list(A_DATES)))
    assert a["prob"].to_list() == [0.5] * 12 and not a["prob_calibrated"].any()
    # folds 2011 and 2012 see only fold 2010's rows: the UP step
    for day in B_DATES + C_DATES:
        assert probs_on(out, day) == [0.0, 0.0, 1.0, 1.0]
    assert out.filter(pl.col("date") >= B_DATES[0])["prob_calibrated"].all()
    # the test is meaningful: with fold 2011's not-yet-matured rows the map would be flat at 0.5
    score = np.tile([0.125, 0.375, 0.625, 0.875], 6)
    y = np.array([0, 0, 1, 1] * 3 + [1, 1, 0, 0] * 3, dtype=float)
    assert IsotonicCalibrator().fit(score, y).predict(np.array([0.125, 0.875])).tolist() == [0.5, 0.5]


def test_a_row_that_matured_one_day_before_the_fold_is_used():
    spec = base_spec()
    spec[3] = (B_DATES[0], d(2012, 1, 30), DOWN)  # label ended the day before 2012-01-31
    out = walk_forward_isotonic(oof_pred(A_DATES + B_DATES + C_DATES), labels_for(spec), FOLDS, min_rows=4, min_dates=1)
    # 3 UP dates + 1 DOWN date: P = 3/4 for the high scores, 1/4 for the low ones
    assert probs_on(out, C_DATES[0]) == [0.25, 0.25, 0.75, 0.75]
    assert probs_on(out, B_DATES[0]) == [0.0, 0.0, 1.0, 1.0]  # fold 2011 itself is unaffected


def test_fold_order_not_dates_alone_decides_earlier():
    # fold 2010's own rows have label ends before its first test date: still never used for fold 2010
    spec = [(t, d(2009, 1, 30), UP) for t in A_DATES]
    out = walk_forward_isotonic(oof_pred(A_DATES), labels_for(spec), FOLDS, min_rows=1, min_dates=1)
    assert out["prob"].to_list() == [0.5] * 12 and not out["prob_calibrated"].any()


def test_min_rows_fallback():
    out = walk_forward_isotonic(oof_pred(A_DATES + B_DATES + C_DATES), labels_for(base_spec()), FOLDS)
    assert out["prob"].to_list() == [0.5] * 36 and not out["prob_calibrated"].any()
    out = walk_forward_isotonic(oof_pred(A_DATES + B_DATES + C_DATES), labels_for(base_spec()), FOLDS, min_rows=13, min_dates=1)
    assert not out["prob_calibrated"].any()  # 12 usable rows at most


def test_missing_labels_are_scored_not_fitted():
    spec = [s for s in base_spec() if s[0] != A_DATES[0]]  # no labels at all for one fold-2010 date
    lab = labels_for(spec).with_columns(
        pl.when(pl.col("date") == A_DATES[1]).then(None).otherwise(pl.col("excess_12m")).alias("excess_12m")
    )
    out = walk_forward_isotonic(oof_pred(A_DATES + B_DATES + C_DATES), lab, FOLDS, min_rows=4, min_dates=1)
    assert out.height == 36
    assert probs_on(out, C_DATES[0]) == [0.0, 0.0, 1.0, 1.0]  # from A_DATES[2] alone
    out = walk_forward_isotonic(oof_pred(A_DATES + B_DATES + C_DATES), lab, FOLDS, min_rows=5, min_dates=1)
    assert not out["prob_calibrated"].any()


def test_oof_date_outside_folds_raises():
    with pytest.raises(ValueError, match="no fold"):
        walk_forward_isotonic(oof_pred((d(2013, 1, 31),)), labels_for(base_spec()), FOLDS, min_rows=1, min_dates=1)


def test_isotonic_monotone_and_bounded():
    rng = np.random.default_rng(0)
    score = rng.uniform(0, 1, 3000)
    y = (rng.uniform(0, 1, 3000) < 0.2 + 0.6 * score).astype(float)
    cal = IsotonicCalibrator().fit(score, y)
    grid = np.linspace(-1.0, 2.0, 601)
    p = cal.predict(grid)
    assert np.all(np.diff(p) >= 0)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert p[0] == cal.predict(np.array([score.min()]))[0]  # clipped outside the fitted range
    assert pickle.loads(pickle.dumps(cal)).predict(grid).tolist() == p.tolist()
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().predict(grid)


def test_fit_isotonic_final_uses_only_matured_rows():
    oof = oof_pred(A_DATES + B_DATES + C_DATES)
    cal = fit_isotonic(oof, labels_for(base_spec()), before=d(2012, 1, 31), min_rows=4, min_dates=1)
    # A (UP, 12 rows) + C (DOWN, bogus early ends, 12 rows): no fold structure here, only label end
    assert cal.n_obs == 24
    assert fit_isotonic(oof, labels_for(base_spec()), before=d(2012, 1, 31), min_rows=25, min_dates=1) is None


# -- conformal ---------------------------------------------------------------------------------------


def test_conformal_offset_rank_by_hand():
    scores = np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6])  # n = 9
    # ceil((9 + 1) * (1 - 0.2)) = 8 -> 8th smallest = 0.8
    assert conformal_offset(scores, 0.2) == 0.8
    # ceil(10 * 0.9) = 9 -> the largest
    assert conformal_offset(scores, 0.1) == 0.9
    # ceil(10 * 0.95) = 10 > 9 -> too few
    assert conformal_offset(scores, 0.05) == math.inf
    assert conformal_offset(np.array([0.1, 0.2, 0.3]), 0.2) == math.inf  # ceil(3.2) = 4 > 3
    assert conformal_offset(np.array([]), 0.2) == math.inf
    assert conformal_offset(np.append(scores, [np.nan, np.inf]), 0.2) == 0.8  # non-finite ignored
    with pytest.raises(ValueError):
        conformal_offset(scores, 1.0)


def test_cqr_scores():
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    assert cqr_scores(lo, hi, np.array([-0.5, 0.25, 1.5])).tolist() == [0.5, -0.25, 0.5]


def toy_q(n_per_date: int, seed: int):
    """Twelve month ends per fold year, y ~ N(0, sigma) with sigma varying by stock; raw range = +-0.5 sigma
    (covers ~38%). Labels end 12 dates later (matured for the fold two years on, not the next one)."""
    rng = np.random.default_rng(seed)
    dates = [d(y, m, 28) for y in (2010, 2011, 2012) for m in range(1, 13)]
    folds = [Fold(str(y), (), tuple(t for t in dates if t.year == y)) for y in (2010, 2011, 2012)]
    n = len(dates) * n_per_date
    sigma = rng.uniform(0.1, 0.5, n)
    y = rng.normal(0.0, sigma)
    date_col = np.repeat(np.array(dates, dtype="datetime64[D]"), n_per_date)
    q = pl.DataFrame({
        "date": date_col, "security_id": np.tile(np.arange(n_per_date), len(dates)),
        "q10": -0.5 * sigma, "q50": np.zeros(n), "q90": 0.5 * sigma,
    }).with_columns(pl.col("date").cast(pl.Date))
    lab = q.select("date", "security_id").with_columns(
        pl.Series("excess_12m", y), pl.col("date").dt.offset_by("1y").dt.offset_by("-1d").alias("label_end_12m")
    )
    return q, lab, folds


def test_cqr_widens_undercovering_ranges_to_target():
    q, lab, folds = toy_q(400, seed=7)
    out = walk_forward_conformal(q, lab, folds, alpha=0.2, window_months=36, min_rows=1)
    assert out.columns == ["date", "security_id", "q10", "q50", "q90", "conformal_offset", "conformal_ok"]
    j = out.join(lab, on=["date", "security_id"])
    raw = q.join(lab, on=["date", "security_id"])
    cover = lambda f: f.select(pl.col("excess_12m").is_between(pl.col("q10"), pl.col("q90")).mean()).item()  # noqa: E731
    # fold 2011's first date 2011-01-28: fold 2010 labels end 2011-01-27 .. 2011-12-27, only the first
    # matured: one date (400 rows) < min_dates=12, so no offset
    f11 = j.filter(pl.col("date").dt.year() == 2011)
    assert not f11["conformal_ok"].any() and (f11["conformal_offset"] == 0).all()
    # fold 2012: all twelve fold-2010 dates plus fold 2011's first date (ends 2012-01-27 < 2012-01-28)
    f12 = j.filter(pl.col("date").dt.year() == 2012)
    assert f12["conformal_ok"].all() and f12["conformal_offset"][0] > 0
    assert cover(raw.filter(pl.col("date").dt.year() == 2012)) < 0.5
    assert abs(cover(f12) - 0.8) < 0.03
    # fold 2010 has nothing earlier
    f10 = j.filter(pl.col("date").dt.year() == 2010)
    assert not f10["conformal_ok"].any() and (f10["conformal_offset"] == 0).all()
    assert f10["q10"].to_list() == raw.filter(pl.col("date").dt.year() == 2010)["q10"].to_list()


def test_fold_offset_matches_hand_selected_rows():
    q, lab, folds = toy_q(50, seed=3)
    out = walk_forward_conformal(q, lab, folds, alpha=0.2, window_months=36, min_rows=1)
    raw = q.join(lab, on=["date", "security_id"])
    use = raw.filter(
        (pl.col("date") < d(2012, 1, 1)) & (pl.col("label_end_12m") < d(2012, 1, 28))
    )
    assert use["date"].n_unique() == 13  # twelve 2010 dates + 2011-01-28 (ends 2012-01-27)
    want = conformal_offset(cqr_scores(use["q10"].to_numpy(), use["q90"].to_numpy(), use["excess_12m"].to_numpy()), 0.2)
    assert out.filter(pl.col("date") == d(2012, 6, 28))["conformal_offset"][0] == pytest.approx(want)


def window_case():
    """Fold 2010: six dates, all ranges [0, 0]. Old dates miss by 100, the last two by 1..5 and 6..10."""
    dates = (d(2010, 1, 29), d(2010, 2, 26), d(2010, 3, 31), d(2010, 4, 30), d(2010, 5, 28), d(2010, 6, 30))
    folds = [Fold("2010", (), dates), Fold("2011", (), (d(2011, 6, 30),))]
    sids = (1, 2, 3, 4, 5)
    rows = [(t, s) for t in dates + folds[1].test_dates for s in sids]
    q = pl.DataFrame(rows, schema={"date": pl.Date, "security_id": pl.Int64}, orient="row").with_columns(
        pl.lit(0.0).alias("q10"), pl.lit(0.0).alias("q50"), pl.lit(0.0).alias("q90")
    )
    y = {t: [100.0] * 5 for t in dates[:4]} | {dates[4]: [1.0, 2.0, 3.0, 4.0, 5.0], dates[5]: [6.0, 7.0, 8.0, 9.0, 10.0]}
    lab = pl.DataFrame(
        [(t, s, y[t][s - 1], d(2011, 1, 31)) for t in dates for s in sids],
        schema={"date": pl.Date, "security_id": pl.Int64, "excess_12m": pl.Float64, "label_end_12m": pl.Date},
        orient="row",
    )
    return q, lab, folds


def test_window_months_truncates_to_recent_dates():
    q, lab, folds = window_case()
    off = lambda w: walk_forward_conformal(q, lab, folds, alpha=0.2, window_months=w, min_rows=1, min_dates=1).filter(  # noqa: E731
        pl.col("date") == d(2011, 6, 30)
    )
    two = off(2)
    # last two dates: scores 1..10, n = 10, rank ceil(11 * 0.8) = 9 -> 9.0
    assert two["conformal_offset"].to_list() == [9.0] * 5 and two["conformal_ok"].all()
    assert two["q10"].to_list() == [-9.0] * 5 and two["q90"].to_list() == [9.0] * 5
    # all six dates: 20 scores of 100 + 1..10, n = 30, rank ceil(31 * 0.8) = 25 -> 100
    assert off(36)["conformal_offset"][0] == 100.0
    # one date: 6..10, n = 5, rank ceil(6 * 0.8) = 5 -> 10
    assert off(1)["conformal_offset"][0] == 10.0
    assert latest_conformal_offset(q, lab, before=d(2011, 6, 30), window_months=2, min_rows=1, min_dates=1) == (9.0, True)
    assert latest_conformal_offset(q, lab, before=d(2011, 1, 31), window_months=2) == (0.0, False)  # none matured


def test_too_few_scores_falls_back_to_zero():
    q, lab, folds = window_case()
    out = walk_forward_conformal(q, lab, folds, alpha=0.05, window_months=1, min_rows=1, min_dates=1)  # n = 5, rank ceil(5.7) = 6
    last = out.filter(pl.col("date") == d(2011, 6, 30))
    assert (last["conformal_offset"] == 0).all() and not last["conformal_ok"].any()
    assert last["q10"].to_list() == [0.0] * 5


def test_negative_offset_narrows_but_keeps_order():
    q, lab, folds = window_case()
    wide = q.with_columns(pl.lit(-1000.0).alias("q10"), pl.lit(1000.0).alias("q90"))
    wide = wide.with_columns(
        pl.when(pl.col("date") == d(2011, 6, 30)).then(pl.lit(-1.0)).otherwise(pl.col("q10")).alias("q10"),
        pl.when(pl.col("date") == d(2011, 6, 30)).then(pl.lit(1.0)).otherwise(pl.col("q90")).alias("q90"),
    )
    last = walk_forward_conformal(wide, lab, folds, window_months=36, min_rows=1, min_dates=1).filter(pl.col("date") == d(2011, 6, 30))
    assert last["conformal_offset"][0] == -900.0  # 1000 - 100
    assert last["q10"].to_list() == [0.0] * 5 and last["q90"].to_list() == [0.0] * 5  # clamped at q50


# -- minimum distinct matured dates --------------------------------------------------------------------
# fold 2010 has n_dates month ends in 2010 (labels end 2010-12-31, all matured before 2011-01-31);
# fold 2011 is the single date 2011-01-31. m securities per date.

TEST_DAY = d(2011, 1, 31)


def month_ends_2010(n_dates: int) -> list[dt.date]:
    return pl.date_range(d(2010, 1, 1), d(2010, 12, 1), "1mo", eager=True).dt.month_end().to_list()[:n_dates]


def many_dates_frames(n_dates: int, m: int):
    dates = month_ends_2010(n_dates)
    folds = [Fold("2010", (), tuple(dates)), Fold("2011", (), (TEST_DAY,))]
    all_dates = np.array([*dates, TEST_DAY], dtype="datetime64[D]")
    sid = np.tile(np.arange(1, m + 1), all_dates.size)
    base = pl.DataFrame({"date": np.repeat(all_dates, m), "security_id": sid}).with_columns(pl.col("date").cast(pl.Date))
    oof = base.with_columns(pl.Series("pred", sid.astype(np.float64)))
    ex = np.where(sid > m // 2, 0.1, -0.1)
    lab = base.with_columns(pl.Series("excess_12m", ex), pl.lit(d(2010, 12, 31)).alias("label_end_12m"))
    lab = lab.filter(pl.col("date") < TEST_DAY)
    rng = np.random.default_rng(0)
    q = base.with_columns(pl.lit(-1.0).alias("q10"), pl.lit(0.0).alias("q50"), pl.lit(1.0).alias("q90"))
    lab_q = lab.with_columns(pl.Series("excess_12m", rng.uniform(-2.0, 2.0, lab.height)))
    return oof, lab, q, lab_q, folds


def test_one_month_of_3000_rows_does_not_calibrate():
    # one usable date, 3000 rows: passes min_rows=2000 but has 1 < 12 distinct dates
    oof, lab, q, lab_q, folds = many_dates_frames(1, 3000)
    out = walk_forward_isotonic(oof, lab, folds, min_rows=2000).filter(pl.col("date") == TEST_DAY)
    assert not out["prob_calibrated"].any() and (out["prob"] == 0.5).all()
    assert walk_forward_isotonic(oof, lab, folds, min_rows=2000, min_dates=1).filter(
        pl.col("date") == TEST_DAY)["prob_calibrated"].all()
    assert fit_isotonic(oof, lab, before=TEST_DAY, min_rows=2000) is None
    cf = walk_forward_conformal(q, lab_q, folds, min_rows=2000).filter(pl.col("date") == TEST_DAY)
    assert not cf["conformal_ok"].any() and (cf["conformal_offset"] == 0).all()
    assert cf["q10"].to_list() == [-1.0] * 3000
    assert latest_conformal_offset(q, lab_q, before=TEST_DAY, min_rows=2000) == (0.0, False)


@pytest.mark.parametrize("n_dates, ok", [(11, False), (12, True)])
def test_min_dates_boundary(n_dates, ok):
    # 200 rows per date: 2200 / 2400 rows both pass min_rows=2000, so the date count decides
    oof, lab, q, lab_q, folds = many_dates_frames(n_dates, 200)
    iso = walk_forward_isotonic(oof, lab, folds, min_rows=2000, min_dates=12).filter(pl.col("date") == TEST_DAY)
    assert iso["prob_calibrated"].all() == ok and iso["prob_calibrated"].any() == ok
    if ok:  # securities 101..200 beat, 1..100 do not: the map is the step 0 -> 1
        assert iso.sort("security_id")["prob"].to_list() == [0.0] * 100 + [1.0] * 100
    assert (fit_isotonic(oof, lab, before=TEST_DAY, min_rows=2000, min_dates=12) is not None) == ok
    cf = walk_forward_conformal(q, lab_q, folds, min_rows=2000, min_dates=12).filter(pl.col("date") == TEST_DAY)
    assert cf["conformal_ok"].all() == ok and cf["conformal_ok"].any() == ok
    assert latest_conformal_offset(q, lab_q, before=TEST_DAY, min_rows=2000, min_dates=12)[1] == ok


def test_conformal_dates_are_counted_after_the_window():
    # 12 matured dates, but window_months=6 keeps only 6 of them: 6 < min_dates=12
    oof, lab, q, lab_q, folds = many_dates_frames(12, 200)
    cf = walk_forward_conformal(q, lab_q, folds, window_months=6, min_rows=1, min_dates=12)
    assert not cf.filter(pl.col("date") == TEST_DAY)["conformal_ok"].any()
    assert walk_forward_conformal(q, lab_q, folds, window_months=6, min_rows=1, min_dates=6).filter(
        pl.col("date") == TEST_DAY)["conformal_ok"].all()
    assert latest_conformal_offset(q, lab_q, before=TEST_DAY, window_months=6, min_rows=1) == (0.0, False)


def test_conformal_min_rows_guard():
    # 12 dates x 10 rows = 120 rows: the offset rank exists, but 120 < min_rows
    oof, lab, q, lab_q, folds = many_dates_frames(12, 10)
    at = lambda **kw: walk_forward_conformal(q, lab_q, folds, **kw).filter(pl.col("date") == TEST_DAY)  # noqa: E731
    assert not at(min_rows=121)["conformal_ok"].any()
    assert at(min_rows=120)["conformal_ok"].all()
    assert latest_conformal_offset(q, lab_q, before=TEST_DAY, min_rows=121) == (0.0, False)
    assert latest_conformal_offset(q, lab_q, before=TEST_DAY, min_rows=120)[1] is True

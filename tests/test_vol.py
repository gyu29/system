"""Volatility model (plan 6.3): point-in-time log vols, the ridge, and its walk-forward fit."""

import datetime as dt
import pickle

import numpy as np
import polars as pl
import pytest
from conftest import business_days

from ltforecast.dataset import HoldoutLockedError
from ltforecast.modelconfig import ValidationRules
from ltforecast.models.vol import (
    LVOL_COLS,
    VolModel,
    VolParams,
    oos_r2,
    oos_r2_by_fold,
    vol_frame,
    vol_inputs,
)
from ltforecast.universe import month_end_dates
from ltforecast.validation.runner import walk_forward
from ltforecast.validation.splitter import walk_forward_folds

d = dt.date


def path_prices(closes, sid=1, start=d(2020, 1, 1)):
    days = business_days(start, start + dt.timedelta(days=3 * len(closes) + 30))[: len(closes)]
    return pl.DataFrame(
        {"security_id": [sid] * len(closes), "date": days, "adj_close": [float(c) for c in closes]},
        schema_overrides={"security_id": pl.Int64},
    )


def hand_lvol(closes, i, w):
    """log(sqrt(252) * std of the non-null log returns among rows i-w+1 .. i)."""
    c = np.asarray(closes, dtype=float)
    lr = np.r_[np.nan, np.log(c[1:] / c[:-1])]
    win = lr[max(0, i - w + 1) : i + 1]
    win = win[np.isfinite(win)]
    return np.log(np.sqrt(252) * win.std(ddof=1)), len(win)


RNG = np.random.default_rng(7)
CLOSES = list(100 * np.exp(np.cumsum(RNG.normal(0, 0.02, 300))))


# -- vol_inputs --------------------------------------------------------------------------------


def test_lvol_matches_hand_computation():
    px = path_prices(CLOSES)
    entries = px.select("security_id", "date")[[14, 15, 39, 40, 125, 126, 299]]
    out = vol_inputs(px, entries)
    assert out.columns == ["security_id", "date", *LVOL_COLS]
    by_row = dict(zip(entries["date"].to_list(), out.iter_rows(named=True), strict=True))
    for i in (14, 15, 39, 40, 125, 126, 299):
        row = by_row[px["date"][i]]
        for w, k, col in ((21, 15, "lvol_21"), (63, 40, "lvol_63"), (252, 126, "lvol_252")):
            want, n = hand_lvol(CLOSES, i, w)
            if n < k:
                assert row[col] is None, (i, col)
            else:
                assert row[col] == pytest.approx(want, rel=1e-9), (i, col)
    # row 14 has 14 returns (row 0 has none): below the 15 minimum; row 15 has 15
    assert by_row[px["date"][14]]["lvol_21"] is None
    assert by_row[px["date"][15]]["lvol_21"] is not None


def test_changing_a_price_after_the_entry_leaves_lvol_unchanged():
    px = path_prices(CLOSES)
    entry = px.select("security_id", "date")[[200]]
    before = vol_inputs(px, entry)
    bumped = px.with_columns(
        pl.when(pl.int_range(pl.len()) > 200).then(pl.col("adj_close") * 3.0).otherwise(pl.col("adj_close"))
    )
    # also scramble the return on the day right after the entry
    bumped = bumped.with_columns(
        pl.when(pl.int_range(pl.len()) == 201).then(1.0).otherwise(pl.col("adj_close")).alias("adj_close")
    )
    after = vol_inputs(bumped, entry)
    assert before.equals(after)
    assert before["lvol_252"][0] is not None


def test_missing_price_on_entry_and_bad_prices():
    px = path_prices(CLOSES).with_columns(
        pl.when(pl.int_range(pl.len()) == 50).then(-1.0).otherwise(pl.col("adj_close")).alias("adj_close")
    )
    entries = pl.DataFrame(
        {"security_id": [1, 1, 2], "date": [px["date"][60], d(2030, 1, 1), px["date"][60]]},
        schema_overrides={"security_id": pl.Int64},
    )
    out = vol_inputs(px, entries).sort("security_id", "date")
    assert out.height == 3
    # rows 50 and 51 have no return (bad price): row 60's 21-row window holds 21 - 2 = 19 returns
    clean = [c if i != 50 else np.nan for i, c in enumerate(CLOSES)]
    want, n = hand_lvol(clean, 60, 21)
    assert n == 19
    assert out["lvol_21"][0] == pytest.approx(want, rel=1e-9)
    assert out.row(1)[2:] == (None, None, None)  # no price row on the entry date
    assert out.row(2)[2:] == (None, None, None)  # unknown security


def test_constant_price_gives_null_not_minus_inf():
    out = vol_inputs(path_prices([50.0] * 40), path_prices([50.0] * 40).select("security_id", "date")[[39]])
    assert out["lvol_21"][0] is None


def test_securities_do_not_bleed_into_each_other():
    a = path_prices(CLOSES[:100], sid=1)
    b = path_prices(list(np.array(CLOSES[100:200]) * 7), sid=2)
    both = pl.concat([b, a])
    e = pl.concat([a, b]).select("security_id", "date").filter(pl.col("date") == a["date"][30])
    out = vol_inputs(both, e).sort("security_id")
    alone = vol_inputs(b, e.filter(pl.col("security_id") == 2))
    assert out.filter(pl.col("security_id") == 2).equals(alone)
    assert out["lvol_21"][0] == pytest.approx(hand_lvol(CLOSES[:100], 30, 21)[0], rel=1e-9)


# -- VolModel ----------------------------------------------------------------------------------


def synthetic_panel(n_sec=150, n_dates=48, seed=0, start=d(2000, 1, 31)):
    """log realized vol = -0.3 + 0.7 * lvol_252 + 0.2 * (sector B) + N(0, 0.15)."""
    rng = np.random.default_rng(seed)
    n = n_sec * n_dates
    lvol_252 = rng.normal(np.log(0.3), 0.4, n)
    lvol_63 = lvol_252 + rng.normal(0, 0.25, n)
    lvol_21 = lvol_63 + rng.normal(0, 0.35, n)
    sector = np.where(np.arange(n) % 3 == 0, "B", "A")
    y = -0.3 + 0.7 * lvol_252 + 0.2 * (sector == "B") + rng.normal(0, 0.15, n)
    dates = np.repeat(np.arange(n_dates), n_sec)
    return pl.DataFrame(
        {
            "date": [start + dt.timedelta(days=31 * int(k)) for k in dates],
            "security_id": np.tile(np.arange(n_sec), n_dates).astype(np.int64),
            "sector": sector,
            "lvol_21": lvol_21,
            "lvol_63": lvol_63,
            "lvol_252": lvol_252,
            "beta_60m": rng.normal(1.0, 0.3, n),
            "log_mcap": rng.normal(21.0, 1.5, n),
            "log_rvol_12m": y,
        }
    )


def test_recovers_planted_coefficients_out_of_sample():
    df = synthetic_panel()
    cut = df["date"].unique().sort()[30]
    train, test = df.filter(pl.col("date") < cut), df.filter(pl.col("date") >= cut)
    m = VolModel().fit(train, None)
    r2 = oos_r2(m.predict_log(test.drop("log_rvol_12m")), test["log_rvol_12m"].to_numpy())
    assert r2 > 0.4
    c = m.coefficients()
    assert c["lvol_252"] == pytest.approx(0.7, abs=0.05)
    assert abs(c["lvol_63"]) < 0.05 and abs(c["lvol_21"]) < 0.05
    assert abs(c["beta_60m"]) < 0.03 and abs(c["log_mcap"]) < 0.02
    assert c["sector=B"] - c["sector=A"] == pytest.approx(0.2, abs=0.03)
    sig = m.predict(test)
    assert sig.shape == (test.height,) and (sig >= 0.05).all() and (sig <= 3.0).all()


def test_predict_does_not_read_the_target():
    df = synthetic_panel(n_sec=40, n_dates=12)
    m = VolModel().fit(df, df)
    assert m.valid_r2_ is not None and m.valid_r2_ > 0.4
    assert np.array_equal(m.predict(df), m.predict(df.drop("log_rvol_12m")))


def test_unseen_and_null_sector_get_zero_dummies():
    df = synthetic_panel(n_sec=40, n_dates=12)
    m = VolModel().fit(df)
    row = df.head(1)
    c = m.coefficients()
    base = m.predict_log(row.with_columns(pl.lit("Z").alias("sector")))
    null = m.predict_log(row.with_columns(pl.lit(None, dtype=pl.String).alias("sector")))
    x = row.row(0, named=True)
    want = c["intercept"] + sum(c[k] * x[k] for k in ("lvol_21", "lvol_63", "lvol_252", "beta_60m", "log_mcap"))
    assert base[0] == pytest.approx(want, abs=1e-9)
    assert null[0] == pytest.approx(want, abs=1e-9)
    as_a = m.predict_log(row.with_columns(pl.lit("A").alias("sector")))
    assert as_a[0] == pytest.approx(want + c["sector=A"], abs=1e-9)


def test_missing_inputs_use_train_medians():
    df = synthetic_panel(n_sec=40, n_dates=12)
    m = VolModel().fit(df)
    row = df.head(1)
    blank = row.with_columns(pl.lit(None, dtype=pl.Float64).alias("lvol_252"))
    filled = row.with_columns(pl.lit(float(df["lvol_252"].median())).alias("lvol_252"))
    assert m.predict_log(blank)[0] == pytest.approx(m.predict_log(filled)[0], abs=1e-12)
    # NaN / inf inputs are missing too
    nan = row.with_columns(pl.lit(float("nan")).alias("lvol_252"))
    assert m.predict_log(nan)[0] == pytest.approx(m.predict_log(filled)[0], abs=1e-12)


def test_null_targets_are_skipped_and_medians_ignore_nulls():
    df = synthetic_panel(n_sec=40, n_dates=12)
    holes = df.with_columns(
        pl.when(pl.int_range(pl.len()) % 5 == 0).then(None).otherwise(pl.col("log_rvol_12m")).alias("log_rvol_12m"),
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("beta_60m")).alias("beta_60m"),
    )
    m = VolModel().fit(holes)
    assert np.isfinite(m.predict(holes)).all()
    with pytest.raises(ValueError, match="no finite targets"):
        VolModel().fit(holes.with_columns(pl.lit(None, dtype=pl.Float64).alias("log_rvol_12m")))


def test_predictions_are_clipped():
    df = synthetic_panel(n_sec=40, n_dates=12)
    m = VolModel().fit(df)
    extreme = df.head(2).with_columns(pl.Series("lvol_252", [-40.0, 40.0]))
    assert m.predict(extreme) == pytest.approx([0.05, 3.0])


def test_deterministic_and_picklable():
    df = synthetic_panel(n_sec=40, n_dates=12)
    a, b = VolModel(VolParams(alpha=0.01)).fit(df), VolModel(VolParams(alpha=0.01)).fit(df)
    assert np.array_equal(a.predict(df), b.predict(df))
    back = pickle.loads(pickle.dumps(a))
    assert np.array_equal(back.predict(df), a.predict(df))
    assert back.coefficients() == a.coefficients()


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        VolModel().predict(synthetic_panel(n_sec=2, n_dates=1))


def test_oos_r2():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert oos_r2(a, a) == 1.0
    assert oos_r2(np.full(4, 2.5), a) == pytest.approx(0.0)
    # sse = 1 + 0 + 0 + 1 = 2, sst = 5
    assert oos_r2(np.array([2.0, 2.0, 3.0, 3.0]), a) == pytest.approx(1 - 2 / 5)
    assert oos_r2(np.array([np.nan, 2.0, 3.0, 4.0]), a) == 1.0
    assert np.isnan(oos_r2(np.ones(3), np.ones(3)))
    with pytest.raises(ValueError):
        oos_r2(np.ones(3), np.ones(4))


# -- vol_frame + walk-forward ------------------------------------------------------------------


RULES = ValidationRules(
    first_test_year=2010, test_months=12, purge_months=12, embargo_months=1,
    min_train_months=60, inner_valid_months=24,
)


def test_vol_frame_and_walk_forward():
    cal = business_days(d(1999, 1, 1), d(2011, 12, 31))
    me = month_end_dates(cal).filter(pl.Series(month_end_dates(cal)) >= d(2000, 1, 1)).to_list()
    rng = np.random.default_rng(3)
    sids = list(range(1, 31))
    sig = {s: 0.1 + 0.6 * (s / 30) for s in sids}  # each stock has its own daily vol level
    px = pl.concat(
        [
            pl.DataFrame(
                {
                    "security_id": s,
                    "date": cal,
                    "adj_close": 50 * np.exp(np.cumsum(rng.normal(0, sig[s] / np.sqrt(252), len(cal)))),
                },
                schema_overrides={"security_id": pl.Int64},
            )
            for s in sids
        ]
    )
    keys = pl.DataFrame({"security_id": [s for s in sids for _ in me], "date": me * len(sids)},
                        schema_overrides={"security_id": pl.Int64})
    ends = pl.DataFrame({"date": me, "label_end_12m": me[12:] + [None] * 12})
    labels = keys.join(ends, on="date", how="left").with_columns(
        pl.col("security_id").replace_strict(sig, return_dtype=pl.Float64).alias("realized_vol_12m"),
        pl.when(pl.col("date") < d(2011, 6, 1)).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("period"),
    ).with_columns(
        pl.when(pl.col("label_end_12m").is_not_null()).then(pl.col("realized_vol_12m")).alias("realized_vol_12m")
    )
    raw = keys.with_columns(pl.lit(1.0).alias("beta_60m"), pl.lit(20.0).alias("log_mcap"), pl.lit(0.0).alias("x"))
    secs = pl.DataFrame({"security_id": sids, "sector": ["A" if s % 2 else "B" for s in sids]},
                        schema_overrides={"security_id": pl.Int64})

    frame = vol_frame(raw, labels, px, secs)
    assert frame.columns == ["date", "security_id", "sector", *LVOL_COLS, "beta_60m", "log_mcap",
                             "log_rvol_12m", "label_end_12m", "period"]
    assert set(frame["period"]) == {"dev"} and frame["date"].max() < d(2011, 6, 1)
    with pytest.raises(HoldoutLockedError):
        vol_frame(raw, labels, px, secs, periods=("dev", "holdout"))
    # lvol equals vol_inputs on the same entries (one pass over prices, joined back)
    direct = vol_inputs(px, frame.select("security_id", "date"))
    assert frame.select("security_id", "date", *LVOL_COLS).sort("security_id", "date").equals(direct)

    folds = walk_forward_folds(frame["date"].unique().sort().to_list(), RULES, d(2010, 12, 31))
    oof = walk_forward(VolModel, frame, folds, RULES, target="log_rvol_12m")
    assert oof["pred"].min() >= 0.05
    per_fold = oos_r2_by_fold(oof, frame)
    assert per_fold["fold"].to_list() == ["2010"]
    assert per_fold["r2"][0] > 0.8  # vol is a stable per-stock level here


def test_refit_on_train_plus_valid_and_valid_r2_is_train_only():
    df = synthetic_panel(n_sec=40, n_dates=24)
    cut = df["date"].unique().sort()[16]
    train, valid = df.filter(pl.col("date") < cut), df.filter(pl.col("date") >= cut)
    only_train = VolModel(VolParams(refit=False)).fit(train)
    # refit=False: valid is scored only, the coefficients are the train-only ones
    nf = VolModel(VolParams(refit=False)).fit(train, valid)
    assert np.array_equal(nf.coef_, only_train.coef_) and nf.intercept_ == only_train.intercept_
    # refit=True (default): same honest valid_r2_, coefficients equal a fit on train + valid
    rf = VolModel().fit(train, valid)
    both = VolModel().fit(pl.concat([train, valid]))
    assert rf.valid_r2_ == nf.valid_r2_
    assert rf.valid_r2_ == oos_r2(only_train.predict_log(valid), valid["log_rvol_12m"].to_numpy())
    assert np.allclose(rf.coef_, both.coef_) and rf.intercept_ == pytest.approx(both.intercept_)
    # changing only valid targets moves the refit model, not the refit=False one
    shifted = valid.with_columns(pl.col("log_rvol_12m") + 0.5)
    assert VolModel().fit(train, shifted).intercept_ > rf.intercept_ + 0.1
    assert VolModel(VolParams(refit=False)).fit(train, shifted).intercept_ == only_train.intercept_


def test_valid_without_finite_targets_does_not_refit():
    df = synthetic_panel(n_sec=40, n_dates=12)
    blank = df.head(40).with_columns(pl.lit(None, dtype=pl.Float64).alias("log_rvol_12m"))
    m = VolModel().fit(df, blank)
    assert m.valid_r2_ is None
    assert np.array_equal(m.coef_, VolModel().fit(df).coef_)

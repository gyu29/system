"""Walk-forward runner: what the model is shown at fit and at predict, with literal dates.

Panel: 4 securities x the 144 month-end trading days 2000-01 .. 2011-12 (Memorial Day 2010 removed,
so 2010-05 ends on the 28th). A row's 12m label ends 12 month ends later; the last 12 dates (2011) have
no label yet. With purge 12 + embargo 1, min 60 train months and 24 inner-validation months:

    fold 2010: inner train 2000-01 .. 2005-10 | inner valid 2006-12 .. 2008-11 | test 2010-01 .. 2010-12
    fold 2011: inner train 2000-01 .. 2006-10 | inner valid 2007-12 .. 2009-11 | test 2011-01 .. 2011-12
"""

import datetime as dt

import numpy as np
import polars as pl
import pytest
from conftest import business_days

from ltforecast.modelconfig import ValidationRules
from ltforecast.universe import month_end_dates
from ltforecast.validation.runner import walk_forward
from ltforecast.validation.splitter import Fold, LeakageError, walk_forward_folds

d = dt.date
RULES = ValidationRules(
    first_test_year=2010, test_months=12, purge_months=12, embargo_months=1,
    min_train_months=60, inner_valid_months=24,
)
CAL = business_days(d(2000, 1, 1), d(2011, 12, 31))
D = month_end_dates(CAL.filter(CAL != d(2010, 5, 31))).to_list()
SECS = (1, 2, 3, 4)
HORIZONS = {"3m": 3, "6m": 6, "12m": 12}
LABEL_KINDS = ("fwd_ret", "bench_ret", "excess", "excess_w", "rank", "realized_vol")
FEATURE_COLS = ["date", "security_id", "sector", "f1", "f2"]


def panel() -> pl.DataFrame:
    """f1 = security_id + (month index) / 1000, so every prediction can be written down by hand."""
    rows = []
    for i, day in enumerate(D):
        for sid in SECS:
            row = {"date": day, "security_id": sid, "sector": "A" if sid < 3 else "B",
                   "f1": sid + i / 1000, "f2": -float(sid)}
            for h, k in HORIZONS.items():
                done = i + k < len(D)
                row |= {f"{c}_{h}": (sid - 0.5) / 4 if done else None for c in LABEL_KINDS}
                row[f"label_end_{h}"] = D[i + k] if done else None
            row["target"] = row["rank_12m"]
            row["period"] = "dev"
            rows.append(row)
    schema = {"security_id": pl.Int64} | {f"{c}_{h}": pl.Float64 for c in LABEL_KINDS for h in HORIZONS}
    return pl.DataFrame(rows, schema_overrides=schema | {"target": pl.Float64})


def without_label(frame: pl.DataFrame, sid: int, day: dt.date) -> pl.DataFrame:
    """The row loses its 12m label entirely (target and label end), as when the window ran past the data."""
    hit = (pl.col("security_id") == sid) & (pl.col("date") == day)
    cols = ["target", "rank_12m", "excess_12m", "label_end_12m"]
    return frame.with_columns(pl.when(hit).then(None).otherwise(pl.col(c)).alias(c) for c in cols)


class Spy:
    """Scores by f1 and records everything it was handed."""

    name = "spy"

    def __init__(self, log: list):
        self.log = log

    def fit(self, train, valid):
        self.log.append(("fit", train, valid))
        return self

    def predict(self, df):
        self.log.append(("predict", df))
        return df["f1"].to_numpy()


def run(frame=None, folds=None, rules=RULES, **kw):
    frame = panel() if frame is None else frame
    folds = walk_forward_folds(D, RULES, d(2011, 12, 31)) if folds is None else folds
    log: list = []
    oof = walk_forward(lambda: Spy(log), frame, folds, rules, **kw)
    return oof, [x for x in log if x[0] == "fit"], [x[1] for x in log if x[0] == "predict"]


def test_panel_dates():
    assert len(D) == 144 and D[0] == d(2000, 1, 31) and D[-1] == d(2011, 12, 30)
    assert D[106] == d(2008, 11, 28) and D[124] == d(2010, 5, 28)


def test_oof_frame():
    oof, _, _ = run()
    assert oof.schema == {"date": pl.Date, "security_id": pl.Int64, "fold": pl.String, "pred": pl.Float64}
    # 24 test dates x 4 securities, each scored exactly once
    assert oof.height == 96 and oof.select("date", "security_id").is_unique().all()
    assert oof.rows() == sorted(oof.rows())
    assert oof["date"].min() == d(2010, 1, 29) and oof["date"].max() == d(2011, 12, 30)
    assert oof.group_by("fold").len().sort("fold").rows() == [("2010", 48), ("2011", 48)]
    assert oof.filter(pl.col("date").dt.year() == 2010)["fold"].unique().to_list() == ["2010"]
    # 2010-05-28 is month index 124: pred = f1 = security_id + 0.124
    may = oof.filter(pl.col("date") == d(2010, 5, 28))
    assert may["security_id"].to_list() == [1, 2, 3, 4]
    assert may["pred"].to_list() == pytest.approx([1.124, 2.124, 3.124, 4.124])
    # first row of the last fold: 2011-01-31 is index 132
    assert oof.filter(pl.col("fold") == "2011").row(0) == (d(2011, 1, 31), 1, "2011", pytest.approx(1.132))


def test_unlabelled_test_rows_are_still_scored():
    # every 2011 row has a null target (its label window runs past the data) and is predicted anyway
    oof, _, _ = run()
    assert panel().filter(pl.col("date").dt.year() == 2011)["target"].null_count() == 48
    assert oof.filter(pl.col("fold") == "2011").height == 48


def test_fit_sees_only_the_purged_past():
    _, fits, _ = run()
    assert len(fits) == 2
    expected = [
        # first test date, inner-train end, inner-valid start, inner-valid end, train dates
        (d(2010, 1, 29), d(2005, 10, 31), d(2006, 12, 29), d(2008, 11, 28), 70),
        (d(2011, 1, 31), d(2006, 10, 31), d(2007, 12, 31), d(2009, 11, 30), 82),
    ]
    for (_, train, valid), (test_start, train_end, valid_start, valid_end, n_train) in zip(fits, expected):
        assert train["date"].min() == d(2000, 1, 31) and train["date"].max() == train_end
        assert valid["date"].min() == valid_start and valid["date"].max() == valid_end
        assert train["date"].n_unique() == n_train and valid["date"].n_unique() == 24
        assert train.height == n_train * 4 and valid.height == 24 * 4
        # no test date, and no date after the test block starts, reaches fit
        assert max(train["date"].max(), valid["date"].max()) < test_start
        # purge: every label the model trains or validates on was complete before the test block ...
        assert max(train["label_end_12m"].max(), valid["label_end_12m"].max()) < test_start
        # ... and every inner-train label was complete before inner validation starts
        assert train["label_end_12m"].max() < valid_start
    # literal label ends for fold 2010: 2005-10 + 12m and 2008-11 + 12m
    assert fits[0][1]["label_end_12m"].max() == d(2006, 10, 31)
    assert fits[0][2]["label_end_12m"].max() == d(2009, 11, 30)


def test_fit_frames_keep_labels_and_have_no_null_targets():
    frame = without_label(without_label(panel(), 4, d(2004, 3, 31)), 3, d(2007, 6, 29))
    _, fits, _ = run(frame)
    _, train, valid = fits[0]
    # one unlabelled row dropped from inner train (70 x 4 - 1) and one from inner valid (24 x 4 - 1)
    assert train.height == 279 and valid.height == 95
    assert train.filter((pl.col("security_id") == 4) & (pl.col("date") == d(2004, 3, 31))).is_empty()
    assert valid.filter((pl.col("security_id") == 3) & (pl.col("date") == d(2007, 6, 29))).is_empty()
    for _, tr, va in fits:
        assert tr["target"].null_count() == 0 and va["target"].null_count() == 0
        assert tr.columns == frame.columns and va.columns == frame.columns


def test_predict_frame_has_no_label_columns():
    _, _, predicts = run()
    assert len(predicts) == 2
    for df in predicts:
        assert df.columns == FEATURE_COLS
        assert df.height == 48
    hidden = set(panel().columns) - set(FEATURE_COLS)
    # 6 kinds + label_end for 3 horizons, plus target and period
    assert len(hidden) == 7 * 3 + 2
    for name in ("fwd_ret_12m", "bench_ret_3m", "excess_6m", "excess_w_12m", "rank_12m", "realized_vol_12m",
                 "label_end_12m", "target", "period"):
        assert name in hidden
    assert predicts[0]["date"].min() == d(2010, 1, 29) and predicts[0]["date"].max() == d(2010, 12, 31)


class Leaky(Spy):
    def __init__(self, col: str):
        super().__init__([])
        self.col = col

    def predict(self, df):
        return df[self.col].fill_null(0.5).to_numpy()


@pytest.mark.parametrize("col", ["target", "excess_12m", "excess_w_12m", "rank_3m", "fwd_ret_6m", "period"])
def test_a_model_that_reads_a_label_at_predict_time_fails(col):
    folds = walk_forward_folds(D, RULES, d(2011, 12, 31))
    with pytest.raises((KeyError, pl.exceptions.ColumnNotFoundError)):
        walk_forward(lambda: Leaky(col), panel(), folds, RULES)


def test_a_custom_target_is_hidden_too():
    # a target outside the usual label names is still removed before predict
    frame = panel().with_columns(pl.col("target").alias("y"))
    folds = walk_forward_folds(D, RULES, d(2011, 12, 31))
    _, _, predicts = run(frame, target="y")
    assert all(df.columns == FEATURE_COLS for df in predicts)
    with pytest.raises((KeyError, pl.exceptions.ColumnNotFoundError)):
        walk_forward(lambda: Leaky("y"), frame, folds, RULES, target="y")


def test_other_horizon_target():
    # 3m target: null-target filtering and the purge check follow the named columns
    frame = panel().with_columns(
        pl.when(pl.col("date") == d(2003, 1, 31)).then(None).otherwise(pl.col("rank_3m")).alias("rank_3m")
    )
    oof, fits, _ = run(frame, target="rank_3m", label_end_col="label_end_3m")
    assert oof.height == 96
    assert fits[0][1].height == 69 * 4 and fits[0][1]["rank_3m"].null_count() == 0
    assert d(2003, 1, 31) not in fits[0][1]["date"].to_list()


class Fixed(Spy):
    def __init__(self, out):
        super().__init__([])
        self.out = out

    def predict(self, df):
        return self.out(df)


def test_wrong_length_prediction_raises():
    folds = walk_forward_folds(D, RULES, d(2011, 12, 31))
    with pytest.raises(ValueError, match="shape"):
        walk_forward(lambda: Fixed(lambda df: np.zeros(df.height - 1)), panel(), folds, RULES)
    with pytest.raises(ValueError, match="shape"):
        walk_forward(lambda: Fixed(lambda df: np.zeros((df.height, 1))), panel(), folds, RULES)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_prediction_raises(bad):
    def predict(df):
        out = np.zeros(df.height)
        out[7] = bad
        return out

    folds = walk_forward_folds(D, RULES, d(2011, 12, 31))
    with pytest.raises(ValueError, match="1 non-finite"):
        walk_forward(lambda: Fixed(predict), panel(), folds, RULES)


def test_a_fresh_model_per_fold():
    made = []

    def make():
        made.append(Spy([]))
        return made[-1]

    walk_forward(make, panel(), walk_forward_folds(D, RULES, d(2011, 12, 31)), RULES)
    assert len(made) == 2 and made[0] is not made[1]
    # each instance was fit once and predicted once
    assert [[kind for kind, *_ in m.log] for m in made] == [["fit", "predict"]] * 2


def test_nothing_after_the_fold_end_is_touched():
    # folds stop at 2010-12-31; the frame still holds 2011, which must never reach the model
    folds = walk_forward_folds(D, RULES, d(2010, 12, 31))
    oof, fits, predicts = run(folds=folds)
    assert [f.name for f in folds] == ["2010"]
    assert oof["date"].max() == d(2010, 12, 31) and oof.height == 48
    seen = pl.concat([fits[0][1]["date"], fits[0][2]["date"], predicts[0]["date"]])
    assert seen.max() == d(2010, 12, 31)


def test_row_order_of_the_input_does_not_matter():
    oof, _, _ = run()
    shuffled, _, _ = run(panel().sample(fraction=1.0, shuffle=True, seed=11))
    assert shuffled.equals(oof)


def test_unpurged_outer_fold_raises():
    # hand-made fold whose training window runs to 2009-06: those labels end in 2010-06, inside the test
    leaky = Fold("2010", tuple(D[:114]), tuple(D[120:132]))
    assert leaky.train_dates[-1] == d(2009, 6, 30)
    with pytest.raises(LeakageError, match="2010-01-29"):
        run(folds=[leaky])
    # a label ending exactly on the first test date (2009-01 + 12m = 2010-01-29) is also a violation
    edge = Fold("2010", tuple(D[:109]), tuple(D[120:132]))
    with pytest.raises(LeakageError):
        run(folds=[edge])
    # one month less is clean
    run(folds=[Fold("2010", tuple(D[:108]), tuple(D[120:132]))])


def test_unpurged_inner_split_raises():
    # correctly purged outer folds, but rules that leave no gap before inner validation
    no_gap = ValidationRules(2010, 12, 0, 0, 60, 24)
    with pytest.raises(LeakageError, match="2006-12-29"):
        run(rules=no_gap)


def test_target_without_a_label_end_raises():
    hit = (pl.col("security_id") == 2) & (pl.col("date") == d(2003, 5, 30))
    frame = panel().with_columns(pl.when(hit).then(None).otherwise(pl.col("label_end_12m")).alias("label_end_12m"))
    with pytest.raises(LeakageError, match="1 training rows"):
        run(frame)


def test_fold_with_no_rows_raises():
    frame = panel().filter(pl.col("date").dt.year() != 2010)
    with pytest.raises(ValueError, match="fold 2010"):
        run(frame)


def test_no_folds_gives_an_empty_frame():
    oof, fits, _ = run(folds=[])
    assert oof.is_empty() and fits == []
    assert oof.schema == {"date": pl.Date, "security_id": pl.Int64, "fold": pl.String, "pred": pl.Float64}

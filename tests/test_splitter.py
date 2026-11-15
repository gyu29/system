"""Walk-forward folds with literal dates (plan 4.1).

The calendar is every weekday 2000-2025 minus Memorial Day 2010, so month ends are trading days that
differ from calendar month ends: 2008-11-28, 2009-02-27, 2010-01-29, 2010-05-28, 2021-10-29.
"""

import datetime as dt

import polars as pl
import pytest
from conftest import business_days

from ltforecast.modelconfig import ValidationRules
from ltforecast.universe import month_end_dates
from ltforecast.validation.splitter import Fold, LeakageError, assert_purged, inner_split, walk_forward_folds

d = dt.date
RULES = ValidationRules(
    first_test_year=2010, test_months=12, purge_months=12, embargo_months=1,
    min_train_months=60, inner_valid_months=24,
)
DEV_END = d(2021, 10, 31)
CAL = business_days(d(2000, 1, 1), d(2025, 12, 31))
MONTH_ENDS = month_end_dates(CAL.filter(CAL != d(2010, 5, 31))).to_list()


@pytest.fixture(scope="module")
def folds() -> list[Fold]:
    return walk_forward_folds(MONTH_ENDS, RULES, DEV_END)


def test_calendar_has_trading_month_ends():
    assert len(MONTH_ENDS) == 26 * 12
    for day in (d(2008, 11, 28), d(2009, 2, 27), d(2010, 1, 29), d(2010, 5, 28), d(2021, 10, 29)):
        assert day in MONTH_ENDS
    assert d(2010, 5, 31) not in MONTH_ENDS and d(2009, 2, 28) not in MONTH_ENDS


def test_one_fold_per_calendar_year(folds):
    assert [f.name for f in folds] == [str(y) for y in range(2010, 2022)]


def test_fold_2010_trains_through_november_2008(folds):
    f = folds[0]
    assert f.test_dates[0] == d(2010, 1, 29) and f.test_dates[-1] == d(2010, 12, 31)
    assert len(f.test_dates) == 12
    assert d(2010, 5, 28) in f.test_dates
    assert f.train_dates[0] == d(2000, 1, 31)
    # purge 12 + embargo 1: the 13 month ends 2008-12 .. 2009-12 are dropped
    assert f.train_dates[-1] == d(2008, 11, 28)
    assert len(f.train_dates) == 8 * 12 + 11
    assert d(2008, 12, 31) not in f.train_dates and d(2009, 2, 27) not in f.train_dates


def test_fold_2011_expands_by_one_year(folds):
    f = folds[1]
    assert f.train_dates[0] == d(2000, 1, 31) and f.train_dates[-1] == d(2009, 11, 30)
    assert len(f.train_dates) == 9 * 12 + 11
    assert d(2009, 2, 27) in f.train_dates
    assert f.test_dates[0] == d(2011, 1, 31) and f.test_dates[-1] == d(2011, 12, 30)


def test_last_fold_is_partial_and_ends_at_dev_end(folds):
    f = folds[-1]
    assert f.name == "2021"
    assert f.test_dates[0] == d(2021, 1, 29) and f.test_dates[-1] == d(2021, 10, 29)
    assert len(f.test_dates) == 10
    assert f.train_dates[-1] == d(2019, 11, 29)


def test_nothing_after_end_appears_anywhere(folds):
    # the calendar runs to 2025-12 and so covers the gap and the holdout; none of it may leak in
    assert max(MONTH_ENDS) == d(2025, 12, 31)
    used = {day for f in folds for day in (*f.train_dates, *f.test_dates)}
    assert max(used) == d(2021, 10, 29)
    assert not any(day > DEV_END for day in used)
    # `end` on a month-end date is inclusive
    assert walk_forward_folds(MONTH_ENDS, RULES, d(2021, 10, 29))[-1].test_dates[-1] == d(2021, 10, 29)
    assert walk_forward_folds(MONTH_ENDS, RULES, d(2021, 10, 28))[-1].test_dates[-1] == d(2021, 9, 30)


def test_folds_are_ordered_disjoint_and_gapped(folds):
    seen: set[dt.date] = set()
    for f in folds:
        assert list(f.train_dates) == sorted(set(f.train_dates))
        assert list(f.test_dates) == sorted(set(f.test_dates))
        assert not seen & set(f.test_dates)
        seen |= set(f.test_dates)
        # exactly 13 month ends lie strictly between the last train date and the first test date
        between = [x for x in MONTH_ENDS if f.train_dates[-1] < x < f.test_dates[0]]
        assert len(between) == 13
    assert len(seen) == 11 * 12 + 10


def test_short_history_folds_are_skipped():
    # history from 2005-01: test year 2010 has 47 train months, 2011 has 59, 2012 has 71
    late = [x for x in MONTH_ENDS if x >= d(2005, 1, 1)]
    got = walk_forward_folds(late, RULES, DEV_END)
    assert [f.name for f in got] == [str(y) for y in range(2012, 2022)]
    assert len(got[0].train_dates) == 71
    assert got[0].train_dates[0] == d(2005, 1, 31) and got[0].train_dates[-1] == d(2010, 11, 30)
    # exactly at the minimum is kept
    exact = walk_forward_folds(late, ValidationRules(2010, 12, 12, 1, 59, 24), DEV_END)
    assert exact[0].name == "2011" and len(exact[0].train_dates) == 59


def test_input_may_be_unsorted_duplicated_or_a_series(folds):
    shuffled = pl.Series(MONTH_ENDS + MONTH_ENDS[:50]).sample(fraction=1.0, shuffle=True, seed=3)
    assert walk_forward_folds(shuffled, RULES, DEV_END) == folds
    assert walk_forward_folds(shuffled.to_list(), RULES, DEV_END) == folds


def test_no_folds_when_end_precedes_the_first_test_year():
    assert walk_forward_folds(MONTH_ENDS, RULES, d(2009, 12, 31)) == []
    assert walk_forward_folds([], RULES, DEV_END) == []


def test_gap_is_counted_in_dates_not_calendar_months():
    # with a month missing from the data (2009-06), 13 DATES are still dropped: train stops one
    # calendar month earlier, which errs on the safe side
    holey = [x for x in MONTH_ENDS if x != d(2009, 6, 30)]
    f = walk_forward_folds(holey, RULES, DEV_END)[0]
    assert f.train_dates[-1] == d(2008, 10, 31)
    assert len(f.train_dates) == 8 * 12 + 10


def test_shorter_test_blocks():
    six = walk_forward_folds(MONTH_ENDS, ValidationRules(2010, 6, 12, 1, 60, 24), d(2011, 2, 28))
    assert [f.name for f in six] == ["2010-01", "2010-07", "2011-01"]
    assert six[1].test_dates == (d(2010, 7, 30), d(2010, 8, 31), d(2010, 9, 30), d(2010, 10, 29),
                                 d(2010, 11, 30), d(2010, 12, 31))
    assert six[1].train_dates[-1] == d(2009, 5, 29)
    assert six[2].test_dates == (d(2011, 1, 31), d(2011, 2, 28))


def test_inner_split_leaves_the_same_gap(folds):
    train, valid = inner_split(folds[0].train_dates, RULES)
    # the last 24 train dates: 2006-12 .. 2008-11
    assert valid[0] == d(2006, 12, 29) and valid[-1] == d(2008, 11, 28) and len(valid) == 24
    # a further 13 dropped (2005-11 .. 2006-11): inner train ends 2005-10, the same distance as
    # 2008-11 -> 2010-01 in the outer split
    assert train[0] == d(2000, 1, 31) and train[-1] == d(2005, 10, 31)
    assert d(2005, 11, 30) not in train and d(2006, 11, 30) not in valid
    assert len(train) == 107 - 24 - 13
    assert set(train) | set(valid) <= set(folds[0].train_dates)
    assert len([x for x in MONTH_ENDS if train[-1] < x < valid[0]]) == 13


def test_inner_split_never_reaches_the_test_year(folds):
    for f in folds:
        train, valid = inner_split(f.train_dates, RULES)
        assert max(valid) == f.train_dates[-1] < f.test_dates[0]
        assert max(train) < min(valid)


def test_inner_split_needs_something_to_fit_on():
    # 37 dates = 24 validation + 13 gap: nothing left
    with pytest.raises(ValueError, match="nothing to fit"):
        inner_split(MONTH_ENDS[:37], RULES)
    train, valid = inner_split(MONTH_ENDS[:38], RULES)
    assert train == (d(2000, 1, 31),) and valid[0] == d(2001, 3, 30)


def purge_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": [1, 2, 3],
            "date": [d(2008, 9, 30), d(2008, 10, 31), d(2008, 11, 28)],
            "label_end_12m": [d(2009, 9, 30), d(2009, 10, 30), d(2009, 11, 30)],
            "target": [0.2, 0.5, 0.8],
        }
    )


def test_assert_purged_passes_when_every_label_ends_before_the_test():
    assert_purged(purge_frame(), d(2010, 1, 29))
    # the day after the last label end is still fine: the label was complete the day before
    assert_purged(purge_frame(), d(2009, 12, 1))


def test_assert_purged_fails_on_one_overlapping_row():
    df = purge_frame().with_columns(
        pl.when(pl.col("security_id") == 2).then(d(2010, 2, 1)).otherwise(pl.col("label_end_12m")).alias("label_end_12m")
    )
    with pytest.raises(LeakageError, match="1 training rows"):
        assert_purged(df, d(2010, 1, 29))


def test_assert_purged_fails_when_a_label_ends_on_the_first_test_date():
    with pytest.raises(LeakageError):
        assert_purged(purge_frame(), d(2009, 11, 30))


def test_assert_purged_null_label_end():
    null_end = pl.when(pl.col("security_id") == 3).then(None).otherwise(pl.col("label_end_12m")).alias("label_end_12m")
    # a target with no known label end cannot be proven clean
    with pytest.raises(LeakageError):
        assert_purged(purge_frame().with_columns(null_end), d(2010, 1, 29))
    # no target either: the row has no label and is dropped before fitting
    no_label = purge_frame().with_columns(
        null_end, pl.when(pl.col("security_id") == 3).then(None).otherwise(pl.col("target")).alias("target")
    )
    assert_purged(no_label, d(2010, 1, 29))
    # without a target column a null label end is always a violation
    with pytest.raises(LeakageError):
        assert_purged(no_label.drop("target"), d(2010, 1, 29))


def test_assert_purged_other_columns():
    df = purge_frame().rename({"label_end_12m": "label_end_3m", "target": "rank_3m"})
    assert_purged(df, d(2010, 1, 29), "label_end_3m", target="rank_3m")
    with pytest.raises(LeakageError):
        assert_purged(df, d(2009, 10, 30), "label_end_3m", target="rank_3m")


def test_leakage_error_is_an_assertion_error():
    assert issubclass(LeakageError, AssertionError)

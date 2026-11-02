import datetime as dt

import polars as pl
import pytest
from conftest import business_days

from ltforecast.pit.asof import LookaheadError, add_usable_from, asof_join, assert_no_lookahead, usable_from

CAL = business_days()
d = dt.date


@pytest.mark.parametrize(
    ("published", "lag", "expected"),
    [
        (d(2019, 8, 2), 1, d(2019, 8, 5)),   # Friday filing -> Monday
        (d(2019, 8, 3), 1, d(2019, 8, 5)),   # Saturday filing -> Monday
        (d(2019, 8, 5), 1, d(2019, 8, 6)),
        (d(2019, 8, 5), 2, d(2019, 8, 7)),
        (d(2019, 8, 5), 0, d(2019, 8, 5)),
        (d(2019, 8, 3), 0, d(2019, 8, 5)),
    ],
)
def test_usable_from(published, lag, expected):
    assert usable_from(pl.Series([published]), CAL, lag)[0] == expected


def test_usable_from_beyond_calendar_is_null():
    assert usable_from(pl.Series([d(2020, 12, 31)]), CAL, 1)[0] is None


def test_asof_join_never_sees_filing_on_its_own_day():
    fund = add_usable_from(
        pl.DataFrame({
            "security_id": [1, 1],
            "available_date": [d(2019, 5, 1), d(2019, 8, 2)],
            "revenue": [100.0, 120.0],
        }),
        CAL,
        lag=1,
    )
    left = pl.DataFrame({"security_id": [1, 1, 1, 1], "date": [d(2019, 4, 30), d(2019, 8, 2), d(2019, 8, 5), d(2019, 8, 30)]})
    out = asof_join(left, fund).sort("date")
    assert out["revenue"].to_list() == [None, 100.0, 120.0, 120.0]


def test_asof_join_requires_usable_from():
    with pytest.raises(ValueError):
        asof_join(pl.DataFrame({"security_id": [1], "date": [d(2019, 1, 2)]}),
                  pl.DataFrame({"security_id": [1], "available_date": [d(2019, 1, 1)]}))


def test_assert_no_lookahead():
    ok = pl.DataFrame({"date": [d(2020, 1, 2)], "src": [d(2020, 1, 2)]})
    assert_no_lookahead(ok, "date", "src")
    with pytest.raises(LookaheadError):
        assert_no_lookahead(ok.with_columns(pl.lit(d(2020, 1, 3)).alias("src")), "date", "src")

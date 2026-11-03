"""Triple-barrier labels on hand-built price paths (plan 2.7).

Entry 100 on 2020-01-31, sigma 0.2, k = 1: TP at 100*e^0.2 = 122.14, SL at 100*e^-0.2 = 81.87.
Window: 5 trading days, Feb 3 - Feb 7.
"""

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest
from conftest import business_days

from ltforecast.config import BarrierRules
from ltforecast.levels import barrier_labels, level_ratio, trailing_sigma

d = dt.date
CAL = business_days(d(2020, 1, 1), d(2020, 3, 31))
ENTRY = d(2020, 1, 31)
TP, SL = math.exp(0.2), math.exp(-0.2)
RULES = BarrierRules(horizon="1w", k_tp=(1.0, None), k_sl=(1.0, None), tp_bounds=(0.10, 1.00),
                     sl_bounds=(0.08, 0.40), sigma_window=252, sigma_min_obs=126)

# security_id: {date: (open, close)} on the days that differ from 100/100; `scale` = close / adj_close
PATHS = {
    1: {d(2020, 2, 3): (100, 105), d(2020, 2, 4): (105, 110), d(2020, 2, 5): (110, 123)},  # TP crossed intraday
    2: {d(2020, 2, 4): (130, 128)},                                                       # TP gap up
    3: {d(2020, 2, 4): (75, 78)},                                                         # SL gap down
    4: {d(2020, 2, 6): (85, 80)},                                                         # SL crossed intraday
    5: {d(2020, 2, 7): (100, 104)},                                                       # neither
    6: {d(2020, 2, 3): (98, 95)},                                                         # delists after Feb 3
}
LAST = {6: d(2020, 2, 3)}
SCALE = {2: 2.0}  # raw prices are twice the total-return path (e.g. dividends since)


def make_prices() -> pl.DataFrame:
    rows = []
    for sid, moves in PATHS.items():
        level = (100.0, 100.0)
        for day in CAL:
            if day > LAST.get(sid, CAL[-1]):
                break
            level = moves.get(day, (level[1], level[1]))
            s = SCALE.get(sid, 1.0)
            rows.append((sid, day, level[0] * s, level[1] * s, float(level[1])))
    return pl.DataFrame(rows, schema=["security_id", "date", "open", "close", "adj_close"], orient="row")


def run(sigma=0.2, rules=RULES):
    prices = make_prices()
    entries = pl.DataFrame({"security_id": list(PATHS), "date": ENTRY, "sigma": sigma})
    delist = pl.DataFrame({"security_id": [6], "date": [d(2020, 2, 4)], "delist_return": [-0.30]})
    return barrier_labels(prices, entries, delist, CAL, 5, rules)


def outcome(df, sid, k_tp=1.0, k_sl=1.0):
    f = pl.col("security_id") == sid
    f &= pl.col("k_tp").is_null() if k_tp is None else pl.col("k_tp") == k_tp
    f &= pl.col("k_sl").is_null() if k_sl is None else pl.col("k_sl") == k_sl
    r = df.filter(f).row(0, named=True)
    return r["outcome"], r["exit_day"], r["exit_ret"]


def test_level_guardrails():
    s = np.array([0.01, 0.2, 2.0])
    assert level_ratio(s, 1.0, (0.10, 1.00), "tp") - 1 == pytest.approx([0.10, TP - 1, 1.00])
    assert level_ratio(s, 1.0, (0.08, 0.40), "sl") - 1 == pytest.approx([-0.08, SL - 1, -0.40])
    assert level_ratio(s, None, (0.08, 0.40), "sl") is None


def test_barrier_outcomes():
    df = run()
    assert df.height == len(PATHS) * 4
    assert outcome(df, 1) == ("tp", 3, pytest.approx(TP - 1))    # crossed during the day: fills at TP
    assert outcome(df, 2) == ("tp", 2, pytest.approx(0.30))      # gapped above TP: fills at the open
    assert outcome(df, 3) == ("sl", 2, pytest.approx(-0.25))     # gapped below SL: fills at the open
    assert outcome(df, 4) == ("sl", 4, pytest.approx(SL - 1))    # crossed during the day: fills at SL
    assert outcome(df, 5) == ("time", 5, pytest.approx(0.04))
    # -30% delisting on Feb 4 gaps through the stop: exit at 95 * 0.7
    assert outcome(df, 6) == ("sl", 2, pytest.approx(0.95 * 0.7 - 1))


def test_missing_barriers():
    df = run()
    assert outcome(df, 1, k_tp=None) == ("time", 5, pytest.approx(0.23))
    assert outcome(df, 3, k_sl=None) == ("time", 5, pytest.approx(-0.22))
    assert outcome(df, 6, k_sl=None) == ("time", 2, pytest.approx(0.95 * 0.7 - 1))
    row = df.filter((pl.col("security_id") == 5) & pl.col("k_tp").is_null() & pl.col("k_sl").is_null()).row(0, named=True)
    assert row["tp_level"] is None and row["sl_level"] is None and row["exit_date"] == d(2020, 2, 7)


def test_entries_without_full_window_or_sigma_are_skipped():
    prices = make_prices()
    entries = pl.DataFrame({"security_id": [5, 5, 4], "date": [ENTRY, d(2020, 3, 31), ENTRY], "sigma": [0.2, 0.2, None]})
    empty = pl.DataFrame(schema={"security_id": pl.Int64, "date": pl.Date, "delist_return": pl.Float64})
    df = barrier_labels(prices, entries, empty, CAL, 5, RULES)
    assert set(df.select("security_id", "date").unique().rows()) == {(5, ENTRY)}


def test_trailing_sigma_uses_only_past_returns():
    rets = [None, 0.01, -0.01, 0.02, -0.02, 0.03]
    p = pl.DataFrame({"security_id": 1, "date": CAL[:6], "ret": rets})
    s = trailing_sigma(p, window=3, min_obs=3)["sigma"].to_list()
    assert s[:3] == [None, None, None]
    lr = np.log1p(np.array(rets[3:6], dtype=float))
    assert s[5] == pytest.approx(lr.std(ddof=1) * math.sqrt(252))

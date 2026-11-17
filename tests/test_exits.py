"""Exit rules, sleeve portfolio and the TP/SL multiplier choice (plan 6.5) on hand-built paths."""

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest
from conftest import business_days

from ltforecast import exits
from ltforecast.config import BarrierRules
from ltforecast.exits import (
    Paths,
    exit_entries,
    fixed,
    hold,
    max_drawdown,
    net_sharpe,
    portfolio_returns,
    position_paths,
    select_pair,
    signal,
    signal_exit_days,
    trailing,
)
from ltforecast.levels import barrier_labels, level_ratio
from ltforecast.validation.experiments import read_runs

d = dt.date
CAL = business_days(d(2020, 1, 1), d(2020, 6, 30))
GRID = BarrierRules(horizon="x", k_tp=(0.75, 1.0, 1.5, 2.0, None), k_sl=(0.5, 0.75, 1.0, 1.5, None),
                    tp_bounds=(0.10, 1.00), sl_bounds=(0.08, 0.40), sigma_window=252, sigma_min_obs=126)
SMALL = BarrierRules(horizon="x", k_tp=(1.0, None), k_sl=(1.0, None), tp_bounds=(0.10, 1.00),
                     sl_bounds=(0.08, 0.40), sigma_window=252, sigma_min_obs=126)
TP, SL = math.exp(0.2), math.exp(-0.2)  # sigma 0.2, k = 1


def mk_paths(close, open_=None, d0=None, extra=None) -> Paths:
    """Paths built by hand: row i entered at calendar index d0[i] (default 0)."""
    close = np.asarray(close, dtype=np.float64)
    n, h = close.shape
    open_ = close.copy() if open_ is None else np.asarray(open_, dtype=np.float64)
    d0 = np.zeros(n, dtype=np.int64) if d0 is None else np.asarray(d0, dtype=np.int64)
    ent = pl.DataFrame({"security_id": np.arange(n, dtype=np.int64), "date": CAL.gather(d0), "d": d0} | (extra or {}))
    return Paths(entries=ent, close=close, open=open_, horizon=h, calendar=CAL)


# --------------------------------------------------------------------------------------------- reproduce levels


def crafted_market():
    """Six crafted stocks (TP crossed / gapped, SL crossed / gapped, neither, a delisting) plus six random
    walks with overnight gaps, missing days, a split-style raw/adj scale and two delistings."""
    rows = []
    moves = {
        1: {3: (100, 105), 4: (105, 110), 5: (110, 123)},
        2: {4: (130, 128)},
        3: {4: (75, 78)},
        4: {6: (85, 80)},
        5: {7: (100, 104)},
        6: {3: (98, 95)},
    }
    for sid, mv in moves.items():
        lvl = (100.0, 100.0)
        for i in range(len(CAL)):
            if sid == 6 and i > 3:
                break
            lvl = mv.get(i, (lvl[1], lvl[1]))
            s = 2.0 if sid == 2 else 1.0
            rows.append((sid, CAL[i], lvl[0] * s, lvl[1] * s, float(lvl[1])))
    rng = np.random.default_rng(7)
    for sid in range(11, 17):
        first, last = int(rng.integers(0, 10)), len(CAL) - 1 - int(rng.integers(0, 10))
        if sid in (12, 15):
            last = 60 + sid
        c = 50.0
        for i in range(first, last + 1):
            o = c * math.exp(rng.normal(0, 0.03))
            c = o * math.exp(rng.normal(0, 0.03))
            if rng.random() < 0.05 and i not in (first, last):
                continue  # untraded day
            rows.append((sid, CAL[i], o * 3, c * 3, c))
    prices = pl.DataFrame(rows, schema=["security_id", "date", "open", "close", "adj_close"], orient="row")
    delist = pl.DataFrame({"security_id": [6, 12, 15], "date": [CAL[4], CAL[73], CAL[78]],
                           "delist_return": [-0.30, -0.5, 0.1]})
    ent = [(sid, CAL[0 if sid < 10 else j], float(rng.uniform(0.05, 0.9)))
           for sid in [*moves, *range(11, 17)] for j in ([0] if sid < 10 else range(10, 100, 7))]
    entries = pl.DataFrame(ent, schema=["security_id", "date", "sigma"], orient="row")
    entries = entries.with_columns(pl.when(pl.col("security_id") < 10).then(0.2).otherwise(pl.col("sigma")).alias("sigma"))
    last = prices.group_by("security_id").agg(pl.col("date").max().alias("last"))
    entries = entries.join(last, on="security_id").filter(pl.col("date") <= pl.col("last")).drop("last")
    return prices, entries, delist


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_fixed_reproduces_barrier_labels(dtype):
    prices, entries, delist = crafted_market()
    for h in (5, 20):
        lab = barrier_labels(prices, entries, delist, CAL, h, GRID)
        assert set(lab["outcome"]) == {"tp", "sl", "time"}
        paths = position_paths(prices, entries, delist, CAL, h, dtype=dtype)
        assert paths.close.dtype == dtype and len(paths) == lab.select("security_id", "date").unique().height
        sigma = paths.entries["sigma"].to_numpy()
        keys = paths.entries.select("security_id", "date")
        for k_tp in GRID.k_tp:
            for k_sl in GRID.k_sl:
                day, ret = fixed(paths, level_ratio(sigma, k_tp, GRID.tp_bounds, "tp"),
                                 level_ratio(sigma, k_sl, GRID.sl_bounds, "sl"))
                f = pl.col("k_tp").is_null() if k_tp is None else pl.col("k_tp") == k_tp
                f &= pl.col("k_sl").is_null() if k_sl is None else pl.col("k_sl") == k_sl
                ref = keys.join(lab.filter(f), on=["security_id", "date"], how="left", maintain_order="left")
                assert ref["exit_day"].to_list() == day.tolist(), (h, k_tp, k_sl)
                exp = ref["exit_ret"].fill_null(np.nan).to_numpy()
                tol = 1e-12 if dtype is np.float64 else 1e-6
                np.testing.assert_allclose(ret, exp, rtol=tol, atol=tol, equal_nan=True)


def test_position_paths_values_delisting_and_drops():
    prices, entries, delist = crafted_market()
    extra = pl.DataFrame({"security_id": [1, 1], "date": [d(2020, 1, 4), CAL[-3]], "sigma": [0.2, 0.2]})  # Saturday; short window
    p = position_paths(prices, pl.concat([entries, extra]), delist, CAL, 5, dtype=np.float64)
    assert p.entries.height == entries.height  # off-calendar and incomplete windows dropped
    assert p.entries["date"].is_sorted()
    row = {sid: i for i, sid in enumerate(p.entries.filter(pl.col("date") == CAL[0])["security_id"])}
    np.testing.assert_allclose(p.close[row[1]], [1.0, 1.0, 1.05, 1.10, 1.23])
    np.testing.assert_allclose(p.open[row[2]], [1.0, 1.0, 1.0, 1.30, 1.28])  # raw open / scale onto adj path
    # delisted after CAL[3]: one final day at 0.95 * 0.7, then nothing
    np.testing.assert_allclose(p.close[row[6]], [1.0, 1.0, 0.95, 0.665, np.nan])
    np.testing.assert_allclose(p.open[row[6]], [1.0, 1.0, 0.98, 0.665, np.nan])
    assert hold(p)[0][row[6]] == 4 and hold(p)[1][row[6]] == pytest.approx(-0.335)


def test_none_none_equals_hold():
    prices, entries, delist = crafted_market()
    p = position_paths(prices, entries, delist, CAL, 20)
    a, b = fixed(p, None, None), hold(p)
    assert (a[0] == b[0]).all()
    np.testing.assert_array_equal(a[1], b[1])


# --------------------------------------------------------------------------------------------- exit rules


def test_hold_on_five_day_path():
    p = mk_paths([[1.01, 0.99, np.nan, 1.04, 1.06],
                  [1.02, 0.97, 0.90, np.nan, np.nan],   # data ends after day 3
                  [np.nan] * 5])                        # no prices at all
    day, ret = hold(p)
    assert day.tolist() == [5, 3, 5]
    np.testing.assert_allclose(ret, [0.06, -0.10, np.nan])
    assert day.dtype == np.int64 and ret.dtype == np.float64


def test_fixed_gap_and_intraday_fills():
    close = [[1.05, 1.10, 1.25, 1.30, 1.00],   # TP crossed intraday on day 3: fills at TP
             [1.00, 1.28, 1.28, 1.28, 1.28],   # opens at 1.30 on day 2: fills at the open
             [0.95, 0.78, 0.78, 0.78, 0.78],   # gaps to 0.75 on day 2: fills at the open
             [1.00, 1.00, 1.00, 1.04, 1.04]]   # neither: time exit
    open_ = np.array(close)
    open_[1, 1], open_[2, 1] = 1.30, 0.75
    open_[0, 2] = 1.12
    day, ret = fixed(mk_paths(close, open_), TP, SL)
    assert day.tolist() == [3, 2, 2, 5]
    np.testing.assert_allclose(ret, [TP - 1, 0.30, -0.25, 0.04])


def test_trailing_stop_with_gap():
    close = [[1.05, 1.20, 1.10, 1.02, 1.30],   # max before day 4 = 1.20, stop 1.08; gap open 1.00
             [1.05, 1.20, 1.10, 1.05, 1.30],   # same, but opens at 1.15: fills at the stop 1.08
             [0.95, 0.89, 0.95, 1.00, 1.01],   # stop from entry 0.90: day 2 close 0.89, open 0.92
             [1.01, 1.02, 1.03, 1.04, 1.05]]   # never stopped: time exit
    open_ = np.array(close)
    open_[0, 3], open_[1, 3], open_[2, 1] = 1.00, 1.15, 0.92
    day, ret = trailing(mk_paths(close, open_), 0.10)
    assert day.tolist() == [4, 4, 2, 5]
    np.testing.assert_allclose(ret, [0.0, 0.08, -0.10, 0.05], atol=1e-12)
    # per-position distances
    day, ret = trailing(mk_paths(close, open_), np.array([0.5, 0.5, 0.5, 0.01]))
    assert day.tolist() == [5, 5, 5, 5]


def test_signal_exit_at_supplied_day():
    close = [[1.01, 1.02, 1.03, 1.04, 1.05],
             [1.01, np.nan, 1.03, 1.04, 1.05],     # no price on the signal day: next close
             [1.01, 1.02, np.nan, np.nan, np.nan],  # data ends before the signal: hold exit
             [1.01, 1.02, 1.03, 1.04, 1.05]]       # h = no signal
    day, ret = signal(mk_paths(close), np.array([2, 2, 4, 5]))
    assert day.tolist() == [2, 3, 2, 5]
    np.testing.assert_allclose(ret, [0.02, 0.03, 0.02, 0.05])


def test_signal_exit_days_from_scores():
    p = mk_paths(np.ones((3, 30)), d0=[0, 0, 5])
    sid = p.entries["security_id"]
    scores = pl.DataFrame({
        "security_id": [0, 0, 0, 1, 2, 2],
        "date": [CAL[0], CAL[21], CAL[42], CAL[21], CAL[5], CAL[26]],
        "rank_pct": [0.1, 0.35, 0.2, 0.55, 0.1, 0.39],
    })
    assert sid.to_list() == [0, 1, 2]
    # sid 0: the entry-day score does not count; first low re-score is day 21. sid 1: none. sid 2: day 21.
    assert signal_exit_days(p, scores).tolist() == [21, 30, 21]


# --------------------------------------------------------------------------------------------- portfolio


def test_portfolio_two_overlapping_sleeves_with_costs():
    c = 0.01
    close = [[1.10, 1.21, 1.331],    # sleeve A (entered day 0), held to the horizon
             [0.90, 0.80, 0.85],     # sleeve A, stopped on day 2 at 0.80
             [1.02, 1.04, 1.00]]     # sleeve B (entered day 1), signal exit on day 1
    p = mk_paths(close, d0=[0, 0, 1])
    out = portfolio_returns(None, p, np.array([3, 2, 1]), np.array([0.331, -0.20, 0.02]), c)
    k = 1 - c
    a = [1.0, (1.10 + 0.90) / 2 * k, (1.21 + 0.80 * k) / 2 * k, (1.331 * k + 0.80 * k) / 2 * k]
    b = [1.0, 1.02 * k * k, 1.02 * k * k, 1.02 * k * k]
    ra = [a[i] / a[i - 1] - 1 for i in (1, 2, 3)]
    rb = [b[i] / b[i - 1] - 1 for i in (1, 2, 3)]
    assert ra[0] == pytest.approx(-0.01)
    assert out["date"].to_list() == list(CAL[1:5])
    np.testing.assert_allclose(out["ret"].to_numpy(), [ra[0], (ra[1] + rb[0]) / 2, (ra[2] + rb[1]) / 2, rb[2]],
                               rtol=1e-12)
    assert rb[1] == 0 and rb[2] == 0  # cash at 0% after the exit
    # a subset of positions (boolean mask) = sleeve A alone
    only_a = portfolio_returns(np.array([True, True, False]), p, np.array([3, 2, 1]), np.array([0.331, -0.20, 0.02]), c)
    np.testing.assert_allclose(only_a["ret"].to_numpy(), ra)


def test_portfolio_skips_nan_positions_and_forward_fills():
    p = mk_paths([[1.1, np.nan, 1.2], [np.nan] * 3])
    out = portfolio_returns(None, p, np.array([3, 3]), np.array([0.2, np.nan]), 0.0)
    np.testing.assert_allclose(out["ret"].to_numpy(), [0.1, 0.0, 1.2 / 1.1 - 1])
    empty = portfolio_returns(np.array([1]), p, np.array([3, 3]), np.array([0.2, np.nan]), 0.0)
    assert empty.is_empty() and empty.schema == {"date": pl.Date, "ret": pl.Float64}


def test_net_sharpe_and_max_drawdown_literal():
    r = [0.01, -0.01, 0.02, 0.0]
    assert net_sharpe(r) == pytest.approx(0.005 / math.sqrt(5e-4 / 3) * math.sqrt(252))
    assert net_sharpe(pl.DataFrame({"ret": r})) == pytest.approx(net_sharpe(r))
    assert math.isnan(net_sharpe([0.01, 0.01])) and math.isnan(net_sharpe([0.01]))
    assert max_drawdown([0.10, -0.20, 0.05, -0.10, 0.30]) == pytest.approx(0.8316 / 1.1 - 1)
    assert max_drawdown([-0.5, 0.1]) == pytest.approx(-0.5)  # from the starting capital
    assert max_drawdown([0.01, 0.02]) == 0.0 and max_drawdown([]) == 0.0


# --------------------------------------------------------------------------------------------- selection


def fold_paths(kinds: list[list[float]], n_folds=3, sleeves=2, per=4) -> Paths:
    """Positions in n_folds folds x sleeves monthly entries x per names, cycling through `kinds`
    (close paths), each scaled slightly so the names differ."""
    close, d0, fold = [], [], []
    for f in range(n_folds):
        for s in range(sleeves):
            for j in range(per):
                base = np.asarray(kinds[(f + s + j) % len(kinds)])
                close.append(1 + (base - 1) * (1 + 0.05 * j + 0.02 * s + 0.01 * f))
                d0.append(10 * (2 * f + s))
                fold.append(f"f{f}")
    n = len(close)
    return mk_paths(np.array(close), d0=d0, extra={"fold": fold, "sigma": np.full(n, 0.2)})


def test_select_pair_take_profit_clearly_wins(tmp_path):
    spike = [1.05, 1.25, 1.00, 0.80, 0.70]   # TP on day 2, then a crash
    dip = [0.80, 1.00, 1.25, 1.00, 0.90]     # a stop would sell the low; TP sells day 3
    p = fold_paths([spike, dip])
    sig_days = np.full(len(p), 2)
    sel = select_pair(p, SMALL, cost=0.002, k_trail=(1.0,), signal_days=sig_days, exp_dir=tmp_path)
    assert (sel.k_tp, sel.k_sl) == (1.0, None)
    assert sel.verdict == "exit_orders"
    s = sel.summary.filter(pl.col("k_tp") == 1.0, pl.col("k_sl").is_null()).row(0, named=True)
    assert s["mean_sharpe"] > sel.hold["sharpe"] and s["mean_max_dd"] >= sel.hold["max_dd"]
    assert sel.table.height == 4 * 3 and sel.table["fold"].unique().sort().to_list() == ["f0", "f1", "f2"]
    hold_rows = sel.table.filter(pl.col("k_tp").is_null() & pl.col("k_sl").is_null())
    assert hold_rows["sharpe"].mean() == pytest.approx(sel.hold["sharpe"])
    v = sel.variants
    assert sorted(v["variant"].unique().to_list()) == ["hold", "signal", "trailing"] and v.height == 9
    assert sel.n_trials == 4 + 1 + 1
    runs = read_runs(tmp_path)
    assert runs.height == 1 and runs["n_trials"].to_list() == [6]


def test_select_pair_none_beats_hold_is_risk_reference():
    up = [0.80, 0.95, 1.05, 1.10, 1.15]     # stops sell the dip; TP (1.22) is never reached
    slow = [0.99, 1.02, 1.04, 1.06, 1.12]
    p = fold_paths([up, slow])
    sel = select_pair(p, SMALL, cost=0.002, k_trail=())
    assert sel.verdict == "risk_reference"
    others = sel.summary.filter(~(pl.col("k_tp").is_null() & pl.col("k_sl").is_null()))
    assert (others["mean_sharpe"] <= sel.hold["sharpe"] + 1e-12).all()
    # TP-only never triggers, so it equals hold and is the reference pair shown
    assert (sel.k_tp, sel.k_sl) == (1.0, None)
    assert sel.n_trials == 4 and set(sel.variants["variant"]) == {"hold"}


def test_select_pair_needs_fold_and_sigma_and_three_folds():
    p = mk_paths(np.ones((2, 5)))
    with pytest.raises(ValueError, match="fold"):
        select_pair(p, SMALL, cost=0.0)
    p2 = fold_paths([[1.01, 1.03, 1.02, 1.05, 1.04]], n_folds=2)
    with pytest.raises(ValueError, match="at least"):
        select_pair(p2, SMALL, cost=0.0, k_trail=())


def test_exit_entries_top_quintile():
    oof = pl.DataFrame({
        "date": [CAL[0]] * 10 + [CAL[20]] * 5,
        "security_id": list(range(10)) + list(range(5)),
        "fold": ["f0"] * 15,
        "pred": [float(x) for x in range(10)] + [0.5, 0.5, None, 0.1, 0.2],
    })
    sigma = pl.DataFrame({"security_id": [9, 8, 0, 1], "date": [CAL[0], CAL[0], CAL[20], CAL[20]],
                          "sigma": [0.3, 0.2, 0.4, 0.4]})
    out = exit_entries(oof, sigma)
    # date 0: top 2 of 10 = ids 9, 8. date 20: 4 valid preds -> 1 name; tie at 0.5 broken by security_id
    assert out.select("date", "security_id").rows() == [(CAL[0], 8), (CAL[0], 9), (CAL[20], 0)]
    assert out.columns == ["security_id", "date", "fold", "pred", "sigma"]


def test_trail_distance_bounds():
    s = np.array([0.01, 0.2, 2.0])
    np.testing.assert_allclose(exits.trail_distance(s, 1.0, (0.08, 0.40)), [0.08, 1 - SL, 0.40])


# --------------------------------------------------------------------------------------------- stale paths


def stale_inputs(h=10):
    """Prices falling 1% a day from 100 with open = previous close. Last price day per sid:
    1 -> CAL[5] (stale), 2 -> CAL[5] + delisting on CAL[6] (known), 3 -> CAL[7] (window index 6 =
    h - 1 - 3: not stale), 4 -> CAL[6] (index 5: stale), 5 -> full path, 6 -> CAL[3] with a delisting
    on CAL[15], after the horizon day CAL[10] (stale, as labels.forward_returns)."""
    last = {1: 5, 2: 5, 3: 7, 4: 6, 5: h + 2, 6: 3}
    rows = [(sid, CAL[i], 100 * 0.99 ** max(i - 1, 0), 100 * 0.99**i, 100 * 0.99**i)
            for sid, n in last.items() for i in range(n + 1)]
    prices = pl.DataFrame(rows, schema=["security_id", "date", "open", "close", "adj_close"], orient="row")
    delist = pl.DataFrame({"security_id": [2, 6], "date": [CAL[6], CAL[15]], "delist_return": [-0.5, -0.2]})
    entries = pl.DataFrame({"security_id": list(last), "date": [CAL[0]] * len(last), "sigma": [0.2] * len(last)})
    return prices, entries, delist


def stale_market(h=10):
    return position_paths(*stale_inputs(h), CAL, h, dtype=np.float64, stale_days=3)


def test_position_paths_marks_stale_like_forward_returns():
    p = stale_market()
    row = {sid: i for i, sid in enumerate(p.entries["security_id"])}
    assert {sid for sid, i in row.items() if p.stale[i]} == {1, 4, 6}
    # off by default
    prices_off = position_paths(*crafted_market(), CAL, 5)
    assert prices_off.stale is None


def test_stale_time_and_hold_exits_are_unknown_but_earlier_hits_are_kept():
    p = stale_market()
    row = {sid: i for i, sid in enumerate(p.entries["security_id"])}
    day, ret = hold(p)
    assert np.isnan(ret[row[1]]) and np.isnan(ret[row[4]]) and np.isnan(ret[row[6]])
    # delisted inside the window: the appended delisting day is the exit (day 6), value 0.99^5 * 0.5
    assert day[row[2]] == 6 and ret[row[2]] == pytest.approx(0.99**5 * 0.5 - 1)
    assert ret[row[3]] == pytest.approx(0.99**7 - 1) and ret[row[5]] == pytest.approx(0.99**10 - 1)
    # SL far away (0.80): never hit, so the stale positions have no known outcome
    day, ret = fixed(p, None, 0.80)
    assert np.isnan(ret[[row[1], row[4], row[6]]]).all() and np.isfinite(ret[[row[2], row[3], row[5]]]).all()
    # SL at 0.97: the close 0.99^4 = 0.9606 on day 4 crosses it; the open 0.99^3 = 0.9703 did not gap,
    # so the fill is the level. The stale sid 1 keeps this known 'sl' exit.
    day, ret = fixed(p, None, 0.97)
    assert day[row[1]] == 4 and ret[row[1]] == pytest.approx(-0.03)
    assert day[row[6]] == 3 and np.isnan(ret[row[6]])  # sid 6's last price is day 3, before the stop
    tday, tret = trailing(p, np.full(len(p), 0.5))
    assert np.isnan(tret[row[1]]) and np.isfinite(tret[row[5]])


def test_select_pair_leaves_stale_positions_out():
    spike = [1.05, 1.25, 1.00, 0.80, 0.70]
    dip = [0.80, 1.00, 1.25, 1.00, 0.90]
    p = fold_paths([spike, dip])
    rng = np.random.default_rng(0)
    stale = rng.random(len(p)) < 0.3
    stale[::4] = False  # every sleeve keeps a position
    junk = p.close.copy()
    junk[stale, 2:] = np.nan  # prices stop early: no TP hit inside, so these would be time exits
    with_stale = Paths(entries=p.entries, close=junk, open=junk.copy(), horizon=p.horizon, calendar=CAL, stale=stale)
    keep = np.flatnonzero(~stale)
    without = Paths(entries=p.entries[keep], close=junk[keep], open=junk[keep].copy(), horizon=p.horizon, calendar=CAL)
    a = select_pair(with_stale, SMALL, cost=0.002, k_trail=())
    b = select_pair(without, SMALL, cost=0.002, k_trail=())
    assert a.table.equals(b.table) and a.verdict == b.verdict
    # without the mask, the stale rows are cashed out at their last print and the table changes
    c = select_pair(Paths(entries=p.entries, close=junk, open=junk.copy(), horizon=p.horizon, calendar=CAL),
                    SMALL, cost=0.002, k_trail=())
    assert not c.table.equals(a.table)


# --------------------------------------------------------------------------------------------- verdict and pick


def test_pick_drawdown_eligibility_and_fallback():
    hold_ = {"k_tp": None, "k_sl": None, "sharpe": 1.0, "max_dd": -0.10}
    deep = {"k_tp": 1.0, "k_sl": None, "sharpe": 2.0, "max_dd": -0.20}     # best Sharpe, worse drawdown
    shallow = {"k_tp": None, "k_sl": 1.0, "sharpe": 1.5, "max_dd": -0.05}
    assert exits._pick([hold_, deep, shallow]) is shallow
    assert exits._pick([hold_, deep]) is deep              # no eligible pair: fall back to all non-hold
    assert exits._pick([hold_]) is None


def verdict_table(sharpe: dict, dd: dict, folds=("f0", "f1", "f2")) -> pl.DataFrame:
    """Rows (k_tp, k_sl, fold, sharpe, max_dd) from {pair: [per-fold sharpe]} and {pair: [per-fold max_dd]}."""
    rows = [{"k_tp": a, "k_sl": b, "fold": f, "sharpe": sharpe[(a, b)][i], "max_dd": dd[(a, b)][i]}
            for (a, b) in sharpe for i, f in enumerate(folds)]
    return pl.DataFrame(rows, schema={"k_tp": pl.Float64, "k_sl": pl.Float64, "fold": pl.String,
                                      "sharpe": pl.Float64, "max_dd": pl.Float64})


def noisy_diffs(pairs, folds=("f0", "f1", "f2"), mean=0.001, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return {(a, b, f): mean + rng.normal(0, 0.001, n) for (a, b) in pairs for f in folds}


H0, A_, B_ = (None, None), (1.0, None), (None, 1.0)


def test_holdout_verdict_leave_one_fold_out_overturns_in_sample_winner():
    # in sample A has the best mean Sharpe (1.833 vs B 1.5 vs hold 1.0): the old rule said exit_orders.
    # Leave f0 out: A (3 - 0.5)/2 = 1.25 < B 1.5 -> B scores 1.5 on f0; same for f1;
    # leave f2 out: A 3.0 -> A scores -0.5 on f2. Held-out mean (1.5 + 1.5 - 0.5)/3 = 0.833 < 1.0.
    sh = {H0: [1.0, 1.0, 1.0], A_: [3.0, 3.0, -0.5], B_: [1.5, 1.5, 1.5]}
    dd = {H0: [-0.1] * 3, A_: [-0.05] * 3, B_: [-0.05] * 3}
    v, info = exits.holdout_verdict(verdict_table(sh, dd), noisy_diffs(sh))
    assert v == "risk_reference"
    assert [(p["fold"], p["k_tp"], p["k_sl"]) for p in info["picks"]] == [
        ("f0", None, 1.0), ("f1", None, 1.0), ("f2", 1.0, None)]
    assert info["sharpe"] == pytest.approx(2.5 / 3) and info["hold_sharpe"] == pytest.approx(1.0)
    assert info["max_dd"] == pytest.approx(-0.05) and info["n_days"] == 600


def test_holdout_verdict_needs_sharpe_drawdown_and_significance():
    sh = {H0: [1.0, 1.0, 1.0], B_: [1.5, 1.5, 1.5]}
    good_dd = {H0: [-0.1] * 3, B_: [-0.05] * 3}
    v, info = exits.holdout_verdict(verdict_table(sh, good_dd), noisy_diffs(sh))
    assert v == "exit_orders" and info["nw_tstat"] >= 2 and info["sharpe"] == pytest.approx(1.5)
    # B is the only non-hold pair, so it is picked although its drawdown is worse than hold's:
    # the verdict's own drawdown clause must say risk_reference
    bad_dd = {H0: [-0.1] * 3, B_: [-0.2] * 3}
    v, info = exits.holdout_verdict(verdict_table(sh, bad_dd), noisy_diffs(sh))
    assert v == "risk_reference" and info["max_dd"] == pytest.approx(-0.2) and info["nw_tstat"] >= 2
    # better Sharpe and drawdown but the daily difference is noise: t < 2
    v, info = exits.holdout_verdict(verdict_table(sh, good_dd), noisy_diffs(sh, mean=0.0, seed=3))
    assert v == "risk_reference" and abs(info["nw_tstat"]) < 2
    # one fold: nothing held out
    one = verdict_table({H0: [1.0], B_: [2.0]}, {H0: [-0.1], B_: [-0.05]}, folds=("f0",))
    v, info = exits.holdout_verdict(one, noisy_diffs(sh, folds=("f0",)))
    assert v == "risk_reference" and info["picks"] == []


def test_select_pair_summary_eligible_flag():
    up = [0.80, 0.95, 1.05, 1.10, 1.15]
    slow = [0.99, 1.02, 1.04, 1.06, 1.12]
    sel = select_pair(fold_paths([up, slow]), SMALL, cost=0.002, k_trail=())
    s = sel.summary
    assert s["eligible"].to_list() == (s["mean_max_dd"] >= sel.hold["max_dd"]).to_list()
    # SL-only sells the 0.80 dip: deeper drawdown than hold, so not eligible
    row = s.filter(pl.col("k_tp").is_null() & (pl.col("k_sl") == 1.0)).row(0, named=True)
    assert row["mean_max_dd"] < sel.hold["max_dd"] and row["eligible"] is False


def test_select_pair_scores_each_fold_on_its_own_positions():
    spike = [1.05, 1.25, 1.00, 0.80, 0.70]
    dip = [0.80, 1.00, 1.25, 1.00, 0.90]
    up = [1.01, 1.03, 1.02, 1.06, 1.08]
    kinds = {"f0": spike, "f1": dip, "f2": up}
    close, d0, fold = [], [], []
    for i, (f, base) in enumerate(kinds.items()):
        for s in range(2):
            for j in range(4):
                close.append(1 + (np.asarray(base) - 1) * (1 + 0.05 * j + 0.02 * s))
                d0.append(10 * (2 * i + s))
                fold.append(f)
    p = mk_paths(np.array(close), d0=d0, extra={"fold": fold, "sigma": np.full(len(close), 0.2)})
    sel = select_pair(p, SMALL, cost=0.002, k_trail=())
    sigma = p.entries["sigma"].to_numpy()
    rules = {(None, None): hold(p), (1.0, None): fixed(p, level_ratio(sigma, 1.0, SMALL.tp_bounds, "tp"), None),
             (None, 1.0): fixed(p, None, level_ratio(sigma, 1.0, SMALL.sl_bounds, "sl"))}
    fold_arr = np.array(fold)
    for (a, b), (day, ret) in rules.items():
        for f in kinds:
            r = portfolio_returns(fold_arr == f, p, day, ret, 0.002)
            f_a = pl.col("k_tp").is_null() if a is None else pl.col("k_tp") == a
            f_b = pl.col("k_sl").is_null() if b is None else pl.col("k_sl") == b
            row = sel.table.filter(f_a, f_b, pl.col("fold") == f).row(0, named=True)
            assert row["sharpe"] == pytest.approx(net_sharpe(r)), (a, b, f)
            assert row["max_dd"] == pytest.approx(max_drawdown(r)), (a, b, f)
    hold_dd = sel.table.filter(pl.col("k_tp").is_null() & pl.col("k_sl").is_null())["max_dd"]
    assert hold_dd.n_unique() == 3  # the folds really differ


# --------------------------------------------------------------------------------------------- null regression

NULL_H, NULL_FOLDS, NULL_SLEEVES, NULL_PER = 42, 4, 6, 12
NULL_CAL = pl.Series("date", [d(2000, 1, 3) + dt.timedelta(days=i)
                              for i in range(21 * NULL_SLEEVES * NULL_FOLDS + NULL_H + 5)])


def martingale_paths(seed: int, bad_drift: float = 0.0) -> Paths:
    """4 folds x 6 monthly sleeves x 12 names; t(4) shocks, a common market factor and an Ito
    correction so every price is a martingale: no exit rule can beat holding in expectation.
    bad_drift < 0 (positive control): half the names fall at that annual log drift from day 6 on,
    which a stop-loss cuts short."""
    rng = np.random.default_rng(seed)
    ds = np.repeat(21 * np.arange(NULL_FOLDS * NULL_SLEEVES), NULL_PER)
    n, h = len(ds), NULL_H
    sig = rng.uniform(0.2, 0.5, n)
    lr = rng.standard_t(4, size=(n, h)) / np.sqrt(2) * (sig / np.sqrt(252))[:, None]
    lr += (rng.standard_normal(len(NULL_CAL)) * 0.01)[ds[:, None] + np.arange(1, h + 1)[None, :]]
    lr -= (0.5 * sig**2 / 252)[:, None] + 0.5 * 0.01**2
    if bad_drift:
        lr[rng.random(n) < 0.5, 5:] += bad_drift / 252
    close = np.exp(np.cumsum(lr, axis=1))
    ent = pl.DataFrame({"security_id": np.arange(n) % NULL_PER, "date": NULL_CAL.gather(ds), "d": ds.astype(np.int64),
                        "fold": [f"f{x // (21 * NULL_SLEEVES)}" for x in ds], "sigma": sig})
    return Paths(entries=ent, close=close, open=close.copy(), horizon=h, calendar=NULL_CAL)


def test_verdict_under_the_null_is_mostly_risk_reference():
    # the old in-sample rule (best of 24 pairs vs hold on the same folds) said exit_orders in ~85%
    verdicts = [select_pair(martingale_paths(s), GRID, cost=0.002, k_trail=()).verdict for s in range(20)]
    assert verdicts.count("risk_reference") >= 16, verdicts


def test_verdict_positive_control_real_stop_benefit():
    for s in range(3):
        sel = select_pair(martingale_paths(100 + s, bad_drift=-3.0), GRID, cost=0.002, k_trail=())
        assert sel.verdict == "exit_orders" and sel.k_sl is not None
        assert sel.holdout["nw_tstat"] >= 2 and len(sel.holdout["picks"]) == 4


def test_barrier_labels_null_the_same_stale_positions_as_position_paths():
    h = 10
    prices, entries, delist = stale_inputs(h)
    p = stale_market(h)
    stale = {sid for sid, flag in zip(p.entries["security_id"], p.stale) if flag}
    # no TP; SL either absent or at exactly -3% (tiny k, lower bound 0.03)
    rules = BarrierRules(horizon="10d", k_tp=(None,), k_sl=(None, 1e-6), tp_bounds=(0.10, 1.00),
                         sl_bounds=(0.03, 0.40), sigma_window=252, sigma_min_obs=126)
    lab = barrier_labels(prices, entries, delist, CAL, h, rules, stale_days=3)
    no_sl = lab.filter(pl.col("k_sl").is_null())
    nulls = no_sl.filter(pl.col("outcome").is_null())
    assert set(nulls["security_id"]) == stale == {1, 4, 6}
    assert nulls.select(pl.col("exit_day", "exit_date", "exit_ret").is_null().all()).row(0) == (True, True, True)
    known = no_sl.filter(pl.col("outcome").is_not_null()).sort("security_id")
    assert known["outcome"].to_list() == ["time"] * 3 and known["security_id"].to_list() == [2, 3, 5]
    # sl at 0.97: sid 1 crosses on day 4 (window index 3) before its prices stop, so the hit is kept;
    # sid 6's prices stop at day 3 above the stop, so it stays unknown; sid 4 hits on day 4 as well
    sl = lab.filter(pl.col("k_sl").is_not_null()).sort("security_id")
    out = dict(zip(sl["security_id"], zip(sl["outcome"], sl["exit_day"])))
    assert out[1] == ("sl", 4) and out[4] == ("sl", 4) and out[6] == (None, None)
    # matches fixed() on the same stale-marked paths, null <-> NaN
    sigma = p.entries["sigma"].to_numpy()
    _, ret = fixed(p, None, level_ratio(sigma, 1e-6, rules.sl_bounds, "sl"))
    ref = p.entries.select("security_id", "date").join(sl, on=["security_id", "date"], how="left",
                                                     maintain_order="left")
    np.testing.assert_allclose(ret, ref["exit_ret"].fill_null(np.nan).to_numpy(), rtol=1e-12, equal_nan=True)
    # off by default: every row has an outcome
    assert barrier_labels(prices, entries, delist, CAL, h, rules)["outcome"].null_count() == 0


def test_a_window_without_any_price_is_stale_in_levels_and_exits_alike():
    # sid 1 trades only on the entry day CAL[0]; sid 2 the same but delists on CAL[3]; sid 3 trades daily
    h = 10
    rows = [(1, CAL[0], 100.0, 100.0, 100.0), (2, CAL[0], 100.0, 100.0, 100.0)]
    rows += [(3, CAL[i], 100.0, 100.0, 100.0) for i in range(h + 1)]
    prices = pl.DataFrame(rows, schema=["security_id", "date", "open", "close", "adj_close"], orient="row")
    delist = pl.DataFrame({"security_id": [2], "date": [CAL[3]], "delist_return": [-0.4]})
    entries = pl.DataFrame({"security_id": [1, 2, 3], "date": [CAL[0]] * 3, "sigma": [0.2] * 3})
    p = position_paths(prices, entries, delist, CAL, h, dtype=np.float64, stale_days=3)
    assert dict(zip(p.entries["security_id"], p.stale.tolist())) == {1: True, 2: False, 3: False}
    rules = BarrierRules(horizon="10d", k_tp=(None,), k_sl=(None,), tp_bounds=(0.10, 1.00),
                         sl_bounds=(0.08, 0.40), sigma_window=252, sigma_min_obs=126)
    lab = barrier_labels(prices, entries, delist, CAL, h, rules, stale_days=3).sort("security_id")
    assert lab["outcome"].to_list() == [None, "time", "time"]
    assert lab["exit_ret"].to_list()[1:] == pytest.approx([-0.4, 0.0])

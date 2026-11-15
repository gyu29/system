"""Risk features (features/risk.py): hand-computed values, null rules and point-in-time behaviour.

The trading calendar is every weekday of 2014-2020 (1827 days, 84 months). T = 2020-12-31 (a
Thursday month end) is the usual prediction date; T1 = 2020-06-30 is an earlier one for the
point-in-time tests. Windows are counted from the end of the calendar: CAL[-252:] are the 252 rows
whose returns form the one-year window at T.
"""

import math

import numpy as np
import polars as pl
import pytest
from conftest import business_days
from feature_helpers import d, make_benchmark, make_ctx, make_prices

from ltforecast.features import risk
from ltforecast.features.catalog import load_catalog

CAL = business_days(d(2014, 1, 1), d(2020, 12, 31))
N = len(CAL)
T = d(2020, 12, 31)
T1 = d(2020, 6, 30)
_BY_MONTH = CAL.to_frame("date").group_by(pl.col("date").dt.truncate("1mo").alias("m")).agg(
    pl.col("date").min().alias("first"), pl.col("date").max().alias("last")
).sort("m")
FIRST_DAYS = _BY_MONTH["first"].to_list()   # first trading day of each of the 84 months
MONTH_ENDS = _BY_MONTH["last"].to_list()
FEATURES = ["vol_1y", "idio_vol", "beta_60m", "max_ret_1m"]


def path(rets, p0=100.0) -> list[float]:
    """Prices p0, p0(1+r1), p0(1+r1)(1+r2), ...: one more price than returns."""
    return [p0, *(p0 * np.cumprod(1.0 + np.asarray(rets, dtype=float))).tolist()]


def tile(pattern, n) -> np.ndarray:
    return np.resize(np.asarray(pattern, dtype=float), n)


def at(out: pl.DataFrame, sid: int, date=T) -> dict:
    rows = out.filter((pl.col("security_id") == sid) & (pl.col("date") == date))
    assert rows.height == 1
    return rows.row(0, named=True)


def alternating_bench(b=0.01) -> pl.DataFrame:
    """Benchmark return +b, -b, +b, ... on every day of the calendar."""
    return make_benchmark(CAL, tile([b, -b], N).tolist())


def monthly_prices(start, rets_by_month: dict) -> dict:
    """{date: price} change points: 100 from `start`, then x(1 + r) on each given day."""
    out, p = {start: 100.0}, 100.0
    for day, r in rets_by_month.items():
        p *= 1.0 + r
        out[day] = p
    return out


# ---------------------------------------------------------------- module contract


def test_families_and_catalog_names():
    assert risk.FAMILIES == ("risk",)
    assert tuple(load_catalog().names(family="risk")) == risk.FEATURES == tuple(FEATURES)


def test_one_row_per_entry_schema_and_source_date():
    rng = np.random.default_rng(1)
    prices = make_prices({1: path(rng.normal(0, 0.02, N - 1)), 2: path(rng.normal(0, 0.01, N - 1))}, CAL)
    entries = [
        (1, T), (1, T1), (2, T),
        (2, d(2020, 12, 26)),   # a Saturday: no price row on that date
        (99, T),                # a security without any price
        (1, CAL[0]),            # first row: no return yet
    ]
    ctx = make_ctx(prices=prices, benchmark=alternating_bench(), entries=entries)
    out = risk.compute(ctx)

    assert out.height == ctx.entries.height == 6
    assert out.select("security_id", "date").is_unique().all()
    assert out.select("security_id", "date").equals(ctx.entries.select("security_id", "date"))
    assert out.columns == ["security_id", "date", *FEATURES, "src_date_risk"]
    assert [out.schema[c] for c in FEATURES] == [pl.Float64] * 4
    assert out.schema["src_date_risk"] == pl.Date
    for c in FEATURES:
        assert not out[c].is_nan().any() and out[c].is_finite().all() in (True, None)

    # full history: everything is set and the latest input used is the price of t itself
    full = at(out, 1, T)
    assert all(full[c] is not None for c in FEATURES) and full["src_date_risk"] == T
    assert at(out, 1, T1)["src_date_risk"] == T1
    # nothing to compute from: every feature and the source date are null
    for sid, day in [(2, d(2020, 12, 26)), (99, T), (1, CAL[0])]:
        row = at(out, sid, day)
        assert all(row[c] is None for c in FEATURES) and row["src_date_risk"] is None
    assert out.filter(pl.col("src_date_risk") > pl.col("date")).height == 0


def test_empty_entries_give_an_empty_frame():
    prices = make_prices({1: 10.0}, CAL)
    ctx = make_ctx(prices=prices, benchmark=alternating_bench(), entries=pl.DataFrame(
        schema={"security_id": pl.Int64, "date": pl.Date}))
    out = risk.compute(ctx)
    assert out.height == 0 and out.columns == ["security_id", "date", *FEATURES, "src_date_risk"]


# ---------------------------------------------------------------- vol_1y


def test_vol_1y_hand_computed():
    a = 0.01
    # 253 rows ending at T -> 252 log returns +a, -a, +a, ... (126 of each)
    # mean = 0; sum of squares = 252 a^2; variance (ddof=1) = 252 a^2 / 251
    # annualized: sqrt(252 a^2 / 251 * 252) = a * 252 / sqrt(251) = 0.1590610...
    alt = [100.0 * math.exp(a * (i % 2)) for i in range(253)]
    # constant growth: every log return is log(1.001), so the std is zero
    steady = [100.0 * 1.001**i for i in range(253)]
    # a flat price: all returns exactly zero
    prices = make_prices({1: alt, 2: steady, 3: 50.0}, CAL, first={1: CAL[-253], 2: CAL[-253]})
    out = risk.compute(make_ctx(prices=prices, benchmark=alternating_bench(), entries=[(1, T), (2, T), (3, T)]))

    assert at(out, 1)["vol_1y"] == pytest.approx(a * 252 / math.sqrt(251), rel=1e-9)
    assert at(out, 1)["vol_1y"] == pytest.approx(0.1590610, rel=1e-6)
    assert at(out, 2)["vol_1y"] == pytest.approx(0.0, abs=1e-9)
    assert at(out, 3)["vol_1y"] == 0.0


def test_vol_1y_uses_log_returns_and_only_the_last_252_rows():
    # older history is 10x as volatile; the window at T must not see it
    a = 0.01
    old = tile([math.expm1(10 * a), math.expm1(-10 * a)], N - 1 - 252)
    new = tile([math.expm1(a), math.expm1(-a)], 252)   # simple returns whose logs are +a, -a
    prices = make_prices({1: path(np.concatenate([old, new]))}, CAL)
    out = risk.compute(make_ctx(prices=prices, benchmark=alternating_bench(), entries=[(1, T)]))
    assert at(out, 1)["vol_1y"] == pytest.approx(a * 252 / math.sqrt(251), rel=1e-9)


def test_vol_and_idio_need_126_observations():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.02, 200)
    # 127 rows = 126 returns (enough); 126 rows = 125 returns (not enough)
    prices = make_prices({1: path(r[:126]), 2: path(r[:125])}, CAL, first={1: CAL[-127], 2: CAL[-126]})
    out = risk.compute(make_ctx(prices=prices, benchmark=alternating_bench(), entries=[(1, T), (2, T)]))
    assert at(out, 1)["vol_1y"] == pytest.approx(np.std(np.log1p(r[:126]), ddof=1) * math.sqrt(252), rel=1e-9)
    assert at(out, 1)["idio_vol"] is not None
    assert at(out, 2)["vol_1y"] is None and at(out, 2)["idio_vol"] is None
    # 125 returns are still plenty for the 21-day maximum
    assert at(out, 2)["max_ret_1m"] == pytest.approx(r[104:125].max(), rel=1e-9)
    assert at(out, 2)["src_date_risk"] == T


def test_non_positive_price_is_a_missing_return_not_nan():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.02, 252)
    p = path(r)             # 253 prices; p[i] / p[i-1] - 1 = r[i-1]
    p[100] = 0.0            # kills the returns into and out of that row: r[99] and r[100]
    p[240] = -5.0           # r[239] and r[240], both inside the last 21 rows too
    prices = make_prices({1: p}, CAL, first={1: CAL[-253]})
    out = risk.compute(make_ctx(prices=prices, benchmark=alternating_bench(), entries=[(1, T)]))
    keep = np.ones(252, dtype=bool)
    keep[[99, 100, 239, 240]] = False
    row = at(out, 1)
    assert row["vol_1y"] == pytest.approx(np.std(np.log1p(r[keep]), ddof=1) * math.sqrt(252), rel=1e-9)
    assert row["max_ret_1m"] == pytest.approx(r[231:][keep[231:]].max(), rel=1e-9)   # 19 valid of 21
    assert all(row[c] is None or math.isfinite(row[c]) for c in FEATURES)


# ---------------------------------------------------------------- idio_vol


def test_idio_vol_hand_computed():
    b, c = 0.01, 0.005
    x = tile([b, -b, b, -b], N)          # benchmark
    e = tile([c, c, -c, -c], N)          # sums to zero and is orthogonal to x over any 4 days
    # stock 1: r = 0.001 + 2 x + e. OLS over 252 days (63 whole cycles) recovers alpha = 0.001 and
    # beta = 2 exactly, so the residuals are e: RSS = 252 c^2, variance = RSS / (252 - 2),
    # annualized std = sqrt(252 c^2 / 250 * 252) = c * 252 / sqrt(250) = 0.0796894...
    # stock 2: r = 0.001 + 2 x exactly -> no residual at all
    # stock 3: r = e only (beta 0): idio vol = same residual std
    prices = make_prices(
        {1: path((0.001 + 2 * x + e)[1:]), 2: path((0.001 + 2 * x)[1:]), 3: path(e[1:])}, CAL
    )
    out = risk.compute(make_ctx(prices=prices, benchmark=make_benchmark(CAL, x.tolist()),
                                entries=[(1, T), (2, T), (3, T)]))
    expected = c * 252 / math.sqrt(250)
    assert expected == pytest.approx(0.0796894, rel=1e-6)
    assert at(out, 1)["idio_vol"] == pytest.approx(expected, rel=1e-8)
    assert at(out, 2)["idio_vol"] == pytest.approx(0.0, abs=1e-7)
    assert at(out, 3)["idio_vol"] == pytest.approx(expected, rel=1e-8)
    # total volatility of stock 1 is far above its idiosyncratic part (most of it is 2 x market)
    assert at(out, 1)["vol_1y"] > 3 * at(out, 1)["idio_vol"]


def test_idio_vol_excludes_days_without_benchmark_members():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.01, N)
    r = 0.8 * x + rng.normal(0, 0.015, N)
    members_from = CAL[-152]             # the last 152 days have members; 100 window days do not
    bench = make_benchmark(CAL, x.tolist()).with_columns(
        pl.when(pl.col("date") >= members_from).then(1).otherwise(0).cast(pl.Int64).alias("n_members"),
        # as labels.benchmark_index writes it: a placeholder 0 on days without members
        pl.when(pl.col("date") >= members_from).then(pl.col("bench_ret")).otherwise(0.0).alias("bench_ret"),
    )
    prices = make_prices({1: path(r[1:])}, CAL)
    out = risk.compute(make_ctx(prices=prices, benchmark=bench, entries=[(1, T), (1, CAL[-27])]))

    # OLS on the 152 member days only: residual variance = RSS / (152 - 2)
    xs, ys = x[-152:], r[-152:]
    slope, icpt = np.polyfit(xs, ys, 1)
    rss = float(((ys - icpt - slope * xs) ** 2).sum())
    assert at(out, 1)["idio_vol"] == pytest.approx(math.sqrt(rss / 150 * 252), rel=1e-8)
    # with the placeholder zeros included the answer would differ
    slope_all, icpt_all = np.polyfit(np.where(np.arange(252) >= 100, x[-252:], 0.0), r[-252:], 1)
    wrong = math.sqrt(float(((r[-252:] - icpt_all - slope_all * np.where(np.arange(252) >= 100, x[-252:], 0.0)) ** 2).sum()) / 250 * 252)
    assert abs(wrong - at(out, 1)["idio_vol"]) > 1e-4

    # 26 days earlier only 126 of the member days have happened (enough); vol_1y does not care
    early = at(out, 1, CAL[-27])
    assert early["idio_vol"] is not None and early["vol_1y"] is not None
    out2 = risk.compute(make_ctx(prices=prices, benchmark=bench, entries=[(1, CAL[-28])]))
    early2 = at(out2, 1, CAL[-28])       # 125 paired days: not enough
    assert early2["idio_vol"] is None and early2["vol_1y"] is not None


def test_regressions_are_null_without_benchmark_variance_or_benchmark():
    rng = np.random.default_rng(5)
    prices = make_prices({1: path(rng.normal(0, 0.02, N - 1))}, CAL)
    flat = risk.compute(make_ctx(prices=prices, benchmark=make_benchmark(CAL, 0.0), entries=[(1, T)]))
    const = risk.compute(make_ctx(prices=prices, benchmark=make_benchmark(CAL, 0.001), entries=[(1, T)]))
    none = risk.compute(make_ctx(prices=prices, entries=[(1, T)]))
    for out in (flat, none):
        row = at(out, 1)
        assert row["idio_vol"] is None and row["beta_60m"] is None
        assert row["vol_1y"] is not None and row["max_ret_1m"] is not None and row["src_date_risk"] == T
    # a benchmark that returns the same 0.1% every day has no daily variance either
    assert at(const, 1)["idio_vol"] is None


# ---------------------------------------------------------------- beta_60m


def _alt(k, size):
    return size if k % 2 == 0 else -size


def test_beta_60m_hand_computed_24_months():
    # 24 monthly returns, 2019-01 .. 2020-12 (months 60..83), each realised on the first trading day
    # benchmark x: +2%, -2%, +2%, ...  (12 of each, mean 0)
    # stock 1 y:   +5% when x = +2%, -1% when x = -2%  (mean 2%)
    #   Sxy = 12 (0.02)(0.05 - 0.02) + 12 (-0.02)(-0.01 - 0.02) = 0.0072 + 0.0072 = 0.0144
    #   Sxx = 24 (0.02)^2 = 0.0096          beta = 0.0144 / 0.0096 = 1.5
    # stock 2 y = -x: beta = -1 (the sign is kept);  stock 3 y = 0.5 x: beta = 0.5
    months = range(60, 84)
    x = {FIRST_DAYS[k]: _alt(k, 0.02) for k in months}
    start = MONTH_ENDS[59]               # 2018-12-31: the price the first monthly return starts from
    prices = make_prices({
        1: monthly_prices(start, {FIRST_DAYS[k]: 0.05 if k % 2 == 0 else -0.01 for k in months}),
        2: monthly_prices(start, {day: -v for day, v in x.items()}),
        3: monthly_prices(start, {day: 0.5 * v for day, v in x.items()}),
        # listed one month later: 23 monthly returns, one short of the minimum
        4: monthly_prices(MONTH_ENDS[60], {FIRST_DAYS[k]: 0.05 if k % 2 == 0 else -0.01 for k in range(61, 84)}),
    }, CAL)
    out = risk.compute(make_ctx(prices=prices, benchmark=make_benchmark(CAL, x),
                                entries=[(1, T), (2, T), (3, T), (4, T), (1, MONTH_ENDS[82])]))
    assert at(out, 1)["beta_60m"] == pytest.approx(0.0144 / 0.0096, rel=1e-9)
    assert at(out, 1)["beta_60m"] == pytest.approx(1.5, rel=1e-9)
    assert at(out, 2)["beta_60m"] == pytest.approx(-1.0, rel=1e-9)
    assert at(out, 3)["beta_60m"] == pytest.approx(0.5, rel=1e-9)
    assert at(out, 4)["beta_60m"] is None
    assert at(out, 1, MONTH_ENDS[82])["beta_60m"] is None     # a month earlier: 23 returns


def test_beta_60m_window_is_the_last_60_months():
    # months 1..23: stock = -3 x benchmark;  months 24..83 (2016-01 .. 2020-12): 0.01 + 1.5 x + e
    # with e = +1%, +1%, -1%, -1% (orthogonal to the alternating x over any 4 months)
    xs = np.array([_alt(k, 0.02) for k in range(84)])
    es = tile([0.01, 0.01, -0.01, -0.01], 84)
    ys = np.where(np.arange(84) >= 24, 0.01 + 1.5 * xs + es, -3.0 * xs)
    prices = make_prices({1: monthly_prices(CAL[0], {FIRST_DAYS[k]: ys[k] for k in range(1, 84)})}, CAL)
    bench = make_benchmark(CAL, {FIRST_DAYS[k]: xs[k] for k in range(84)})
    out = risk.compute(make_ctx(prices=prices, benchmark=bench,
                                entries=[(1, T), (1, MONTH_ENDS[82]), (1, MONTH_ENDS[24]), (1, MONTH_ENDS[23])]))
    # at T the window is months 24..83: 60 months of the second regime only -> exactly 1.5
    assert at(out, 1)["beta_60m"] == pytest.approx(1.5, rel=1e-9)
    # one month earlier it is months 23..82: one month of the old regime is still inside
    assert at(out, 1, MONTH_ENDS[82])["beta_60m"] == pytest.approx(np.polyfit(xs[23:83], ys[23:83], 1)[0], rel=1e-9)
    assert abs(at(out, 1, MONTH_ENDS[82])["beta_60m"] - 1.5) > 0.01
    # month 0 has no return (nothing before it): months 1..24 are 24 returns, months 1..23 only 23
    assert at(out, 1, MONTH_ENDS[24])["beta_60m"] == pytest.approx(np.polyfit(xs[1:25], ys[1:25], 1)[0], rel=1e-9)
    assert at(out, 1, MONTH_ENDS[23])["beta_60m"] is None


def test_beta_60m_skips_months_with_member_less_benchmark_days_and_missing_month_ends():
    months = range(60, 84)
    x = {FIRST_DAYS[k]: _alt(k, 0.02) for k in months}
    y = {FIRST_DAYS[k]: 0.05 if k % 2 == 0 else -0.01 for k in months}
    prices = make_prices({1: monthly_prices(MONTH_ENDS[59], y)}, CAL)
    bench = make_benchmark(CAL, x)
    ok = risk.compute(make_ctx(prices=prices, benchmark=bench, entries=[(1, T)]))
    assert at(ok, 1)["beta_60m"] == pytest.approx(1.5, rel=1e-9)

    # one day of January 2019 has no members: that month is not an observation -> 23 left
    hole = bench.with_columns(
        pl.when(pl.col("date") == d(2019, 1, 15)).then(0).otherwise(pl.col("n_members")).alias("n_members")
    )
    out = risk.compute(make_ctx(prices=prices, benchmark=hole, entries=[(1, T)]))
    assert at(out, 1)["beta_60m"] is None

    # the stock has no price on the month end of 2019-06: the returns of June and July are both
    # undefined (nothing is carried forward) -> 22 left
    gap = prices.filter(pl.col("date") != MONTH_ENDS[65])
    out = risk.compute(make_ctx(prices=gap, benchmark=bench, entries=[(1, T)], calendar=CAL))
    assert at(out, 1)["beta_60m"] is None


# ---------------------------------------------------------------- max_ret_1m


def test_max_ret_1m_hand_computed():
    # flat at 100 except three days. The window at T is the last 21 rows, CAL[-21:].
    #   CAL[-22]: +8%  (one row too old)   CAL[-21]: +5%  (oldest row inside)   CAL[-5]: -3%
    # -> max = 0.05
    stock = {CAL[0]: 100.0, CAL[-22]: 108.0, CAL[-21]: 113.4, CAL[-5]: 113.4 * 0.97}
    # falls 1% every day: the largest daily return is -1% (negative, kept as is)
    falling = [100.0 * 0.99**i for i in range(N)]
    prices = make_prices({1: stock, 2: falling}, CAL)
    out = risk.compute(make_ctx(prices=prices, benchmark=alternating_bench(),
                                entries=[(1, T), (2, T), (1, CAL[-2])]))
    assert at(out, 1)["max_ret_1m"] == pytest.approx(0.05, rel=1e-9)
    assert at(out, 2)["max_ret_1m"] == pytest.approx(-0.01, rel=1e-9)
    # one day earlier the +8% day is still inside the window
    assert at(out, 1, CAL[-2])["max_ret_1m"] == pytest.approx(0.08, rel=1e-9)


def test_max_ret_1m_needs_15_returns():
    rng = np.random.default_rng(6)
    r = rng.normal(0, 0.02, 15)
    # 16 rows = 15 returns (enough); 15 rows = 14 returns (not enough)
    prices = make_prices({1: path(r), 2: path(r[:14])}, CAL, first={1: CAL[-16], 2: CAL[-15]})
    out = risk.compute(make_ctx(prices=prices, benchmark=alternating_bench(), entries=[(1, T), (2, T)]))
    assert at(out, 1)["max_ret_1m"] == pytest.approx(r.max(), rel=1e-9)
    assert at(out, 1)["vol_1y"] is None and at(out, 1)["src_date_risk"] == T
    row = at(out, 2)
    assert all(row[c] is None for c in FEATURES) and row["src_date_risk"] is None


# ---------------------------------------------------------------- sign conventions


def test_raw_values_are_not_sign_flipped():
    rng = np.random.default_rng(7)
    x = rng.normal(0.0003, 0.01, N)
    noise = rng.normal(0, 0.01, N)
    prices = make_prices({
        1: path((2.0 * x + 3 * noise)[1:]),    # high beta, noisy
        2: path((0.5 * x + noise)[1:]),        # low beta, quiet
        3: path((-1.0 * x)[1:]),               # moves against the market
    }, CAL)
    out = risk.compute(make_ctx(prices=prices, benchmark=make_benchmark(CAL, x.tolist()),
                                entries=[(1, T), (2, T), (3, T)]))
    hi, lo, inv = at(out, 1), at(out, 2), at(out, 3)
    assert hi["vol_1y"] > lo["vol_1y"] > 0
    assert hi["idio_vol"] > lo["idio_vol"] > 0
    assert hi["beta_60m"] > lo["beta_60m"] > 0 > inv["beta_60m"]
    assert hi["max_ret_1m"] > lo["max_ret_1m"] > 0
    assert inv["idio_vol"] == pytest.approx(0.0, abs=1e-6)     # an exact linear function of the market
    # the catalog records the direction instead: more risk, lower expected return
    assert all(load_catalog().spec(name).sign == -1 for name in FEATURES)


# ---------------------------------------------------------------- brute-force reference


def _random_market(seed=11):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0004, 0.011, N)
    spec = {s: path((beta * x + rng.normal(0, 0.02, N))[1:]) for s, beta in [(1, 1.3), (2, 0.6), (3, 1.0)]}
    spec[4] = path((0.9 * x + rng.normal(0, 0.03, N))[1:400])
    prices = make_prices(spec, CAL, first={4: d(2019, 6, 3)})
    # security 3 does not trade for ten weeks (two month ends are missing)
    prices = prices.filter(
        ~((pl.col("security_id") == 3) & pl.col("date").is_between(d(2019, 3, 11), d(2019, 5, 20)))
    )
    members_from = d(2014, 7, 1)
    bench = make_benchmark(CAL, np.where(np.array(CAL.to_list()) >= members_from, x, 0.0).tolist()).with_columns(
        (pl.col("date") >= members_from).cast(pl.Int64).alias("n_members")
    )
    return prices, bench


def _reference(prices: pl.DataFrame, bench: pl.DataFrame, sid: int, t) -> dict:
    """The four features by plain loops over one security's rows."""
    null = dict.fromkeys(FEATURES)
    px = prices.filter(pl.col("security_id") == sid).sort("date")
    dates, p = px["date"].to_list(), px["adj_close"].to_list()
    if t not in dates:
        return null
    i = dates.index(t)
    ret = [None] + [p[j] / p[j - 1] - 1 if p[j] > 0 and p[j - 1] > 0 else None for j in range(1, len(p))]
    bx = {r["date"]: r["bench_ret"] for r in bench.iter_rows(named=True) if r["n_members"] > 0}
    out = dict(null)

    window = range(max(0, i - 251), i + 1)
    logs = [math.log1p(ret[j]) for j in window if ret[j] is not None]
    if len(logs) >= 126:
        out["vol_1y"] = float(np.std(logs, ddof=1) * math.sqrt(252))
    pairs = np.array([(bx[dates[j]], ret[j]) for j in window if ret[j] is not None and dates[j] in bx])
    if len(pairs) >= 126:
        slope, icpt = np.polyfit(pairs[:, 0], pairs[:, 1], 1)
        rss = float(((pairs[:, 1] - icpt - slope * pairs[:, 0]) ** 2).sum())
        out["idio_vol"] = math.sqrt(rss / (len(pairs) - 2) * 252)
    last = [ret[j] for j in range(max(0, i - 20), i + 1) if ret[j] is not None]
    if len(last) >= 15:
        out["max_ret_1m"] = max(last)

    if t in MONTH_ENDS:
        price = dict(zip(dates, p))
        level = dict(zip(bench["date"].to_list(), bench["bench_level"].to_list()))
        no_members = [r["date"] for r in bench.iter_rows(named=True) if r["n_members"] <= 0]
        k = MONTH_ENDS.index(t)
        obs = []
        for m in range(max(1, k - 59), k + 1):
            a, b = MONTH_ENDS[m - 1], MONTH_ENDS[m]
            if a in price and b in price and not any(a < day <= b for day in no_members):
                obs.append((level[b] / level[a] - 1, price[b] / price[a] - 1))
        if len(obs) >= 24:
            obs = np.array(obs)
            out["beta_60m"] = float(np.polyfit(obs[:, 0], obs[:, 1], 1)[0])
    return out


def test_matches_a_brute_force_reference_on_random_data():
    prices, bench = _random_market()
    ctx = make_ctx(prices=prices, benchmark=bench, calendar=CAL)
    out = risk.compute(ctx)
    assert out.height == ctx.entries.height > 250
    seen = dict.fromkeys(FEATURES, 0)
    for row in out.iter_rows(named=True):
        ref = _reference(prices, bench, row["security_id"], row["date"])
        for c in FEATURES:
            if ref[c] is None:
                assert row[c] is None, (row, c)
            else:
                assert row[c] == pytest.approx(ref[c], rel=1e-7, abs=1e-10), (row, c)
                seen[c] += 1
    assert all(n > 50 for n in seen.values())
    assert out.filter(pl.col("src_date_risk") > pl.col("date")).height == 0


def test_result_does_not_depend_on_row_order_or_chunking(monkeypatch):
    prices, bench = _random_market()
    base = risk.compute(make_ctx(prices=prices, benchmark=bench, calendar=CAL))
    shuffled = risk.compute(make_ctx(
        prices=prices.sample(fraction=1.0, shuffle=True, seed=1),
        benchmark=bench.sample(fraction=1.0, shuffle=True, seed=2),
        calendar=CAL,
    ))
    assert shuffled.equals(base)
    monkeypatch.setattr(risk, "CHUNK_SECURITIES", 1)
    assert risk.compute(make_ctx(prices=prices, benchmark=bench, calendar=CAL)).equals(base)


# ---------------------------------------------------------------- point in time


def _pit_entries():
    return [(s, day) for s in (1, 2, 3) for day in (T1, T)]


def _assert_same(a: dict, b: dict):
    for c in [*FEATURES, "src_date_risk"]:
        if a[c] is None or c == "src_date_risk":
            assert b[c] == a[c], c
        else:
            assert b[c] == pytest.approx(a[c], rel=1e-12, abs=1e-14), c


@pytest.mark.parametrize("feature", FEATURES)
def test_later_prices_do_not_change_the_value(feature):
    prices, bench = _random_market()
    base = risk.compute(make_ctx(prices=prices, benchmark=bench, entries=_pit_entries(), calendar=CAL))
    # every price after T1 is different (a new path), starting with the very next trading day
    after = pl.col("date") > T1
    bump = 1 + 0.5 * ((pl.col("date").dt.ordinal_day() % 7) / 7.0)
    changed = prices.with_columns(pl.when(after).then(pl.col("adj_close") * bump).otherwise(pl.col("adj_close")).alias("adj_close"))
    out = risk.compute(make_ctx(prices=changed, benchmark=bench, entries=_pit_entries(), calendar=CAL))
    # only the next trading day changes
    nxt = CAL.filter(CAL > T1)[0]
    one = prices.with_columns(pl.when(pl.col("date") == nxt).then(pl.col("adj_close") * 3).otherwise(pl.col("adj_close")).alias("adj_close"))
    out_one = risk.compute(make_ctx(prices=one, benchmark=bench, entries=_pit_entries(), calendar=CAL))
    # nothing after T1 exists at all (the calendar ends at T1 too)
    cut = risk.compute(make_ctx(prices=prices.filter(~after), benchmark=bench.filter(~after),
                                entries=[(s, T1) for s in (1, 2, 3)]))
    for sid in (1, 2, 3):
        ref = at(base, sid, T1)
        assert ref[feature] is not None
        for other in (out, out_one, cut):
            _assert_same(ref, at(other, sid, T1))
            assert at(other, sid, T1)[feature] == pytest.approx(ref[feature], rel=1e-12)
        # the test has teeth: the value at T, whose window holds the changed prices, does move
        assert at(out, sid, T)[feature] != pytest.approx(at(base, sid, T)[feature], rel=1e-6)


@pytest.mark.parametrize("feature", ["idio_vol", "beta_60m"])
def test_later_benchmark_does_not_change_the_value(feature):
    prices, bench = _random_market()
    base = risk.compute(make_ctx(prices=prices, benchmark=bench, entries=_pit_entries(), calendar=CAL))
    # the benchmark after T1 is rewritten: returns tripled and sign-flipped on odd days, levels rebuilt
    after = pl.col("date") > T1
    flip = pl.when(pl.col("date").dt.ordinal_day() % 2 == 1).then(-3.0).otherwise(3.0)
    changed = bench.with_columns(
        pl.when(after).then(pl.col("bench_ret") * flip).otherwise(pl.col("bench_ret")).alias("bench_ret")
    ).with_columns((1 + pl.col("bench_ret")).cum_prod().alias("bench_level"))
    out = risk.compute(make_ctx(prices=prices, benchmark=changed, entries=_pit_entries(), calendar=CAL))
    # the benchmark stops having members right after T1
    dropped = bench.with_columns(pl.when(after).then(0).otherwise(pl.col("n_members")).alias("n_members"))
    out_dropped = risk.compute(make_ctx(prices=prices, benchmark=dropped, entries=_pit_entries(), calendar=CAL))
    for sid in (1, 2, 3):
        ref = at(base, sid, T1)
        assert ref[feature] is not None
        _assert_same(ref, at(out, sid, T1))
        _assert_same(ref, at(out_dropped, sid, T1))
        assert at(out, sid, T)[feature] != pytest.approx(at(base, sid, T)[feature], rel=1e-6)


def test_a_change_on_the_prediction_date_itself_is_used():
    # windows end at t inclusive: the close of t is an input (unlike a filing made on t)
    prices, bench = _random_market()
    base = risk.compute(make_ctx(prices=prices, benchmark=bench, entries=[(1, T1)], calendar=CAL))
    on_t = prices.with_columns(
        pl.when((pl.col("date") == T1) & (pl.col("security_id") == 1)).then(pl.col("adj_close") * 1.5)
        .otherwise(pl.col("adj_close")).alias("adj_close")
    )
    out = risk.compute(make_ctx(prices=on_t, benchmark=bench, entries=[(1, T1)], calendar=CAL))
    a, b = at(base, 1, T1), at(out, 1, T1)
    assert b["max_ret_1m"] > 0.4 > a["max_ret_1m"]
    for c in ("vol_1y", "idio_vol", "beta_60m"):
        assert b[c] != pytest.approx(a[c], rel=1e-6)
    assert b["src_date_risk"] == T1

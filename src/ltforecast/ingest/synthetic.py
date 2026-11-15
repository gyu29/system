"""Synthetic Sharadar/FRED-shaped raw data with a planted, modest return signal (DESIGN section 8).

No vendor keys exist in the development environment, so every phase is built and tested on data
that has the vendor's shapes and the cases the point-in-time layer must survive: IPOs, bankruptcies,
mergers, splits, dividends, late and amended filings, restated (MR*) rows, non-stock categories and
names too small, cheap or illiquid for the universe.

How the pieces depend on each other (everything is simulated from two years before `start`):
  market + industry + idiosyncratic shocks, delisting events      -> exogenous daily log returns
  exogenous returns (lagged), own investment and issuance shocks  -> balance sheets and income
  characteristics that were public at each month end              -> planted drift for the next month
  exogenous returns + planted drift                               -> prices, splits, dividends
  prices                                                          -> dividends paid, net issuance, per-share items

The planted signal: next month's drift is a sum of LOADINGS times the cross-sectional ranks of
gross profitability (gp TTM / assets), book/price, 12-1 momentum and asset growth as of the month
end, using only fundamentals for fiscal quarters that ended at least three months earlier.
Ranks move slowly, so the drift persists and shows up in 12-month excess returns with a rank IC of
roughly 0.05-0.10 for the combined signal (averaged over seeds; one 150-stock sample is noisy):
detectable, far from a toy. The volatility effect is a drag instead of a rank loading: a stock's
log return loses VOL_DRAG x sigma^2 a year, so volatile stocks have lower median returns.

Conventions: SEP close/open/high/low/volume are split-adjusted, closeunadj is the traded price,
closeadj is total-return adjusted with (close_t + dividend_t) / close_{t-1} = closeadj_t / closeadj_{t-1}.
ACTIONS dividend and SF1 per-share values (eps, dps, sharesbas, shareswa) are on the share basis of
their own date, not adjusted for later splits. DAILY.marketcap is in millions of dollars.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import lfilter

from ltforecast.ingest.sharadar import TABLES
from ltforecast.store import Store

BURN_IN_YEARS = 2
YEAR = 252

# Market regimes: annualized log drift and mean spell length in trading days.
BULL_DRIFT, BEAR_DRIFT = 0.17, -0.28
BULL_DAYS, BEAR_DAYS = 560, 170
MARKET_DRIFT = (BULL_DRIFT * BULL_DAYS + BEAR_DRIFT * BEAR_DAYS) / (BULL_DAYS + BEAR_DAYS)  # long-run mean

# Annualized log-return drift per unit of each characteristic's cross-sectional rank (ranks are
# scaled to unit variance). Profitability is the strongest so that one raw SF1 ratio is detectable.
LOADINGS = {"profitability": 0.024, "value": 0.018, "momentum": 0.025, "asset_growth": -0.018}
# The volatility effect is not a rank loading: a stock's log return loses VOL_DRAG x sigma^2 a year.
# 0.5 would give every idiosyncratic shock a zero expected simple return.
VOL_DRAG = 0.25
NONCURRENT_DEBT_SHARE = 0.8  # SF1 debtnc as a share of debt

CATEGORIES = (
    "Domestic Common Stock",
    "Domestic Common Stock Primary Class",
    "Domestic Common Stock Secondary Class",
    "ADR Common Stock",
    "ETF",
)
CATEGORY_SHARES = (None, 0.06, 0.02, 0.05, 0.05)  # the first category takes the remainder

# sector, industry, famaindustry, siccode, asset turnover, gross margin, operating margin,
# liabilities/assets, volatility scale, book/price
INDUSTRIES = (
    ("Technology", "Software - Application", "Computer Software", "7372", 0.60, 0.70, 0.10, 0.40, 1.25, 0.25),
    ("Technology", "Semiconductors", "Electronic Equipment", "3674", 0.70, 0.50, 0.12, 0.35, 1.30, 0.35),
    ("Technology", "Computer Hardware", "Computers", "3571", 1.00, 0.35, 0.08, 0.45, 1.15, 0.40),
    ("Healthcare", "Biotechnology", "Pharmaceutical Products", "2834", 0.40, 0.75, 0.02, 0.35, 1.50, 0.25),
    ("Healthcare", "Medical Devices", "Medical Equipment", "3841", 0.70, 0.60, 0.12, 0.40, 1.10, 0.30),
    ("Financial Services", "Banks - Regional", "Banking", "6022", 0.07, 0.80, 0.30, 0.90, 0.85, 0.90),
    ("Financial Services", "Insurance - Property & Casualty", "Insurance", "6331", 0.30, 0.30, 0.10, 0.75, 0.80, 0.80),
    ("Consumer Cyclical", "Specialty Retail", "Retail", "5651", 1.80, 0.35, 0.06, 0.55, 1.05, 0.40),
    ("Consumer Cyclical", "Restaurants", "Restaurants, Hotels, Motels", "5812", 1.10, 0.30, 0.10, 0.60, 0.95, 0.35),
    ("Consumer Defensive", "Packaged Foods", "Food Products", "2000", 1.00, 0.35, 0.11, 0.55, 0.70, 0.40),
    ("Industrials", "Specialty Industrial Machinery", "Machinery", "3560", 0.90, 0.32, 0.10, 0.55, 1.00, 0.45),
    ("Energy", "Oil & Gas E&P", "Petroleum and Natural Gas", "1311", 0.40, 0.55, 0.15, 0.50, 1.30, 0.60),
    ("Basic Materials", "Specialty Chemicals", "Chemicals", "2860", 0.80, 0.30, 0.10, 0.55, 1.00, 0.50),
    ("Utilities", "Utilities - Regulated Electric", "Utilities", "4911", 0.30, 0.40, 0.18, 0.68, 0.60, 0.65),
)
_INDUSTRY_PARAMS = np.array([row[4:] for row in INDUSTRIES])

SURVIVES, BANKRUPT, MERGER = 0, 1, 2

# SF1 metrics. Flows are per quarter in ARQ and trailing four quarters in ART / ARY; balance-sheet
# items are as of the period end in every dimension.
FLOWS = (
    "revenue", "gp", "opinc", "ebit", "ebitda", "netinc", "ncfo", "capex", "fcf", "ncfdiv", "ncfcommon",
    "intexp", "depamor", "taxexp",
)
PER_SHARE = ("eps", "epsdil", "dps", "shareswa")
BALANCE = (
    "assets", "assetsc", "liabilities", "liabilitiesc", "equity", "debt", "debtnc", "cashneq", "retearn",
    "workingcapital", "invcap", "sharesbas",
)
SF1_METRICS = (*FLOWS, *PER_SHARE, *BALANCE)
# A restatement or amendment rescales the income and cash-flow statement, never the balance sheet.
RESTATED = (*FLOWS, "eps", "epsdil", "dps")

TICKER_NULL_COLS = (
    "cusips", "sicsector", "sicindustry", "famasector", "scalemarketcap", "scalerevenue", "relatedtickers",
    "location", "companysite",
)


@dataclass(frozen=True)
class _Grid:
    """Simulation calendar: weekdays from January 1st two years before `start` through `end`."""

    days: np.ndarray            # datetime64[D]
    first_out: int              # index of the first day that is written out (the first weekday >= start)
    month_id: np.ndarray        # per day, 0 = the first simulated month
    month_starts: np.ndarray    # index of the first day of each month
    quarter_id: np.ndarray      # per day, month_id // 3
    quarter_starts: np.ndarray
    quarter_ends: np.ndarray    # datetime64[D], calendar quarter end = fiscal period end


def _grid(start: dt.date, end: dt.date) -> _Grid:
    first = np.datetime64(dt.date(start.year - BURN_IN_YEARS, 1, 1), "D")
    days = np.arange(first, np.datetime64(end, "D") + np.timedelta64(1, "D"))
    days = days[np.is_busday(days)]
    month_id = days.astype("datetime64[M]").astype(np.int64)
    month_id -= month_id[0]
    quarter_id = month_id // 3
    n_quarters = quarter_id[-1] + 1
    quarter_ends = (first.astype("datetime64[M]") + _months(3 * np.arange(1, n_quarters + 1))).astype("datetime64[D]") - _days(1)
    return _Grid(
        days=days,
        first_out=int(np.searchsorted(days, np.datetime64(start, "D"))),
        month_id=month_id,
        month_starts=np.flatnonzero(np.diff(month_id, prepend=-1)),
        quarter_id=quarter_id,
        quarter_starts=np.flatnonzero(np.diff(quarter_id, prepend=-1)),
        quarter_ends=quarter_ends,
    )


def _days(n) -> np.ndarray:
    return np.asarray(n).astype("timedelta64[D]")


def _months(n) -> np.ndarray:
    return np.asarray(n).astype("timedelta64[M]")


def _ar1(rng: np.random.Generator, shape, phi: float, sd: float) -> np.ndarray:
    """AR(1) along axis 0 with innovation sd `sd`, started at zero."""
    return lfilter([1.0], [1.0, -phi], sd * rng.standard_normal(shape), axis=0)


def _period_sum(x: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(x, starts, axis=0)


def _trailing4(x: np.ndarray) -> np.ndarray:
    """Sum of the last four rows; the first three rows are NaN."""
    c = np.cumsum(x, axis=0)
    out = c.copy()
    out[4:] -= c[:-4]
    out[:3] = np.nan
    return out


def _roster(rng: np.random.Generator, n: int, g: _Grid) -> dict[str, np.ndarray]:
    """Per-stock constants: identity, life cycle, risk, size and accounting structure."""
    n_days = len(g.days)
    counts = [max(1, round(s * n)) if s else 0 for s in CATEGORY_SHARES]
    counts[2] = round(CATEGORY_SHARES[2] * n)  # the secondary share class may be absent in small samples
    counts[0] = n - sum(counts)
    n_common, n_etf = counts[0] + counts[1], counts[4]
    category = np.repeat(np.arange(len(CATEGORIES)), counts)
    # Common stocks are dealt round-robin so every industry has names; ETFs have no industry (-1).
    industry = np.concatenate([
        np.arange(n_common) % len(INDUSTRIES),
        rng.integers(0, len(INDUSTRIES), n - n_common - n_etf),
        np.full(n_etf, -1),
    ])
    order = rng.permutation(n)
    category, industry = category[order], industry[order]
    is_etf = industry < 0
    turnover0, gm0, om0, lev0, vol0, bp0 = _INDUSTRY_PARAMS[np.maximum(industry, 0)].T

    codes = rng.choice(26**4, n, replace=False)
    ticker = np.array(["".join(chr(65 + (c // 26**k) % 26) for k in range(4)) for c in codes])

    # Life cycle, in day indices. Seasoned names trade from the first output day.
    span = n_days - g.first_out
    ipo = rng.random(n) < 0.35
    first = np.where(ipo, g.first_out + rng.integers(21, max(22, int(0.7 * span)), n), g.first_out)
    last = np.full(n, n_days - 1)
    fate = np.full(n, SURVIVES)
    eligible = np.flatnonzero(~is_etf & (first + 300 < n_days - 40))
    n_bankrupt = min(max(1, round(0.08 * n)), len(eligible) // 2)
    n_merger = min(max(1, round(0.12 * n)), len(eligible) - n_bankrupt)
    leavers = rng.permutation(eligible)[: n_bankrupt + n_merger]
    last[leavers] = rng.integers(first[leavers] + 300, n_days - 40)
    fate[leavers[:n_bankrupt]] = BANKRUPT
    fate[leavers[n_bankrupt:]] = MERGER

    z = rng.standard_normal((10, n))
    u = rng.random((12, n))
    sigma = np.where(is_etf, 0.04, np.clip(0.30 * np.exp(0.45 * z[0]) * vol0, 0.12, 0.9))
    beta = np.where(is_etf, 1.0, np.clip(1.0 + 0.5 * np.log(sigma / 0.30) + 0.2 * z[1], 0.3, 2.2))

    # Deliberate universe failures: micro caps, low-priced and illiquid names.
    micro, low_priced, illiquid = u[0] < 0.12, u[1] < 0.08, u[2] < 0.06
    market_cap0 = np.where(micro, 1.0e8 * np.exp(0.5 * z[2]), 2.0e9 * np.exp(1.2 * z[2]))
    price0 = np.where(low_priced, 1.5 + 3.0 * u[3], np.clip(28.0 * np.exp(0.55 * z[3]), 6.0, 90.0))
    turnover = 0.005 * np.exp(0.5 * z[4]) * np.where(illiquid, 0.03, 1.0)

    asset_turnover = turnover0 * np.exp(0.25 * z[5])
    gross_margin = np.clip(gm0 + 0.08 * z[6], 0.08, 0.92)
    op_margin = np.minimum(om0 + 0.08 * z[7], gross_margin - 0.04)
    leverage = np.clip(lev0 + 0.08 * z[8], 0.10, 0.93)
    book_to_price = bp0 * np.exp(0.35 * z[9])
    # Deposits and insurance float are liabilities but not debt.
    debt_share = (0.2 + 0.4 * u[4]) * np.where(lev0 > 0.7, 0.3, 1.0)
    interest_rate = 0.04 + 0.04 * u[5]
    # Steady-state earnings yield decides who can afford a dividend.
    roa = 0.75 * (op_margin * asset_turnover - interest_rate * debt_share * leverage)
    earnings_yield = roa * book_to_price / (1.0 - leverage)
    payer = ~is_etf & (earnings_yield > 0.02) & (u[6] < 0.6)
    div_yield = np.where(payer, np.clip((0.2 + 0.4 * u[7]) * earnings_yield, 0.0, 0.06), 0.0)
    div_yield = np.where(is_etf, 0.015, div_yield)

    return {
        "ticker": ticker,
        "permaticker": 100001 + np.arange(n),
        "category": category,
        "industry": industry,
        "has_fund": ~is_etf,  # ETFs file no 10-Q: no SF1 and no DAILY rows
        "exchange": rng.integers(0, 3, n),
        "first": first,
        "last": last,
        "fate": fate,
        "announce": last - rng.integers(30, 100, n),  # mergers: the deal is announced this day
        "premium": 0.15 + 0.30 * u[8],
        "decline": 1.5 + 1.5 * u[8],                 # bankruptcies: log price lost over the final year
        "crash": 0.30 + 0.30 * u[9],
        "crash_day": last - rng.integers(1, 40, n),
        "sigma": sigma,
        "beta": beta,
        "market_cap0": market_cap0,
        "price0": price0,
        "turnover": turnover,
        "split_at": np.where(u[10] < 0.3, np.inf, 100.0 + 80.0 * u[11]),
        "div_yield": div_yield,
        "div_day": rng.integers(5, 55, n),
        "buyback": np.where(payer, 0.006 * rng.random(n), 0.0),
        "fye": rng.choice([12, 3, 6, 9], n, p=[0.7, 0.1, 0.1, 0.1]),
        "asset_turnover": asset_turnover,
        "gross_margin": gross_margin,
        "op_margin": op_margin,
        "leverage": leverage,
        "book_to_price": book_to_price,
        "debt_share": debt_share,
        "interest_rate": interest_rate,
        "growth": 0.02 * rng.standard_normal(n),
        "dep_rate": np.where(lev0 > 0.7, 0.005, 0.02 + 0.04 * rng.random(n)),
        "capex_mult": 0.8 + rng.random(n),
        "ca_share": 0.25 + 0.35 * rng.random(n),
        "cash_share": 0.15 + 0.45 * rng.random(n),
        "cl_share": 0.20 + 0.30 * rng.random(n),
        "seasonality": 0.15 * rng.random(n),
    }


def _market(rng: np.random.Generator, n_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daily market log return, its log-volatility state and the bear-spell flag.
    Bull and bear spells alternate with geometric lengths; volatility clusters and is higher in bears."""
    n_spells = n_days // 150 + 4
    spells = np.column_stack([rng.geometric(1 / BULL_DAYS, n_spells), rng.geometric(1 / BEAR_DAYS, n_spells)]).ravel()
    edges = np.cumsum(spells) - rng.integers(0, BULL_DAYS)
    bear = np.searchsorted(edges, np.arange(n_days), side="right") % 2 == 1
    log_vol = _ar1(rng, n_days, 0.985, 0.05)
    vol = np.where(bear, 0.24, 0.13) / np.sqrt(YEAR) * np.exp(log_vol)
    ret = np.where(bear, BEAR_DRIFT, BULL_DRIFT) / YEAR + vol * rng.standard_normal(n_days)
    return ret, log_vol, bear


def _exogenous_returns(rng: np.random.Generator, g: _Grid, ros: dict, market: np.ndarray, log_vol: np.ndarray) -> np.ndarray:
    """Daily log total returns before the planted drift: beta x market + industry + idiosyncratic,
    plus the price path of a bankruptcy (decline and a crash day) or a merger (jump, then a quiet spread)."""
    n_days, n = len(g.days), len(ros["sigma"])
    own_vol = _ar1(rng, (n_days, n), 0.97, 0.06) + 0.5 * log_vol[:, None]
    # exp(h - var(h)) has a mean square of one, so `sigma` stays the stock's long-run volatility
    var = 0.06**2 / (1 - 0.97**2) + 0.25 * 0.05**2 / (1 - 0.985**2)
    idio_sd = ros["sigma"] / np.sqrt(YEAR) * np.exp(own_vol - var)
    # The volatility drag: without it every shock would have a zero expected log return, and the
    # most volatile stocks would compound upward and take over the cap-weighted benchmark.
    idio = idio_sd * rng.standard_normal((n_days, n)) - VOL_DRAG * idio_sd**2
    industry = 0.08 / np.sqrt(YEAR) * rng.standard_normal((n_days, len(INDUSTRIES) + 1))
    industry[:, -1] = 0.0  # industry -1 (ETFs) indexes this column
    # Beta scales the market's swings around its long-run drift, not the drift itself: a high-beta
    # stock falls harder in a bear spell but does not compound at beta times the equity premium.
    swing = market - MARKET_DRIFT / YEAR
    exo = MARKET_DRIFT / YEAR + ros["beta"] * swing[:, None] + industry[:, ros["industry"]] + idio

    for i in np.flatnonzero(ros["fate"] == BANKRUPT):
        end = ros["last"][i]
        begin = max(end - YEAR, 0)
        ramp = np.linspace(0.5, 1.5, end - begin + 1)
        exo[begin : end + 1, i] -= ros["decline"][i] * ramp / ramp.sum()
        exo[ros["crash_day"][i], i] -= ros["crash"][i]
    for i in np.flatnonzero(ros["fate"] == MERGER):
        a = ros["announce"][i]
        exo[a, i] = np.log1p(ros["premium"][i])
        exo[a + 1 :, i] *= 0.08
    return exo


def _fundamentals(rng: np.random.Generator, g: _Grid, ros: dict, exo: np.ndarray) -> dict[str, np.ndarray]:
    """Quarterly statements, (quarters, stocks), that do not depend on the planted drift.

    Assets follow the stock's own past returns with a lag (the mean of the eight prior quarters), share
    issuance and a persistent investment shock, so prices and book values stay tied together in the
    long run. Liabilities are a slowly moving share of assets and equity is the remainder.
    `shares_index` has one extra leading row: row q is the share count at the end of quarter q - 1.
    """
    n_q, n = len(g.quarter_ends), exo.shape[1]
    shape = (n_q, n)
    r_q = _period_sum(exo, g.quarter_starts)
    csum = np.vstack([np.zeros((1, n)), np.cumsum(r_q, axis=0)])
    qi = np.arange(n_q)
    tracked = (csum[qi] - csum[np.maximum(qi - 8, 0)]) / 8

    invest = _ar1(rng, shape, 0.5, 0.02)
    secondary = (rng.random(shape) < 0.03) * rng.uniform(0.05, 0.20, shape)
    d_shares = 0.0015 + 0.4 * np.maximum(invest, 0.0) + secondary - ros["buyback"]
    shares_index = np.exp(np.vstack([np.zeros((1, n)), np.cumsum(d_shares, axis=0)]))

    assets0 = ros["market_cap0"] * ros["book_to_price"] / (1.0 - ros["leverage"])
    d_assets = tracked - ros["div_yield"] / 4 + d_shares + invest + ros["growth"] / 4
    assets = assets0 * np.exp(np.cumsum(d_assets, axis=0))
    assets_avg = (np.vstack([assets0, assets[:-1]]) + assets) / 2

    # Margins erode and leverage climbs over the two years before a bankruptcy.
    distress = np.zeros(shape)
    for i in np.flatnonzero(ros["fate"] == BANKRUPT):
        q_end = g.quarter_id[ros["last"][i]]
        q_begin = max(q_end - 8, 0)
        distress[q_begin : q_end + 1, i] = np.linspace(0.0, 1.0, q_end - q_begin + 1)

    def wobble() -> np.ndarray:
        return np.exp(_ar1(rng, shape, 0.8, 0.03))

    leverage = np.clip(ros["leverage"] + _ar1(rng, shape, 0.9, 0.01) + 0.25 * distress, 0.08, 0.97)
    liabilities = leverage * assets
    equity = assets - liabilities
    debt = ros["debt_share"] * liabilities * wobble()
    liabilitiesc = ros["cl_share"] * liabilities * wobble()
    assetsc = ros["ca_share"] * assets * wobble()
    cashneq = ros["cash_share"] * assetsc * wobble()

    season = 1.0 + ros["seasonality"] * np.array([-0.5, 0.0, 0.0, 0.5])[qi % 4][:, None]
    revenue = ros["asset_turnover"] / 4 * assets_avg * season * np.exp(0.03 * rng.standard_normal(shape))
    gross_margin = np.clip(ros["gross_margin"] + _ar1(rng, shape, 0.9, 0.01), 0.03, 0.95)
    op_margin = ros["op_margin"] + _ar1(rng, shape, 0.9, 0.015) + 0.01 * rng.standard_normal(shape) - 0.25 * distress
    gp = gross_margin * revenue
    opinc = np.minimum(op_margin, gross_margin - 0.02) * revenue
    depamor = ros["dep_rate"] / 4 * assets_avg
    ebit = opinc + 0.001 * assets_avg * rng.standard_normal(shape)
    intexp = ros["interest_rate"] / 4 * debt
    ebt = ebit - intexp
    taxexp = 0.25 * np.maximum(ebt, 0.0)
    netinc = ebt - taxexp
    accruals = assets_avg * _ar1(rng, shape, 0.3, 0.008)
    ncfo = netinc + depamor - accruals
    capex = -ros["dep_rate"] * ros["capex_mult"] / 4 * assets_avg * np.exp(0.25 * rng.standard_normal(shape))

    return {
        "shares_index": shares_index,
        "revenue": revenue, "gp": gp, "opinc": opinc, "ebit": ebit, "ebitda": ebit + depamor, "netinc": netinc,
        "ncfo": ncfo, "capex": capex, "fcf": ncfo + capex, "intexp": intexp, "depamor": depamor, "taxexp": taxexp,
        "assets": assets, "assetsc": assetsc, "liabilities": liabilities, "liabilitiesc": liabilitiesc,
        "equity": equity, "debt": debt, "cashneq": cashneq,
        # long-term debt (the Piotroski leverage signal reads it): a fixed share, so no extra random draws
        "debtnc": NONCURRENT_DEBT_SHARE * debt,
        "workingcapital": assetsc - liabilitiesc,
        # Sharadar: invcap = debt + assets - intangibles - cashneq - liabilitiesc (no intangibles here)
        "invcap": debt + assets - cashneq - liabilitiesc,
    }


def _dividend_schedule(g: _Grid, ros: dict) -> np.ndarray:
    """(days, stocks) dividend as a fraction of the previous close; non-zero on ex-dates only.
    Payers pay a quarter of their yield on a fixed trading day of each quarter, and stop once a
    bankruptcy decline starts."""
    n_days, n = len(g.days), len(ros["sigma"])
    delta = np.zeros((n_days, n))
    payers = np.flatnonzero(ros["div_yield"] > 0)
    day = g.quarter_starts[:, None] + ros["div_day"][payers]
    quarter_stop = np.append(g.quarter_starts[1:], n_days)[:, None]
    stop = np.where(ros["fate"][payers] == BANKRUPT, ros["last"][payers] - YEAR, n_days)
    ok = (day < quarter_stop) & (day < stop)
    cols = np.broadcast_to(payers, day.shape)
    delta[day[ok], cols[ok]] = np.broadcast_to(ros["div_yield"][payers] / 4, day.shape)[ok]
    return delta


def _centered_rank(x: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Rank among active, finite values, centered and scaled to unit variance; 0 elsewhere."""
    ok = active & np.isfinite(x)
    out = np.zeros(len(x))
    k = int(ok.sum())
    if k > 1:
        rank = np.empty(k)
        rank[np.argsort(x[ok])] = np.arange(k)
        out[ok] = ((rank + 0.5) / k - 0.5) * np.sqrt(12.0)
    return out


def _planted_alpha(g: _Grid, ros: dict, fund: dict, exo: np.ndarray, delta: np.ndarray, signal: float) -> np.ndarray:
    """(months, stocks) annualized log-return drift applied during each month.

    Month m's drift is set from what was known at the end of month m - 1: fundamentals of the fiscal
    quarter that ended at least three months before (quarter m // 3 - 2, public for all but late
    filers), the market cap and trailing returns at that month end. Book/price and momentum depend
    on earlier drift through the price, hence the loop over months.
    """
    n_months, n = len(g.month_starts), exo.shape[1]
    r_month = _period_sum(exo, g.month_starts)
    month_last = np.append(g.month_starts[1:], len(g.days)) - 1
    # ex-dividend price level; exact up to a second-order term (see _prices)
    log_div = np.vstack([np.zeros((1, n)), np.cumsum(np.log1p(-delta), axis=0)[month_last]])
    # The drift stops when a merger is announced: the price is then pinned to the deal.
    stop_day = np.where(ros["fate"] == MERGER, ros["announce"] - 1, ros["last"])
    stop_month = g.month_id[stop_day]

    gp_assets = _trailing4(fund["gp"]) / fund["assets"]
    asset_growth = np.full_like(fund["assets"], np.nan)
    asset_growth[4:] = fund["assets"][4:] / fund["assets"][:-4] - 1.0
    log_equity = np.log(fund["equity"])
    log_shares = np.log(fund["shares_index"])
    log_cap0 = np.log(ros["market_cap0"])
    nan = np.full(n, np.nan)

    log_tr = np.zeros((n_months + 1, n))  # row m: cumulative log total return at the end of month m - 1
    alpha = np.zeros((n_months, n))
    for m in range(n_months):
        active = ros["has_fund"] & (m <= stop_month)
        q = m // 3 - 2
        known = q >= 0
        log_mcap = log_cap0 + log_tr[m] + log_div[m] + log_shares[m // 3]
        chars = {
            "profitability": gp_assets[q] if known else nan,
            "value": log_equity[q] - log_mcap if known else nan,
            "momentum": log_tr[m - 1] - log_tr[m - 12] if m >= 12 else nan,
            "asset_growth": asset_growth[q] if known else nan,
        }
        alpha[m] = signal * sum(LOADINGS[k] * _centered_rank(x, active) for k, x in chars.items())
        log_tr[m + 1] = log_tr[m] + r_month[m] + alpha[m] / 12
    return alpha


def _prices(g: _Grid, ros: dict, fund: dict, ret: np.ndarray, delta: np.ndarray) -> dict[str, np.ndarray]:
    """Price paths from daily log total returns `ret`. All arrays are (days, stocks) unless noted.

    `price` is per share on the share basis of the listing day; `n_splits` counts the 2:1 splits since
    listing (a split happens whenever the traded price reaches the stock's `split_at`), so the traded
    price is price / 2**n_splits and the split-adjusted close is price / 2**(splits at the last day).
    `shares` (quarters + 1, stocks) is the share count on the listing basis at the end of the previous quarter.
    """
    n_days, n = ret.shape
    cols = np.arange(n)
    t = np.arange(n_days)[:, None]
    # ex-date: close_t = close_{t-1} * exp(ret_t) - dividend_t, with dividend_t = delta * close_{t-1}
    log_price = np.cumsum(np.where(delta > 0, np.log(np.exp(ret) - delta), ret), axis=0)
    log_tr = np.cumsum(ret, axis=0)
    first, last = ros["first"], ros["last"]
    price = ros["price0"] * np.exp(log_price - log_price[first, cols])
    listed = (t >= first) & (t <= last)

    peak = np.maximum.accumulate(np.where(listed, price, 0.0), axis=0)
    with np.errstate(divide="ignore"):
        n_splits = np.clip(np.floor(np.log2(peak / ros["split_at"])) + 1.0, 0.0, None)
    final_splits = n_splits[last, cols]
    close = price / 2.0**final_splits
    shares = fund["shares_index"] * ros["market_cap0"] * np.exp(log_price[first, cols]) / ros["price0"]
    return {
        "listed": listed,
        "price": price,
        "n_splits": n_splits,
        "final_splits": final_splits,
        "close": close,
        "closeunadj": price / 2.0**n_splits,
        "closeadj": close[last, cols] * np.exp(log_tr - log_tr[last, cols]),
        "log_move": np.diff(log_price, axis=0, prepend=log_price[:1]),
        "shares": shares,
    }


def _price_linked_fundamentals(g: _Grid, ros: dict, fund: dict, px: dict, delta: np.ndarray) -> None:
    """Adds the statement lines that depend on prices: dividends paid, net issuance at the quarter's
    average price, retained earnings, and share counts / per-share items on each period end's own share basis.
    `_sf1` moves the share counts and per-share items to the basis of their filing date (`_filing_basis`)."""
    price, shares = px["price"], px["shares"]
    days_in_quarter = np.diff(np.append(g.quarter_starts, len(g.days)))[:, None]
    quarter_last = np.append(g.quarter_starts[1:], len(g.days)) - 1

    div_per_share = delta * np.vstack([price[:1], price[:-1]])  # listing share basis
    fund["ncfdiv"] = -_period_sum(div_per_share * shares[g.quarter_id], g.quarter_starts)
    fund["ncfcommon"] = (shares[1:] - shares[:-1]) * _period_sum(price, g.quarter_starts) / days_in_quarter
    retained0 = 0.4 * ros["market_cap0"] * ros["book_to_price"]
    fund["retearn"] = retained0 + np.cumsum(fund["netinc"] + fund["ncfdiv"], axis=0)

    basis = 2.0 ** px["n_splits"][quarter_last]  # shares now per listing-day share, at each quarter end
    shares_avg = (shares[1:] + shares[:-1]) / 2
    fund["sharesbas"] = shares[1:] * basis
    fund["shareswa"] = shares_avg * basis
    fund["eps"] = fund["netinc"] / fund["shareswa"]
    fund["dps"] = _period_sum(div_per_share, g.quarter_starts) / basis

    # Trailing-four-quarter versions for ART / ARY, per-share items on the latest period end's basis.
    ttm = {k: _trailing4(fund[k]) for k in FLOWS}
    ttm["shareswa"] = _trailing4(shares_avg) / 4 * basis
    ttm["eps"] = ttm["netinc"] / ttm["shareswa"]
    ttm["dps"] = _trailing4(fund["dps"] * basis) / basis
    for d in (fund, ttm):
        d["epsdil"] = np.where(d["eps"] > 0, 0.98 * d["eps"], d["eps"])
    fund["ttm"] = ttm


def _names(ros: dict) -> np.ndarray:
    etf = ros["category"] == CATEGORIES.index("ETF")
    return np.where(etf, np.char.add(ros["ticker"], " Index Fund"), np.char.add(ros["ticker"], " Corp"))


def _sep_and_daily(rng: np.random.Generator, g: _Grid, ros: dict, px: dict, end: dt.date) -> tuple[pl.DataFrame, pl.DataFrame]:
    ii, tt = np.nonzero(px["listed"].T)  # stock-major: one contiguous block of days per ticker
    ticker = pl.Series("ticker", ros["ticker"]).gather(ii)
    close = px["close"][tt, ii]
    daily_sd = ros["sigma"][ii] / np.sqrt(YEAR)
    move = px["log_move"][tt, ii]
    k = len(ii)
    # Part of the day's move happens overnight; the range extends beyond open and close.
    open_ = close * np.exp(-rng.uniform(0.4, 0.9, k) * move + 0.2 * daily_sd * rng.standard_normal(k))
    high = np.maximum(open_, close) * np.exp(0.6 * daily_sd * np.abs(rng.standard_normal(k)))
    low = np.minimum(open_, close) * np.exp(-0.6 * daily_sd * np.abs(rng.standard_normal(k)))
    shares = px["shares"][g.quarter_id[tt], ii]
    volume = np.round(
        shares * 2.0 ** px["final_splits"][ii] * ros["turnover"][ii]
        * np.exp(0.5 * rng.standard_normal(k)) * (1.0 + np.abs(move) / (2.0 * daily_sd))
    )
    sep = pl.DataFrame({
        "ticker": ticker, "date": g.days[tt], "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "closeadj": px["closeadj"][tt, ii], "closeunadj": px["closeunadj"][tt, ii],
    }).with_columns(pl.lit(end).alias("lastupdated"))

    has_fund = ros["has_fund"][ii]
    daily = pl.DataFrame({
        "ticker": ticker.filter(has_fund),
        "date": g.days[tt][has_fund],
        "marketcap": np.round(px["price"][tt, ii][has_fund] * shares[has_fund] / 1.0e6, 1),
    }).select("ticker", "date", pl.lit(end).alias("lastupdated"), "marketcap")
    return sep, daily


def _actions(rng: np.random.Generator, g: _Grid, ros: dict, px: dict, delta: np.ndarray) -> pl.DataFrame:
    n = len(ros["ticker"])
    t = np.arange(len(g.days))[:, None]
    after_listing = (t > ros["first"]) & (t <= ros["last"])
    tickers, names = pl.Series(ros["ticker"]), pl.Series(_names(ros))
    schema = {"date": pl.Date, "action": pl.String, "ticker": pl.String, "name": pl.String, "value": pl.Float64,
              "contraticker": pl.String, "contraname": pl.String}

    def rows(action: str, ii, tt, value, contra=None) -> pl.DataFrame:
        k = len(ii)
        return pl.DataFrame({
            "date": g.days[tt],
            "action": [action] * k,
            "ticker": tickers.gather(ii),
            "name": names.gather(ii),
            "value": value,
            "contraticker": tickers.gather(contra) if contra is not None else [None] * k,
            "contraname": names.gather(contra) if contra is not None else [None] * k,
        }, schema=schema)

    n_splits = px["n_splits"]
    ii, tt = np.nonzero(((delta > 0) & after_listing).T)
    # dollars per share actually paid: the previous close restated to the ex-date's share basis
    dividends = rows("dividend", ii, tt, delta[tt, ii] * px["price"][tt - 1, ii] / 2.0 ** n_splits[tt, ii])
    step = np.diff(n_splits, axis=0, prepend=n_splits[:1])
    ii, tt = np.nonzero(((step > 0) & after_listing).T)
    splits = rows("split", ii, tt, 2.0 ** step[tt, ii])

    bankrupt = np.flatnonzero(ros["fate"] == BANKRUPT)
    merged = np.flatnonzero(ros["fate"] == MERGER)
    gone = np.concatenate([bankrupt, merged])
    deal_price = px["closeunadj"][ros["last"][merged], merged] * (1.0 + 0.02 * rng.random(len(merged)))
    acquirer = (merged + 1 + rng.integers(0, n - 1, len(merged))) % n
    return pl.concat([
        dividends,
        splits,
        rows("bankruptcyliquidation", bankrupt, ros["last"][bankrupt], [None] * len(bankrupt)),
        rows("acquisitionby", merged, ros["last"][merged], deal_price, acquirer),
        rows("delisted", gone, ros["last"][gone], [None] * len(gone)),
    ]).sort("date", "ticker", "action")


SHARE_COUNTS = ("sharesbas", "shareswa")
PER_SHARE_VALUES = ("eps", "epsdil", "dps")


def _filing_basis(g: _Grid, px: dict, datekey: np.ndarray) -> np.ndarray:
    """(quarters, stocks) factor that moves a period end's share basis to its filing date's: a figure
    filed on day F reflects every split with ex-date < F (a split on F itself is not yet in it), as the
    feature code assumes (`features.investment.split_adjusted_context`)."""
    n_splits = px["n_splits"]
    cols = np.arange(n_splits.shape[1])
    quarter_last = np.append(g.quarter_starts[1:], len(g.days)) - 1
    before_filing = np.clip(np.searchsorted(g.days, datekey) - 1, 0, len(g.days) - 1)
    return 2.0 ** (n_splits[before_filing, cols] - n_splits[quarter_last][:, cols])


def _sf1(rng: np.random.Generator, g: _Grid, ros: dict, fund: dict, px: dict, end: dt.date) -> pl.DataFrame:
    """ARQ / ART / ARY rows as first filed, plus amendments (same period, later datekey, changed
    income and cash-flow numbers) and restated MR* rows dated at the period end, as the vendor does.
    Share counts and per-share items are on the share basis of the first filing's datekey (amendments
    and MR* rows keep it)."""
    n_q, n = len(g.quarter_ends), len(ros["ticker"])
    shape = (n_q, n)
    q = np.arange(n_q)[:, None]
    end_month = g.quarter_ends.astype("datetime64[M]").astype(np.int64)  # months since 1970
    to_fy_end = (ros["fye"] - (end_month % 12 + 1)[:, None]) % 12        # months until the fiscal year end
    is_fy_end = to_fy_end == 0

    # 10-Q: 30-50 days after the period end, 10-K: 50-75; about 3% are late.
    lag = np.where(is_fy_end, rng.integers(50, 76, shape), rng.integers(30, 51, shape))
    lag = np.where(rng.random(shape) < 0.03, rng.integers(92, 161, shape), lag)
    datekey = np.busday_offset(g.quarter_ends[:, None] + _days(lag), 0, roll="forward")
    # The data is a snapshot taken at `end`; a delisted company stops filing.
    cutoff = g.days[ros["last"]]
    q_first = np.maximum(g.quarter_id[ros["first"]] - 8, 0)
    reported = ros["has_fund"] & (q >= q_first) & (datekey <= cutoff)
    has_ttm = reported & (q >= q_first + 3)
    rebase = _filing_basis(g, px, datekey)

    tickers = pl.Series(ros["ticker"])
    parts = []
    for dim, mask, trailing in (("ARQ", reported, False), ("ART", has_ttm, True), ("ARY", has_ttm & is_fy_end, True)):
        ii, qq = np.nonzero(mask.T)
        source = {**fund, **fund["ttm"]} if trailing else fund
        cols = {
            "ticker": tickers.gather(ii),
            "dimension": [dim] * len(ii),
            "reportperiod": g.quarter_ends[qq],
            "datekey": datekey[qq, ii],
            "_cutoff": cutoff[ii],
            "_fy": 1970 + (end_month[qq] + to_fy_end[qq, ii]) // 12,
            "_fq": 4 - to_fy_end[qq, ii] // 3,
        }
        cols |= {m: source[m][qq, ii] for m in SF1_METRICS}
        rb = rebase[qq, ii]
        cols |= {m: cols[m] * rb for m in SHARE_COUNTS} | {m: cols[m] / rb for m in PER_SHARE_VALUES}
        parts.append(pl.DataFrame(cols))
    base = pl.concat(parts)

    k = base.height
    later = np.busday_offset(base["datekey"].to_numpy() + _days(rng.integers(20, 240, k)), 0, roll="forward")
    base = base.with_columns(
        pl.Series("_u", rng.random(k)),
        pl.Series("_later", later),
        pl.Series("_bump", 1.0 + 0.06 * rng.standard_normal(k)),
    )
    bumped = [pl.col(c) * pl.col("_bump") for c in RESTATED]
    amended = (
        base.filter((pl.col("_u") < 0.015) & (pl.col("_later") <= pl.col("_cutoff")))
        .with_columns(pl.col("_later").alias("datekey"), *bumped)
    )
    restated = base.filter(pl.col("_u") > 0.92).with_columns(
        ("MR" + pl.col("dimension").str.slice(2)).alias("dimension"),
        pl.col("reportperiod").alias("datekey"),
        *bumped,
    )
    period = (
        pl.when(pl.col("dimension").str.ends_with("Y"))
        .then(pl.format("{}-FY", "_fy"))
        .otherwise(pl.format("{}-Q{}", "_fy", "_fq"))
    )
    return (
        pl.concat([base, amended, restated])
        .select(
            "ticker", "dimension", pl.col("reportperiod").alias("calendardate"), "datekey", "reportperiod",
            pl.lit(end).alias("lastupdated"), period.alias("fiscalperiod"), *SF1_METRICS,
        )
        .sort("ticker", "dimension", "datekey", "reportperiod")
    )


def _tickers(g: _Grid, ros: dict, sf1: pl.DataFrame, end: dt.date) -> pl.DataFrame:
    """Security master: one table=SEP row per ticker, and a table=SF1 duplicate for companies with
    fundamentals (the vendor lists a ticker once per table it appears in)."""
    n = len(ros["ticker"])
    ind = ros["industry"]
    described = ind >= 0

    def industry_col(j: int) -> list:
        return [INDUSTRIES[i][j] if ok else None for i, ok in zip(ind, described)]

    first_price = g.days[ros["first"]]
    quarters = sf1.group_by("ticker").agg(
        pl.col("reportperiod").min().alias("firstquarter"), pl.col("reportperiod").max().alias("lastquarter")
    )
    sep_rows = pl.DataFrame({
        "permaticker": ros["permaticker"],
        "ticker": ros["ticker"],
        "name": _names(ros),
        "exchange": np.array(["NYSE", "NASDAQ", "NYSEMKT"])[ros["exchange"]],
        "isdelisted": np.where(ros["fate"] == SURVIVES, "N", "Y"),
        "category": np.array(CATEGORIES)[ros["category"]],
        "siccode": industry_col(3),
        "famaindustry": industry_col(2),
        "sector": industry_col(0),
        "industry": industry_col(1),
        "currency": ["USD"] * n,
        "secfilings": [f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={1000000 + i:010d}" for i in range(n)],
        "firstadded": first_price,
        "firstpricedate": first_price,
        "lastpricedate": g.days[ros["last"]],
        "_has_fund": ros["has_fund"],
    }, schema_overrides={"siccode": pl.String, "famaindustry": pl.String, "sector": pl.String, "industry": pl.String})
    sep_rows = sep_rows.join(quarters, on="ticker", how="left").with_columns(
        pl.lit(end).alias("lastupdated"), *[pl.lit(None, pl.String).alias(c) for c in TICKER_NULL_COLS]
    )
    both = pl.concat([
        sep_rows.with_columns(pl.lit("SEP").alias("table")),
        sep_rows.filter("_has_fund").with_columns(pl.lit("SF1").alias("table")),
    ])
    return both.select(
        "table", "permaticker", "ticker", "name", "exchange", "isdelisted", "category", "cusips", "siccode",
        "sicsector", "sicindustry", "famasector", "famaindustry", "sector", "industry", "scalemarketcap",
        "scalerevenue", "relatedtickers", "currency", "location", "lastupdated", "firstadded", "firstpricedate",
        "lastpricedate", "firstquarter", "lastquarter", "secfilings", "companysite",
    )


def _fred(rng: np.random.Generator, g: _Grid, log_vol: np.ndarray, bear: np.ndarray, end: dt.date) -> pl.DataFrame:
    """macro_pit rows. Rates are in percent. Short rates fall and credit spreads widen in bear spells.
    Daily market series are known the next day; FEDFUNDS (monthly average, dated the 1st) after 35
    days; CPI (dated the 1st) after 45. Only observations already released at `end` are included."""
    n_days = len(g.days)
    stress = lfilter([1 / 120], [1.0, -(1 - 1 / 120)], bear.astype(float))  # slow memory of bear spells
    short = np.clip(4.5 - 4.0 * stress + _ar1(rng, n_days, 0.995, 0.03), 0.05, None)
    term = 1.2 + 1.8 * stress + _ar1(rng, n_days, 0.99, 0.03)
    credit = np.clip(1.8 + 2.5 * stress + 0.8 * log_vol + _ar1(rng, n_days, 0.98, 0.02), 0.6, None)

    month_first = (g.days[0].astype("datetime64[M]") + _months(np.arange(len(g.month_starts)))).astype("datetime64[D]")
    days_in_month = np.diff(np.append(g.month_starts, n_days))
    fedfunds = _period_sum(short, g.month_starts) / days_in_month
    cpi = 170.0 * np.exp(np.cumsum(0.025 / 12 + 0.002 * rng.standard_normal(len(month_first))))

    series = (
        ("CPIAUCSL", month_first, cpi, 45, 3),
        ("DGS10", g.days, short + term, 1, 2),
        ("DGS3MO", g.days, short, 1, 2),
        ("BAA10Y", g.days, credit, 1, 2),
        ("FEDFUNDS", month_first, fedfunds, 35, 2),
    )
    frames = [
        pl.DataFrame({
            "series": [name] * len(dates),
            "observation_date": dates,
            "available_date": dates + _days(lag),
            "value": np.round(values, decimals),
        })
        for name, dates, values, lag, decimals in series
    ]
    return (
        pl.concat(frames)
        .filter(pl.col("available_date") <= end)
        .with_columns(pl.lit(True).alias("first_release"))
        .sort("series", "available_date", "observation_date")
    )


def _truth(g: _Grid, ros: dict, px: dict, alpha: np.ndarray) -> pl.DataFrame:
    """The expected return for the month after each month end, for listed companies: ticker, date
    (last trading day of the month), alpha (annualized expected log return relative to other stocks:
    the planted drift minus the stock's volatility drag)."""
    month_end = g.month_starts[1:] - 1
    mask = px["listed"][month_end] & ros["has_fund"]
    ii, mm = np.nonzero(mask.T)
    return pl.DataFrame({
        "ticker": pl.Series(ros["ticker"]).gather(ii),
        "date": g.days[month_end[mm]],
        "alpha": alpha[1:][mm, ii] - VOL_DRAG * ros["sigma"][ii] ** 2,
    })


def generate(
    n_stocks: int = 150,
    start: dt.date = dt.date(2000, 1, 3),
    end: dt.date = dt.date(2012, 12, 31),
    seed: int = 0,
    *,
    signal: float = 1.0,
    with_truth: bool = False,
) -> dict[str, pl.DataFrame]:
    """Raw vendor-shaped frames keyed TICKERS, SEP, SF1, DAILY, ACTIONS (as `sharadar.normalize`
    returns them) and FRED_OBSERVATIONS (macro_pit schema). Deterministic for a given seed.

    `signal` scales the planted drift (0 = only the volatility drag separates expected returns).
    `with_truth` adds a TRUTH frame (ticker, date, alpha) with the planted expected return; it is not
    vendor data and `write_raw` skips it.
    Prices start at `start` (fundamentals and macro series reach back two more years).
    """
    if n_stocks < 10:
        raise ValueError("n_stocks must be at least 10")
    if end <= start:
        raise ValueError("end must be after start")
    rng = np.random.default_rng(seed)
    g = _grid(start, end)
    ros = _roster(rng, n_stocks, g)
    market, log_vol, bear = _market(rng, len(g.days))
    exo = _exogenous_returns(rng, g, ros, market, log_vol)
    fund = _fundamentals(rng, g, ros, exo)
    delta = _dividend_schedule(g, ros)
    alpha = _planted_alpha(g, ros, fund, exo, delta, signal)
    days_in_month = np.diff(np.append(g.month_starts, len(g.days)))
    ret = exo + (alpha / (12 * days_in_month[:, None]))[g.month_id]
    px = _prices(g, ros, fund, ret, delta)
    _price_linked_fundamentals(g, ros, fund, px, delta)

    sep, daily = _sep_and_daily(rng, g, ros, px, end)
    sf1 = _sf1(rng, g, ros, fund, px, end)
    frames = {
        "TICKERS": _tickers(g, ros, sf1, end),
        "SEP": sep,
        "SF1": sf1,
        "DAILY": daily,
        "ACTIONS": _actions(rng, g, ros, px, delta),
        "FRED_OBSERVATIONS": _fred(rng, g, log_vol, bear, end),
    }
    if with_truth:
        frames["TRUTH"] = _truth(g, ros, px, alpha)
    return frames


MARKER = "SYNTHETIC.json"


def marker_path(store: Store) -> Path:
    """raw/SYNTHETIC.json: present exactly when the raw vendor files are (all) synthetic."""
    return store.root / "raw" / MARKER


def raw_is_synthetic(store: Store) -> bool:
    return marker_path(store).exists()


def _raw_files(store: Store) -> list[Path]:
    """Every raw vendor file `write_raw` would (over)write that exists now."""
    paths = [store.raw_path("sharadar", t) for t in TABLES] + [store.raw_path("fred", "OBSERVATIONS")]
    return [p for p in paths if p.exists()]


def write_raw(
    store: Store, frames: dict[str, pl.DataFrame], params: dict | None = None, force: bool = False
) -> Path:
    """Save the frames where `ltforecast build` reads them and return the marker path.

    Refuses (FileExistsError) to overwrite raw vendor files that are not marked synthetic, unless
    `force`. The marker (JSON of `params`) is written before any data, so a write that fails halfway
    still leaves the raw directory flagged as synthetic.
    """
    vendor = [] if raw_is_synthetic(store) else _raw_files(store)
    if vendor and not force:
        names = ", ".join(p.relative_to(store.root).as_posix() for p in vendor)
        raise FileExistsError(f"raw vendor data exists ({names}); pass force=True to replace it with synthetic data")
    marker = marker_path(store)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(params or {}, indent=2) + "\n")
    for table in TABLES:
        store.write_raw("sharadar", table, frames[table])
    store.write_raw("fred", "OBSERVATIONS", frames["FRED_OBSERVATIONS"])
    return marker


def guard_vendor_write(store: Store, replace_synthetic: bool = False) -> None:
    """Call before writing any real vendor raw file (`ingest sharadar`, `ingest fred`).

    If the raw directory holds synthetic data, a vendor write would leave real and synthetic tables
    side by side, so raise RuntimeError unless `replace_synthetic`, which deletes every synthetic raw
    file first and the marker last (the marker never disappears while a synthetic table remains).
    """
    if not raw_is_synthetic(store):
        return
    if not replace_synthetic:
        raise RuntimeError(
            f"{marker_path(store)} says data/raw holds synthetic data; writing vendor tables over part of it "
            "would mix real and synthetic data. Remove data/raw or pass --replace-synthetic."
        )
    for p in _raw_files(store):
        p.unlink()
    marker_path(store).unlink()

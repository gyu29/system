"""The synthetic raw data must have the vendor's shapes, run through build -> universe -> labels
unchanged, contain the awkward cases on purpose, and carry a planted signal of realistic size.

Sizes: SMALL (40 stocks, 6 years) for everything structural; SIGNAL (150 stocks, 13 years, the
generator's defaults) only for the two tests that measure the planted signal.
"""

import dataclasses
import datetime as dt
import time
from types import SimpleNamespace

import polars as pl
import pytest
import yaml
from polars.testing import assert_frame_equal

from ltforecast import quality
from ltforecast.config import DEFAULT_CONFIG, load_config
from ltforecast.ingest import sharadar, synthetic
from ltforecast.labels import build_labels
from ltforecast.pit import tables
from ltforecast.store import Store
from ltforecast.universe import build_universe

d = dt.date
SMALL = dict(n_stocks=40, start=d(2004, 1, 1), end=d(2009, 12, 31))
SIGNAL = dict(n_stocks=150, start=d(2000, 1, 3), end=d(2012, 12, 31))
VENDOR_KEYS = {"TICKERS", "SEP", "SF1", "DAILY", "ACTIONS", "FRED_OBSERVATIONS"}
AS_REPORTED = ("ARQ", "ART", "ARY")
REQUIRED_METRICS = (
    "revenue", "gp", "opinc", "ebit", "ebitda", "netinc", "eps", "ncfo", "capex", "fcf", "ncfdiv", "ncfcommon",
    "assets", "assetsc", "liabilities", "liabilitiesc", "equity", "debt", "cashneq", "intexp", "retearn",
    "workingcapital", "invcap", "sharesbas", "shareswa", "dps", "debtnc",
)
FLOWS = ("revenue", "gp", "netinc", "ncfo", "capex", "ncfdiv")


def make_cfg(root, start: dt.date, end: dt.date):
    """Same approach as the conftest `cfg` fixture: the default config with the data directory and dates overridden."""
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["data_dir"] = str(root / "data")
    raw["dates"]["history_start"] = start
    raw["universe"]["cpi_base_date"] = end
    path = root / "config" / "universe.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def run_pipeline(frames: dict[str, pl.DataFrame], cfg, horizons: dict[str, int] | None = None) -> SimpleNamespace:
    """What `ltforecast build` and `ltforecast labels` do, in memory."""
    sec = tables.build_securities(frames["TICKERS"], frames["ACTIONS"], cfg.delisting)
    prices = tables.build_prices(frames["SEP"], frames["ACTIONS"], sec)
    fund = tables.build_fundamentals(frames["SF1"], sec, cfg.pit.sf1_dimensions)
    mcap = tables.build_marketcap(frames["DAILY"], sec, cfg.universe.marketcap_unit)
    macro = tables.build_macro([frames["FRED_OBSERVATIONS"]])
    universe = build_universe(prices, sec, mcap, macro, cfg.universe, cfg.dates.history_start)
    rules = cfg.labels if horizons is None else dataclasses.replace(cfg.labels, horizons=horizons)
    labels, bench = build_labels(prices, universe, mcap, sec, rules, cfg.delisting, cfg.dates)
    return SimpleNamespace(sec=sec, prices=prices, fund=fund, mcap=mcap, universe=universe, labels=labels, bench=bench)


def mean_rank_ic(df: pl.DataFrame, x: str, y: str = "excess_12m", min_names: int = 20) -> float:
    """Average over dates of the cross-sectional Spearman correlation of x and y."""
    ic = (
        df.drop_nulls([x, y])
        .group_by("date")
        .agg(pl.corr(x, y, method="spearman").alias("ic"), pl.len().alias("n"))
        .filter(pl.col("n") >= min_names)
    )
    assert ic.height >= 100  # months
    return ic["ic"].mean()


def first_filed(sf1: pl.DataFrame, dimension: str) -> pl.DataFrame:
    return (
        sf1.filter(pl.col("dimension") == dimension)
        .sort("datekey")
        .unique(["ticker", "reportperiod"], keep="first", maintain_order=True)
    )


@pytest.fixture(scope="module")
def frames():
    return synthetic.generate(**SMALL, seed=0)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    return make_cfg(tmp_path_factory.mktemp("small"), SMALL["start"], SMALL["end"])


@pytest.fixture(scope="module")
def built(frames, cfg):
    return run_pipeline(frames, cfg)


@pytest.fixture(scope="module")
def signal_run(tmp_path_factory):
    frames = synthetic.generate(**SIGNAL, seed=0, with_truth=True)
    cfg = make_cfg(tmp_path_factory.mktemp("signal"), SIGNAL["start"], SIGNAL["end"])
    return frames, run_pipeline(frames, cfg, horizons={"12m": 252})


# determinism and speed

def test_same_seed_same_frames(frames):
    again = synthetic.generate(**SMALL, seed=0)
    assert set(again) == set(frames) == VENDOR_KEYS
    for name in frames:
        assert_frame_equal(again[name], frames[name])


def test_different_seed_different_frames(frames):
    other = synthetic.generate(**SMALL, seed=1)
    assert other["TICKERS"]["ticker"].to_list() != frames["TICKERS"]["ticker"].to_list()
    assert other["SEP"]["close"].sum() != frames["SEP"]["close"].sum()
    assert other["FRED_OBSERVATIONS"]["value"].sum() != frames["FRED_OBSERVATIONS"]["value"].sum()


def test_generation_is_fast():
    # typically 0.05 s and 0.15 s; the bounds leave room for a busy machine
    t0 = time.perf_counter()
    synthetic.generate(**SMALL, seed=2)
    t1 = time.perf_counter()
    synthetic.generate(**SIGNAL, seed=2)
    t2 = time.perf_counter()
    assert t1 - t0 < 1.0
    assert t2 - t1 < 5.0


def test_rejects_degenerate_requests():
    with pytest.raises(ValueError, match="n_stocks"):
        synthetic.generate(n_stocks=5)
    with pytest.raises(ValueError, match="end"):
        synthetic.generate(start=d(2005, 1, 3), end=d(2005, 1, 3))


# shapes

def test_frames_are_typed_like_normalized_vendor_tables(frames):
    for table in sharadar.TABLES:
        df = frames[table]
        for col, dtype in df.schema.items():
            if col in sharadar.DATE_COLS[table]:
                assert dtype == pl.Date, (table, col)
            elif col in sharadar.STRING_COLS[table]:
                assert dtype == pl.String, (table, col)
            elif col in sharadar.INT_COLS:
                assert dtype == pl.Int64, (table, col)
            else:
                assert dtype == pl.Float64, (table, col)
    assert frames["SEP"].columns == [
        "ticker", "date", "open", "high", "low", "close", "volume", "closeadj", "closeunadj", "lastupdated"
    ]
    assert frames["DAILY"].columns == ["ticker", "date", "lastupdated", "marketcap"]
    assert frames["ACTIONS"].columns == ["date", "action", "ticker", "name", "value", "contraticker", "contraname"]
    assert set(sharadar.DATE_COLS["TICKERS"]) | set(sharadar.STRING_COLS["TICKERS"]) | {"permaticker"} == set(
        frames["TICKERS"].columns
    )
    assert {"ticker", "dimension", "calendardate", "datekey", "reportperiod", "lastupdated", "fiscalperiod"} <= set(
        frames["SF1"].columns
    )


def test_identifiers_are_unique(frames):
    t = frames["TICKERS"].filter(pl.col("table") == "SEP")
    assert t.height == SMALL["n_stocks"]
    assert t["ticker"].n_unique() == t.height and t["permaticker"].n_unique() == t.height
    # companies with fundamentals are listed a second time for the SF1 table, as the vendor does
    assert frames["TICKERS"].filter(pl.col("table") == "SF1").height > 0
    assert set(frames["SEP"]["ticker"]) == set(t["ticker"])
    assert set(frames["SF1"]["ticker"]) <= set(t["ticker"])
    assert frames["SEP"].select("ticker", "date").is_duplicated().sum() == 0
    assert frames["DAILY"].select("ticker", "date").is_duplicated().sum() == 0


def test_sectors_and_industries_are_populated():
    t = synthetic.generate(n_stocks=100, start=d(2006, 1, 2), end=d(2007, 12, 31), seed=0)["TICKERS"]
    t = t.filter((pl.col("table") == "SEP") & pl.col("famaindustry").is_not_null())
    fama = t.group_by("famaindustry").len()
    assert fama.height >= 12 and fama["len"].min() >= 5
    assert t["sector"].n_unique() >= 6


def test_pipeline_outputs(frames, built):
    assert built.sec.height == SMALL["n_stocks"]
    assert built.prices.height == frames["SEP"].height
    assert built.prices["adj_close"].null_count() == 0 and built.prices["adj_close"].min() > 0
    assert built.mcap.height == frames["DAILY"].height
    assert set(built.fund["dimension"]) == set(AS_REPORTED)
    assert set(REQUIRED_METRICS) <= set(built.fund["metric"])
    assert built.labels.height == built.universe["in_universe"].sum()
    assert built.labels["excess_12m"].is_not_null().sum() > 500
    assert built.labels["target"].drop_nulls().is_between(0, 1).all()
    assert built.bench["bench_level"].is_finite().all()


# prices

def test_trading_calendar_and_lifetimes(frames):
    sep, t = frames["SEP"], frames["TICKERS"].filter(pl.col("table") == "SEP")
    assert sep["date"].dt.weekday().max() <= 5
    assert sep["date"].min() == d(2004, 1, 1) and sep["date"].max() == SMALL["end"]
    life = sep.group_by("ticker").agg(pl.col("date").min().alias("first"), pl.col("date").max().alias("last"), pl.len())
    life = life.join(t, on="ticker")
    assert (life["first"] == life["firstpricedate"]).all() and (life["last"] == life["lastpricedate"]).all()
    # no gaps: every stock trades on every market day of its life
    calendar = sep.select(pl.col("date").unique().sort()).with_row_index("i")
    span = life.join(calendar.rename({"date": "first", "i": "i0"}), on="first").join(
        calendar.rename({"date": "last", "i": "i1"}), on="last"
    )
    assert (span["i1"] - span["i0"] + 1 == span["len"]).all()
    assert t["firstpricedate"].n_unique() >= 5  # staggered IPOs
    assert (t["firstpricedate"] > d(2004, 1, 1)).sum() >= 5


def test_ohlc_and_volume_are_consistent(frames):
    sep = frames["SEP"]
    assert sep.null_count().sum_horizontal()[0] == 0
    assert (sep["low"] > 0).all() and (sep["volume"] >= 0).all()
    assert sep.select((pl.col("high") >= pl.max_horizontal("open", "close")).all()).item()
    assert sep.select((pl.col("low") <= pl.min_horizontal("open", "close")).all()).item()
    assert (sep["volume"] == sep["volume"].round()).all()


def test_splits_separate_adjusted_from_traded_price(frames):
    splits = frames["ACTIONS"].filter(pl.col("action") == "split")
    assert splits.height >= 3 and (splits["value"] >= 2.0).all()
    ratio = pl.col("closeunadj") / pl.col("close")
    sep = (
        frames["SEP"].sort("ticker", "date")
        .with_columns((ratio.shift(1) / ratio).over("ticker").alias("step"))
        .join(splits.select("ticker", "date", "value"), on=["ticker", "date"], how="left")
    )
    on_split = sep.filter(pl.col("value").is_not_null())
    assert on_split.height == splits.height
    assert ((on_split["step"] / on_split["value"] - 1).abs() < 1e-9).all()
    off_split = sep.filter(pl.col("value").is_null()).drop_nulls("step")
    assert ((off_split["step"] - 1).abs() < 1e-9).all()
    # on its last day a stock's adjusted and traded prices agree
    last = sep.group_by("ticker").agg(pl.all().sort_by("date").last())
    assert ((last["closeunadj"] / last["close"] - 1).abs() < 1e-9).all()


def test_dividends_are_in_the_total_return_price(frames):
    divs = frames["ACTIONS"].filter(pl.col("action") == "dividend")
    splits = frames["ACTIONS"].filter(pl.col("action") == "split").select("ticker", "date")
    assert divs.height > 100 and (divs["value"] > 0).all()
    sep = frames["SEP"].sort("ticker", "date").with_columns(
        (pl.col("closeadj") / pl.col("closeadj").shift(1)).over("ticker").alias("total"),
        pl.col("closeunadj").shift(1).over("ticker").alias("prev"),
    )
    paid = divs.join(sep, on=["ticker", "date"]).join(splits, on=["ticker", "date"], how="anti")
    assert paid.height > 100
    with_dividend = (paid["closeunadj"] + paid["value"]) / paid["prev"]
    assert ((with_dividend / paid["total"] - 1).abs() < 1e-9).all()
    # non-payers: the total-return price moves exactly like the close
    payers = divs["ticker"].unique()
    rest = frames["SEP"].filter(~pl.col("ticker").is_in(payers.implode()))
    assert rest["ticker"].n_unique() >= 5
    assert ((rest["closeadj"] / rest["close"] - 1).abs() < 1e-9).all()
    assert (frames["SEP"]["closeadj"] != frames["SEP"]["close"]).mean() > 0.1


def test_market_has_bull_and_bear_spells_and_stocks_differ_in_risk(frames, built):
    level = built.bench.sort("date")["bench_level"]
    year = level / level.shift(252) - 1
    assert year.max() > 0.15 and year.min() < -0.15
    vol = (
        built.prices.join(built.sec.filter(pl.col("category") == "Domestic Common Stock"), on="security_id")
        .group_by("security_id")
        .agg(pl.col("ret").std().alias("sd"))["sd"]
    )
    assert vol.quantile(0.9) / vol.quantile(0.1) > 2.0


def test_marketcap_is_price_times_shares_in_millions(frames):
    arq = first_filed(frames["SF1"], "ARQ").select("ticker", "reportperiod", "datekey", "sharesbas")
    splits = frames["ACTIONS"].filter(pl.col("action") == "split").select("ticker", pl.col("date").alias("split"))
    day = frames["DAILY"].join(frames["SEP"].select("ticker", "date", "closeunadj"), on=["ticker", "date"])
    assert day.height == frames["DAILY"].height
    # sharesbas is on the share basis of its filing (splits with ex-date < datekey): compare it with the
    # last trading day before the filing, while that day is still in the quarter after the period end
    # (the market cap's share count changes each quarter)
    before = (
        arq.sort("datekey")
        .join_asof(day.sort("date"), left_on="datekey", right_on="date", by="ticker", strategy="backward",
                   allow_exact_matches=False, tolerance="5d", check_sortedness=False)
        .drop_nulls("date")
        .filter(pl.col("date") <= pl.col("reportperiod").dt.offset_by("3mo").dt.month_end())
        .filter(pl.col("marketcap") > 50)  # marketcap is rounded to $0.1M
    )
    assert before.height > 300
    implied = before["closeunadj"] * before["sharesbas"] / 1.0e6
    assert ((implied / before["marketcap"] - 1).abs() < 0.01).all()
    # the case that tells the two conventions apart occurs: a split between the period end and the filing
    between = before.join(splits, on="ticker").filter(
        (pl.col("split") > pl.col("reportperiod")) & (pl.col("split") < pl.col("datekey"))
    )
    assert between.height > 0
    assert 100 < frames["DAILY"]["marketcap"].median() < 100_000


def test_share_counts_pass_the_share_basis_check(built):
    r = quality.check_share_basis(built.fund, built.prices)
    assert r.status == "pass", r.detail


# life cycle, categories and the universe

def test_both_delisting_kinds_occur(frames, built):
    reasons = built.sec.drop_nulls("delisted_date")["delist_reason"].to_list()
    assert reasons.count("bankruptcyliquidation") >= 2 and reasons.count("acquisitionby") >= 2
    assert set(reasons) == {"bankruptcyliquidation", "acquisitionby"}
    t = frames["TICKERS"].filter(pl.col("table") == "SEP")
    gone = t.filter(pl.col("isdelisted") == "Y")
    assert gone.height == len(reasons) and (gone["lastpricedate"] < SMALL["end"]).all()
    assert (t.filter(pl.col("isdelisted") == "N")["lastpricedate"] == SMALL["end"]).all()
    # every delisting has the generic row on the last price date as well as the specific one
    generic = frames["ACTIONS"].filter(pl.col("action") == "delisted")
    assert generic.sort("ticker").select("ticker", "date").equals(
        gone.sort("ticker").select("ticker", pl.col("lastpricedate").alias("date"))
    )
    # a delisted company files nothing after its last trading day
    late = frames["SF1"].filter(pl.col("dimension").is_in(AS_REPORTED)).join(gone, on="ticker")
    assert (late["datekey"] <= late["lastpricedate"]).all()


def test_bankruptcies_decay_and_mergers_converge_on_the_deal(frames, built):
    sep = frames["SEP"].sort("ticker", "date")
    final_year = sep.group_by("ticker").agg(
        (pl.col("closeadj").last() / pl.col("closeadj").tail(253).first() - 1).alias("ret"),
        pl.col("closeunadj").last().alias("last_price"),
        (pl.col("closeadj") / pl.col("closeadj").shift(1) - 1).tail(20).abs().max().alias("late_move"),
    )
    acts = frames["ACTIONS"]
    bankrupt = final_year.join(acts.filter(pl.col("action") == "bankruptcyliquidation"), on="ticker")
    assert (bankrupt["ret"] < -0.5).all()
    merged = final_year.join(acts.filter(pl.col("action") == "acquisitionby"), on="ticker")
    assert merged["contraticker"].null_count() == 0
    assert (merged["value"] / merged["last_price"]).is_between(1.0, 1.03).all()  # deal price just above the last trade
    assert (merged["late_move"] < 0.05).all()  # pinned to the deal: only the spread moves


def test_universe_is_non_trivial_and_filters_bite(built):
    u = built.universe.join(built.sec.select("security_id", "category"), on="security_id")
    assert u["in_universe"].mean() >= 0.30
    per_month = u.group_by("date").agg(pl.col("in_universe").sum())["in_universe"]
    assert per_month.median() >= 15
    # ETFs, ADRs and secondary share classes are present and never admitted
    other = u.filter(~pl.col("category").is_in(["Domestic Common Stock", "Domestic Common Stock Primary Class"]))
    assert {"ETF", "ADR Common Stock"} <= set(other["category"])
    assert not other["in_universe"].any() and not other["is_common"].any()
    # each of the three screens rejects some common stocks
    common = u.filter(pl.col("is_common"))
    assert (common["close_unadj"] < 5).sum() >= 20
    assert (common["market_cap"] < common["mcap_threshold"]).sum() >= 20
    assert (common["adv"] < 1.0e6).sum() >= 20
    # delisted names were members while they traded
    gone = built.sec.drop_nulls("delisted_date")["security_id"]
    assert u.filter(pl.col("security_id").is_in(gone.implode()) & pl.col("in_universe")).height > 0


# fundamentals

def test_sf1_has_every_metric_in_every_as_reported_dimension(frames):
    sf1 = frames["SF1"]
    assert set(REQUIRED_METRICS) <= set(sf1.columns)
    assert set(sf1["dimension"]) >= {"ARQ", "ART", "ARY", "MRQ"}
    reported = sf1.filter(pl.col("dimension").is_in(AS_REPORTED))
    assert reported.select(REQUIRED_METRICS).null_count().sum_horizontal()[0] == 0
    assert reported.select(pl.col(REQUIRED_METRICS).is_finite().all()).row(0) == (True,) * len(REQUIRED_METRICS)
    assert (reported["capex"] < 0).all()
    assert (reported["ncfdiv"] <= 0).all() and (reported["ncfdiv"] < 0).any()
    assert (reported["dps"] >= 0).all() and ((reported["dps"] > 0) == (reported["ncfdiv"] < 0)).all()
    assert (reported["ncfcommon"] > 0).any() and (reported["ncfcommon"] < 0).any()  # issuers and buyers-back
    assert (reported.select(pl.min_horizontal("revenue", "assets", "equity", "sharesbas", "shareswa"))[:, 0] > 0).all()
    assert (reported["netinc"] < 0).any()


def test_accounting_identities(frames):
    r = frames["SF1"].filter(pl.col("dimension").is_in(AS_REPORTED))
    scale = r["assets"]

    def same(a: pl.Series, b: pl.Series) -> bool:
        return ((a - b).abs() <= 1e-9 * scale).all()

    assert same(r["assets"], r["liabilities"] + r["equity"])
    assert same(r["fcf"], r["ncfo"] + r["capex"])
    assert same(r["workingcapital"], r["assetsc"] - r["liabilitiesc"])
    assert ((r["eps"] * r["shareswa"] / r["netinc"] - 1).abs() < 1e-9).all()
    assert (r["gp"] < r["revenue"]).all() and (r["opinc"] < r["gp"]).all() and (r["ebitda"] > r["ebit"]).all()
    assert (r["assetsc"] < r["assets"]).all() and (r["cashneq"] < r["assetsc"]).all()
    assert (r["liabilitiesc"] < r["liabilities"]).all() and (r["debt"] < r["liabilities"]).all()


def test_trailing_and_annual_flows_add_up(frames):
    sf1 = frames["SF1"]
    arq, art, ary = (first_filed(sf1, dim) for dim in AS_REPORTED)
    rolled = arq.sort("ticker", "reportperiod").with_columns(
        *[pl.col(c).rolling_sum(4).over("ticker").name.suffix("_sum") for c in FLOWS],
        (pl.col("reportperiod") - pl.col("reportperiod").shift(3).over("ticker")).dt.total_days().alias("span"),
    )
    # four consecutive quarters only (a late filing missing at the snapshot leaves a hole)
    both = art.join(rolled.filter(pl.col("span") < 280), on=["ticker", "reportperiod"], suffix="_q")
    assert both.height > 0.9 * art.height
    for c in FLOWS:
        assert ((both[c] - both[f"{c}_sum"]).abs() <= 1e-9 * both["assets"]).all(), c
        assert not ((both[c] - both[f"{c}_q"]).abs() <= 1e-9 * both["assets"]).all(), c  # a year is not a quarter
    # balance-sheet items are point-in-time: identical in every dimension
    for c in ("assets", "equity", "debt", "sharesbas"):
        assert (both[c] == both[f"{c}_q"]).all(), c

    # ARY: one row per fiscal year, equal to the trailing year at the fiscal year end
    annual = ary.join(art, on=["ticker", "reportperiod"], suffix="_t")
    assert annual.height == ary.height
    for c in (*FLOWS, "assets", "eps", "dps"):
        assert (annual[c] == annual[f"{c}_t"]).all(), c
    gaps = ary.sort("ticker", "reportperiod").select(
        (pl.col("reportperiod") - pl.col("reportperiod").shift(1).over("ticker")).dt.total_days()
    )[:, 0].drop_nulls()
    assert gaps.is_between(365, 366).all()
    assert ary["fiscalperiod"].str.ends_with("-FY").all() and arq["fiscalperiod"].str.contains(r"^\d{4}-Q[1-4]$").all()
    assert ary["reportperiod"].dt.month().n_unique() > 1  # not every fiscal year ends in December


def test_filing_lags(frames):
    r = first_filed(frames["SF1"], "ARQ")
    lag = (r["datekey"] - r["reportperiod"]).dt.total_days()
    assert lag.min() >= 30
    assert 35 <= lag.median() <= 60
    on_time = lag.filter(lag <= 90)
    assert on_time.max() <= 77  # 75 days, rolled forward off a weekend
    assert 0.005 < (lag > 90).mean() < 0.10  # some late filers
    assert r["datekey"].dt.weekday().max() <= 5
    assert (frames["SF1"]["datekey"] <= SMALL["end"]).all()


def test_amendments_and_restatements_are_dropped_by_the_build(frames, built):
    sf1 = frames["SF1"]
    key = ["ticker", "dimension", "reportperiod"]
    reported = sf1.filter(pl.col("dimension").is_in(AS_REPORTED))
    refiled = reported.filter(pl.struct(key).is_duplicated()).sort(*key, "datekey")
    assert refiled.height >= 10
    pairs = refiled.group_by(key, maintain_order=True).agg(
        pl.col("datekey").first().alias("first_date"), pl.col("datekey").last().alias("amended_date"),
        pl.col("revenue").first().alias("first_revenue"), pl.col("revenue").last().alias("amended_revenue"),
        pl.col("assets").n_unique().alias("n_assets"),
    )
    assert (pairs["amended_date"] > pairs["first_date"]).all()
    assert (pairs["amended_revenue"] != pairs["first_revenue"]).all()

    ids = built.sec.select("ticker", "security_id")
    got = built.fund.filter(pl.col("metric") == "revenue").join(ids, on="security_id")
    # one value per period, and it is the first one filed
    assert got.select("security_id", "dimension", "fiscal_period_end").is_duplicated().sum() == 0
    assert got.height == reported.unique(key).height
    check = pairs.join(
        got, left_on=key, right_on=["ticker", "dimension", "fiscal_period_end"], how="left"
    )
    assert (check["value"] == check["first_revenue"]).all()
    assert (check["available_date"] == check["first_date"]).all()

    # restated rows exist in the raw data, dated at the period end (not point-in-time), and are excluded
    restated = sf1.filter(pl.col("dimension").str.starts_with("MR"))
    assert restated.height >= 20 and (restated["datekey"] == restated["reportperiod"]).all()
    assert not built.fund["dimension"].str.starts_with("MR").any()
    assert (built.fund["available_date"] > built.fund["fiscal_period_end"]).all()


# macro

def test_fred_observations(frames, cfg):
    m = frames["FRED_OBSERVATIONS"]
    assert m.schema == pl.Schema({
        "series": pl.String, "observation_date": pl.Date, "available_date": pl.Date, "value": pl.Float64,
        "first_release": pl.Boolean,
    })
    assert set(m["series"]) == {s.id for s in cfg.macro} == {"CPIAUCSL", "DGS10", "DGS3MO", "BAA10Y", "FEDFUNDS"}
    assert m["first_release"].all() and m["value"].is_finite().all()
    assert m.select("series", "observation_date").is_duplicated().sum() == 0
    assert (m["available_date"] <= SMALL["end"]).all()
    lag = m.with_columns((pl.col("available_date") - pl.col("observation_date")).dt.total_days().alias("lag"))
    lags = {r["series"]: (r["lo"], r["hi"]) for r in lag.group_by("series").agg(
        pl.col("lag").min().alias("lo"), pl.col("lag").max().alias("hi")).iter_rows(named=True)}
    assert lags == {"CPIAUCSL": (45, 45), "FEDFUNDS": (35, 35), "DGS10": (1, 1), "DGS3MO": (1, 1), "BAA10Y": (1, 1)}

    cpi = m.filter(pl.col("series") == "CPIAUCSL").sort("observation_date")
    assert cpi["observation_date"].min() <= d(2002, 1, 1)  # two years before the first price
    assert (cpi["observation_date"].dt.day() == 1).all()
    assert cpi.height == cpi["observation_date"].dt.truncate("1mo").n_unique() >= 90
    assert cpi["value"][-1] > cpi["value"][0] > 0
    daily = m.filter(pl.col("series") == "DGS10")
    assert daily.height > 1500 and daily["observation_date"].dt.weekday().max() <= 5
    assert (m.filter(pl.col("series") != "CPIAUCSL")["value"] > 0).all()


def test_write_raw_puts_frames_where_the_build_reads_them(tmp_path):
    frames = synthetic.generate(n_stocks=12, start=d(2006, 1, 2), end=d(2007, 6, 29), seed=3, with_truth=True)
    store = Store(tmp_path / "data")
    synthetic.write_raw(store, frames)
    for table in sharadar.TABLES:
        assert_frame_equal(store.scan_raw("sharadar", table).collect(), frames[table])
    assert_frame_equal(store.scan_raw("fred", "OBSERVATIONS").collect(), frames["FRED_OBSERVATIONS"])
    written = sorted(p.relative_to(store.root).as_posix() for p in store.root.rglob("*.parquet"))
    assert written == sorted(
        [f"raw/sharadar/{t}.parquet" for t in sharadar.TABLES] + ["raw/fred/OBSERVATIONS.parquet"]
    )  # TRUTH is not vendor data


# provenance: synthetic data must never silently replace or mix with vendor data

@pytest.fixture(scope="module")
def tiny():
    return synthetic.generate(n_stocks=10, start=d(2006, 1, 2), end=d(2006, 6, 30), seed=1)


def test_write_raw_records_a_marker(tmp_path, tiny):
    store = Store(tmp_path / "data")
    assert not synthetic.raw_is_synthetic(store)
    params = {"n_stocks": 10, "start": "2006-01-02", "end": "2006-06-30", "seed": 1}
    path = synthetic.write_raw(store, tiny, params)
    assert path == store.root / "raw" / "SYNTHETIC.json"
    assert yaml.safe_load(path.read_text()) == params  # JSON is YAML
    assert synthetic.raw_is_synthetic(store)


def test_write_raw_refuses_to_overwrite_vendor_data(tmp_path, tiny):
    store = Store(tmp_path / "data")
    vendor_sep = pl.DataFrame({"ticker": ["REAL"], "close": [1.0]})
    store.write_raw("sharadar", "SEP", vendor_sep)
    with pytest.raises(FileExistsError, match="vendor"):
        synthetic.write_raw(store, tiny)
    assert_frame_equal(store.scan_raw("sharadar", "SEP").collect(), vendor_sep)  # untouched
    assert not synthetic.raw_is_synthetic(store)
    synthetic.write_raw(store, tiny, force=True)
    assert store.scan_raw("sharadar", "SEP").collect().height == tiny["SEP"].height
    assert synthetic.raw_is_synthetic(store)


def test_vendor_fred_alone_also_blocks(tmp_path, tiny):
    store = Store(tmp_path / "data")
    store.write_raw("fred", "OBSERVATIONS", tiny["FRED_OBSERVATIONS"])
    with pytest.raises(FileExistsError):
        synthetic.write_raw(store, tiny)


def test_rewriting_synthetic_over_synthetic_is_allowed(tmp_path, tiny):
    store = Store(tmp_path / "data")
    synthetic.write_raw(store, tiny, {"seed": 1})
    synthetic.write_raw(store, tiny, {"seed": 2})
    assert yaml.safe_load(synthetic.marker_path(store).read_text()) == {"seed": 2}


def test_vendor_ingest_guard(tmp_path, tiny):
    store = Store(tmp_path / "data")
    synthetic.guard_vendor_write(store)  # empty raw dir: nothing to protect
    store.write_raw("sharadar", "SEP", tiny["SEP"])
    synthetic.guard_vendor_write(store)  # vendor over vendor: fine
    synthetic.write_raw(store, tiny, force=True)
    # a partial vendor refresh over synthetic raw data would mix the two: refuse
    with pytest.raises(RuntimeError, match="synthetic"):
        synthetic.guard_vendor_write(store)
    assert synthetic.raw_is_synthetic(store)
    assert store.raw_path("sharadar", "SF1").exists()
    # explicit replacement removes every synthetic raw file, then the marker
    synthetic.guard_vendor_write(store, replace_synthetic=True)
    assert not synthetic.raw_is_synthetic(store)
    assert not any((store.root / "raw").rglob("*.parquet"))


def test_marker_written_before_data(tmp_path, tiny, monkeypatch):
    # a crash halfway through must still leave the raw directory flagged as synthetic
    store = Store(tmp_path / "data")
    calls = []

    def failing_write(vendor, table, df):
        calls.append(table)
        if len(calls) == 2:
            raise OSError("disk full")
        return Store.write_raw(store, vendor, table, df)

    monkeypatch.setattr(store, "write_raw", failing_write)
    with pytest.raises(OSError):
        synthetic.write_raw(store, tiny)
    assert synthetic.raw_is_synthetic(store)


# the planted signal (150 stocks, 13 years; about 1.5 s for both tests together)

def test_planted_signal_has_realistic_strength(signal_run):
    frames, built = signal_run
    truth = frames["TRUTH"].join(built.sec.select("ticker", "security_id"), on="ticker")
    df = built.labels.join(truth, on=["security_id", "date"], how="left")
    # the truth covers every label except the sample's last day: no following month was simulated
    assert df.filter(pl.col("date") < SIGNAL["end"])["alpha"].null_count() == 0
    ic = mean_rank_ic(df, "alpha")
    # about 0.09 for this seed, 0.09 on average over seeds: clearly there, nowhere near a toy
    assert 0.03 < ic < 0.20


def test_profitability_known_today_predicts_the_next_year(signal_run):
    frames, built = signal_run
    # gp / assets from the latest trailing-twelve-month filing made strictly before each month end
    gpa = (
        built.fund.filter((pl.col("dimension") == "ART") & pl.col("metric").is_in(["gp", "assets"]))
        .pivot("metric", index=["security_id", "fiscal_period_end", "available_date"], values="value")
        .select("security_id", pl.col("available_date").alias("date"), (pl.col("gp") / pl.col("assets")).alias("gpa"))
        .sort("date")
    )
    df = built.labels.sort("date").join_asof(
        gpa, on="date", by="security_id", strategy="backward", allow_exact_matches=False, check_sortedness=False
    )
    assert df["gpa"].is_not_null().mean() > 0.95
    assert mean_rank_ic(df, "gpa") > 0.01  # about 0.05 for this seed


def test_truth_frame_is_optional_and_signal_can_be_switched_off(frames):
    kwargs = dict(n_stocks=20, start=d(2005, 1, 3), end=d(2007, 12, 31), seed=4)
    off = synthetic.generate(**kwargs, signal=0.0, with_truth=True)
    on = synthetic.generate(**kwargs, with_truth=True)
    assert set(on) == VENDOR_KEYS | {"TRUTH"}
    assert on["TRUTH"].columns == ["ticker", "date", "alpha"]
    # without the planted drift, expected returns differ only by each stock's constant volatility drag
    assert off["TRUTH"].group_by("ticker").agg(pl.col("alpha").n_unique())["alpha"].max() == 1
    assert on["TRUTH"].group_by("ticker").agg(pl.col("alpha").n_unique())["alpha"].min() > 1
    assert_frame_equal(on["TICKERS"].drop("firstquarter", "lastquarter"), off["TICKERS"].drop("firstquarter", "lastquarter"))
    assert on["SEP"]["close"].sum() != off["SEP"]["close"].sum()

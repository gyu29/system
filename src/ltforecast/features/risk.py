"""Risk features (plan 3.1; Ang, Hodrick, Xing & Zhang 2006; Frazzini & Pedersen 2014; Bali, Cakici
& Whitelaw 2011).

    vol_1y      sqrt(252) x std (ddof=1) of daily log total returns over the trailing 252 price rows
                (at least 126 returns)
    idio_vol    sqrt(252) x residual std of an OLS (with intercept) of the daily simple return on the
                same-day universe benchmark return over the trailing 252 price rows (at least 126
                paired days). Residual variance = RSS / (n - 2).
    beta_60m    slope (with intercept) of the monthly total return on the benchmark's monthly return
                over the trailing 60 calendar months (at least 24 paired months)
    max_ret_1m  largest daily simple return over the trailing 21 price rows (at least 15 returns)

Conventions
- Every window ends at the prediction date t, inclusive. An entry gets values only if the security
  has a price row dated exactly t (nothing is carried forward); otherwise all four are null.
- A daily return is adj_close / previous row's adj_close - 1 within the security; it is null on the
  first row and when either price is missing or not positive. Daily windows count the security's
  own price rows, so a null return uses up a slot of the window without being an observation.
- Benchmark days with n_members = 0 (before the universe exists, bench_ret is a placeholder 0) are
  not observations: they are left out of the daily regression, and a month that contains one has no
  benchmark return.
- A monthly return runs from the month-end trading day of the previous calendar month to this one
  (`panel.months_back`); it is null if the security has no price on either day. The benchmark's is
  the ratio of bench_level between the same two days.
- A regression on a benchmark with no variance in the window has no slope: idio_vol / beta_60m are
  null. Raw values are not sign-flipped (the catalog's `sign` says all four predict lower returns).
- Output is Float64 with null for missing, never NaN or inf. `src_date_risk` is the latest date of
  an observation that went into a non-null feature (null if all four are null), never after t.

Scale: daily windows are differences of per-security cumulative sums (no per-row Python), computed
for `CHUNK_SECURITIES` securities at a time so that only a slice of prices_daily is widened at once.
"""

from __future__ import annotations

import polars as pl

from ltforecast.features.panel import FeatureContext, month_end_prices, months_back

FAMILIES = ("risk",)
FEATURES = ("vol_1y", "idio_vol", "beta_60m", "max_ret_1m")
SRC_DATE = "src_date_risk"

TRADING_DAYS = 252          # annualization
VOL_WINDOW, VOL_MIN_OBS = 252, 126
BETA_WINDOW, BETA_MIN_OBS = 60, 24
MAX_WINDOW, MAX_MIN_OBS = 21, 15
CHUNK_SECURITIES = 1000
# A centred sum of squares this small relative to the raw one is rounding noise of the cumulative
# sums, i.e. a series that is constant over the window.
_REL_EPS = 1e-10

_KEYS = ["security_id", "date"]
_SID = "security_id"


def _finite(e: pl.Expr) -> pl.Expr:
    return pl.when(e.is_finite()).then(e)


def _trailing_sum(e: pl.Expr, window: int) -> pl.Expr:
    """Sum of the non-null values of `e` over the trailing `window` rows of the security."""
    c = e.fill_null(0.0).cum_sum()
    return (c - c.shift(window).fill_null(0.0)).over(_SID)


def _last_date(valid: pl.Expr) -> pl.Expr:
    """Date of the latest row of the security, up to this one, where `valid` holds."""
    return pl.when(valid).then(pl.col("date")).forward_fill().over(_SID)


def _bench_daily(benchmark: pl.DataFrame) -> pl.DataFrame:
    """date, _x: the benchmark return on days it has members (null otherwise)."""
    return benchmark.select(
        "date",
        pl.when(pl.col("n_members") > 0).then(_finite(pl.col("bench_ret"))).alias("_x"),
    ).drop_nulls("_x")


def _daily(prices: pl.DataFrame, bench: pl.DataFrame, entries: pl.DataFrame) -> pl.DataFrame:
    """vol_1y, idio_vol, max_ret_1m and their source dates at the entries of one chunk of securities
    (only entries that have a price row on that date come back)."""
    w = VOL_WINDOW
    adj, prev = pl.col("adj_close"), pl.col("adj_close").shift(1).over(_SID)
    ret, lr, x, y = pl.col("_ret"), pl.col("_lr"), pl.col("_xp"), pl.col("_yp")
    n, sl, sll = pl.col("_n"), pl.col("_sl"), pl.col("_sll")
    m, sx, sy, sxx, sxy, syy = (pl.col(c) for c in ("_m", "_sx", "_sy", "_sxx", "_sxy", "_syy"))

    var_sum = sll - sl * sl / n               # centred sum of squares of the log returns
    cxx = sxx - sx * sx / m
    cxy = sxy - sx * sy / m
    cyy = syy - sy * sy / m
    rss = cyy - cxy * cxy / cxx
    annual = float(TRADING_DAYS)

    out = (
        prices.lazy()
        .select(_SID, "date", "adj_close")
        .join(bench.lazy(), on="date", how="left")
        .sort(_SID, "date")  # after the join: a join does not keep row order, the windows rely on it
        .with_columns(_finite(pl.when((adj > 0) & (prev > 0)).then(adj / prev - 1)).alias("_ret"))
        .with_columns(
            (1 + ret).log().alias("_lr"),
            pl.when(ret.is_not_null()).then(pl.col("_x")).alias("_xp"),
            pl.when(pl.col("_x").is_not_null()).then(ret).alias("_yp"),
        )
        .with_columns(
            _trailing_sum(lr.is_not_null().cast(pl.Float64), w).alias("_n"),
            _trailing_sum(lr, w).alias("_sl"),
            _trailing_sum(lr * lr, w).alias("_sll"),
            _trailing_sum(x.is_not_null().cast(pl.Float64), w).alias("_m"),
            _trailing_sum(x, w).alias("_sx"),
            _trailing_sum(y, w).alias("_sy"),
            _trailing_sum(x * x, w).alias("_sxx"),
            _trailing_sum(x * y, w).alias("_sxy"),
            _trailing_sum(y * y, w).alias("_syy"),
            _trailing_sum(ret.is_not_null().cast(pl.Float64), MAX_WINDOW).alias("_k"),
            ret.fill_null(float("-inf")).rolling_max(MAX_WINDOW, min_samples=1).over(_SID).alias("_max"),
            _last_date(ret.is_not_null()).alias("_d_ret"),
            _last_date(x.is_not_null()).alias("_d_pair"),
        )
        .join(entries.lazy().select(_KEYS), on=_KEYS, how="semi")
        .select(
            *_KEYS,
            pl.when(n >= VOL_MIN_OBS)
            .then(
                pl.when(var_sum > _REL_EPS * sll).then((var_sum / (n - 1) * annual).sqrt()).otherwise(0.0)
            )
            .alias("vol_1y"),
            pl.when((m >= VOL_MIN_OBS) & (cxx > _REL_EPS * sxx))
            .then(pl.when(rss > _REL_EPS * cyy).then((rss / (m - 2) * annual).sqrt()).otherwise(0.0))
            .alias("idio_vol"),
            pl.when(pl.col("_k") >= MAX_MIN_OBS).then(pl.col("_max")).alias("max_ret_1m"),
            "_d_ret",
            "_d_pair",
        )
        .collect()
    )
    return out


def _monthly_beta(ctx: FeatureContext, entries: pl.DataFrame) -> pl.DataFrame:
    """beta_60m and the date of the latest month used, per (security, month-end trading day)."""
    back = months_back(ctx, 1)
    sids = entries.select(_SID).unique()
    px = month_end_prices(ctx).select(_SID, "date", "adj_close").join(sids, on=_SID, how="semi")
    stock = (
        px.join(back, on="date", how="left")
        .join(
            px.select(_SID, pl.col("date").alias("date_k"), pl.col("adj_close").alias("_prev")),
            on=[_SID, "date_k"],
            how="left",
        )
        .select(
            *_KEYS,
            _finite(
                pl.when((pl.col("adj_close") > 0) & (pl.col("_prev") > 0)).then(
                    pl.col("adj_close") / pl.col("_prev") - 1
                )
            ).alias("_y"),
        )
    )
    # _z counts the member-less days so far: equal at both ends of a month = none inside it
    levels = ctx.benchmark.sort("date").select(
        "date",
        pl.col("bench_level").alias("_lvl"),
        (pl.col("n_members").fill_null(0) <= 0).cast(pl.Int64).cum_sum().alias("_z"),
    )
    bench = (
        back.join(levels, on="date", how="inner")
        .join(levels.rename({"date": "date_k", "_lvl": "_lvl_k", "_z": "_z_k"}), on="date_k", how="inner")
        .select(
            "date",
            _finite(
                pl.when((pl.col("_z") == pl.col("_z_k")) & (pl.col("_lvl") > 0) & (pl.col("_lvl_k") > 0)).then(
                    pl.col("_lvl") / pl.col("_lvl_k") - 1
                )
            ).alias("_x"),
        )
    )
    x = pl.when(pl.col("_y").is_not_null()).then(pl.col("_x"))
    y = pl.when(pl.col("_x").is_not_null()).then(pl.col("_y"))
    month = (pl.col("date").dt.year() * 12 + pl.col("date").dt.month()).cast(pl.Int32)
    sums = (
        stock.join(bench, on="date", how="left")
        .select(*_KEYS, month.alias("_mi"), x.alias("_xp"), y.alias("_yp"))
        .sort(_SID, "_mi")
        .rolling(index_column="_mi", period=f"{BETA_WINDOW}i", group_by=_SID)
        .agg(
            pl.col("date").last(),
            pl.col("_xp").count().cast(pl.Float64).alias("_m"),
            pl.col("_xp").sum().alias("_sx"),
            pl.col("_yp").sum().alias("_sy"),
            (pl.col("_xp") * pl.col("_xp")).sum().alias("_sxx"),
            (pl.col("_xp") * pl.col("_yp")).sum().alias("_sxy"),
            pl.col("date").filter(pl.col("_xp").is_not_null()).max().alias("_d_beta"),
        )
    )
    m, sx, sy, sxx, sxy = (pl.col(c) for c in ("_m", "_sx", "_sy", "_sxx", "_sxy"))
    cxx = sxx - sx * sx / m
    cxy = sxy - sx * sy / m
    return sums.select(
        *_KEYS,
        pl.when((m >= BETA_MIN_OBS) & (cxx > _REL_EPS * sxx)).then(cxy / cxx).alias("beta_60m"),
        "_d_beta",
    )


def compute(ctx: FeatureContext) -> pl.DataFrame:
    """Columns: security_id, date, vol_1y, idio_vol, beta_60m, max_ret_1m (Float64), src_date_risk.

    One row per row of `ctx.entries`, sorted by security_id, date. `src_date_risk` is the latest
    observation date behind a non-null feature (null when all four are null), never after `date`.
    """
    entries = ctx.entries.select(_KEYS).sort(_KEYS)
    bench = _bench_daily(ctx.benchmark)
    sids = entries[_SID].unique().sort()
    parts = []
    for i in range(0, len(sids), CHUNK_SECURITIES):
        chunk = sids.slice(i, CHUNK_SECURITIES).to_frame()
        parts.append(
            _daily(
                ctx.prices.join(chunk, on=_SID, how="semi"),
                bench,
                entries.join(chunk, on=_SID, how="semi"),
            )
        )
    schema = {_SID: pl.Int64, "date": pl.Date, "vol_1y": pl.Float64, "idio_vol": pl.Float64,
              "max_ret_1m": pl.Float64, "_d_ret": pl.Date, "_d_pair": pl.Date}
    daily = pl.concat(parts) if parts else pl.DataFrame(schema=schema)

    def used(feature: str, date_col: str) -> pl.Expr:
        return pl.when(pl.col(feature).is_not_null()).then(pl.col(date_col))

    out = (
        entries.join(daily, on=_KEYS, how="left")
        .join(_monthly_beta(ctx, entries), on=_KEYS, how="left")
        .with_columns(_finite(pl.col(f).cast(pl.Float64)).alias(f) for f in FEATURES)
        .with_columns(
            pl.max_horizontal(
                used("vol_1y", "_d_ret"),
                used("max_ret_1m", "_d_ret"),
                used("idio_vol", "_d_pair"),
                used("beta_60m", "_d_beta"),
            ).alias(SRC_DATE)
        )
        .select(*_KEYS, *FEATURES, SRC_DATE)
        .sort(_KEYS)
    )
    if out.height != entries.height:
        raise AssertionError(f"risk features: {out.height} rows for {entries.height} entries")
    if out.filter(pl.col(SRC_DATE) > pl.col("date")).height:
        raise AssertionError("risk features: a source date is after its prediction date")
    return out

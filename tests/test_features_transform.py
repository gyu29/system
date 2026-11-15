"""Hand-computed ranks, sector-relative ranks, has_ flags and the full transform on a tiny catalog.

Rank convention: (average rank - 0.5) / n - 0.5 over the n non-null values of a date.
"""

import datetime as dt
import math
import random

import polars as pl
import pytest

from ltforecast.features.catalog import Catalog, FamilySpec, FeatureSpec, load_catalog
from ltforecast.features.transform import has_flags, impute_for_linear, sector_rank, transform, xs_rank

d = dt.date
D1, D2 = d(2020, 1, 31), d(2020, 2, 28)


def spec(name: str, family: str, kind: str = "stock") -> FeatureSpec:
    return FeatureSpec(name=name, family=family, label=name, definition="-", source=("-",), lookback="-",
                       lag="-", citation="-", sign=1, kind=kind)


CATALOG = Catalog(
    features=(spec("bp", "value"), spec("ey", "value"), spec("mom", "momentum"), spec("term", "context", "context")),
    families={
        "value": FamilySpec("value", "Value", sector_relative=True),
        "momentum": FamilySpec("momentum", "Momentum"),
        "context": FamilySpec("context", "Context"),
    },
)


def frame(**cols) -> pl.DataFrame:
    return pl.DataFrame(cols)


def test_xs_rank_exact_values_with_ties_and_nulls():
    df = frame(
        date=[D1] * 5 + [D2] * 4,
        security_id=[1, 2, 3, 4, 5, 1, 2, 3, 4],
        x=[10.0, 20.0, 20.0, None, 40.0, 3.0, 1.0, 2.0, 5.0],
    )
    out = df.with_columns(xs_rank("x"))
    # D1: four values; ranks 1, 2.5, 2.5, 4 -> (r - 0.5) / 4 - 0.5.  D2: ranks 3, 1, 2, 4
    assert out["x"].to_list() == [-0.375, 0.0, 0.0, None, 0.375, 0.125, -0.375, -0.125, 0.375]
    assert out.columns == ["date", "security_id", "x"]


def test_xs_rank_edge_cases():
    df = frame(date=[D1, D2, D2, D2, D2], x=[7.0, float("nan"), float("inf"), 1.0, 2.0])
    # a lone value sits at the centre; NaN and inf are not ranked (and do not count towards n)
    assert df.with_columns(xs_rank("x"))["x"].to_list() == [0.0, None, None, -0.25, 0.25]
    all_null = frame(date=[D1, D1], x=pl.Series([None, None], dtype=pl.Float64))
    assert all_null.with_columns(xs_rank("x"))["x"].to_list() == [None, None]
    wide = frame(date=[D1] * 100, x=[float(i) for i in range(100)]).with_columns(xs_rank("x"))
    assert wide["x"].min() == pytest.approx(-0.495) and wide["x"].max() == pytest.approx(0.495)
    assert wide["x"].mean() == pytest.approx(0.0)


def test_sector_rank_within_date_and_sector():
    df = frame(
        date=[D1] * 7,
        sector=["A", "A", "A", "B", None, None, None],
        x=[1.0, 3.0, 2.0, 9.0, 4.0, 5.0, 6.0],
    )
    out = df.with_columns(sector_rank("x", min_group=2))
    third = 1 / 3
    # A: ranks 1, 3, 2 of three.  B: a single name is below min_group.  No sector: no sector rank.
    assert out["x_sr"].to_list() == pytest.approx([-third, third, 0.0, None, None, None, None])
    assert out["x"].to_list() == df["x"].to_list()   # the raw column is untouched


def test_sector_rank_needs_min_group_values():
    five = [-0.4, -0.2, 0.0, 0.2, 0.4]
    df = frame(
        date=[D1] * 14,
        sector=["A"] * 5 + ["B"] * 4 + ["C"] * 5,
        # C has five names but only four values
        x=[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, None],
    )
    assert df.with_columns(sector_rank("x"))["x_sr"].to_list() == pytest.approx(five + [None] * 9)
    # the same sector on another date is a separate group
    two_dates = frame(date=[D1] * 3 + [D2] * 2, sector=["A"] * 5, x=[1.0, 2.0, 3.0, 4.0, 5.0])
    assert two_dates.with_columns(sector_rank("x", min_group=3))["x_sr"].to_list() == pytest.approx(
        [-1 / 3, 0.0, 1 / 3, None, None])


def test_has_flags():
    df = frame(bp=[1.0, None, None, 0.0], ey=[2.0, 3.0, None, None], mom=[None, None, 0.5, None], term=[1.0] * 4)
    out = has_flags(df, CATALOG)
    assert out.columns == ["bp", "ey", "mom", "term", "has_value", "has_momentum"]   # no flag for context
    assert out["has_value"].to_list() == [1, 1, 0, 1]      # any feature of the family; 0.0 is a value
    assert out["has_momentum"].to_list() == [0, 0, 1, 0]
    assert out.schema["has_value"] == pl.Int8 and out["has_value"].null_count() == 0


RAW = frame(
    # deliberately unsorted
    security_id=[3, 1, 2, 4, 5, 2, 1],
    date=[D1, D1, D1, D1, D1, D2, D2],
    sector=["A", "A", "A", "B", None, "A", "A"],
    bp=[0.9, 0.1, 0.5, 0.3, None, 2.0, 1.0],
    ey=[None, 0.05, 0.05, None, None, 0.1, None],
    mom=[0.2, -0.1, None, 0.4, None, 0.0, 0.0],
    term=[1.5, 1.5, 1.5, 1.5, 1.5, 0.8, 0.8],
    src_date_value=[D1] * 5 + [D2] * 2,   # extra columns are dropped
)


def test_transform_end_to_end():
    out = transform(RAW, CATALOG, min_group=2)
    assert out.columns == ["date", "security_id", "sector", "bp", "ey", "mom", "bp_sr", "ey_sr", "has_value",
                           "has_momentum", "term"]
    assert out.select("date", "security_id").rows() == [(D1, 1), (D1, 2), (D1, 3), (D1, 4), (D1, 5), (D2, 1), (D2, 2)]
    col = {c: out[c].to_list() for c in out.columns}
    # D1 bp: 0.1, 0.5, 0.9, 0.3, null -> ranks 1, 3, 4, 2 of four.  D2: 1.0, 2.0 -> ranks 1, 2 of two
    assert col["bp"] == [-0.375, 0.125, 0.375, -0.125, None, -0.25, 0.25]
    # D1 ey: a tie between the only two values; D2: a single value
    assert col["ey"] == [0.0, 0.0, None, None, None, None, 0.0]
    # D1 mom: -0.1, null, 0.2, 0.4 -> ranks 1, 2, 3 of three.  D2: a tie
    assert col["mom"] == pytest.approx([-1 / 3, None, 0.0, 1 / 3, None, 0.0, 0.0])
    # sector A on D1 holds stocks 1, 2, 3 (bp ranks 1, 2, 3 of three); B has one name; stock 5 has no sector
    assert col["bp_sr"] == pytest.approx([-1 / 3, 0.0, 1 / 3, None, None, -0.25, 0.25])
    assert col["ey_sr"] == [0.0, 0.0, None, None, None, None, None]   # D2: one value in the sector
    assert col["has_value"] == [1, 1, 1, 1, 0, 1, 1]
    assert col["has_momentum"] == [1, 0, 1, 1, 0, 1, 1]
    assert col["term"] == [1.5, 1.5, 1.5, 1.5, 1.5, 0.8, 0.8]         # context: passed through unranked
    assert col["sector"] == ["A", "A", "A", "B", None, "A", "A"]
    assert out.schema["bp"] == pl.Float64 and out.schema["bp_sr"] == pl.Float64 and out.schema["term"] == pl.Float64


def test_transform_default_min_group_is_five():
    assert transform(RAW, CATALOG)["bp_sr"].null_count() == RAW.height


def test_transform_requires_every_catalog_column():
    with pytest.raises(ValueError, match=r"\['mom'\]"):
        transform(RAW.drop("mom"), CATALOG)
    with pytest.raises(ValueError, match="sector"):
        transform(RAW.drop("sector"), CATALOG)


def test_impute_for_linear():
    out = transform(RAW, CATALOG, min_group=2)
    filled = impute_for_linear(out, ["bp", "bp_sr"])
    assert filled["bp"].to_list() == [-0.375, 0.125, 0.375, -0.125, 0.0, -0.25, 0.25]
    assert filled["bp_sr"].to_list() == pytest.approx([-1 / 3, 0.0, 1 / 3, 0.0, 0.0, -0.25, 0.25])
    assert filled["ey"].to_list() == out["ey"].to_list()   # only the named columns
    assert filled.columns == out.columns


def test_transform_with_the_real_catalog():
    catalog = load_catalog()
    rng = random.Random(3)
    n_stocks, dates = 40, [D1, D2]
    rows = {
        "date": [day for day in dates for _ in range(n_stocks)],
        "security_id": list(range(n_stocks)) * len(dates),
        "sector": [("Tech", "Energy", "Banks", None)[i % 4] for i in range(n_stocks)] * len(dates),
    }
    for name in catalog.names(kind="stock"):
        rows[name] = [None if rng.random() < 0.2 else rng.gauss(0, 1) for _ in range(n_stocks * len(dates))]
    for i, name in enumerate(catalog.names(kind="context")):
        rows[name] = [float(i + j) for j in range(len(dates)) for _ in range(n_stocks)]
    raw = pl.DataFrame(rows)
    out = transform(raw, catalog)

    assert out.columns == ["date", "security_id", "sector", *catalog.model_columns()]
    assert out.height == raw.height
    for name in catalog.linear_columns():
        ranked = out[name].drop_nulls()
        assert ranked.len() > 0 and -0.5 < ranked.min() and ranked.max() < 0.5, name
    for name in catalog.names(kind="stock"):
        assert out[name].null_count() == raw[name].null_count()
        # continuous values have no ties, so the centred ranks of a date sum to zero
        assert all(math.isclose(v, 0.0, abs_tol=1e-9) for v in out.group_by("date").agg(pl.col(name).sum())[name])
    assert out.filter(pl.col("sector").is_null()).select(pl.col("^.*_sr$").is_null().all()).row(0) == (True,) * 11
    for name in catalog.names(kind="context"):
        assert out[name].to_list() == raw[name].to_list()
    imputed = impute_for_linear(out, catalog.linear_columns())
    assert sum(imputed.select(catalog.linear_columns()).null_count().row(0)) == 0
    assert all(out[f].dtype == pl.Int8 for f in catalog.flag_names())

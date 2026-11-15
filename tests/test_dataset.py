"""modeling frame = features joined to labels, with the gap and the holdout locked.

Five prediction dates, one or two per period under the default config dates
(dev_end 2021-10-31, holdout 2022-11-01 -> 2025-10-31):
    2021-09-30 dev   2021-10-29 dev   2021-11-30 gap   2022-11-30 holdout   2025-11-28 live
"""

import datetime as dt

import polars as pl
import pytest

from ltforecast.dataset import (
    HoldoutLockedError,
    HoldoutToken,
    frame_from_tables,
    label_columns,
    modeling_frame,
)

d = dt.date
DATES = {
    d(2021, 9, 30): "dev",
    d(2021, 10, 29): "dev",
    d(2021, 11, 30): "gap",
    d(2022, 11, 30): "holdout",
    d(2025, 11, 28): "live",
}
MODEL_COLS = ["book_to_price", "book_to_price_sr", "has_value", "term_spread"]
HORIZON_COLS = ["fwd_ret", "bench_ret", "excess", "excess_w", "rank", "realized_vol", "label_end"]
LABEL_COLS = [f"{c}_{h}" for h in ("3m", "6m", "12m") for c in HORIZON_COLS] + ["target"]


class StubCatalog:
    """Stands in for features.catalog.Catalog: the dataset only needs model_columns()."""

    def __init__(self, cols=MODEL_COLS):
        self.cols = list(cols)

    def model_columns(self) -> list[str]:
        return self.cols


def features() -> pl.DataFrame:
    """Securities 1 and 2 on every date, deliberately unsorted, plus a column no model uses."""
    rows = [
        {
            "security_id": sid, "date": day, "sector": "Tech" if sid == 1 else "Energy",
            "book_to_price": 0.1 * sid, "book_to_price_sr": -0.1 * sid, "has_value": 1,
            "term_spread": 1.5, "not_a_model_column": 99.0,
        }
        for day in reversed(DATES)
        for sid in (2, 1)
    ]
    return pl.DataFrame(rows).with_columns(pl.col("security_id").cast(pl.Int64), pl.col("has_value").cast(pl.Int8))


def labels() -> pl.DataFrame:
    rows = []
    for day, per in DATES.items():
        for sid in (1, 2):
            row = {"security_id": sid, "date": day, "period": per}
            for h, days in (("3m", 91), ("6m", 182), ("12m", 365)):
                row |= {f"{c}_{h}": 0.01 * sid for c in HORIZON_COLS[:-1]}
                row[f"label_end_{h}"] = day + dt.timedelta(days=days)
            row["target"] = row["rank_12m"]
            rows.append(row)
    return pl.DataFrame(rows)


def test_default_is_dev_only_with_the_contract_columns():
    out = frame_from_tables(features(), labels(), StubCatalog())
    assert out.columns == ["date", "security_id", "sector", *MODEL_COLS, *LABEL_COLS, "period"]
    # 2 dev dates x 2 securities, sorted by date then security
    assert out.select("date", "security_id").rows() == [
        (d(2021, 9, 30), 1), (d(2021, 9, 30), 2), (d(2021, 10, 29), 1), (d(2021, 10, 29), 2),
    ]
    assert out["period"].unique().to_list() == ["dev"]
    assert out["date"].max() == d(2021, 10, 29)
    assert out["book_to_price"].to_list() == [0.1, 0.2, 0.1, 0.2]
    assert out["target"].to_list() == [0.01, 0.02, 0.01, 0.02]
    assert out["label_end_12m"][0] == d(2022, 9, 30)
    assert "not_a_model_column" not in out.columns


@pytest.mark.parametrize("periods", [("holdout",), ("gap",), ("dev", "holdout"), ("dev", "gap", "live")])
def test_gap_and_holdout_are_locked(periods):
    with pytest.raises(HoldoutLockedError):
        frame_from_tables(features(), labels(), StubCatalog(), periods=periods)


@pytest.mark.parametrize("fake", [True, "token", object(), HoldoutToken])
def test_only_a_real_token_unlocks(fake):
    with pytest.raises(HoldoutLockedError):
        frame_from_tables(features(), labels(), StubCatalog(), periods=("holdout",), unlock=fake)


def test_a_token_cannot_be_constructed_directly():
    with pytest.raises(HoldoutLockedError):
        HoldoutToken()
    with pytest.raises(HoldoutLockedError):
        HoldoutToken(object())


def test_issued_token_unlocks_the_holdout():
    token = HoldoutToken._issue()
    out = frame_from_tables(features(), labels(), StubCatalog(), periods=("holdout",), unlock=token)
    assert out.select("date", "security_id", "period").rows() == [
        (d(2022, 11, 30), 1, "holdout"), (d(2022, 11, 30), 2, "holdout"),
    ]
    both = frame_from_tables(features(), labels(), StubCatalog(), periods=("dev", "gap", "holdout"), unlock=token)
    assert both["period"].unique(maintain_order=True).to_list() == ["dev", "gap", "holdout"]
    assert both.height == 8


def test_live_rows_need_no_token():
    # the production job scores dates after the holdout; they are not locked
    out = frame_from_tables(features(), labels(), StubCatalog(), periods=("live",))
    assert out["date"].unique().to_list() == [d(2025, 11, 28)]


def test_unknown_or_empty_periods_raise():
    with pytest.raises(ValueError, match="periods"):
        frame_from_tables(features(), labels(), StubCatalog(), periods=("development",))
    with pytest.raises(ValueError, match="periods"):
        frame_from_tables(features(), labels(), StubCatalog(), periods=())


def test_inner_join_drops_rows_missing_on_either_side():
    f = features().filter(~((pl.col("security_id") == 2) & (pl.col("date") == d(2021, 9, 30))))
    lab = labels().filter(~((pl.col("security_id") == 1) & (pl.col("date") == d(2021, 10, 29))))
    out = frame_from_tables(f, lab, StubCatalog())
    assert out.select("date", "security_id").rows() == [(d(2021, 9, 30), 1), (d(2021, 10, 29), 2)]


def test_null_features_and_labels_are_kept():
    # missing values are the model's and the runner's business, not the loader's
    f = features().with_columns(pl.lit(None, dtype=pl.Float64).alias("book_to_price"))
    lab = labels().with_columns(pl.lit(None, dtype=pl.Float64).alias("target"))
    out = frame_from_tables(f, lab, StubCatalog())
    assert out.height == 4 and out["book_to_price"].null_count() == 4 and out["target"].null_count() == 4


def test_missing_model_column_raises():
    with pytest.raises(ValueError, match="roic"):
        frame_from_tables(features(), labels(), StubCatalog([*MODEL_COLS, "roic"]))
    with pytest.raises(ValueError, match="sector"):
        frame_from_tables(features().drop("sector"), labels(), StubCatalog())


def test_label_with_an_unrecognised_name_raises():
    # the runner hides labels from predict() by name; a label it cannot name would be a leak
    lab = labels().with_columns(pl.col("fwd_ret_12m").alias("next_year_return"))
    with pytest.raises(ValueError, match="next_year_return"):
        frame_from_tables(features(), lab, StubCatalog())


def test_feature_named_like_a_label_raises():
    f = features().with_columns(pl.col("book_to_price").alias("rank_value"))
    with pytest.raises(ValueError, match="rank_value"):
        frame_from_tables(f, labels(), StubCatalog([*MODEL_COLS, "rank_value"]))


def test_duplicate_keys_raise():
    with pytest.raises(pl.exceptions.ComputeError, match="1:1"):
        frame_from_tables(pl.concat([features(), features().head(1)]), labels(), StubCatalog())
    with pytest.raises(pl.exceptions.ComputeError, match="1:1"):
        frame_from_tables(features(), pl.concat([labels(), labels().head(1)]), StubCatalog())


def test_label_columns_by_name():
    cols = ["date", "security_id", "sector", "mom_12_1", "excess_w_12m", "rank_3m", "bench_ret_6m",
            "realized_vol_12m", "label_end_3m", "fwd_ret_12m", "excess_6m", "target", "period", "has_value"]
    assert label_columns(cols) == ["excess_w_12m", "rank_3m", "bench_ret_6m", "realized_vol_12m",
                                   "label_end_3m", "fwd_ret_12m", "excess_6m", "target", "period"]
    assert label_columns(labels().columns) == [c for c in labels().columns if c not in ("security_id", "date")]


class FakeStore:
    """scan(table) over in-memory frames, recording which tables were read."""

    def __init__(self, **tables: pl.DataFrame):
        self.tables, self.scanned = tables, []

    def scan(self, table: str) -> pl.LazyFrame:
        self.scanned.append(table)
        return self.tables[table].lazy()


def test_modeling_frame_reads_the_stored_tables(cfg):
    store = FakeStore(features=features(), labels=labels())
    out = modeling_frame(store, cfg, StubCatalog())
    assert out.equals(frame_from_tables(features(), labels(), StubCatalog()))
    assert sorted(store.scanned) == ["features", "labels"]
    assert out["date"].max() == d(2021, 10, 29) <= cfg.dates.dev_end


def test_modeling_frame_refuses_the_holdout_before_reading_anything(cfg):
    store = FakeStore(features=features(), labels=labels())
    with pytest.raises(HoldoutLockedError):
        modeling_frame(store, cfg, StubCatalog(), periods=("dev", "holdout"))
    assert store.scanned == []
    out = modeling_frame(store, cfg, StubCatalog(), periods=("holdout",), unlock=HoldoutToken._issue())
    assert out["date"].unique().to_list() == [d(2022, 11, 30)]


def test_modeling_frame_rejects_labels_stamped_under_other_dates(cfg):
    # labels built when dev_end was later: a 2022-11-30 row stamped "dev" sits inside today's holdout
    stale = labels().with_columns(
        pl.when(pl.col("date") == d(2022, 11, 30)).then(pl.lit("dev")).otherwise(pl.col("period")).alias("period")
    )
    with pytest.raises(HoldoutLockedError, match="2 label rows"):
        modeling_frame(FakeStore(features=features(), labels=stale), cfg, StubCatalog())

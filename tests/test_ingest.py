import datetime as dt
import io
import zipfile

import polars as pl
import pytest
from conftest import cpi_payload

from ltforecast.config import MacroSeries
from ltforecast.ingest import fred, sharadar

SEP_CSV = b"""ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated
AAA,2020-01-02,10,11,9,10.5,1000,10.4,21,2020-01-03
AAA,2020-01-03,10.5,11,10,10.8,,10.7,21.6,2020-01-04
"""


def test_normalize_types():
    df = sharadar.read_csv(SEP_CSV, "SEP")
    assert df.schema["date"] == pl.Date
    assert df.schema["close"] == pl.Float64
    assert df["volume"].to_list() == [1000.0, None]  # empty string -> null


def test_normalize_rejects_unexpected_text_column():
    bad = b"ticker,date,close,note\nAAA,2020-01-02,1.0,hello\n"
    with pytest.raises(pl.exceptions.InvalidOperationError):
        sharadar.read_csv(bad, "SEP")


def test_load_zip_export(tmp_path):
    path = tmp_path / "SHARADAR_SEP_2020.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("SHARADAR_SEP.csv", SEP_CSV)
    assert sharadar.load_export(path, "SEP").height == 2


class FakeResponse:
    def __init__(self, payload=None, content=b""):
        self._payload, self.content = payload, content

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


def _status(status, link=None):
    return FakeResponse({"datatable_bulk_download": {"file": {"status": status, "link": link}}})


def test_export_link_polls_until_fresh():
    session = FakeSession([_status("creating"), _status("fresh", "https://example/file.zip")])
    assert sharadar.export_link("SEP", "k", session, poll_seconds=0) == "https://example/file.zip"
    assert session.calls[0][1]["qopts.export"] == "true"


def test_download_table():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", SEP_CSV)
    session = FakeSession([_status("fresh", "L"), FakeResponse(content=buf.getvalue())])
    assert sharadar.download_table("SEP", key="k", session=session).height == 2


def test_fred_first_release_and_fallback():
    cpi = MacroSeries("CPIAUCSL", vintage=True, release_lag_days=50)
    df = fred.parse_observations(cpi_payload(), cpi)
    real = df.filter(pl.col("observation_date") == dt.date(2019, 1, 1)).row(0, named=True)
    assert real["first_release"] and real["available_date"] == dt.date(2019, 2, 15)
    # 2017 observations only have a 2018 'vintage': fall back to observation date + lag
    stale = df.filter(pl.col("observation_date") == dt.date(2017, 1, 1)).row(0, named=True)
    assert not stale["first_release"] and stale["available_date"] == dt.date(2017, 2, 20)


def test_fred_market_series_and_missing_values():
    s = MacroSeries("DGS10", vintage=False, release_lag_days=1)
    payload = {"observations": [
        {"realtime_start": "2026-01-01", "realtime_end": "9999-12-31", "date": "2020-01-02", "value": "1.88"},
        {"realtime_start": "2026-01-01", "realtime_end": "9999-12-31", "date": "2020-01-03", "value": "."},
    ]}
    df = fred.parse_observations(payload, s)
    assert df.height == 1
    assert df["available_date"][0] == dt.date(2020, 1, 3)

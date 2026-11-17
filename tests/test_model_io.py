"""models/io.py: pickle + JSON sidecar round trip."""

import datetime as dt
import json
import pickle

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge

from ltforecast.models.io import load_model, save_model, sidecar_path


class Linear:
    name = "linear"

    def __init__(self, cols):
        self.cols = cols
        self.model = Ridge(alpha=1.0)

    def fit(self, train, valid):
        self.model.fit(train.select(self.cols).to_numpy(), train["target"].to_numpy())
        return self

    def predict(self, df):
        return self.model.predict(df.select(self.cols).to_numpy())


def frame(seed=0, n=200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    return pl.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "target": x @ [0.5, -0.2, 0.1] + rng.normal(size=n)})


META = {"trained_through": dt.date(2022, 10, 31), "columns": ["a", "b", "c"], "params": {"alpha": 1.0}}


def test_round_trip_preserves_predictions_exactly(tmp_path):
    m = Linear(["a", "b", "c"]).fit(frame(), None)
    path = tmp_path / "sub" / "ridge.pkl"
    side = save_model(m, path, META)
    obj, meta = load_model(path)
    test = frame(seed=1)
    assert np.array_equal(obj.predict(test), m.predict(test))
    assert meta == side
    assert meta["trained_through"] == "2022-10-31" and meta["columns"] == ["a", "b", "c"]
    assert meta["params"] == {"alpha": 1.0}
    assert isinstance(meta["git_commit"], str) and meta["git_commit"]
    created = dt.datetime.fromisoformat(meta["created_utc"])
    assert created.utcoffset() == dt.timedelta(0)
    assert sidecar_path(path) == tmp_path / "sub" / "ridge.pkl.json"
    assert json.loads(sidecar_path(path).read_text()) == meta


def test_missing_sidecar_raises(tmp_path):
    path = tmp_path / "m.pkl"
    path.write_bytes(pickle.dumps({"x": 1}))
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_model(path)
    save_model({"x": 1}, path, META)
    sidecar_path(path).unlink()
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_model(path)


def test_tampered_pickle_raises(tmp_path):
    path = tmp_path / "m.pkl"
    save_model({"x": 1}, path, META)
    path.write_bytes(pickle.dumps({"x": 2}))
    with pytest.raises(ValueError, match="sha256"):
        load_model(path)


def test_reserved_meta_keys_are_refused(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        save_model({}, tmp_path / "m.pkl", {"git_commit": "abc"})
    assert not (tmp_path / "m.pkl").exists()


def test_overwrite_replaces_both_files(tmp_path):
    path = tmp_path / "m.pkl"
    save_model({"v": 1}, path, META)
    save_model({"v": 2}, path, META | {"params": {"alpha": 2.0}})
    obj, meta = load_model(path)
    assert obj == {"v": 2} and meta["params"] == {"alpha": 2.0}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.pkl", "m.pkl.json"]

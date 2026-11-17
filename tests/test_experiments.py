"""Run log and trial counter (plan 4.3): append-only CSV, stable config hash, mean-across-folds selection."""

import datetime as dt
import hashlib
import json
import os
import subprocess

import numpy as np
import polars as pl
import pytest

from ltforecast.config import REPO_ROOT
from ltforecast.validation import experiments as ex

HEADER = "timestamp_utc,name,git_commit,git_dirty,code_hash,config_hash,n_trials,config_json,metrics_json"
LEGACY_HEADER = "timestamp_utc,name,git_commit,git_dirty,config_hash,n_trials,config_json,metrics_json"
SHA = "a" * 40
CODE = "c" * 64


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """A log directory that does not exist yet, with the git lookup pinned to a clean fake commit."""
    monkeypatch.setattr(ex, "git_state", lambda log_path=None: (SHA, False))
    monkeypatch.setattr(ex, "code_hash", lambda log_path=None, root=None: CODE)
    return tmp_path / "nested" / "experiments"


def fake_git(monkeypatch, *, commit=None, status=None, fail=None):
    """Replace subprocess.run inside the module; returns the list of git argument lists it was asked for."""
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["cwd"] == REPO_ROOT
        if fail is not None:
            raise fail
        out = commit if cmd[1] == "rev-parse" else status
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(ex.subprocess, "run", run)
    return calls


def test_two_runs_append_two_rows(log_dir):
    first = ex.log_run(log_dir, "ridge", {"alpha": 1.0}, {"mean_ic": 0.03})
    second = ex.log_run(log_dir, "lgbm", {"leaves": 15, "lr": 0.05}, {"mean_ic": 0.05}, n_trials=20)

    lines = (log_dir / "runs.csv").read_text().splitlines()
    assert lines[0] == HEADER
    assert len(lines) == 3                      # the header is written once, by the first run only

    runs = ex.read_runs(log_dir)
    assert runs.columns == HEADER.split(",")
    assert runs["name"].to_list() == ["ridge", "lgbm"]
    assert runs["git_commit"].to_list() == [SHA, SHA]
    assert runs["git_dirty"].to_list() == [False, False]
    assert runs["code_hash"].to_list() == [CODE, CODE]
    assert runs["n_trials"].to_list() == [1, 20]
    assert runs["config_json"].to_list() == ['{"alpha": 1.0}', '{"leaves": 15, "lr": 0.05}']
    assert runs["metrics_json"].to_list() == ['{"mean_ic": 0.03}', '{"mean_ic": 0.05}']
    assert runs.to_dicts() == [first, second]   # the returned row is exactly what was logged

    stamps = [dt.datetime.fromisoformat(t) for t in runs["timestamp_utc"]]
    assert all(t.utcoffset() == dt.timedelta(0) for t in stamps)
    assert stamps[0] <= stamps[1]
    assert abs(dt.datetime.now(dt.timezone.utc) - stamps[1]) < dt.timedelta(minutes=1)


def test_trial_count_sums_n_trials(log_dir):
    ex.log_run(log_dir, "b2", {}, {})                         # default: one trial
    ex.log_run(log_dir, "grid", {"k": 1}, {}, n_trials=20)
    ex.log_run(log_dir, "re-report", {"k": 1}, {}, n_trials=0)
    ex.log_run(log_dir, "exits", {"pairs": 25}, {}, n_trials=np.int64(25))
    assert ex.trial_count(log_dir) == 1 + 20 + 0 + 25
    assert isinstance(ex.trial_count(log_dir), int)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_bad_n_trials_is_rejected_before_writing(log_dir, bad):
    with pytest.raises((ValueError, TypeError)):
        ex.log_run(log_dir, "x", {}, {}, n_trials=bad)
    assert not (log_dir / "runs.csv").exists()


def test_config_hash_ignores_key_order(log_dir):
    a = {"a": 1, "b": {"x": 2, "y": [3, 4]}}
    b = {"b": {"y": [3, 4], "x": 2}, "a": 1}
    canonical = '{"a": 1, "b": {"x": 2, "y": [3, 4]}}'
    assert hashlib.sha1(canonical.encode()).hexdigest() == "0bfc57cb897d0f3bf0bd65246240714122169b83"

    assert ex.canonical_json(a) == ex.canonical_json(b) == canonical
    assert ex.config_hash(a) == ex.config_hash(b) == "0bfc57cb897d0f3bf0bd65246240714122169b83"
    assert ex.config_hash({"a": 1, "b": {"x": 2, "y": [4, 3]}}) != ex.config_hash(a)   # list order is content
    assert ex.config_hash({"a": 2, "b": {"x": 2, "y": [3, 4]}}) != ex.config_hash(a)

    rows = [ex.log_run(log_dir, "a", a, {}), ex.log_run(log_dir, "b", b, {})]
    assert [r["config_hash"] for r in rows] == ["0bfc57cb897d0f3bf0bd65246240714122169b83"] * 2
    assert ex.read_runs(log_dir)["config_json"].to_list() == [canonical, canonical]


def test_config_with_dates_paths_and_tuples_hashes_and_round_trips(log_dir, tmp_path):
    config = {"dev_end": dt.date(2021, 10, 29), "dir": tmp_path, "horizons": (63, 126, 252), "seed": None}
    row = ex.log_run(log_dir, "cfg", config, {})
    stored = ex.read_runs(log_dir)["config_json"][0]
    assert json.loads(stored) == {
        "dev_end": "2021-10-29", "dir": str(tmp_path), "horizons": [63, 126, 252], "seed": None,
    }
    assert row["config_hash"] == hashlib.sha1(stored.encode()).hexdigest()


def test_metrics_with_nan_none_dates_and_numpy_scalars_serialize(log_dir):
    metrics = {
        "mean_ic": np.float64(0.05),
        "nw_tstat": float("nan"),
        "worst": -np.inf,
        "ic32": np.float32(0.25),
        "n_months": np.int64(120),
        "passed": np.bool_(True),
        "skipped": None,
        "first_test": dt.date(2010, 1, 29),
        "fitted_at": dt.datetime(2026, 1, 2, 3, 4, 5),
        "last_obs": np.datetime64("2021-10-29"),
        "by_fold": {2010: np.float64(0.02), 2011: float("nan")},
        "spreads": np.array([0.1, np.nan, -0.2]),
        "folds": ("2010", "2011"),
        "deciles": pl.DataFrame({"decile": [1, 2], "mean_ret": [-0.01, None]}),
        "ic_series": pl.Series([0.5, None]),
    }
    ex.log_run(log_dir, "m", {}, metrics)
    text = ex.read_runs(log_dir)["metrics_json"][0]

    def strict(token):
        raise AssertionError(f"non-standard JSON token {token}")

    assert json.loads(text, parse_constant=strict) == {
        "mean_ic": 0.05,
        "nw_tstat": None,
        "worst": None,
        "ic32": 0.25,
        "n_months": 120,
        "passed": True,
        "skipped": None,
        "first_test": "2010-01-29",
        "fitted_at": "2026-01-02T03:04:05",
        "last_obs": "2021-10-29",
        "by_fold": {"2010": 0.02, "2011": None},
        "spreads": [0.1, None, -0.2],
        "folds": ["2010", "2011"],
        "deciles": [{"decile": 1, "mean_ret": -0.01}, {"decile": 2, "mean_ret": None}],
        "ic_series": [0.5, None],
    }


def test_awkward_text_survives_the_csv_round_trip(log_dir):
    nasty = 'top "decile", net\nof costs \\ é\tx'
    ex.log_run(log_dir, nasty, {"note": nasty}, {"note": nasty, "k": [1, 2]})
    ex.log_run(log_dir, "", {}, {})
    ex.log_run(log_dir, "plain", {"a": 1}, {"b": 2}, n_trials=3)

    runs = ex.read_runs(log_dir)
    assert runs["name"].to_list() == [nasty, "", "plain"]
    assert json.loads(runs["config_json"][0]) == {"note": nasty}
    assert json.loads(runs["metrics_json"][0]) == {"note": nasty, "k": [1, 2]}
    assert runs["config_json"].to_list()[1:] == ["{}", '{"a": 1}']
    assert ex.trial_count(log_dir) == 5


def test_existing_log_is_appended_to_not_truncated(log_dir):
    log_dir.mkdir(parents=True)
    old = HEADER + '\n2020-01-01T00:00:00.000000+00:00,old,' + "b" * 40 + ',true,' + "d" * 64 + ',deadbeef,7,"{""a"": 1}",{}\n'
    (log_dir / "runs.csv").write_text(old)

    ex.log_run(log_dir, "new", {"a": 2}, {"ic": 0.1}, n_trials=2)

    text = (log_dir / "runs.csv").read_text()
    assert text.startswith(old)                 # the old bytes are untouched
    assert text.count(HEADER) == 1              # no second header
    runs = ex.read_runs(log_dir)
    assert runs["name"].to_list() == ["old", "new"]
    assert runs["git_dirty"].to_list() == [True, False]
    assert runs["config_json"].to_list() == ['{"a": 1}', '{"a": 2}']
    assert ex.trial_count(log_dir) == 7 + 2


def test_existing_log_without_trailing_newline_gets_a_new_line(log_dir):
    log_dir.mkdir(parents=True)
    old = HEADER + "\n2020-01-01T00:00:00.000000+00:00,old," + "b" * 40 + ",false," + "d" * 64 + ",deadbeef,7,{},{}"
    (log_dir / "runs.csv").write_text(old)
    ex.log_run(log_dir, "new", {}, {})
    assert (log_dir / "runs.csv").read_text().startswith(old + "\n")
    assert ex.read_runs(log_dir)["name"].to_list() == ["old", "new"]
    assert ex.trial_count(log_dir) == 8


def test_a_file_that_is_not_a_run_log_is_left_alone(log_dir):
    log_dir.mkdir(parents=True)
    foreign = "date,value\n2020-01-01,1\n"
    (log_dir / "runs.csv").write_text(foreign)
    with pytest.raises(ValueError, match="not a run log"):
        ex.log_run(log_dir, "x", {}, {})
    with pytest.raises(ValueError, match="not a run log"):
        ex.read_runs(log_dir)
    assert (log_dir / "runs.csv").read_text() == foreign


def test_missing_or_empty_log_reads_as_an_empty_frame(log_dir):
    expected = {c: pl.String for c in HEADER.split(",")} | {"git_dirty": pl.Boolean, "n_trials": pl.Int64}

    missing = ex.read_runs(log_dir)             # the directory itself does not exist
    assert missing.height == 0 and dict(missing.schema) == expected
    assert ex.trial_count(log_dir) == 0
    assert not log_dir.exists()                 # reading never creates anything

    log_dir.mkdir(parents=True)
    (log_dir / "runs.csv").write_text("")       # zero bytes
    assert ex.read_runs(log_dir).height == 0 and dict(ex.read_runs(log_dir).schema) == expected
    assert ex.trial_count(log_dir) == 0

    (log_dir / "runs.csv").write_text(HEADER + "\n")   # header only
    assert ex.read_runs(log_dir).height == 0 and dict(ex.read_runs(log_dir).schema) == expected
    assert ex.trial_count(log_dir) == 0

    ex.log_run(log_dir, "first", {}, {}, n_trials=4)
    assert dict(ex.read_runs(log_dir).schema) == expected   # same schema with and without rows
    assert ex.trial_count(log_dir) == 4


def test_zero_byte_log_gets_its_header_on_first_run(log_dir):
    log_dir.mkdir(parents=True)
    (log_dir / "runs.csv").write_text("")
    ex.log_run(log_dir, "first", {}, {})
    assert (log_dir / "runs.csv").read_text().splitlines()[0] == HEADER
    assert ex.read_runs(log_dir)["name"].to_list() == ["first"]


def test_git_unavailable_is_recorded_as_unknown(tmp_path, monkeypatch):
    for failure in (
        FileNotFoundError("git"),                               # git is not installed
        subprocess.CalledProcessError(128, ["git"]),            # not a repository
        subprocess.TimeoutExpired(["git"], 30),
    ):
        fake_git(monkeypatch, fail=failure)
        assert ex.git_state() == ("unknown", None)

    row = ex.log_run(tmp_path, "no-git", {"a": 1}, {"ic": 0.1})
    assert (row["git_commit"], row["git_dirty"]) == ("unknown", None)
    line = (tmp_path / "runs.csv").read_text().splitlines()[1]
    assert ",no-git,unknown,unknown," in line
    assert len(row["code_hash"]) == 64             # still hashed, from the files on disk
    runs = ex.read_runs(tmp_path)
    assert runs["git_commit"].to_list() == ["unknown"]
    assert runs["git_dirty"].to_list() == [None]
    assert ex.trial_count(tmp_path) == 1


def test_git_commit_and_dirty_flag(monkeypatch):
    fake_git(monkeypatch, commit=SHA + "\n", status="")
    assert ex.git_state() == (SHA, False)
    fake_git(monkeypatch, commit=SHA + "\n", status=" M src/ltforecast/labels.py\n?? scratch.py\n")
    assert ex.git_state() == (SHA, True)
    fake_git(monkeypatch, commit="", status="")                 # a repository with no commits yet
    assert ex.git_state() == ("unknown", False)


def test_the_run_log_itself_does_not_make_the_tree_dirty(tmp_path, monkeypatch):
    calls = fake_git(monkeypatch, commit=SHA + "\n", status="")
    ex.git_state(REPO_ROOT / "experiments" / "runs.csv")
    assert calls[0] == ["git", "rev-parse", "HEAD"]
    assert calls[1] == [
        "git", "status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude,literal)experiments/runs.csv",
    ]
    # a log outside the repository needs no exclusion
    calls = fake_git(monkeypatch, commit=SHA + "\n", status="")
    ex.git_state(tmp_path / "runs.csv")
    assert calls[1] == ["git", "status", "--porcelain", "--untracked-files=all"]


def test_real_git_lookup_never_raises(tmp_path):
    commit, dirty = ex.git_state(tmp_path / "runs.csv")
    assert commit == "unknown" or (len(commit) == 40 and set(commit) <= set("0123456789abcdef"))
    assert dirty in (True, False, None)


def code_tree(root):
    """A small source tree (no git repository) for the content hash."""
    (root / "src" / "ltforecast" / "__pycache__").mkdir(parents=True)
    (root / "config").mkdir()
    (root / "src" / "ltforecast" / "value.py").write_text("x = 1\n")
    (root / "src" / "ltforecast" / "__pycache__" / "value.cpython-314.pyc").write_bytes(b"\x00junk")
    (root / "config" / "model.yaml").write_text("alpha: 1\n")
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "notes.txt").write_text("not code\n")
    return root


def expected_hash(root, rels):
    h = hashlib.sha256()
    for rel in sorted(rels):
        data = (root / rel).read_bytes()
        h.update(rel.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()


def test_code_hash_is_a_content_hash_of_the_source_tree(tmp_path):
    root = code_tree(tmp_path / "repo")
    code = ["config/model.yaml", "pyproject.toml", "src/ltforecast/value.py"]
    before = ex.code_hash(root=root)
    assert before == expected_hash(root, code)              # bytecode and files outside src/config ignored
    (root / "notes.txt").write_text("changed\n")
    (root / "src" / "ltforecast" / "__pycache__" / "value.cpython-314.pyc").write_bytes(b"other")
    assert ex.code_hash(root=root) == before
    # an uncommitted edit to a source file changes the hash although HEAD would not change
    (root / "src" / "ltforecast" / "value.py").write_text("x = 2\n")
    assert ex.code_hash(root=root) != before
    (root / "src" / "ltforecast" / "value.py").write_text("x = 1\n")
    assert ex.code_hash(root=root) == before                # same tree, same hash
    (root / "src" / "ltforecast" / "new.py").write_text("")  # a new untracked file counts too
    assert ex.code_hash(root=root) != before


def test_code_hash_uses_git_file_list_and_skips_the_run_log(tmp_path, monkeypatch):
    root = code_tree(tmp_path / "repo")
    (root / "config" / "runs.csv").write_text("log\n")
    calls = []

    def git(*args, cwd=None):
        calls.append((args, cwd))
        # git leaves ignored files out; it may list a tracked file that was deleted from disk
        return "src/ltforecast/value.py\0config/model.yaml\0config/runs.csv\0src/gone.py\0"

    monkeypatch.setattr(ex, "_git", git)
    got = ex.code_hash(root / "config" / "runs.csv", root=root)
    assert calls == [(("ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *ex.CODE_PATHS), root)]
    assert got == expected_hash(root, ["config/model.yaml", "src/ltforecast/value.py"])


def test_two_runs_on_different_uncommitted_code_get_different_code_hashes(tmp_path, monkeypatch):
    root = code_tree(tmp_path / "repo")
    monkeypatch.setattr(ex, "git_state", lambda log_path=None: (SHA, True))
    monkeypatch.setattr(ex, "REPO_ROOT", root)
    a = ex.log_run(tmp_path / "log", "a", {"k": 1}, {})
    (root / "src" / "ltforecast" / "value.py").write_text("x = 3\n")
    b = ex.log_run(tmp_path / "log", "b", {"k": 1}, {})
    runs = ex.read_runs(tmp_path / "log")
    assert runs["git_commit"].to_list() == [SHA, SHA]
    assert runs["code_hash"].to_list() == [a["code_hash"], b["code_hash"]] and a["code_hash"] != b["code_hash"]


def test_real_code_hash_is_stable(tmp_path):
    first = ex.code_hash(tmp_path / "runs.csv")
    assert len(first) == 64 and set(first) <= set("0123456789abcdef")
    assert ex.code_hash(tmp_path / "runs.csv") == first


def test_legacy_log_reads_with_unknown_code_hash_and_is_migrated_on_append(log_dir):
    log_dir.mkdir(parents=True)
    legacy = (
        LEGACY_HEADER + '\n2020-01-01T00:00:00.000000+00:00,old,' + "b" * 40 + ',true,deadbeef,7,"{""a"": 1}",{}\n'
        + "2020-01-02T00:00:00.000000+00:00,older," + "b" * 40 + ",false,feedface,3,{},{}"
    )
    (log_dir / "runs.csv").write_text(legacy)
    runs = ex.read_runs(log_dir)
    assert runs.columns == HEADER.split(",")
    assert runs["code_hash"].to_list() == ["unknown", "unknown"]
    assert ex.trial_count(log_dir) == 10
    assert (log_dir / "runs.csv").read_text() == legacy     # reading never rewrites

    ex.log_run(log_dir, "new", {"a": 2}, {}, n_trials=2)
    lines = (log_dir / "runs.csv").read_text().splitlines()
    assert lines[0] == HEADER and len(lines) == 4
    runs = ex.read_runs(log_dir)
    assert runs["name"].to_list() == ["old", "older", "new"]
    assert runs["code_hash"].to_list() == ["unknown", "unknown", CODE]
    assert runs["config_hash"].to_list()[:2] == ["deadbeef", "feedface"]
    assert runs["git_dirty"].to_list() == [True, False, False]
    assert ex.trial_count(log_dir) == 7 + 3 + 2             # trial history survives the migration
    assert not [f for f in os.listdir(log_dir) if f != "runs.csv"]   # no temp file left behind


def test_selection_guard_returns_the_mean_across_folds():
    assert ex.selection_guard({"2010": 0.02, "2011": 0.04, "2012": 0.06}) == pytest.approx(0.04)
    assert ex.selection_guard({"2010": 0.10, "2011": -0.05, "2012": 0.01, "2013": 0.02}) == pytest.approx(0.02)

    # one spectacular year does not win: A's best year beats every year of B, B has the higher mean
    a = {"2010": 0.10, "2011": -0.02, "2012": -0.02}   # mean 0.02
    b = {"2010": 0.03, "2011": 0.03, "2012": 0.03}     # mean 0.03
    scores = {"a": ex.selection_guard(a), "b": ex.selection_guard(b)}
    assert scores == {"a": pytest.approx(0.02), "b": pytest.approx(0.03)}
    assert max(scores, key=scores.get) == "b"


def test_selection_guard_refuses_too_few_or_broken_folds():
    for folds in ({}, {"2010": 0.5}, {"2010": 0.5, "2011": 0.4}):
        with pytest.raises(ValueError, match="at least 3"):
            ex.selection_guard(folds)
    with pytest.raises(ValueError, match="2011"):
        ex.selection_guard({"2010": 0.02, "2011": float("nan"), "2012": 0.06})
    with pytest.raises(ValueError, match="2012"):
        ex.selection_guard({"2010": 0.02, "2011": 0.04, "2012": None})
    with pytest.raises(ValueError, match="2010"):
        ex.selection_guard({"2010": float("inf"), "2011": 0.04, "2012": 0.06})


def test_log_new_counts_a_rerun_of_the_same_config_once(log_dir):
    ex.log_new(log_dir, "baseline/b2", {"a": 1}, {"ic": 0.01}, n_trials=1)
    ex.log_new(log_dir, "baseline/b2", {"a": 1}, {"ic": 0.01}, n_trials=1)   # same name, config, code
    ex.log_new(log_dir, "baseline/b2", {"a": 2}, {"ic": 0.02}, n_trials=1)   # new config: a new trial
    ex.log_new(log_dir, "baseline/b3", {"a": 1}, {"ic": 0.03}, n_trials=1)   # other name: a new trial
    runs = ex.read_runs(log_dir)
    assert runs.height == 4 and runs["n_trials"].to_list() == [1, 0, 1, 1]
    assert ex.trial_count(log_dir) == 3

"""Append-only experiment log and trial counter (plan 4.3).

Every run appends one row to <dir>/runs.csv; existing rows are never rewritten. The trial count
feeds the deflated-Sharpe haircut (validation.metrics.deflated_sharpe), so every config scored on
the walk-forward test folds must be logged, including the ones that lost.

Columns: timestamp_utc (ISO 8601, +00:00), name, git_commit (HEAD sha | "unknown"), git_dirty
(true | false | "unknown" in the file; Boolean with null = unknown in read_runs), code_hash (sha256
of the code that actually ran, see `code_hash`; "unknown" for rows logged before the column existed),
config_hash (sha1 of config_json), n_trials, config_json, metrics_json.

A log written with the older header (no code_hash) is read with code_hash = "unknown", and is
rewritten once with that value filled in the first time a new run is appended to it; every other
field, n_trials included, is kept byte for byte.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ltforecast.config import REPO_ROOT

RUNS_FILE = "runs.csv"
COLUMNS = (
    "timestamp_utc", "name", "git_commit", "git_dirty", "code_hash", "config_hash", "n_trials", "config_json",
    "metrics_json",
)
LEGACY_COLUMNS = tuple(c for c in COLUMNS if c != "code_hash")
CODE_PATHS = ("src", "config", "pyproject.toml", "uv.lock")
UNKNOWN = "unknown"
MIN_SELECTION_FOLDS = 3

_DIRTY_TEXT = {True: "true", False: "false", None: UNKNOWN}
_SCHEMA = {c: pl.String for c in COLUMNS} | {"git_dirty": pl.Boolean, "n_trials": pl.Int64}


def canonical_json(config: dict) -> str:
    """Sorted-key JSON: the text that is hashed and stored as config_json."""
    return json.dumps(config, sort_keys=True, default=str)


def config_hash(config: dict) -> str:
    """sha1 of the canonical JSON, so key order never changes the hash."""
    return hashlib.sha1(canonical_json(config).encode()).hexdigest()


def _jsonable(obj: Any) -> Any:
    """Strict-JSON types only: NaN/inf -> None, numpy scalars -> Python, dates -> ISO strings,
    arrays / Series -> lists, DataFrames -> lists of row dicts, anything else -> str."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, pl.DataFrame):
        return _jsonable(obj.to_dicts())
    if isinstance(obj, pl.Series):
        return _jsonable(obj.to_list())
    if isinstance(obj, (dt.date, dt.time)):
        return obj.isoformat()
    return str(obj)


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """stdout of a git command run in `cwd` (the repo root by default); None if git or the repository
    is unavailable."""
    try:
        done = subprocess.run(
            ["git", *args], cwd=REPO_ROOT if cwd is None else cwd, capture_output=True, text=True, check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def git_state(log_path: Path | None = None) -> tuple[str, bool | None]:
    """(HEAD commit, dirty flag) of the repo; ("unknown", None) when git cannot answer.

    Dirty = any modified or untracked file other than `log_path`: the run log changes on every
    run, so counting it would mark every run after the first as dirty.
    """
    commit = _git("rev-parse", "HEAD")
    args = ["status", "--porcelain", "--untracked-files=all"]
    if log_path is not None and log_path.resolve().is_relative_to(REPO_ROOT):
        rel = log_path.resolve().relative_to(REPO_ROOT).as_posix()
        args += ["--", ".", f":(exclude,literal){rel}"]
    status = _git(*args)
    return (commit.strip() or UNKNOWN) if commit else UNKNOWN, None if status is None else bool(status.strip())


def _is_code_file(rel: str) -> bool:
    parts = rel.split("/")
    return not (
        rel.endswith(".pyc") or "__pycache__" in parts or parts[-1] == ".DS_Store"
        or any(part.endswith(".egg-info") for part in parts)
    )


def code_hash(log_path: Path | None = None, root: Path | None = None) -> str:
    """sha256 over the code that would run now: the sorted (relative path, bytes) of every file under
    CODE_PATHS (src/, config/, pyproject.toml, uv.lock) in `root` (the repo root by default).

    The file list comes from `git ls-files --cached --others --exclude-standard` (tracked plus
    untracked, minus ignored), or from a directory walk when git cannot answer; bytecode, egg-info and
    `log_path` are skipped either way. Unlike the HEAD commit it changes with every uncommitted edit
    or new file, and two identical trees get the same hash whatever commit they sit on.
    """
    root = REPO_ROOT if root is None else Path(root)
    listed = _git("ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *CODE_PATHS, cwd=root)
    if listed is not None:
        rels = [r for r in listed.split("\0") if r]
    else:
        rels = []
        for name in CODE_PATHS:
            top = root / name
            if top.is_file():
                rels.append(name)
            elif top.is_dir():
                rels += [f.relative_to(root).as_posix() for f in top.rglob("*") if f.is_file()]
    skip = None
    if log_path is not None and Path(log_path).resolve().is_relative_to(root.resolve()):
        skip = Path(log_path).resolve().relative_to(root.resolve()).as_posix()
    h = hashlib.sha256()
    for rel in sorted({r for r in rels if r != skip and _is_code_file(r)}):
        f = root / rel
        if not f.is_file():                      # tracked but deleted in the working tree
            continue
        data = f.read_bytes()
        h.update(rel.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()


def _migrate_legacy(path: Path) -> None:
    """Rewrite a log with the pre-code_hash header to the current one, code_hash = "unknown" on every
    old row and all other fields unchanged. Written to a temp file and swapped in atomically."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r]
    at = COLUMNS.index("code_hash")
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(r[:at] + [UNKNOWN] + r[at:] for r in rows[1:])
    tmp = path.with_name(path.name + ".migrating")
    tmp.write_text(buf.getvalue(), encoding="utf-8", newline="")
    os.replace(tmp, path)


def _append(path: Path, row: dict) -> None:
    """Append one CSV line (plus the header on first use) in a single write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as f:
            header = f.readline().decode("utf-8").strip()
        if header == ",".join(LEGACY_COLUMNS):
            _migrate_legacy(path)
        with path.open("rb") as f:
            header = f.readline().decode("utf-8").strip()
            f.seek(-1, 2)
            ends_with_newline = f.read(1) == b"\n"
        if header != ",".join(COLUMNS):
            raise ValueError(f"{path} is not a run log (header {header!r}); refusing to append")
        if not ends_with_newline:
            buf.write("\n")
    else:
        writer.writerow(COLUMNS)
    writer.writerow([_DIRTY_TEXT[row[c]] if c == "git_dirty" else row[c] for c in COLUMNS])
    with path.open("a", newline="", encoding="utf-8") as f:
        f.write(buf.getvalue())


def log_run(dir: str | Path, name: str, config: dict, metrics: dict, n_trials: int = 1) -> dict:
    """Append one run to <dir>/runs.csv and return the row (keys = COLUMNS, typed as in read_runs).

    n_trials = how many configurations this run scored on walk-forward test folds (a grid of 20
    logs 20; a re-report of an already-counted config logs 0).
    """
    if isinstance(n_trials, bool) or int(n_trials) != n_trials or n_trials < 0:
        raise ValueError(f"n_trials must be a non-negative integer, got {n_trials!r}")
    path = Path(dir) / RUNS_FILE
    commit, dirty = git_state(path)
    config_json = canonical_json(config)
    row = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds"),
        "name": str(name),
        "git_commit": commit,
        "git_dirty": dirty,
        "code_hash": code_hash(path),
        "config_hash": hashlib.sha1(config_json.encode()).hexdigest(),
        "n_trials": int(n_trials),
        "config_json": config_json,
        "metrics_json": json.dumps(_jsonable(metrics), sort_keys=True, allow_nan=False),
    }
    _append(path, row)
    return row


def log_new(dir: str | Path, name: str, config: dict, metrics: dict, n_trials: int = 1) -> dict:
    """`log_run`, except that a re-run of an already-counted configuration logs n_trials = 0.

    "Already counted" = a row with the same name, config_hash and code_hash exists: the same config on
    the same code is no new chance to select on noise (rerunning `train`, or `baselines` then `train`).
    Changed code (code_hash) counts again, since the result can differ. The run is always appended."""
    path = Path(dir) / RUNS_FILE
    runs = read_runs(dir)
    if runs.height:
        seen = runs.filter(
            (pl.col("name") == str(name))
            & (pl.col("config_hash") == config_hash(config))
            & (pl.col("code_hash") == code_hash(path))
        )
        if seen.height:
            n_trials = 0
    return log_run(dir, name, config, metrics, n_trials=n_trials)


def read_runs(dir: str | Path) -> pl.DataFrame:
    """The run log in file order; an empty frame with the same schema if there is no log yet.

    config_json / metrics_json stay JSON text (json.loads them); git_dirty is null when unknown;
    code_hash is "unknown" for rows of a log written before that column existed.
    """
    path = Path(dir) / RUNS_FILE
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame(schema=_SCHEMA)
    runs = pl.read_csv(path, infer_schema=False, glob=False)
    if tuple(runs.columns) == LEGACY_COLUMNS:
        runs = runs.with_columns(pl.lit(UNKNOWN).alias("code_hash")).select(COLUMNS)
    if tuple(runs.columns) != COLUMNS:
        raise ValueError(f"{path} is not a run log (columns {runs.columns})")
    # every field is written non-empty except a blank name, which the reader returns as null
    return runs.with_columns(
        pl.col("name").fill_null(""),
        pl.col("git_dirty").replace_strict({"true": True, "false": False}, default=None, return_dtype=pl.Boolean),
        pl.col("n_trials").cast(pl.Int64),
    )


def trial_count(dir: str | Path) -> int:
    """Total configurations tried so far (sum of n_trials); 0 if nothing has been logged."""
    return int(read_runs(dir)["n_trials"].sum())


def selection_guard(per_fold_metric: dict[str, float]) -> float:
    """The mean of a metric across walk-forward folds: the only sanctioned score for picking a config.

    Plan 4.3: a config is never selected on a single test year. Raises ValueError for fewer than
    MIN_SELECTION_FOLDS folds or a missing / non-finite fold value (a fold that failed to evaluate
    must not silently drop out of the mean).
    """
    if len(per_fold_metric) < MIN_SELECTION_FOLDS:
        raise ValueError(
            f"selection needs at least {MIN_SELECTION_FOLDS} walk-forward folds, got {len(per_fold_metric)}"
        )
    bad = [fold for fold, v in per_fold_metric.items() if v is None or not math.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite metric in folds {bad}; fix the evaluation before selecting")
    return math.fsum(per_fold_metric.values()) / len(per_fold_metric)

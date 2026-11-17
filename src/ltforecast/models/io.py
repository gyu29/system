"""Model persistence: a pickle plus a JSON sidecar describing what was trained and from which code.

`save_model(obj, "m.pkl", meta)` writes `m.pkl` and `m.pkl.json`. The sidecar holds the caller's
`meta` (conventionally trained_through, columns, params) plus git_commit, git_dirty, created_utc
(ISO 8601, +00:00) and the sha256 of the pickle. `load_model` refuses a pickle with no sidecar or
whose bytes do not match the recorded hash. Pickles execute code on load: only load files this
project wrote.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

from ltforecast.validation.experiments import git_state

RESERVED = ("git_commit", "git_dirty", "created_utc", "sha256")


def sidecar_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".json")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.asdict(obj)
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    return str(obj)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_model(obj: Any, path: str | Path, meta: dict) -> dict:
    """Pickle `obj` to `path` and write the sidecar; returns the sidecar dict (as JSON round-trips it)."""
    clash = sorted(set(RESERVED) & set(meta))
    if clash:
        raise ValueError(f"meta keys {clash} are reserved for the sidecar")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    commit, dirty = git_state()
    side = dict(meta) | {
        "git_commit": commit,
        "git_dirty": dirty,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }
    text = json.dumps(side, default=_json_default, indent=2, sort_keys=True)
    _atomic_write(path, blob)
    _atomic_write(sidecar_path(path), text.encode())
    return json.loads(text)


def load_model(path: str | Path) -> tuple[Any, dict]:
    """(obj, meta) from a pickle written by `save_model`. Raises FileNotFoundError without the sidecar
    and ValueError if the pickle's bytes differ from the hash recorded in it."""
    path = Path(path)
    side = sidecar_path(path)
    if not side.exists():
        raise FileNotFoundError(f"{path}: no metadata sidecar {side.name}; refusing to load an undocumented model")
    meta = json.loads(side.read_text())
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != meta.get("sha256"):
        raise ValueError(f"{path}: contents do not match the sha256 in {side.name}")
    return pickle.loads(blob), meta

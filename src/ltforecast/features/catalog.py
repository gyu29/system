"""The feature catalog: config/features.yaml loaded and validated (plan section 3).

The catalog is the single list of feature names. Family modules must produce exactly these raw
columns; transform, the models and the reports take their column lists from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ltforecast.config import REPO_ROOT

DEFAULT_CATALOG = REPO_ROOT / "config" / "features.yaml"

KINDS = ("stock", "context")  # stock: ranked across stocks on each date; context: one value per date, unranked
MIN_FEATURES, MAX_FEATURES = 30, 45  # plan 3.1: "roughly 35-45 features. Resist adding more."
SR_SUFFIX = "_sr"
HAS_PREFIX = "has_"

_FEATURE_KEYS = ("name", "family", "label", "definition", "source", "lookback", "lag", "citation", "sign", "kind")
_TEXT_KEYS = ("name", "family", "label", "definition", "lookback", "lag", "citation", "kind")


@dataclass(frozen=True)
class FamilySpec:
    name: str
    label: str
    sector_relative: bool = False  # also rank its stock features within sector (`<name>_sr`)


@dataclass(frozen=True)
class FeatureSpec:
    name: str        # the column name everywhere
    family: str
    label: str       # plain-English name for explanations and reports
    definition: str
    source: tuple[str, ...]
    lookback: str
    lag: str         # availability lag
    citation: str
    sign: int        # +1: high values historically precede high returns; -1: the opposite
    kind: str        # stock | context


@dataclass(frozen=True)
class Catalog:
    features: tuple[FeatureSpec, ...]
    families: dict[str, FamilySpec]  # in report order

    def names(self, kind: str | None = None, family: str | None = None) -> list[str]:
        """Feature names in catalog order, optionally only one kind and/or one family."""
        return [
            f.name
            for f in self.features
            if (kind is None or f.kind == kind) and (family is None or f.family == family)
        ]

    def spec(self, name: str) -> FeatureSpec:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(f"no feature {name!r} in the catalog")

    def stock_families(self) -> list[str]:
        """Families with at least one stock feature, in family order (each gets a `has_<family>` flag)."""
        used = {f.family for f in self.features if f.kind == "stock"}
        return [name for name in self.families if name in used]

    def sector_relative_names(self) -> list[str]:
        """`<name>_sr` for the stock features of families marked sector_relative."""
        return [
            f.name + SR_SUFFIX
            for f in self.features
            if f.kind == "stock" and self.families[f.family].sector_relative
        ]

    def flag_names(self) -> list[str]:
        return [HAS_PREFIX + family for family in self.stock_families()]

    def linear_columns(self) -> list[str]:
        """Inputs of the linear models: ranked stock features, then their sector-relative variants."""
        return self.names(kind="stock") + self.sector_relative_names()

    def model_columns(self) -> list[str]:
        """Every model input: linear_columns(), then the has_<family> flags, then the context features."""
        return self.linear_columns() + self.flag_names() + self.names(kind="context")


def _fail(msg: str):
    raise ValueError(f"feature catalog: {msg}")


def _family(name: str, raw: object) -> FamilySpec:
    if not isinstance(raw, dict) or not isinstance(raw.get("label"), str) or not raw["label"].strip():
        _fail(f"family {name!r} needs a label")
    unknown = set(raw) - {"label", "sector_relative"}
    if unknown:
        _fail(f"family {name!r} has unknown keys {sorted(unknown)}")
    relative = raw.get("sector_relative", False)
    if not isinstance(relative, bool):
        _fail(f"family {name!r}: sector_relative must be true or false")
    return FamilySpec(name=str(name), label=raw["label"], sector_relative=relative)


def _feature(raw: object, families: dict[str, FamilySpec]) -> FeatureSpec:
    if not isinstance(raw, dict):
        _fail(f"each feature must be a mapping, got {raw!r}")
    name = raw.get("name")
    missing, unknown = [k for k in _FEATURE_KEYS if k not in raw], sorted(set(raw) - set(_FEATURE_KEYS))
    if missing or unknown:
        _fail(f"feature {name!r}: missing keys {missing}, unknown keys {unknown}")
    for key in _TEXT_KEYS:
        if not isinstance(raw[key], str) or not raw[key].strip():
            _fail(f"feature {name!r}: {key} must be a non-empty string")
    if raw["family"] not in families:
        _fail(f"feature {name!r}: unknown family {raw['family']!r}")
    if raw["kind"] not in KINDS:
        _fail(f"feature {name!r}: kind must be one of {KINDS}")
    # `type(...) is int` so that YAML true/false (bools are ints in Python) is not read as a sign
    if type(raw["sign"]) is not int or raw["sign"] not in (-1, 1):
        _fail(f"feature {name!r}: sign must be +1 or -1, got {raw['sign']!r}")
    source = [raw["source"]] if isinstance(raw["source"], str) else raw["source"]
    if not isinstance(source, list) or not source or not all(isinstance(s, str) and s.strip() for s in source):
        _fail(f"feature {name!r}: source must be a non-empty list of field names")
    return FeatureSpec(**{k: raw[k] for k in _TEXT_KEYS}, source=tuple(source), sign=raw["sign"])


def parse_catalog(raw: object) -> Catalog:
    """Validate the parsed YAML: every key present, unique names, known families, sign in {-1, +1},
    MIN_FEATURES <= number of features <= MAX_FEATURES. Raises ValueError otherwise."""
    if not isinstance(raw, dict) or set(raw) != {"families", "features"}:
        _fail("expected exactly the top-level keys `families` and `features`")
    if not isinstance(raw["families"], dict) or not isinstance(raw["features"], list):
        _fail("`families` must be a mapping and `features` a list")
    families = {str(name): _family(name, fam) for name, fam in raw["families"].items()}
    features = tuple(_feature(f, families) for f in raw["features"])

    names = [f.name for f in features]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        _fail(f"duplicate feature names {dupes}")
    if not MIN_FEATURES <= len(features) <= MAX_FEATURES:
        _fail(f"{len(features)} features; the plan allows {MIN_FEATURES} to {MAX_FEATURES}")
    catalog = Catalog(features=features, families=families)
    columns = catalog.model_columns()
    clashes = sorted({c for c in columns if columns.count(c) > 1})
    if clashes:
        _fail(f"feature names collide with derived columns: {clashes}")
    return catalog


def load_catalog(path: Path | str = DEFAULT_CATALOG) -> Catalog:
    return parse_catalog(yaml.safe_load(Path(path).read_text()))

"""The feature catalog: config/features.yaml holds exactly the contracted feature set, and the
loader rejects malformed catalogs."""

import copy

import pytest
import yaml

from ltforecast.features.catalog import DEFAULT_CATALOG, Catalog, FeatureSpec, load_catalog, parse_catalog

# family -> feature names, in catalog order (build contract, section 3.1)
EXPECTED = {
    "value": ["book_to_price", "earnings_yield", "fcf_yield", "ebitda_to_ev", "sales_to_ev", "shareholder_yield"],
    "quality": ["gross_profitability", "roic", "op_margin_stability", "piotroski_f", "accruals"],
    "investment": ["asset_growth", "net_issuance", "capex_growth"],
    "momentum": ["mom_12_1", "mom_6_1", "ind_mom_6m", "high_52w"],
    "reversal": ["rev_36_12"],
    "earnings": ["sue", "eps_growth_stability"],
    "risk": ["vol_1y", "idio_vol", "beta_60m", "max_ret_1m"],
    "health": ["debt_to_assets", "interest_coverage", "altman_z", "cash_to_assets"],
    "size": ["log_mcap", "amihud", "turnover"],
    "context": ["term_spread", "credit_spread", "mkt_ret_12m", "xs_dispersion"],
}
NEGATIVE_SIGN = {
    "op_margin_stability", "accruals", "asset_growth", "net_issuance", "capex_growth", "rev_36_12",
    "eps_growth_stability", "vol_1y", "idio_vol", "beta_60m", "max_ret_1m", "debt_to_assets", "log_mcap",
    "turnover",
}
STOCK = [n for family, names in EXPECTED.items() if family != "context" for n in names]


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog()


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(DEFAULT_CATALOG.read_text())


def test_real_catalog_has_exactly_the_contracted_features(catalog):
    assert len(catalog.features) == 36
    assert list(catalog.families) == list(EXPECTED)
    assert {family: catalog.names(family=family) for family in catalog.families} == EXPECTED
    assert catalog.names() == [n for names in EXPECTED.values() for n in names]
    assert len(STOCK) == 32 and catalog.names(kind="stock") == STOCK
    assert catalog.names(kind="context") == EXPECTED["context"]
    assert catalog.names(kind="context", family="value") == []
    assert {f.name for f in catalog.features if f.sign == -1} == NEGATIVE_SIGN
    assert all(f.sign in (-1, 1) for f in catalog.features)
    assert [name for name, fam in catalog.families.items() if fam.sector_relative] == ["value", "quality"]


def test_every_entry_is_documented(catalog):
    for f in catalog.features:
        assert isinstance(f, FeatureSpec)
        for text in (f.label, f.definition, f.lookback, f.lag, f.citation):
            assert isinstance(text, str) and len(text.strip()) >= 3, f.name
        assert f.source and all(isinstance(s, str) and s for s in f.source), f.name
        assert any(ch.isdigit() for ch in f.citation), f"{f.name}: a citation carries a year"
    labels = [f.label for f in catalog.features]
    assert len(set(labels)) == len(labels)
    assert all(fam.label for fam in catalog.families.values())


def test_spec_lookup(catalog):
    spec = catalog.spec("book_to_price")
    assert (spec.family, spec.label, spec.sign, spec.kind) == ("value", "Book/price", 1, "stock")
    assert spec.source == ("SF1.equity (ARQ)", "DAILY.marketcap")
    assert spec.citation == "Fama & French (1992)"
    assert catalog.spec("term_spread").kind == "context"
    with pytest.raises(KeyError):
        catalog.spec("book_to_price_sr")


def test_model_and_linear_columns(catalog):
    sector_relative = [f"{n}_sr" for n in EXPECTED["value"] + EXPECTED["quality"]]
    flags = [f"has_{family}" for family in EXPECTED if family != "context"]
    assert catalog.sector_relative_names() == sector_relative
    assert catalog.stock_families() == [family for family in EXPECTED if family != "context"]
    assert catalog.flag_names() == flags
    assert catalog.linear_columns() == STOCK + sector_relative
    assert catalog.model_columns() == STOCK + sector_relative + flags + EXPECTED["context"]
    assert len(catalog.linear_columns()) == 32 + 11 and len(catalog.model_columns()) == 32 + 11 + 9 + 4
    assert len(set(catalog.model_columns())) == len(catalog.model_columns())
    assert catalog.model_columns() == load_catalog().model_columns()   # deterministic


def test_load_from_a_path(tmp_path, raw):
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert load_catalog(path) == load_catalog()
    assert load_catalog(str(path)).names() == load_catalog().names()


def _feature(raw: dict, name: str) -> dict:
    return next(f for f in raw["features"] if f["name"] == name)


def _duplicate_name(raw):
    _feature(raw, "mom_6_1")["name"] = "mom_12_1"


def _collides_with_derived_column(raw):
    _feature(raw, "mom_6_1")["name"] = "has_value"


def _unknown_family(raw):
    _feature(raw, "sue")["family"] = "sentiment"


def _missing_key(raw):
    del _feature(raw, "sue")["citation"]


def _unknown_key(raw):
    _feature(raw, "sue")["note"] = "typo for definition?"


def _empty_definition(raw):
    _feature(raw, "sue")["definition"] = " "


def _empty_source(raw):
    _feature(raw, "sue")["source"] = []


def _bad_kind(raw):
    _feature(raw, "sue")["kind"] = "macro"


def _too_few(raw):
    raw["features"] = raw["features"][:29]


def _too_many(raw):
    raw["features"] += [{**copy.deepcopy(raw["features"][0]), "name": f"extra_{i}"} for i in range(10)]


def _family_without_label(raw):
    raw["families"]["value"] = {"sector_relative": True}


def _family_typo(raw):
    raw["families"]["value"]["sector_relativ"] = True


def _extra_top_level_key(raw):
    raw["featurs"] = []


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_duplicate_name, "duplicate feature names"),
        (_collides_with_derived_column, "collide"),
        (_unknown_family, "unknown family 'sentiment'"),
        (_missing_key, "missing keys ['citation']"),
        (_unknown_key, "unknown keys ['note']"),
        (_empty_definition, "definition must be a non-empty string"),
        (_empty_source, "source must be a non-empty list"),
        (_bad_kind, "kind must be one of"),
        (_too_few, "29 features"),
        (_too_many, "46 features"),
        (_family_without_label, "needs a label"),
        (_family_typo, "unknown keys ['sector_relativ']"),
        (_extra_top_level_key, "top-level keys"),
    ],
)
def test_malformed_catalog_is_rejected(raw, corrupt, message):
    corrupt(raw)
    with pytest.raises(ValueError) as err:
        parse_catalog(raw)
    assert message in str(err.value)


@pytest.mark.parametrize("sign", [0, 2, -2, 1.0, True, "1", "+", None])
def test_bad_sign_is_rejected(raw, sign):
    _feature(raw, "sue")["sign"] = sign
    with pytest.raises(ValueError, match="sign must be"):
        parse_catalog(raw)


def test_limits_are_inclusive(raw):
    raw["features"] = raw["features"][:30]
    assert len(parse_catalog(raw).features) == 30
    raw["features"] += [{**copy.deepcopy(raw["features"][0]), "name": f"extra_{i}"} for i in range(15)]
    assert len(parse_catalog(raw).features) == 45


def test_single_source_may_be_a_string(raw):
    _feature(raw, "log_mcap")["source"] = "DAILY.marketcap"
    assert parse_catalog(raw).spec("log_mcap").source == ("DAILY.marketcap",)

import json

from rifteye_ml import catalog as cat


def test_parse_code_and_variants():
    cases = {
        "OGN-001/298": ("OGN-001", "standard"),
        "OGN-007a/298": ("OGN-007a", "alt_art"),
        "OGN-304*/298": ("OGN-304*", "signature"),
        "UNL-238/219": ("UNL-238", "overnumbered"),
        "UNL-T01": ("UNL-T01", "token"),
        "VEN-SP1/006": ("VEN-SP1", "standard"),
        "VEN-R01": ("VEN-R01", "standard"),
    }
    for code, (pid, variant) in cases.items():
        parsed = cat.parse_code(code)
        assert parsed is not None, code
        assert parsed["printing_id"] == pid
        assert cat.variant_of(parsed, "Unit", False) == variant, code
    assert cat.parse_code("not a code") is None


def test_slugify_groups_printings_of_one_card():
    assert cat.slugify("Fake Hero") == cat.slugify("Fake Hero (Alternate Art)") == "fake-hero"


# Gallery-shaped rows with invented cards: the real gallery's field names, none of its content.
GALLERY = [
    {"publicCode": "FAK-001/100", "name": "Fake Hero", "type": "Unit", "domains": ["Fury"], "energy": "3",
     "might": "2", "orientation": "portrait", "imageUrl": "https://example.invalid/1.png", "isAltArt": False, "set": "FAK"},
    {"publicCode": "FAK-001a/100", "name": "Fake Hero", "type": "Unit", "domains": ["Fury"], "energy": "3",
     "might": "2", "orientation": "portrait", "imageUrl": "https://example.invalid/1a.png", "isAltArt": True, "set": "FAK"},
    {"publicCode": "FAK-050/100", "name": "Fake Field", "type": "Battlefield", "domains": [], "energy": None,
     "might": None, "orientation": "landscape", "imageUrl": "https://example.invalid/50.png", "set": "FAK"},
    {"publicCode": "FAK-001/100", "name": "Duplicate", "type": "Unit", "imageUrl": "https://example.invalid/dup.png"},
    {"publicCode": "", "name": "No code", "imageUrl": "https://example.invalid/x.png"},
]


def test_from_gallery_normalises_and_dedupes():
    rows = cat.from_gallery(GALLERY)
    by = {r["printing_id"]: r for r in rows}
    assert set(by) == {"FAK-001", "FAK-001a", "FAK-050"}
    assert by["FAK-001"]["name"] == "Fake Hero"  # first occurrence wins
    assert by["FAK-001a"]["variant"] == "alt_art"
    assert by["FAK-001"]["card_id"] == by["FAK-001a"]["card_id"]
    assert by["FAK-050"]["orientation"] == "landscape"
    assert by["FAK-001"]["energy"] == 3.0


def test_from_jsonl_and_merge():
    lines = [
        json.dumps({"external_id": "9", "name": "Fake Hero", "number": "FAK-001/100", "image": "https://example.invalid/t1.jpg"}),
        json.dumps({"external_id": "10", "name": "Fake Promo", "number": "FAK-P07", "image": "https://example.invalid/t2.jpg"}),
        json.dumps({"external_id": "11", "name": "No number", "image": "https://example.invalid/t3.jpg"}),
    ]
    extra = cat.from_jsonl(lines)
    assert {r["printing_id"] for r in extra} == {"FAK-001", "FAK-P07", "tcg-11"}
    merged = cat.merge(cat.from_gallery(GALLERY), extra)
    by = {r["printing_id"]: r for r in merged}
    assert by["FAK-001"]["image_url"].endswith("/1.png")  # the gallery wins
    assert "FAK-P07" in by and "tcg-11" in by


def test_catalog_roundtrip_and_cache_layout(tmp_path):
    rows = cat.from_gallery(GALLERY)
    p = tmp_path / "catalog.jsonl"
    cat.write_catalog(rows, p)
    assert cat.read_catalog(p) == sorted(rows, key=lambda r: r["printing_id"])
    cp = cat.cache_path(tmp_path, "https://example.invalid/1.png")
    assert cp.suffix == ".png" and cp.parent.name == cp.stem[:2]

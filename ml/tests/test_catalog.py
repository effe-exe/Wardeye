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


# Feed-shaped items with invented cards: the official feed's field names, none of its content.
def _feed_item(code, name, types, domains, **extra):
    item = {
        "id": code.lower().replace("/", "-"),
        "publicCode": code,
        "name": name,
        "set": {"label": "Card Set", "value": {"id": code.split("-")[0], "label": "Fake Set"}},
        "cardType": {"label": "Card Type", "type": [{"id": t.lower(), "label": t} for t in types]},
        "domain": {"label": "Domain", "values": [{"id": d.lower(), "label": d} for d in domains]},
        "rarity": {"label": "Rarity", "value": {"id": extra.pop("rarity", "common"), "label": "Common"}},
        "cardImage": {"type": "image", "url": f"https://example.invalid/{code.replace('/', '_')}.png"},
        "orientation": extra.pop("orientation", "portrait"),
    }
    for key in ("energy", "might", "power"):
        if key in extra:
            item[key] = {"label": key.title(), "value": {"id": extra.pop(key), "label": "x"}}
    if "tags" in extra:
        item["tags"] = {"label": "Tags", "tags": extra.pop("tags")}
    item.update(extra)
    return item


FEED = [
    _feed_item("FAK-001/100", "Fake Hero", ["Unit"], ["Fury"], energy=3, might=2, power=1,
               subtitle="Brave", tags=["Fake Hero", "Fakeland"]),
    _feed_item("FAK-001a/100", "Fake Hero", ["Unit"], ["Fury"], energy=3, might=2, subtitle="Brave",
               rarity="showcase"),
    _feed_item("FAK-002/100", "Fake Hero", ["Unit"], ["Fury"], energy=5, might=4, subtitle="Wise"),
    _feed_item("FAK-101*/100", "Fake Title", ["Legend"], ["Fury", "Calm"], tags=["Fake Hero"], subtitle="Starter"),
    _feed_item("FAK-030/100", "Fake Bolt", ["Spell"], ["Fury"], energy=1, subtitle="Fake Hero", tags=["Fake Hero"]),
    _feed_item("FAK-050/100", "Fake Field", ["Battlefield"], ["Colorless"], orientation="landscape"),
    _feed_item("FAK-T01", "Fake Marker", [], []),
    _feed_item("FAK-001/100", "Duplicate", ["Unit"], []),
    {"publicCode": "FAK-060/100", "name": "No image", "cardImage": {}},
]


def test_from_feed_normalises():
    rows = cat.from_feed(FEED)
    by = {r["printing_id"]: r for r in rows}
    assert set(by) == {"FAK-001", "FAK-001a", "FAK-002", "FAK-030", "FAK-101*", "FAK-050", "FAK-T01"}
    hero = by["FAK-001"]
    assert hero["name"] == "Fake Hero, Brave" and hero["card_id"] == "fake-hero-brave"
    assert (hero["type"], hero["domains"], hero["energy"], hero["might"], hero["power"]) == ("Unit", ["Fury"], 3.0, 2.0, 1.0)
    assert hero["tags"] == ["Fake Hero", "Fakeland"] and hero["set_code"] == "FAK" and hero["variant"] == "standard"
    # Same name, different subtitle: a different card. Same subtitle, alt art: the same card.
    assert by["FAK-002"]["card_id"] == "fake-hero-wise"
    assert by["FAK-001a"]["card_id"] == hero["card_id"]
    assert (by["FAK-001a"]["variant"], by["FAK-001a"]["rarity"]) == ("alt_art", "showcase")
    assert by["FAK-101*"]["variant"] == "signature" and by["FAK-101*"]["domains"] == ["Fury", "Calm"]
    # Subtitles on other types are annotations, not part of the name.
    assert (by["FAK-101*"]["name"], by["FAK-030"]["name"]) == ("Fake Title", "Fake Bolt")
    assert by["FAK-050"]["orientation"] == "landscape" and by["FAK-050"]["energy"] is None
    assert (by["FAK-T01"]["variant"], by["FAK-T01"]["type"]) == ("token", "")
    # Every adapter writes the same row shape.
    assert set(hero) == set(cat.from_gallery(GALLERY)[0]) == set(cat.from_jsonl([json.dumps(
        {"name": "Fake Hero", "number": "FAK-001/100", "image": "https://example.invalid/t.jpg"})])[0])


def test_load_feed_reads_pages_lists_and_directories(tmp_path):
    (tmp_path / "en_US-0000.json").write_text(json.dumps({"data": FEED[:3], "metadata": {"totalPages": 2}}))
    (tmp_path / "en_US-0200.json").write_text(json.dumps({"data": FEED[3:], "metadata": {"totalPages": 2}}))
    assert len(cat.load_feed(tmp_path)) == len(FEED)
    assert len(cat.load_feed(tmp_path / "en_US-0200.json")) == len(FEED) - 3
    bare = tmp_path / "list" / "items.json"
    bare.parent.mkdir()
    bare.write_text(json.dumps(FEED[:2]))
    assert len(cat.load_feed(bare)) == 2


def test_fetch_feed_follows_pagination(tmp_path, monkeypatch):
    requested = []

    def fake_get(url):
        requested.append(url)
        start = int(url.split("from=")[1].split("&")[0])
        return {"data": FEED[start:start + 3], "metadata": {"totalPages": 3, "from": start}}

    monkeypatch.setattr(cat, "_get_json", fake_get)
    monkeypatch.setattr(cat.time, "sleep", lambda s: None)
    pages = cat.fetch_feed(tmp_path, locale="en_US", limit=3)
    assert [p.name for p in pages] == ["en_US-0000.json", "en_US-0003.json", "en_US-0006.json"]
    assert all(u.startswith(cat.FEED_URL + "?locale=en_US&from=") and u.endswith("&limit=3") for u in requested)
    assert len(cat.load_feed(tmp_path)) == len(FEED)


def test_localised_printings_adopt_reference_card_ids():
    assert cat.slugify("Fake Hé-ro") == "fake-he-ro"
    assert cat.slugify("假英雄") == "假英雄"  # other scripts are kept, not collapsed to "unknown"
    en = cat.from_feed(FEED)
    zh = cat.from_feed([{**FEED[0], "name": "假英雄", "subtitle": "勇敢"}, {**FEED[2], "name": "假英雄"}], language="zh-Hans")
    assert zh[0]["card_id"] != en[0]["card_id"]
    adopted = {r["printing_id"]: r for r in cat.adopt_card_ids(zh, en)}
    assert adopted["FAK-001"]["card_id"] == "fake-hero-brave" and adopted["FAK-002"]["card_id"] == "fake-hero-wise"
    assert adopted["FAK-001"]["name"] == "假英雄, 勇敢" and adopted["FAK-001"]["language"] == "zh-Hans"

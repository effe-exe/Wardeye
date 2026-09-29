# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""Build a normalised Riftbound catalogue for research, and cache its images locally.

Everything here runs on the researcher's machine. Card images and text are Riot Games IP:
they are cached locally for training and evaluation and never committed, uploaded or
redistributed (decisions D-006 and D-015).

Input formats:

* **feed**: pages of the official card gallery feed (the public JSON behind
  playriftbound.com's card gallery), saved by `fetch-feed`. This is the primary source.
* **gallery**: a JSON list in the flattened shape saved by public mirrors of the gallery,
  such as github.com/slimtreble/Riftbound-card-data (`cards.json`). Fields used:
  `publicCode`/`code`, `name`, `type`, `domains`, `energy`, `might`, `orientation`,
  `imageUrl`, `isAltArt`, `set`.
* **jsonl**: one object per line with `name`, `number` (collector code) and `image`, the
  shape written by TCGplayer/TCGCSV sync scripts. Useful for promos the gallery lacks.

Output: JSONL, one printing per line:
    {printing_id, card_id, name, type, domains, energy, might, power, tags, rarity, set_code,
     collector_number, variant, language, orientation, image_url}

    python -m rifteye_ml.catalog fetch-feed --out ~/rifteye-data/catalog/feed
    python -m rifteye_ml.catalog build --feed ~/rifteye-data/catalog/feed --out catalog.jsonl
    python -m rifteye_ml.catalog download --catalog catalog.jsonl --cache ~/.cache/rifteye/art
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import sys
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator

USER_AGENT = "WardeyeResearch/0.1 (+https://github.com/effe-exe/wardeye)"
FEED_URL = ("https://content.publishing.riotgames.com/publishing-content/v2.0/public/channel/"
            "riftbound_website/list/riftbound_gallery_cards")


def _ssl_context() -> ssl.SSLContext:
    """The system's certificates, plus certifi's when it is installed: Python from python.org on macOS
    has none of its own until its 'Install Certificates' step is run, and then every download fails."""
    ctx = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        return ctx
    ctx.load_verify_locations(certifi.where())
    return ctx


_SSL = _ssl_context()
_CODE = re.compile(r"^(?P<set>[A-Z]{2,4})-(?P<num>[A-Z]*\d+[a-z]?\*?)(?:/(?P<total>\d+))?$")


def slugify(name: str) -> str:
    """Gameplay identity from a card name: same name, same card (Core Rules 132.3).

    Accents fold to ASCII; letters of other scripts (e.g. Chinese names) are kept."""
    s = re.sub(r"\s*\((?:alternate art|alt art|showcase|signature|overnumbered|promo|foil)[^)]*\)\s*", " ", name, flags=re.I)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"[\W_]+", "-", s.lower()).strip("-")
    return s or "unknown"


def parse_code(code: str) -> dict | None:
    """'OGN-007a/298' -> {set: 'OGN', num: '007a', total: 298, printing_id: 'OGN-007a'}."""
    m = _CODE.match((code or "").strip())
    if not m:
        return None
    total = int(m["total"]) if m["total"] else None
    return {"set": m["set"], "num": m["num"], "total": total, "printing_id": f"{m['set']}-{m['num']}"}


def variant_of(code: dict, card_type: str, is_alt_art: bool) -> str:
    num = code["num"]
    if card_type.lower() == "token" or num.startswith("T"):
        return "token"
    if num.endswith("*"):
        return "signature"
    if is_alt_art or num[-1:] in ("a", "b"):
        return "alt_art"
    digits = re.sub(r"\D", "", num)
    if code["total"] and digits and int(digits) > code["total"]:
        return "overnumbered"
    return "standard"


def _as_number(v) -> float | None:
    try:
        return float(v) if v not in (None, "", "-") else None
    except (TypeError, ValueError):
        return None


def _feed_value(field) -> float | None:
    """{"label": "Energy", "value": {"id": 3, "label": "3"}} -> 3.0"""
    value = (field or {}).get("value") if isinstance(field, dict) else None
    return _as_number(value.get("id")) if isinstance(value, dict) else None


def from_feed(items: Iterable[dict], language: str = "en") -> list[dict]:
    """Normalise items of the official card gallery feed (see `fetch_feed`)."""
    out: dict[str, dict] = {}
    for it in items:
        code = parse_code(it.get("publicCode") or "")
        url = (it.get("cardImage") or {}).get("url")
        name = (it.get("name") or "").strip()
        if not code or not url or not name:
            continue
        card_type = " ".join(t.get("label") or "" for t in ((it.get("cardType") or {}).get("type") or [])).strip()
        # Champion units share a name and differ by subtitle: "Ahri, Alluring" and
        # "Ahri, Inquisitive" are different cards. Other subtitles are annotations, such as
        # the champion on a signature spell or "Starter" on a starter-deck legend.
        subtitle = (it.get("subtitle") or "").strip()
        full_name = f"{name}, {subtitle}" if subtitle and card_type == "Unit" else name
        set_value = (it.get("set") or {}).get("value") or {}
        rarity = ((it.get("rarity") or {}).get("value") or {}).get("id") or ""
        out.setdefault(code["printing_id"], {
            "printing_id": code["printing_id"],
            "card_id": slugify(full_name),
            "name": full_name,
            "type": card_type,
            "domains": [v.get("label") for v in ((it.get("domain") or {}).get("values") or []) if v.get("label")],
            "energy": _feed_value(it.get("energy")),
            "might": _feed_value(it.get("might")),
            "power": _feed_value(it.get("power")),
            "tags": list((it.get("tags") or {}).get("tags") or []),
            "rarity": rarity,
            "set_code": set_value.get("id") or code["set"],
            "collector_number": code["num"],
            "variant": variant_of(code, card_type, False),
            "language": language,
            "orientation": "landscape" if it.get("orientation") == "landscape" else "portrait",
            "image_url": url,
        })
    return list(out.values())


def load_feed(path: str | Path) -> list[dict]:
    """Items from a saved feed page, a bare JSON list of items, or a directory of pages."""
    p = Path(path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    items: list[dict] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        items.extend(data.get("data", []) if isinstance(data, dict) else data)
    return items


def _get_json(url: str, tries: int = 4) -> dict:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=45, context=_SSL) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - retry any network failure, re-raise the last one
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def fetch_feed(out_dir: str | Path, locale: str = "en_US", limit: int = 200, delay: float = 1.0) -> list[Path]:
    """Save every page of the official card gallery feed to `out_dir`, on this machine only."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    start, total_pages = 0, 1
    while len(pages) < total_pages:
        data = _get_json(f"{FEED_URL}?locale={locale}&from={start}&limit={limit}")
        total_pages = int((data.get("metadata") or {}).get("totalPages") or 1)
        dest = out / f"{locale}-{start:04d}.json"
        dest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        pages.append(dest)
        start += limit
        if len(pages) < total_pages:
            time.sleep(delay)  # be gentle with the host
    return pages


def from_gallery(rows: Iterable[dict]) -> list[dict]:
    out: dict[str, dict] = {}
    for r in rows:
        code = parse_code(r.get("publicCode") or r.get("code") or "")
        url = r.get("imageUrl")
        name = (r.get("name") or "").strip()
        if not code or not url or not name:
            continue
        card_type = r.get("type") or ""
        out.setdefault(code["printing_id"], {
            "printing_id": code["printing_id"],
            "card_id": slugify(name),
            "name": name,
            "type": card_type,
            "domains": [d for d in (r.get("domains") or []) if d],
            "energy": _as_number(r.get("energy")),
            "might": _as_number(r.get("might")),
            "power": None,
            "tags": [],
            "rarity": "",
            "set_code": r.get("set") or code["set"],
            "collector_number": code["num"],
            "variant": variant_of(code, card_type, bool(r.get("isAltArt"))),
            "language": "en",
            "orientation": "landscape" if r.get("orientation") == "landscape" or card_type == "Battlefield" else "portrait",
            "image_url": url,
        })
    return list(out.values())


def from_jsonl(lines: Iterable[str]) -> list[dict]:
    out: dict[str, dict] = {}
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        name = (r.get("name") or "").strip()
        url = r.get("image") or r.get("image_url")
        if not name or not url:
            continue
        code = parse_code(r.get("number") or "")
        pid = code["printing_id"] if code else f"tcg-{r.get('external_id')}"
        out.setdefault(pid, {
            "printing_id": pid,
            "card_id": slugify(name),
            "name": name,
            "type": r.get("type") or "",
            "domains": [],
            "energy": None,
            "might": None,
            "power": None,
            "tags": [],
            "rarity": "",
            "set_code": code["set"] if code else "",
            "collector_number": code["num"] if code else "",
            "variant": variant_of(code, r.get("type") or "", False) if code else "standard",
            "language": "en",
            "orientation": "portrait",
            "image_url": url,
        })
    return list(out.values())


def adopt_card_ids(rows: list[dict], reference: list[dict]) -> list[dict]:
    """Give localised printings the gameplay identity of the reference printing with the same code."""
    ref = {r["printing_id"]: r["card_id"] for r in reference}
    return [{**r, "card_id": ref.get(r["printing_id"], r["card_id"])} for r in rows]


def merge(primary: list[dict], extra: list[dict]) -> list[dict]:
    """Primary rows win; extra rows only fill printings the primary source lacks."""
    seen = {r["printing_id"] for r in primary}
    return primary + [r for r in extra if r["printing_id"] not in seen]


def read_catalog(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_catalog(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["printing_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------------------
# Image cache (local only)
# ------------------------------------------------------------------------------------

def cache_path(cache_dir: str | Path, url: str) -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()
    ext = ".png" if ".png" in url.lower() else ".webp" if ".webp" in url.lower() else ".jpg"
    return Path(cache_dir) / h[:2] / f"{h}{ext}"


def _fetch(url: str, dest: Path, tries: int = 4) -> bool:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=45, context=_SSL) as resp:
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True
        except Exception as e:  # noqa: BLE001 - report and retry any network failure
            if attempt == tries - 1:
                print(f"  failed {url}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return False


def download_images(rows: list[dict], cache_dir: str | Path, workers: int = 4, delay: float = 0.1) -> dict[str, Path]:
    """Fetch every row's image into the local cache. Skips files that are already cached."""
    todo = [r for r in rows if not cache_path(cache_dir, r["image_url"]).exists()]
    print(f"{len(rows) - len(todo)} cached, {len(todo)} to download")

    def job(r: dict) -> None:
        _fetch(r["image_url"], cache_path(cache_dir, r["image_url"]))
        time.sleep(delay)  # be gentle with the host

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(job, todo), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")
    return {r["printing_id"]: cache_path(cache_dir, r["image_url"]) for r in rows if cache_path(cache_dir, r["image_url"]).exists()}


SUPPLEMENT_SCHEME = "supplement://"


def supplement(spec: list[dict], reference: list[dict], cache_dir: str | Path) -> list[dict]:
    """Rows for printings the official gallery lacks (promos, some alt arts, tokens), each with a picture
    from elsewhere, e.g. the clearest reviewed crop of it on a broadcast. A spec entry has `printing_id`,
    `image` (a path), optionally `rotate` (degrees counter-clockwise that make it upright) and `like` (a
    printing whose fields it copies), and any fields to set. The picture is stored in the art cache under
    supplement://<printing_id>.png, so every tool loads it like gallery art. Like the art, it stays private."""
    from PIL import Image

    by_id = {r["printing_id"]: r for r in reference}
    known = {r["printing_id"] for r in reference}
    out = []
    for e in spec:
        pid = e["printing_id"]
        if pid in known:
            raise ValueError(f"{pid} is already in the catalogue")
        base = dict(by_id[e["like"]]) if e.get("like") else {}
        row = {**base, **{k: v for k, v in e.items() if k not in ("image", "rotate", "like")}}
        row["image_url"] = f"{SUPPLEMENT_SCHEME}{pid}.png"
        row.setdefault("variant", "standard")
        row.setdefault("language", "en")
        row.setdefault("orientation", "portrait")
        missing = [k for k in ("card_id", "name", "set_code") if not row.get(k)]
        if missing:
            raise ValueError(f"{pid}: give {', '.join(missing)} (or `like` a printing to copy them from)")
        im = Image.open(e["image"]).convert("RGB")
        if e.get("rotate"):
            im = im.rotate(int(e["rotate"]), expand=True)
        dest = cache_path(cache_dir, row["image_url"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest)
        out.append(row)
    return out


def iter_images(rows: list[dict], cache_dir: str | Path) -> Iterator[tuple[dict, Path]]:
    for r in rows:
        p = cache_path(cache_dir, r["image_url"])
        if p.exists():
            yield r, p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.catalog", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    fp = sub.add_parser("fetch-feed", help="save every page of the official card gallery feed locally")
    fp.add_argument("--out", required=True, help="directory for the page files")
    fp.add_argument("--locale", default="en_US")
    b = sub.add_parser("build", help="normalise feed pages, gallery JSON and/or JSONL into catalog.jsonl")
    b.add_argument("--feed", help="a directory of pages saved by fetch-feed, or one page file")
    b.add_argument("--language", default="en", help="BCP-47 language of the feed pages")
    b.add_argument("--gallery", help="mirror-shaped JSON list (e.g. a mirror's cards.json)")
    b.add_argument("--jsonl", help="extra TCGplayer/TCGCSV-style JSONL, fills printings the others lack")
    b.add_argument("--card-ids", help="reference catalogue (e.g. English) whose card_id each printing adopts")
    b.add_argument("--out", required=True)
    d = sub.add_parser("download", help="cache every catalogue image locally")
    d.add_argument("--catalog", required=True)
    d.add_argument("--cache", required=True)
    d.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("supplement", help="printings the official gallery lacks, with pictures from elsewhere")
    s.add_argument("--spec", required=True, help="JSON list of entries (see catalog.supplement)")
    s.add_argument("--catalog", required=True, help="the official catalogue")
    s.add_argument("--cache", required=True, help="the art cache the pictures go into")
    s.add_argument("--out", required=True, help="the supplement rows (JSONL)")
    s.add_argument("--merged", help="also write the catalogue with the supplement appended")
    a = ap.parse_args(argv)

    if a.cmd == "fetch-feed":
        pages = fetch_feed(a.out, locale=a.locale)
        print(f"{len(pages)} pages, {len(load_feed(a.out))} items -> {a.out}")
    elif a.cmd == "build":
        if not a.feed and not a.gallery and not a.jsonl:
            ap.error("give --feed, --gallery and/or --jsonl")
        rows: list[dict] = []
        if a.feed:
            rows = from_feed(load_feed(a.feed), language=a.language)
        if a.gallery:
            data = json.loads(Path(a.gallery).read_text(encoding="utf-8"))
            rows = merge(rows, from_gallery(data if isinstance(data, list) else data.get("cards", [])))
        if a.jsonl:
            with open(a.jsonl, encoding="utf-8") as f:
                rows = merge(rows, from_jsonl(f))
        if a.card_ids:
            rows = adopt_card_ids(rows, read_catalog(a.card_ids))
        write_catalog(rows, a.out)
        cards = len({r["card_id"] for r in rows})
        print(f"{len(rows)} printings, {cards} cards -> {a.out}")
    elif a.cmd == "supplement":
        official = read_catalog(a.catalog)
        extra = supplement(json.loads(Path(a.spec).read_text(encoding="utf-8")), official, a.cache)
        write_catalog(extra, a.out)
        if a.merged:
            write_catalog(merge(official, extra), a.merged)
        print(f"{len(extra)} supplement printings -> {a.out}" + (f"; {len(official) + len(extra)} in {a.merged}" if a.merged else ""))
    else:
        rows = [r for r in read_catalog(a.catalog) if not r["image_url"].startswith(SUPPLEMENT_SCHEME)]
        got = download_images(rows, a.cache, workers=a.workers)
        print(f"{len(got)}/{len(rows)} images cached in {a.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

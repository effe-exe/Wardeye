# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Build a normalised Riftbound catalogue for research, and cache its images locally.

Everything here runs on the researcher's machine. Card images and text are Riot Games IP:
they are cached locally for training and evaluation and never committed, uploaded or
redistributed (decisions D-006 and D-015).

Input formats:

* **gallery**: a JSON list in the shape of the official card gallery, as saved by public
  mirrors such as github.com/slimtreble/Riftbound-card-data (`cards.json`). Fields used:
  `publicCode`/`code`, `name`, `type`, `domains`, `energy`, `might`, `orientation`,
  `imageUrl`, `isAltArt`, `set`.
* **jsonl**: one object per line with `name`, `number` (collector code) and `image`, the
  shape written by TCGplayer/TCGCSV sync scripts. Useful for promos the gallery lacks.

Output: JSONL, one printing per line:
    {printing_id, card_id, name, type, domains, energy, might, set_code, collector_number,
     variant, language, orientation, image_url}

    python -m rifteye_ml.catalog build --gallery cards.json --out catalog.jsonl
    python -m rifteye_ml.catalog download --catalog catalog.jsonl --cache ~/.cache/rifteye/art
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator

USER_AGENT = "RiftEyeResearch/0.1 (+https://github.com/effe-exe/RiftEye)"
_CODE = re.compile(r"^(?P<set>[A-Z]{2,4})-(?P<num>[A-Z]*\d+[a-z]?\*?)(?:/(?P<total>\d+))?$")


def slugify(name: str) -> str:
    """Gameplay identity from a card name: same name, same card (Core Rules 132.3)."""
    s = re.sub(r"\s*\((?:alternate art|alt art|showcase|signature|overnumbered|promo|foil)[^)]*\)\s*", " ", name, flags=re.I)
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
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
            "set_code": code["set"] if code else "",
            "collector_number": code["num"] if code else "",
            "variant": variant_of(code, r.get("type") or "", False) if code else "standard",
            "language": "en",
            "orientation": "portrait",
            "image_url": url,
        })
    return list(out.values())


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
            with urllib.request.urlopen(req, timeout=45) as resp:
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


def iter_images(rows: list[dict], cache_dir: str | Path) -> Iterator[tuple[dict, Path]]:
    for r in rows:
        p = cache_path(cache_dir, r["image_url"])
        if p.exists():
            yield r, p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.catalog", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="normalise gallery JSON (and optional JSONL) into catalog.jsonl")
    b.add_argument("--gallery", help="gallery-shaped JSON list (e.g. a mirror's cards.json)")
    b.add_argument("--jsonl", help="extra TCGplayer/TCGCSV-style JSONL, fills printings the gallery lacks")
    b.add_argument("--out", required=True)
    d = sub.add_parser("download", help="cache every catalogue image locally")
    d.add_argument("--catalog", required=True)
    d.add_argument("--cache", required=True)
    d.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    if a.cmd == "build":
        if not a.gallery and not a.jsonl:
            ap.error("give --gallery and/or --jsonl")
        rows: list[dict] = []
        if a.gallery:
            data = json.loads(Path(a.gallery).read_text(encoding="utf-8"))
            rows = from_gallery(data if isinstance(data, list) else data.get("cards", []))
        if a.jsonl:
            with open(a.jsonl, encoding="utf-8") as f:
                rows = merge(rows, from_jsonl(f))
        write_catalog(rows, a.out)
        cards = len({r["card_id"] for r in rows})
        print(f"{len(rows)} printings, {cards} cards -> {a.out}")
    else:
        rows = read_catalog(a.catalog)
        got = download_images(rows, a.cache, workers=a.workers)
        print(f"{len(got)}/{len(rows)} images cached in {a.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

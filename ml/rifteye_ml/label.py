# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Model-assisted labeling of real crops for the evaluation set.

`propose` ranks catalogue candidates for every crop in a folder and renders verification
sheets: the crop, enlarged, next to its top candidates' clean art. A person then writes
`labels.csv` (`file,printing_id`), confirming a candidate by eye or naming the right
printing. The model only proposes; it never labels. Use `?` for crops that cannot be
identified and `none` for cards missing from the catalogue.

Crops, sheets and labels are private: they come from broadcasts and show Riot's card
art (decisions D-006 and D-015). Only accuracy numbers leave the machine.

    python -m rifteye_ml.label propose --catalog catalog.jsonl --cache ~/rifteye-data/art \\
        --crops ~/rifteye-data/real-crops --encoder colorgrid --out ~/rifteye-data/real-crops/proposals.csv \\
        --sheets ~/rifteye-data/real-crops/sheets
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from . import catalog as cat
from .encoders import get_encoder
from .retrieval import search
from .spike import _cached_loader, _gallery, _ints, catalog_key

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def crop_files(folder: Path) -> list[Path]:
    """Every crop image in `folder`, skipping masks and sheets."""
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMAGE_SUFFIXES and not p.stem.endswith("_mask") and p.is_file())


def candidates(idx_row: Sequence[int], rows: Sequence[dict], k: int) -> list[int]:
    """Gallery rows in rank order, one per card, at most k."""
    seen: set[str] = set()
    out: list[int] = []
    for i in idx_row:
        cid = rows[int(i)]["card_id"]
        if cid not in seen:
            seen.add(cid)
            out.append(int(i))
        if len(out) == k:
            break
    return out


def sheet(crop: Image.Image, cands: Sequence[tuple[str, Image.Image]], title: str, tile_h: int = 240) -> Image.Image:
    """The crop, enlarged, then each candidate's art with its rank and printing id."""
    def fit(im: Image.Image) -> Image.Image:
        s = tile_h / im.height
        return im.convert("RGB").resize((max(1, round(im.width * s)), tile_h), Image.LANCZOS)

    tiles = [fit(crop)] + [fit(im) for _, im in cands]
    width = sum(t.width for t in tiles) + 12 * (len(tiles) + 1)
    out = Image.new("RGB", (width, tile_h + 52), (28, 28, 32))
    d = ImageDraw.Draw(out)
    d.text((12, 6), title, fill=(235, 235, 235))
    x = 12
    for i, t in enumerate(tiles):
        out.paste(t, (x, 24))
        label = "crop" if i == 0 else f"{i}. {cands[i - 1][0]}"
        d.text((x, tile_h + 30), label, fill=(255, 210, 90) if i == 0 else (220, 220, 220))
        x += t.width + 12
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.label", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose", help="rank candidates for every crop and render verification sheets")
    p.add_argument("--catalog", required=True, action="append",
                   help="repeatable: e.g. the English catalogue plus a localised one")
    p.add_argument("--cache", required=True)
    p.add_argument("--crops", required=True)
    p.add_argument("--encoder", default="colorgrid")
    p.add_argument("--gallery-scales", default="")
    p.add_argument("--embed-cache")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--out", required=True, help="proposals CSV")
    p.add_argument("--sheets", help="folder for verification sheets (private)")
    a = ap.parse_args(argv)

    rows: list[dict] = []
    for c in a.catalog:
        rows += [r for r in cat.read_catalog(c) if cat.cache_path(a.cache, r["image_url"]).exists()]
    # A localised row with the same image as an English row adds nothing.
    seen: set[str] = set()
    rows = [r for r in rows if not (r["image_url"] in seen or seen.add(r["image_url"]))]
    load = _cached_loader(a.cache, 512)
    images = [load(r) for r in rows]
    enc = get_encoder(a.encoder)
    gallery = _gallery(enc, images, _ints(a.gallery_scales), Path(a.embed_cache) if a.embed_cache else None,
                       key=catalog_key(rows, 512))
    files = crop_files(Path(a.crops))
    crops = [Image.open(f).convert("RGB") for f in files]
    idx, scores, rots = search(enc, gallery, crops, k=min(60, len(rows)), rotation_invariant=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheets = Path(a.sheets) if a.sheets else None
    if sheets:
        sheets.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "rank", "printing_id", "card_id", "language", "score", "rotation"])
        for n, (file, crop) in enumerate(zip(files, crops)):
            ranked = candidates(idx[n], rows, a.k)
            for rank, i in enumerate(ranked, 1):
                pos = list(idx[n]).index(i)
                w.writerow([file.name, rank, rows[i]["printing_id"], rows[i]["card_id"], rows[i]["language"],
                            f"{float(scores[n][pos]):.4f}", int(rots[n][pos])])
            if sheets:
                cands = [(f"{rows[i]['printing_id']} {rows[i]['language']}", images[i]) for i in ranked]
                # Show the crop the way the top candidate matched it, so the two can be compared directly.
                r0 = int(rots[n][list(idx[n]).index(ranked[0])]) if ranked else 0
                shown = crop.rotate(r0, expand=True) if r0 else crop
                sheet(shown, cands, f"{file.name}  ({crop.width}x{crop.height} px)").save(sheets / f"{file.stem}.jpg", quality=88)
    print(f"{len(files)} crops, {len(rows)} gallery rows -> {out}" + (f", sheets in {sheets}" if sheets else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

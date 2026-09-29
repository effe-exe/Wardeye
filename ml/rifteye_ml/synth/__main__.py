# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""Write synthetic broadcast frames of Riftbound boards, with annotations.

    python -m rifteye_ml.synth --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art \
        --boards 200 --out ~/rifteye-data/synth/v0
    python -m rifteye_ml.synth --fixtures 60 --boards 4 --out /tmp/synth-demo     # no data needed

Boards are grouped into clips of one resolution and bitrate. Each board is held for a few frames
with sensor noise, the clip goes through a real libx264 encode and decode, and the last frame of
each board is kept, as in the M0 stream simulator. Output, all private when made from Riot's art:

* `frames/NNNNNN.png`: the decoded frame, lossless (JPEG would add the wrong artifacts);
* `ids/NNNNNN.png`: 16-bit, card id + 1 where that card is uppermost, 65535 under a hand, die or
  counter, 65534 on a broadcast graphic, 0 elsewhere;
* `annotations.jsonl`: one board per line (see `compose.render`);
* `manifest.json`: the settings, the catalogue key and counts.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw

from .. import catalog as cat
from ..degrade import StreamSettings, h264_roundtrip
from ..fixtures import load_fixture_image, synthetic_catalog
from .compose import render, sample_shot
from .layout import sample_board


def _weighted(spec: str) -> tuple[list[str], np.ndarray]:
    """'a:0.6,b:0.4' -> (['a', 'b'], [0.6, 0.4] normalised)."""
    keys, w = [], []
    for part in spec.split(","):
        k, _, v = part.strip().partition(":")
        keys.append(k); w.append(float(v) if v else 1.0)
    p = np.asarray(w, np.float64)
    return keys, p / p.sum()


def _preview(frame: np.ndarray, ann: dict) -> Image.Image:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    for c in ann["cards"]:
        q = c["quad"]; pts = [(q[i], q[i + 1]) for i in range(0, 8, 2)]
        col = (0, 255, 0) if c["visible"] > 0.9 else (255, 200, 0) if c["visible"] > 0.3 else (255, 0, 0)
        if c["kind"] == "card_back":
            col = (80, 160, 255)
        d.line(pts + [pts[0]], fill=col, width=2)
        d.ellipse([pts[0][0] - 3, pts[0][1] - 3, pts[0][0] + 3, pts[0][1] + 3], fill=(255, 255, 255))
    return im


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.synth", description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--catalog", help="catalog.jsonl (with --cache): Riot's card art, kept private")
    src.add_argument("--fixtures", type=int, help="use N procedural fake cards instead (no Riot content)")
    ap.add_argument("--cache", help="image cache written by `catalog download`")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--boards", type=int, default=20)
    ap.add_argument("--clip", type=int, default=8, help="boards per encode (one resolution and bitrate)")
    ap.add_argument("--frames-per-board", type=int, default=10, help="the encoder settles; the last frame is kept")
    ap.add_argument("--sizes", default="1920x1080:0.6,1280x720:0.3,854x480:0.1")
    ap.add_argument("--layouts", default="rq:0.5,full:0.3,pip:0.2")
    ap.add_argument("--bitrates", default="2500,4000,6000,8000", help="kbps, one per clip at random")
    ap.add_argument("--view-mm", default="550,950", help="table seen by the camera, vertically (log-uniform)")
    ap.add_argument("--mats", help="folder of playmat images of your own (official mats are Riot IP: never bundled)")
    ap.add_argument("--previews", type=int, default=0, help="also write N JPEG previews with the quads drawn")
    ap.add_argument("--no-codec", action="store_true", help="skip the H.264 pass (debugging only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.catalog:
        if not a.cache:
            ap.error("--catalog needs --cache")
        from ..spike import _cached_loader, catalog_key
        rows = [r for r in cat.read_catalog(a.catalog) if cat.cache_path(a.cache, r["image_url"]).exists()]
        base_load, key = _cached_loader(a.cache, a.max_side), catalog_key(rows, a.max_side)
    else:
        rows, base_load, key = synthetic_catalog(a.fixtures), load_fixture_image, f"fixtures:{a.fixtures}"
    if not rows:
        ap.error("no cards with cached images")
    art: dict[str, Image.Image] = {}

    def load(row: dict) -> Image.Image:
        k = row["image_url"]
        if k not in art:
            art[k] = base_load(row)
        return art[k]

    mats = [Image.open(p).convert("RGB") for p in sorted(Path(a.mats).glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")] if a.mats else []
    sizes, size_p = _weighted(a.sizes)
    layouts, layout_p = _weighted(a.layouts)
    bitrates = [int(b) for b in a.bitrates.split(",")]
    vmin, vmax = (float(v) for v in a.view_mm.split(","))

    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True); (out / "ids").mkdir(exist_ok=True)
    if a.previews:
        (out / "previews").mkdir(exist_ok=True)
    plan = np.random.default_rng(a.seed)
    t0, n, counts = time.time(), 0, {"cards": 0, "card_backs": 0, "covered": 0, "occluders": 0}
    with open(out / "annotations.jsonl", "w", encoding="utf-8") as fa:
        for c0 in range(0, a.boards, a.clip):
            w, h = (int(v) for v in sizes[int(plan.choice(len(sizes), p=size_p))].split("x"))
            br = bitrates[int(plan.integers(len(bitrates)))]
            boards = []
            for bi in range(c0, min(a.boards, c0 + a.clip)):
                rng = np.random.default_rng([a.seed, bi])  # each board reproducible on its own
                board = sample_board(rows, rng)
                layout = layouts[int(plan.choice(len(layouts), p=layout_p))]
                shot = sample_shot(rng, w, h, layout, (vmin, vmax))
                frame, ids, ann = render(board, rows, load, shot, rng, mats)
                boards.append((bi, frame, ids, ann))
            s = StreamSettings(frame_w=w, frame_h=h, bitrate_kbps=br, fps=30, frames_per_board=a.frames_per_board)
            if a.no_codec:
                decoded = {k: b[1] for k, b in enumerate(boards)}
            else:
                noise = np.random.default_rng([a.seed, c0, 7])
                bank = [noise.standard_normal((h, w, 3), dtype=np.float32) * s.noise_sigma for _ in range(a.frames_per_board)]

                def frames() -> Iterator[np.ndarray]:
                    for _, frame, _, _ in boards:
                        base = frame.astype(np.float32)
                        for i in range(a.frames_per_board):
                            yield np.clip(base + bank[i], 0, 255).astype(np.uint8)

                last = {(k + 1) * a.frames_per_board - 1: k for k in range(len(boards))}
                got = h264_roundtrip(frames(), w, h, s, keep=set(last))
                decoded = {last[fi]: fr for fi, fr in got.items()}
            for k, (bi, frame, ids, ann) in enumerate(boards):
                name = f"{bi:06d}.png"
                Image.fromarray(decoded[k]).save(out / "frames" / name)
                Image.fromarray(ids).save(out / "ids" / name)
                ann.update({"image": f"frames/{name}", "ids": f"ids/{name}", "board": bi, "clip": c0 // a.clip,
                            "bitrate_kbps": None if a.no_codec else br, "frames_per_board": a.frames_per_board})
                fa.write(json.dumps(ann, separators=(",", ":")) + "\n")
                if bi < a.previews:
                    _preview(decoded[k], ann).save(out / "previews" / f"{bi:06d}.jpg", quality=85)
                n += 1
                counts["cards"] += sum(1 for c in ann["cards"] if c["kind"] == "card")
                counts["card_backs"] += sum(1 for c in ann["cards"] if c["kind"] == "card_back")
                counts["covered"] += sum(1 for c in ann["cards"] if 0 < c["visible"] < 0.9 and not c["truncated"])
                counts["occluders"] += len(ann["occluders"])
            print(f"  {n}/{a.boards} boards ({w}x{h}, {br} kbps) in {time.time() - t0:.0f} s", flush=True)
    manifest = {"schema": "rifteye.synth", "version": 1, "boards": n, "seed": a.seed, "catalog": a.catalog or "fixtures",
                "catalog_key": key, "codec": not a.no_codec, "args": vars(a), "counts": counts,
                "stream": {k: v for k, v in asdict(StreamSettings()).items() if k in ("fps", "preset", "noise_sigma")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{n} boards -> {out} ({counts['cards']} face-up, {counts['card_backs']} face-down, "
          f"{counts['covered']} partly covered cards) in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

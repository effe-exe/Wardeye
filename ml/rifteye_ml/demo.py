# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""A demo bundle for apps/viewer: a clip where every recognised card can be hovered, plus
the timeline the change gate and the identifier produce together.

It runs the M0 pipeline offline on a window of a VOD: the bootstrap mat detector a few times
a second, the colour-grid identifier with a gallery pyramid, tracks of one physical card,
and, from a change-gate run over the same window, events that name the card involved.
Face-down cards are shown as face-down, never identified (D-005).

This is a preview of the extension's behaviour on a recorded match, not the live product.
The bundle holds broadcast footage and card art, so it is private (D-006, D-015).

    python -m rifteye_ml.demo --video seg.mp4 --start 90 --duration 120 --table 0.17,0.09,0.86,0.884 \\
        --long 131 --catalog catalog.jsonl --cache art --embed-cache embed-cache \\
        --gate changegate/r11g1-b.json --title "Swiss round 11, game 1" --out demo/r11g1
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image

from .reviewpack import card_scores, fit_long, link_tracks, softmax, view_to_frame

TEMPERATURE = 0.0212  # fitted on the M0 real labels (reviewpack identity)


def frames(video: str, start: float, duration: float, fps: float, ffmpeg: str | None = None) -> Iterator[tuple[float, Image.Image]]:
    """Full-resolution RGB frames at `fps`, with their time from `start`."""
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", video], capture_output=True, text=True).stderr
    size = re.search(r"Video: .*?, (\d{2,5})x(\d{2,5})", probe)
    if not size:
        raise ValueError(f"no video stream in {video}")
    w, h = int(size.group(1)), int(size.group(2))
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(start), "-t", str(duration), "-i", video,
           "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    k = 0
    while True:
        buf = proc.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        yield k / fps, Image.frombytes("RGB", (w, h), buf)
        k += 1
    proc.wait()


def looks_face_down(crop: Image.Image) -> bool:
    """A plain sleeve back: nearly one colour over the whole card. Card faces never are."""
    a = np.asarray(crop.convert("RGB").resize((24, 32)), np.float32)
    inner = a[4:-4, 4:-4].reshape(-1, 3)
    return float(inner.std(axis=0).mean()) < 14.0


def corners(box: dict) -> list[tuple[float, float]]:
    """The rotated rectangle's corners (frame px), from matcrops' centre, sides and long-side angle."""
    cx, cy = box["centre"]
    a = math.radians(box["angle_deg"])
    ux, uy = math.cos(a) * box["long_px"] / 2, math.sin(a) * box["long_px"] / 2
    vx, vy = -math.sin(a) * box["short_px"] / 2, math.cos(a) * box["short_px"] / 2
    return [(cx + ux + vx, cy + uy + vy), (cx + ux - vx, cy + uy - vy), (cx - ux - vx, cy - uy - vy), (cx - ux + vx, cy - uy + vy)]


def attach_events(events: Sequence[dict], tracks: Sequence[dict], window: float = 3.0, grow: float = 0.3) -> list[dict]:
    """Name the card in each change-gate event: the track seen inside the event's box just after
    the change (a card put down or changed) or just before it (a card taken away). Events with no
    recognised card are dropped, as the event engine would drop them."""
    out = []
    for e in events:
        x0, y0, x1, y1 = e["box"]  # frame fractions
        gx, gy = (x1 - x0) * grow, (y1 - y0) * grow
        lo, hi = (e["t"] - window - 2, e["t"]) if e["kind"] == "disappeared" else (e["t"], e["t"] + window)
        best, best_n = None, 0
        for tr in tracks:
            if tr.get("faceDown"):
                continue
            n = sum(1 for s in tr["samples"] if lo <= s[0] <= hi and x0 - gx <= s[1] <= x1 + gx and y0 - gy <= s[2] <= y1 + gy)
            if n > best_n:
                best, best_n = tr, n
        if best is None:
            continue
        kind = {"appeared": "played", "disappeared": "left", "changed": "changed"}[e["kind"]]
        if out and out[-1]["track"] == best["id"] and out[-1]["kind"] == kind and e["t"] - out[-1]["t"] < 5:
            continue  # the same change reported twice
        out.append({"t": round(e["t"], 2), "tBefore": round(e.get("tBefore", e["t"]), 2), "kind": kind,
                    "track": best["id"], "box": [round(v, 4) for v in e["box"]]})
    return out


def grab(video: str, t: float, ffmpeg: str | None = None) -> Image.Image:
    from .reviewpack import grab_frame

    return grab_frame(video, t, ffmpeg)


def box_view(frame: Image.Image, box: Sequence[float], pad: float = 0.08) -> Image.Image:
    """The region of a change-gate box (frame fractions), upright: cards stand portrait."""
    W, H = frame.size
    x0, y0, x1, y1 = box
    gx, gy = (x1 - x0) * pad, (y1 - y0) * pad
    im = frame.crop((round((x0 - gx) * W), round((y0 - gy) * H), round((x1 + gx) * W), round((y1 + gy) * H)))
    return im.rotate(90, expand=True) if im.width > im.height * 1.15 else im


def main(argv: Sequence[str] | None = None) -> int:
    from . import catalog as cat
    from .encoders import get_encoder
    from .matcrops import find_cards, upright_crop
    from .retrieval import search
    from .spike import _cached_loader, _gallery, _ints, catalog_key

    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.demo", description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True, help="the full-resolution source")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--fps", type=float, default=3.0, help="detection rate")
    ap.add_argument("--table", default="0,0,1,1", help="x0,y0,x1,y1 fractions: the mat area free of overlays")
    ap.add_argument("--long", type=float, required=True, help="a card's long side in source px")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--embed-cache")
    ap.add_argument("--encoder", default="colorgrid/trim0.03")
    ap.add_argument("--gallery-scales", default="120,140")
    ap.add_argument("--gate", help="a rifteye_ml.changegate --out file for the same window (its start must match)")
    ap.add_argument("--min-samples", type=int, default=3, help="drop tracks seen fewer times (passing hands, glare)")
    ap.add_argument("--box-min-p", type=float, default=0.6, help="confidence needed to name an event's card from its box")
    ap.add_argument("--title", default="")
    ap.add_argument("--clip", default="clip.mp4", help="the clip's file name inside the bundle (made separately)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    out = Path(a.out)
    (out / "art").mkdir(parents=True, exist_ok=True)
    rows = [r for r in cat.read_catalog(a.catalog) if cat.cache_path(a.cache, r["image_url"]).exists()]
    load = _cached_loader(a.cache, 512)
    art = [load(r) for r in rows]
    enc = get_encoder(a.encoder)
    gallery = _gallery(enc, art, _ints(a.gallery_scales), Path(a.embed_cache) if a.embed_cache else None,
                       key=catalog_key(rows, 512))

    fx0, fy0, fx1, fy1 = (float(v) for v in a.table.split(","))
    meta, crops = [], []
    W = H = 0
    for t, frame in frames(a.video, a.start, a.duration, a.fps):
        W, H = frame.size
        x0, y0 = round(fx0 * W), round(fy0 * H)
        roi = np.asarray(frame)[y0:round(fy1 * H), x0:round(fx1 * W)]
        for b in find_cards(roi, a.long):
            b.centre = (b.centre[0] + x0, b.centre[1] + y0)
            crops.append(upright_crop(frame, b))
            meta.append({"t": t, "segment": "clip", "centre": b.centre, "long_px": b.long_px,
                         "short_px": b.short_px, "angle_deg": b.angle_deg})
    print(f"{len(crops)} card sightings in {a.duration:.0f} s at {a.fps} fps")
    idx, scores, _ = search(enc, gallery, crops, k=min(60, len(rows)), rotation_invariant=True)
    e0 = enc.embed(crops)
    e180 = enc.embed([c.rotate(180) for c in crops])
    sim = np.maximum(e0 @ e0.T, e0 @ e180.T)
    groups = link_tracks(meta, sim, max_gap=6.0, min_sim=0.9)

    per_crop = [card_scores(idx[n], scores[n], rows) for n in range(len(crops))]
    floor = scores[:, -1]
    first_row = {}
    for i, r in enumerate(rows):
        first_row.setdefault(r["card_id"], i)
    cards_used: dict[str, dict] = {}

    def card_entry(row_i: int) -> str:
        r = rows[row_i]
        pid = r["printing_id"]
        if pid not in cards_used:
            path = f"art/{pid}.jpg"
            fit_long(art[row_i], 360).convert("RGB").save(out / path, "JPEG", quality=84)
            cards_used[pid] = {"name": r["name"], "type": r.get("type", ""), "art": path, "printing": pid}
        return pid

    tracks = []
    for g in groups:
        if len(g) < a.min_samples:
            continue
        samples = [[round(meta[n]["t"], 2), round(meta[n]["centre"][0] / W, 4), round(meta[n]["centre"][1] / H, 4),
                    round(meta[n]["long_px"] / H, 4), round(meta[n]["short_px"] / H, 4), round(meta[n]["angle_deg"], 1)]
                   for n in g]
        tr: dict = {"id": f"k{len(tracks):03d}", "samples": samples}
        down = sum(looks_face_down(crops[n]) for n in g)
        if down * 2 > len(g):
            tr["faceDown"] = True  # never identified: hidden information
        else:
            cids = sorted({c for n in g for c in per_crop[n]})
            mat = np.array([[per_crop[n][c][0] if c in per_crop[n] else floor[n] for c in cids] for n in g])
            prob = softmax(mat, TEMPERATURE).mean(axis=0)
            order = np.argsort(-prob)[:3]
            # Show each card's best-matching printing (alt arts look different).
            best_row = {c: max((per_crop[n][c] for n in g if c in per_crop[n]), default=(0.0, first_row[c]))[1] for c in cids}
            tr["guesses"] = [{"card": card_entry(best_row[cids[j]]), "p": round(float(prob[j]), 3)} for j in order]
        tracks.append(tr)

    events = []
    if a.gate:
        run = json.loads(Path(a.gate).read_text(encoding="utf-8"))
        if abs(float(run["start"]) - a.start) > 0.01:
            raise SystemExit(f"the gate run starts at {run['start']} s, the demo at {a.start} s")
        s = run["settings"]
        raw = []
        for e in run["events"]:
            if e["kind"] == "cut" or e["t"] > a.duration:
                continue
            bx = view_to_frame(e["box"], run["table"], s["width"], W, H)
            raw.append({"t": e["t"], "tBefore": e.get("extra", {}).get("t_before", e["t"]), "kind": e["kind"],
                        "box": [bx[0] / W, bx[1] / H, bx[2] / W, bx[3] / H]})
        events = attach_events(raw, tracks)
        # Events whose card no track holds (the bootstrap detector sees isolated cards only): identify
        # the gate's own box after the change, as the pipeline does, and keep only confident answers.
        named = {(round(e["t"], 2), e["kind"]) for e in events}
        kinds = {"appeared": "played", "changed": "changed"}
        extra = []
        for e in raw:
            if e["kind"] not in kinds or (round(e["t"], 2), kinds[e["kind"]]) in named:
                continue
            view = box_view(grab(a.video, a.start + e["t"] + 0.3), e["box"])
            if min(view.size) < a.long * 0.5:
                continue  # smaller than half a card: a counter or a die, not a card
            i2, s2, _ = search(enc, gallery, [view], k=min(60, len(rows)), rotation_invariant=True)
            cs = sorted(card_scores(i2[0], s2[0], rows).items(), key=lambda kv: -kv[1][0])
            prob = softmax(np.array([v[0] for _, v in cs]), TEMPERATURE)
            if prob[0] < a.box_min_p:
                continue
            extra.append({"t": round(e["t"], 2), "tBefore": round(e["tBefore"], 2), "kind": kinds[e["kind"]],
                          "guesses": [{"card": card_entry(cs[j][1][1]), "p": round(float(prob[j]), 3)} for j in range(3)],
                          "box": [round(v, 4) for v in e["box"]]})
        events = sorted(events + extra, key=lambda e: e["t"])
        print(f"{len(raw)} gate events: {len(events) - len(extra)} named by a track, {len(extra)} by their own box")

    bundle = {"schema": "rifteye.demo", "version": 1, "title": a.title, "video": a.clip, "frame": [W, H],
              "detectFps": a.fps, "cards": cards_used, "tracks": tracks, "events": events}
    (out / "data.js").write_text("window.RIFTEYE_DEMO = " + json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + ";\n",
                                 encoding="utf-8")
    n_id = sum(1 for t in tracks if "guesses" in t)
    print(f"{len(tracks)} tracks ({n_id} identified, {len(tracks) - n_id} face-down), {len(cards_used)} card images -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Covered cards: name each one from the part a stack leaves visible, and review the guesses.

The amodal detector gives every card's full quad and, per corner, the chance it shows. In a frame:

1. **Order.** Of two overlapping cards, the one whose corners inside the other still show is on top.
2. **Visible part.** A card minus the cards on top of it, drawn in the card's own upright frame.
3. **Band.** The largest band along one edge that shows (a quarter, 40%, 60% of the card), which is
   what M0 measured on real strips (M0 §7).
4. **Name.** That band against the same band of every gallery card, colour grid + dHash, for both
   ways up the card could lie (the printed top is unknown).

Sightings of one covered card across frames join into a track, and `pack` writes the least sure
tracks as an identity pack for apps/reviewer. `reviewpack apply` turns the answers into labels of
real covered cards, the first real stack labels (M1). Face-down cards are never named.

    python -m rifteye_ml.stacks pack --dets dets.jsonl --catalog catalog.jsonl --cache art \\
        --embed-cache embed-cache --crops-out real-crops/stacks-la --out packs/stacks-la.json
    python -m rifteye_ml.stacks synth-dets --run synth/v0 --out synth-dets.jsonl   # check it on exact truth
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .degrade import _perspective_coeffs
from .detect.geometry import canonical_quad
from .synth.compose import apply_h, homography

FRACTIONS = (0.6, 0.4, 0.25)
TURN_EDGE = {"top": "left", "left": "bottom", "bottom": "right", "right": "top"}  # PIL rotate(90) is counter-clockwise


def inside(p, q: np.ndarray) -> bool:
    """Point inside a convex quad (either winding)."""
    s = [(q[(i + 1) % 4][0] - q[i][0]) * (p[1] - q[i][1]) - (q[(i + 1) % 4][1] - q[i][1]) * (p[0] - q[i][0]) for i in range(4)]
    return all(v >= 0 for v in s) or all(v <= 0 for v in s)


def on_top(a: dict, b: dict) -> bool | None:
    """Whether card a lies on card b, from the corners of each that fall inside the other."""
    qa, qb = np.asarray(a["quad"], float).reshape(4, 2), np.asarray(b["quad"], float).reshape(4, 2)
    va = [a["visible"][k] for k in range(4) if inside(qa[k], qb)]
    vb = [b["visible"][k] for k in range(4) if inside(qb[k], qa)]
    if va and vb:
        return float(np.mean(va)) > float(np.mean(vb))
    if va:
        return float(np.mean(va)) >= 0.5
    if vb:
        return float(np.mean(vb)) < 0.5
    return None  # edges cross with no corner inside: no call


def covers(cards: list[dict]) -> list[list[int]]:
    """For each card, the cards lying on it."""
    boxes = [(np.asarray(c["quad"], float).reshape(4, 2).min(0), np.asarray(c["quad"], float).reshape(4, 2).max(0)) for c in cards]
    out: list[list[int]] = [[] for _ in cards]
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            (lo_i, hi_i), (lo_j, hi_j) = boxes[i], boxes[j]
            if (hi_i < lo_j).any() or (hi_j < lo_i).any():
                continue
            top = on_top(cards[i], cards[j])
            if top is True:
                out[j].append(i)
            elif top is False:
                out[i].append(j)
    return out


def upright_size(q: np.ndarray) -> tuple[int, int]:
    return max(2, round(float(np.linalg.norm(q[1] - q[0])))), max(2, round(float(np.linalg.norm(q[2] - q[1]))))


def rectify(frame: Image.Image, q: np.ndarray) -> Image.Image:
    """The card through its quad, in image order (corner 0 at the top left)."""
    w, h = upright_size(q)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    return frame.transform((w, h), Image.PERSPECTIVE, _perspective_coeffs(dst, q), Image.BICUBIC)


def visible_mask(q: np.ndarray, over: list[np.ndarray], window=None) -> np.ndarray:
    """1 where the card shows, in its rectified frame: the card minus the quads lying on it, and
    minus anything outside the camera `window` (x0, y0, x1, y1), if given."""
    w, h = upright_size(q)
    hmat = homography(q, np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    m = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(m)
    for o in over:
        d.polygon([tuple(p) for p in apply_h(hmat, o)], fill=0)
    mask = np.asarray(m) > 127
    if window is not None:
        x0, y0, x1, y1 = window
        win = Image.new("L", (w, h), 0)
        ImageDraw.Draw(win).polygon([tuple(p) for p in apply_h(hmat, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))], fill=255)
        mask &= np.asarray(win) > 127
    return mask


def best_band(mask: np.ndarray, min_share: float = 0.92) -> tuple[str, float] | None:
    """The largest edge band that shows: ('top', 0.4) etc., ('full', 1.0) if the whole card does,
    None if not even a quarter along any edge."""
    if mask.mean() >= min_share:
        return ("full", 1.0)
    h, w = mask.shape
    for f in FRACTIONS:
        bw, bh = max(1, round(w * f)), max(1, round(h * f))
        cand = {"top": mask[:bh], "bottom": mask[h - bh:], "left": mask[:, :bw], "right": mask[:, w - bw:]}
        best = max(cand, key=lambda e: cand[e].mean())
        if cand[best].mean() >= min_share:
            return (best, f)
    return None


def views_for(crop: Image.Image, band: tuple[str, float]) -> list[tuple[int, Image.Image, str]]:
    """(turn, band image, gallery view) for each way up a portrait card could be: the crop turned by
    `turn` degrees, and the band's edge where it lands on the printed card."""
    edge, f = band
    out = []
    for k in range(4):
        turned = crop.rotate(90 * k, expand=True) if k else crop
        if turned.height < turned.width:
            continue  # a portrait card lies either way up, never sideways once upright
        e = edge
        for _ in range(k):
            e = TURN_EDGE[e] if e != "full" else e
        view = "full" if e == "full" else f"{e}:{f}"
        w, h = turned.size
        if e == "full":
            part = turned
        else:
            bw, bh = max(2, round(w * f)), max(2, round(h * f))
            part = turned.crop({"top": (0, 0, w, bh), "bottom": (0, h - bh, w, h), "left": (0, 0, bw, h), "right": (w - bw, 0, w, h)}[e])
        out.append((90 * k, part, view))
    return out


def frame_meta(path: str, fallback: int) -> tuple[str, float]:
    """(segment, seconds) from a VOD frame name; synthetic frames each get a segment of their own."""
    from .reviewpack import frame_time

    p = Path(path)
    try:
        return p.parent.name, frame_time(p.name)
    except ValueError:
        return f"{p.parent.name}/{p.stem}", float(fallback)


def build_pack(a: argparse.Namespace) -> int:
    from . import catalog as cat
    from .encoders import get_encoder
    from .retrieval import search
    from .reviewpack import (card_scores, context_view, data_uri, fit_long, hms, link_tracks, option_label,
                             pack_doc, review_order, softmax, _write)
    from .spike import _cached_loader, _gallery, _ints, catalog_key

    rows = []
    for c in a.catalog:
        rows += [r for r in cat.read_catalog(c) if cat.cache_path(a.cache, r["image_url"]).exists()]
    seen: set[str] = set()
    rows = [r for r in rows if not (r["image_url"] in seen or seen.add(r["image_url"]))]
    names = {r["printing_id"]: r["name"] for r in rows}
    load = _cached_loader(a.cache, 512)
    art = [load(r) for r in rows]
    enc = get_encoder(a.encoder)
    cache = Path(a.embed_cache) if a.embed_cache else None
    key = catalog_key(rows, 512)
    galleries: dict[str, object] = {}

    def gallery(view: str):
        if view not in galleries:
            galleries[view] = _gallery(enc, art, _ints(a.gallery_scales), cache, key, view=view)
        return galleries[view]

    crops_out = Path(a.crops_out)
    crops_out.mkdir(parents=True, exist_ok=True)
    meta, crops, ranked, truth = [], [], [], []
    frame_no = 0
    for path in a.dets:
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f]
        for rec in recs:
            frame_no += 1
            cards = [c for c in rec["cards"] if c["cls"] == "card" and c.get("score", 1.0) >= a.threshold]
            if not cards:
                continue
            over = covers(cards)
            frame = None
            segment, t = frame_meta(rec["frame"], frame_no)
            for i, c in enumerate(cards):
                q = canonical_quad(c["quad"])
                win = rec.get("window")
                cut = win is not None and not all(win[0] <= x <= win[2] and win[1] <= y <= win[3] for x, y in q)
                if not over[i] and not cut and not a.include_whole:
                    continue
                mask = visible_mask(q, [canonical_quad(cards[j]["quad"]) for j in over[i]], rec.get("window"))
                band = best_band(mask)
                if band is None or (band[0] == "full" and not a.include_whole):
                    continue
                if frame is None:
                    frame = Image.open(rec["frame"]).convert("RGB")
                crop = rectify(frame, q)
                best = None
                for turn, part, view in views_for(crop, band):
                    idx, sc, _ = search(enc, gallery(view), [part], k=min(60, len(rows)), rotation_invariant=False)
                    if best is None or sc[0][0] > best[1][0]:
                        best = (idx[0], sc[0], turn, view)
                name = f"s{len(meta):05d}.png"
                crop.save(crops_out / name)
                w, h = crop.size
                long_px = max(w, h)
                meta.append({"file": name, "frame": rec["frame"], "segment": segment, "t": t,
                             "centre": [float(v) for v in q.mean(axis=0)], "long_px": float(long_px),
                             "quad": [round(float(v), 1) for v in q.ravel()], "band": band[0] if band[0] == "full" else f"{band[0]}:{band[1]}",
                             "view": best[3], "turn": best[2], "shown": round(float(mask.mean()), 3),
                             "covers": [[round(float(v), 1) for v in canonical_quad(cards[j]["quad"]).ravel()] for j in over[i]],
                             "window": rec.get("window")})
                crops.append(crop)
                ranked.append(card_scores(best[0], best[1], rows))
                truth.append(c.get("truth_card"))
    if not meta:
        print("no covered face-up cards in these detections")
        return 1
    (crops_out / "stacks.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{len(meta)} covered sightings; bands: " + ", ".join(f"{k} {v}" for k, v in sorted(
        defaultdict(int, {m['band']: sum(1 for x in meta if x['band'] == m['band']) for m in meta}).items())))

    if any(t is not None for t in truth):  # synthetic check: the exact answer is known
        by_band: dict[str, list[bool]] = defaultdict(list)
        for m, r, tc in zip(meta, ranked, truth):
            if tc:
                top = max(r.items(), key=lambda kv: kv[1][0])[0]
                edge, _, frac = m["view"].partition(":")  # the band as printed: top/bottom are the card's ends
                by_band[f"{'end' if edge in ('top', 'bottom') else 'side' if edge != 'full' else 'whole'} {frac or '1'}"].append(top == tc)
        print("  named right (exact truth): " + "  ".join(f"{b}: {np.mean(v):.1%} (n={len(v)})" for b, v in sorted(by_band.items())))

    e0 = enc.embed(crops)
    e180 = enc.embed([c.rotate(180) for c in crops])
    sim = np.maximum(e0 @ e0.T, e0 @ e180.T)
    tracks = link_tracks(meta, sim, max_gap=a.max_gap, max_dist=a.max_dist, max_dlong=a.max_dlong, min_sim=a.min_sim)
    floor = [min(v[0] for v in r.values()) for r in ranked]
    items_all = []
    for t_i, members in enumerate(tracks):
        cards_ = sorted({c for n in members for c in ranked[n]})
        mat = np.array([[ranked[n][c][0] if c in ranked[n] else floor[n] for c in cards_] for n in members])
        prob = softmax(mat, a.temperature).mean(axis=0)
        order = np.argsort(-prob)
        top = [cards_[j] for j in order[: a.k + 1]]
        rep = max(members, key=lambda n: (meta[n]["shown"], ranked[n].get(top[0], (-1.0, 0))[0]))
        items_all.append({"track": f"k{t_i:04d}", "members": members, "rep": rep, "cards": top, "confidence": float(prob[order[0]])})
    order, audit = review_order([it["confidence"] for it in items_all], a.max_items, a.audit, a.seed)
    print(f"{len(tracks)} tracks; reviewing {len(order)} ({len(audit)} audit)")

    files, items, sidecar, frames = {}, [], {}, {}
    for pos in order:
        it = items_all[pos]
        n = it["rep"]
        m = meta[n]
        opts = []
        for c in it["cards"]:
            cands = [ranked[j][c] for j in it["members"] if c in ranked[j]]
            r = rows[max(cands)[1]]
            p = f"art/{r['printing_id']}-{r.get('language', 'en')}.jpg"
            if p not in files:
                files[p] = data_uri(fit_long(art[max(cands)[1]], a.art_px), a.quality)
            opts.append({"value": r["printing_id"], "label": option_label(r, names), "image": p})
        # The crop the way up it matched, the covered part dimmed.
        mask = visible_mask(np.asarray(m["quad"]).reshape(4, 2), [np.asarray(o).reshape(4, 2) for o in m["covers"]], m.get("window"))
        shade = Image.fromarray(np.where(mask, 255, 90).astype(np.uint8))
        crop = Image.composite(crops[n], Image.new("RGB", crops[n].size, (20, 20, 28)), shade)
        crop = crop.rotate(m["turn"], expand=True) if m["turn"] else crop
        files[f"crop/{it['track']}.jpg"] = data_uri(fit_long(crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS), 360), a.quality + 6)
        images = [f"crop/{it['track']}.jpg"]
        if m["frame"] not in frames:
            frames.clear()
            frames[m["frame"]] = Image.open(m["frame"]).convert("RGB")
        fr = frames[m["frame"]]
        q = np.asarray(m["quad"]).reshape(4, 2)
        cx, cy = q.mean(axis=0)
        lng = m["long_px"] * 2.2
        view = fr.crop((round(cx - lng), round(cy - lng), round(cx + lng), round(cy + lng))).copy()
        d = ImageDraw.Draw(view)
        d.polygon([(x - cx + lng, y - cy + lng) for x, y in q], outline=(138, 127, 240), width=3)
        files[f"context/{it['track']}.jpg"] = data_uri(fit_long(view, 420), a.quality)
        images.append(f"context/{it['track']}.jpg")
        span = meta[it["members"][-1]]["t"] - meta[it["members"][0]]["t"]
        when = "" if "/" in m["segment"] else f"VOD {hms(m['t'])} · "  # synthetic frames have no VOD time
        items.append({"id": it["track"], "images": images, "proposal": opts[0], "confidence": round(it["confidence"], 4),
                      "alternatives": opts[1:],
                      "note": f"{when}covered card, {m['band'].replace(':', ' ')} showing · seen in {len(it['members'])} frame(s)"
                              + (f" over {hms(span)[2:]}" if len(it["members"]) > 1 else "")})
        sidecar[it["track"]] = {"files": [meta[j]["file"] for j in it["members"]], "rep": m["file"], "audit": pos in audit,
                                "confidence": it["confidence"],
                                "proposal": {"printing_id": opts[0]["value"], "card_id": it["cards"][0]},
                                "alternatives": [{"printing_id": o["value"], "card_id": c} for o, c in zip(opts[1:], it["cards"][1:])]}
    vocab = {}
    for r in rows:
        vocab.setdefault(r["printing_id"], r)
    vocabulary = sorted(({"value": pid, "label": option_label(r, names)} for pid, r in vocab.items()), key=lambda o: o["label"])
    pack_id = a.id or Path(a.out).name.removesuffix(".json")
    doc = pack_doc(pack_id, "identity", "Is the covered card this one?", items, files, vocabulary)
    _write(a.out, doc, {"pack": pack_id, "kind": "identity", "encoder": enc.name, "temperature": a.temperature,
                        "catalog": a.catalog, "crops": str(crops_out), "card_ids": {r["printing_id"]: r["card_id"] for r in rows},
                        "tracks": {it["track"]: [meta[j]["file"] for j in it["members"]] for it in items_all}, "items": sidecar})
    return 0


def synth_dets(a: argparse.Namespace) -> int:
    """Detections from a synthetic run's exact annotations, with the true card, to check `pack` against."""
    run = Path(a.run)
    n = 0
    with open(run / "annotations.jsonl", encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as out:
        for line in f:
            ann = json.loads(line)
            if a.max_frames and n >= a.max_frames:
                break
            with Image.open(run / ann["ids"]) as im:
                ids = np.asarray(im).astype(np.int32)
            cards = []
            for c in ann["cards"]:
                if c["visible"] < 0.08:
                    continue
                q = canonical_quad(c["quad"])
                centre = q.mean(axis=0)
                vis = []
                for corner in q:
                    ix, iy = np.round(corner + 0.1 * (centre - corner)).astype(int)
                    ok = 0 <= iy < ids.shape[0] and 0 <= ix < ids.shape[1] and ids[iy, ix] == c["id"] + 1
                    vis.append(1.0 if ok else 0.0)
                d = {"cls": c["kind"], "score": 1.0, "quad": [round(float(v), 1) for v in q.ravel()], "found": [1.0] * 4, "visible": vis}
                if c["kind"] == "card":
                    d["truth_card"] = c.get("card_id")
                cards.append(d)
            out.write(json.dumps({"frame": str(run / ann["image"]), "width": ann["width"], "height": ann["height"],
                                  "window": ann["window"], "cards": cards}) + "\n")
            n += 1
    print(f"{n} frames -> {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.stacks", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="detections -> a review pack of covered cards")
    p.add_argument("--dets", nargs="+", required=True, help="JSONL from `python -m rifteye_ml.detect run`")
    p.add_argument("--catalog", required=True, action="append")
    p.add_argument("--cache", required=True)
    p.add_argument("--embed-cache")
    p.add_argument("--encoder", default="colorgrid/trim0.03+dhash/trim0.03")
    p.add_argument("--gallery-scales", default="120,140,160")
    p.add_argument("--threshold", type=float, default=0.5, help="detection score to keep")
    p.add_argument("--include-whole", action="store_true", help="also name cards nothing covers")
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--max-gap", type=float, default=30.0)
    p.add_argument("--max-dist", type=float, default=15.0)
    p.add_argument("--max-dlong", type=float, default=8.0)
    p.add_argument("--min-sim", type=float, default=0.85)
    p.add_argument("--max-items", type=int, default=300)
    p.add_argument("--audit", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--art-px", type=int, default=320)
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--crops-out", required=True, help="where the rectified crops and stacks.json go (private)")
    p.add_argument("--id")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=build_pack)
    s = sub.add_parser("synth-dets", help="exact detections from a synthetic run, with the true cards")
    s.add_argument("--run", required=True)
    s.add_argument("--max-frames", type=int)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=synth_dets)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

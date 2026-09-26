# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Review packs: the model proposes, a person says correct or wrong.

Naming thousands of crops by hand is slow. Confirming a guess takes about two seconds.
`identity` turns crops into a pack of the model's guesses for apps/reviewer. Each item is one
*track*: the same physical card seen in consecutive frames, so one answer labels every crop
of it. Items are ordered least confident first. A random share of the confident rest is mixed
in as an audit, which measures how often the guesses nobody checks are right.
`events` builds the same kind of pack from change-gate events. `apply` turns the exported
answers into labels.

Packs, their sidecars and answers are private: they hold broadcast crops and card art
(decisions D-006 and D-015). Only accuracy numbers leave the machine.

    python -m rifteye_ml.reviewpack identity --crops real-crops/v2/crops --catalog catalog.jsonl \\
        --cache art --embed-cache embed-cache --labels real-crops/v1/labels.csv --out packs/identity-v2.json
    python -m rifteye_ml.reviewpack events --events changegate/r11g1-b.json --video seg.mp4 --out packs/events.json
    python -m rifteye_ml.reviewpack apply --pack packs/identity-v2.json --answers identity-v2.answers.json \\
        --out real-crops/v2/labels-review.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import random
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

SCHEMA_VERSION = 1
ACCENT = (138, 127, 240)
FRAME_NAME = re.compile(r"(\d+)h(\d+)m(\d+)s")


# ---------------------------------------------------------------------------------------
# Small pure helpers (unit-tested)
# ---------------------------------------------------------------------------------------

def frame_time(name: str) -> float:
    """Seconds into the VOD from a name such as 't03h20m40s_02.png' or 'seg-03h19m00s-...' (first match)."""
    m = FRAME_NAME.search(name)
    if not m:
        raise ValueError(f"no HHhMMmSSs time in {name!r}")
    h, mi, s = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s


def hms(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"


def softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = (x - x.max(axis=-1, keepdims=True)) / temperature
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def card_scores(idx_row: Sequence[int], score_row: Sequence[float], rows: Sequence[dict]) -> dict[str, tuple[float, int]]:
    """Best score and gallery row per card, from one query's ranked gallery rows."""
    out: dict[str, tuple[float, int]] = {}
    for i, sc in zip(idx_row, score_row):
        cid = rows[int(i)]["card_id"]
        if cid not in out or sc > out[cid][0]:
            out[cid] = (float(sc), int(i))
    return out


def fit_temperature(score_lists: Sequence[np.ndarray], truth_pos: Sequence[int],
                    grid: Sequence[float] = tuple(np.geomspace(0.001, 0.2, 60))) -> float:
    """The softmax temperature that best explains labelled crops (maximum likelihood).
    `score_lists[n]` are crop n's distinct-card scores, `truth_pos[n]` the true card's position."""
    best, best_ll = grid[0], -math.inf
    for t in grid:
        ll = sum(math.log(max(1e-12, float(softmax(s, t)[p]))) for s, p in zip(score_lists, truth_pos))
        if ll > best_ll:
            best, best_ll = t, ll
    return float(best)


class _Union:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def link_tracks(meta: Sequence[dict], sim: np.ndarray, max_gap: float = 30.0, max_dist: float = 15.0,
                max_dlong: float = 6.0, min_sim: float = 0.9) -> list[list[int]]:
    """Group crops of one physical card: same segment, a later frame within `max_gap` seconds,
    centre within `max_dist` px, long side within `max_dlong` px, and crop-to-crop similarity
    at least `min_sim` (so a new card played on the same spot starts a new track).
    `meta[i]` needs 'segment', 't', 'centre' and 'long_px'; `sim` is crops × crops.
    Returns tracks as lists of crop indices, each in time order, tracks in order of first sight."""
    n = len(meta)
    u = _Union(n)
    by_seg: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        by_seg[m["segment"]].append(i)
    for members in by_seg.values():
        members.sort(key=lambda i: meta[i]["t"])
        for a_pos, i in enumerate(members):
            best, best_sim = None, min_sim
            for j in reversed(members[:a_pos]):  # earlier crops, nearest in time first
                dt = meta[i]["t"] - meta[j]["t"]
                if dt > max_gap:
                    break
                if dt <= 0:
                    continue  # same frame: different cards
                if math.dist(meta[i]["centre"], meta[j]["centre"]) > max_dist:
                    continue
                if abs(meta[i]["long_px"] - meta[j]["long_px"]) > max_dlong:
                    continue
                if sim[i, j] >= best_sim:
                    best, best_sim = j, float(sim[i, j])
            if best is not None:
                u.join(i, best)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in sorted(range(n), key=lambda i: (meta[i]["t"], i)):
        groups[u.find(i)].append(i)
    return sorted(groups.values(), key=lambda g: (meta[g[0]]["t"], g[0]))


def review_order(confidence: Sequence[float], n: int, audit: float, seed: int = 0) -> tuple[list[int], set[int]]:
    """Which items to review, in order: the least confident first, with a seeded random `audit`
    share of the rest spread evenly through the list. Returns (order, audit members)."""
    total = len(confidence)
    by_conf = sorted(range(total), key=lambda i: (confidence[i], i))
    if total <= n:
        return by_conf, set()
    n_audit = round(n * audit)
    uncertain = by_conf[: n - n_audit]
    rest = by_conf[n - n_audit:]
    picked = random.Random(seed).sample(rest, n_audit)
    order = list(uncertain)
    for k, i in enumerate(picked):  # spread: one audit item every len/n_audit places
        order.insert(min(len(order), round((k + 1) * len(order) / (n_audit + 1)) + k), i)
    return order, set(picked)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def data_uri(im: Image.Image, quality: int = 82) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def fit_long(im: Image.Image, long_side: int) -> Image.Image:
    s = long_side / max(im.size)
    if s >= 1:
        return im
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)


def pack_doc(pack_id: str, kind: str, question: str, items: list[dict], files: dict[str, str],
             vocabulary: list[dict] | None = None) -> dict:
    doc = {"schema": "rifteye.reviewpack", "version": SCHEMA_VERSION, "id": pack_id, "kind": kind,
           "question": question, "items": items}
    if vocabulary:
        doc["vocabulary"] = vocabulary
    doc["files"] = files
    doc["createdAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return doc


# ---------------------------------------------------------------------------------------
# Identity packs
# ---------------------------------------------------------------------------------------

def option_label(row: dict, names: dict[str, str]) -> str:
    """'Ahri, Alluring · OGN-066', plus the variant and language when they are not the default."""
    bits = [names.get(row["printing_id"], row["name"]), row["printing_id"]]
    if row.get("variant", "standard") != "standard":
        bits.append(row["variant"].replace("_", " "))
    if row.get("language", "en") != "en":
        bits.append(row["language"])
    return " · ".join(bits)


def context_view(frame: Image.Image, crop: dict, rotation: int, scale: float = 2.6, long_side: int = 360) -> Image.Image:
    """The card's surroundings, upright like the crop, with the card outlined."""
    cx, cy = crop["centre"]
    rot = frame.rotate(crop["angle_deg"] - 90, resample=Image.BICUBIC, center=(cx, cy))
    w, h = crop["short_px"] * scale, crop["long_px"] * scale
    view = rot.crop((round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2)))
    d = ImageDraw.Draw(view)
    bw, bh = crop["short_px"], crop["long_px"]
    d.rectangle(((w - bw) / 2 - 3, (h - bh) / 2 - 3, (w + bw) / 2 + 3, (h + bh) / 2 + 3), outline=ACCENT, width=2)
    if rotation:
        view = view.rotate(rotation, expand=True)
    return fit_long(view, long_side)


def build_identity(a: argparse.Namespace) -> int:
    from . import catalog as cat
    from .encoders import get_encoder
    from .retrieval import search
    from .spike import _cached_loader, _gallery, _ints, catalog_key

    crops_dir = Path(a.crops)
    meta = json.loads((crops_dir / "crops.json").read_text(encoding="utf-8"))
    for m in meta:
        m["segment"] = Path(m["frame"]).parent.name
        m["t"] = frame_time(m["file"])
    rows: list[dict] = []
    for c in a.catalog:
        rows += [r for r in cat.read_catalog(c) if cat.cache_path(a.cache, r["image_url"]).exists()]
    seen: set[str] = set()
    rows = [r for r in rows if not (r["image_url"] in seen or seen.add(r["image_url"]))]
    names = {}
    for r in rows:  # English names where there are any
        if r.get("language") == "en" or r["printing_id"] not in names:
            names[r["printing_id"]] = r["name"]
    load = _cached_loader(a.cache, 512)
    art = [load(r) for r in rows]
    crops = [Image.open(crops_dir / m["file"]).convert("RGB") for m in meta]
    enc = get_encoder(a.encoder)
    gallery = _gallery(enc, art, _ints(a.gallery_scales), Path(a.embed_cache) if a.embed_cache else None,
                       key=catalog_key(rows, 512))
    k_search = min(60, len(rows))
    idx, scores, rots = search(enc, gallery, crops, k=k_search, rotation_invariant=True)
    print(f"{len(crops)} crops, {len(rows)} gallery rows, encoder {enc.name}")

    # Crop-to-crop similarity for tracking, allowing a half turn (an exhausted card turns in place).
    e0 = enc.embed(crops)
    e180 = enc.embed([c.rotate(180) for c in crops])
    sim = np.maximum(e0 @ e0.T, e0 @ e180.T)
    tracks = link_tracks(meta, sim, max_gap=a.max_gap, min_sim=a.min_sim)

    per_crop = [card_scores(idx[n], scores[n], rows) for n in range(len(crops))]
    floor = scores[:, -1]  # a card missing from a crop's top list scores at most its last entry

    labels: dict[str, str] = {}
    if a.labels:
        for r in csv.DictReader(open(a.labels, encoding="utf-8")):
            if r.get("card_id") and r["card_id"] not in ("", "?", "none"):
                labels[r["file"]] = r["card_id"]
    file_pos = {m["file"]: n for n, m in enumerate(meta)}
    temperature = a.temperature
    if temperature <= 0:
        lists, pos = [], []
        for f, cid in labels.items():
            n = file_pos.get(f)
            if n is None or cid not in per_crop[n]:
                continue
            ranked = sorted(per_crop[n].items(), key=lambda kv: -kv[1][0])
            lists.append(np.array([v[0] for _, v in ranked]))
            pos.append([c for c, _ in ranked].index(cid))
        temperature = fit_temperature(lists, pos) if lists else 0.01
        print(f"temperature {temperature:.4f} fitted on {len(lists)} labelled crops" if lists else "no labels: temperature 0.01")

    items_all = []
    for t_i, members in enumerate(tracks):
        cards = sorted({c for n in members for c in per_crop[n]})
        mat = np.array([[per_crop[n][c][0] if c in per_crop[n] else floor[n] for c in cards] for n in members])
        prob = softmax(mat, temperature).mean(axis=0)
        order = np.argsort(-prob)
        top = [cards[j] for j in order[: a.k + 1]]
        # The printing and view to show: the member crop that scores the top card best.
        best_n = max(members, key=lambda n: per_crop[n].get(top[0], (-1.0, 0))[0])
        items_all.append({
            "track": f"t{t_i:04d}", "members": members, "rep": best_n, "cards": top,
            "confidence": float(prob[order[0]]), "labelled": any(meta[n]["file"] in labels for n in members),
        })
    todo = [it for it in items_all if not it["labelled"]] if not a.include_labelled else items_all
    order, audit = review_order([it["confidence"] for it in todo], a.max_items, a.audit, a.seed)
    print(f"{len(tracks)} tracks ({len(items_all) - len(todo)} already labelled); "
          f"reviewing {len(order)} ({len(audit)} audit)")

    files: dict[str, str] = {}
    frames: dict[str, Image.Image] = {}
    items, sidecar = [], {}
    for pos_i in order:
        it = todo[pos_i]
        n = it["rep"]
        m = meta[n]
        opts = []
        for c in it["cards"]:
            sc_row = per_crop[n].get(c)
            if sc_row is None:  # not in the representative's list: take the best member's row
                cands = [per_crop[j][c] for j in it["members"] if c in per_crop[j]]
                sc_row = max(cands)
            r = rows[sc_row[1]]
            path = f"art/{r['printing_id']}-{r.get('language', 'en')}.jpg"
            if path not in files:
                files[path] = data_uri(fit_long(art[sc_row[1]], a.art_px), a.quality)
            opts.append({"value": r["printing_id"], "label": option_label(r, names), "image": path})
        # Show the crop turned the way it matched the proposal.
        r0 = int(rots[n][list(idx[n]).index(per_crop[n][it["cards"][0]][1])]) if it["cards"][0] in per_crop[n] else 0
        crop_path = f"crop/{it['track']}.jpg"
        shown = crops[n].rotate(r0, expand=True) if r0 else crops[n]
        files[crop_path] = data_uri(shown, a.quality + 6)
        images = [crop_path]
        if a.context:
            fp = m["frame"]
            if fp not in frames:
                frames.clear()  # tracks run in review order, not frame order: keep one frame
                frames[fp] = Image.open(fp).convert("RGB")
            ctx_path = f"context/{it['track']}.jpg"
            files[ctx_path] = data_uri(context_view(frames[fp], m, r0), a.quality)
            images.append(ctx_path)
        span = meta[it["members"][-1]]["t"] - meta[it["members"][0]]["t"]
        seen_in = f"seen in {len(it['members'])} frames over {hms(span)[2:]}" if len(it["members"]) > 1 else "seen in 1 frame"
        items.append({"id": it["track"], "images": images, "proposal": opts[0],
                      "confidence": round(it["confidence"], 4), "alternatives": opts[1:],
                      "note": f"VOD {hms(m['t'])} · {seen_in}"})
        sidecar[it["track"]] = {"files": [meta[j]["file"] for j in it["members"]], "rep": m["file"],
                                "audit": pos_i in audit, "confidence": it["confidence"],
                                "proposal": {"printing_id": opts[0]["value"], "card_id": it["cards"][0]},
                                "alternatives": [{"printing_id": o["value"], "card_id": c}
                                                 for o, c in zip(opts[1:], it["cards"][1:])]}
    vocab_rows = {}
    for r in rows:
        vocab_rows.setdefault(r["printing_id"], r)
    vocabulary = sorted(({"value": pid, "label": option_label(r, names)} for pid, r in vocab_rows.items()),
                        key=lambda o: o["label"])
    pack_id = a.id or Path(a.out).name.removesuffix(".json").removesuffix(".reviewpack")
    doc = pack_doc(pack_id, "identity", "Is this the card?", items, files, vocabulary)
    _write(a.out, doc, {"pack": pack_id, "kind": "identity", "encoder": enc.name, "temperature": temperature,
                        "catalog": a.catalog, "crops": str(crops_dir), "card_ids": {r["printing_id"]: r["card_id"] for r in rows},
                        "tracks": {it["track"]: [meta[j]["file"] for j in it["members"]] for it in items_all},
                        "track_proposals": {it["track"]: it["cards"][0] for it in items_all},
                        "track_confidence": {it["track"]: it["confidence"] for it in items_all},
                        "items": sidecar})
    return 0


def _write(out: str, doc: dict, sidecar: dict) -> None:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    p.write_text(text, encoding="utf-8")
    side = p.with_name(p.name.removesuffix(".json") + ".meta.json")
    side.write_text(json.dumps(sidecar, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(doc['items'])} items, {len(doc['files'])} pictures, {len(text) / 1e6:.1f} MB -> {p} (sidecar {side.name})")


# ---------------------------------------------------------------------------------------
# Event packs (change gate)
# ---------------------------------------------------------------------------------------

EVENT_OPTIONS = {
    "appeared": "A card was put here",
    "disappeared": "A card was taken away from here",
    "changed": "The card here changed (turned, replaced or covered)",
    "none": "No card changed (a hand, shadow, camera or overlay)",
}


def grab_frame(video: str, t: float, ffmpeg: str | None = None) -> Image.Image:
    """One full-resolution frame at `t` seconds (accurate seek)."""
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}", "-i", video,
                          "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(out)).convert("RGB")


def view_to_frame(box: Sequence[int], table: Sequence[float], view_w: int, frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
    """A box in the gate's downscaled table view -> frame pixels (the view keeps the table's aspect)."""
    tx0, ty0, tx1, ty1 = table
    view_h = round(view_w * (ty1 - ty0) * 1080 / ((tx1 - tx0) * 1920) / 2) * 2
    sx = (tx1 - tx0) * frame_w / view_w
    sy = (ty1 - ty0) * frame_h / view_h
    x0, y0, x1, y1 = box
    return (tx0 * frame_w + x0 * sx, ty0 * frame_h + y0 * sy, tx0 * frame_w + x1 * sx, ty0 * frame_h + y1 * sy)


def event_view(frame: Image.Image, box: Sequence[float], min_side: float, long_side: int = 420) -> Image.Image:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(min_side, (x1 - x0) * 2.2, (y1 - y0) * 2.2) / 2
    region = (round(cx - half * 1.2), round(cy - half), round(cx + half * 1.2), round(cy + half))
    view = frame.crop(region)
    d = ImageDraw.Draw(view)
    d.rectangle((x0 - region[0] - 4, y0 - region[1] - 4, x1 - region[0] + 4, y1 - region[1] + 4), outline=ACCENT, width=3)
    return fit_long(view, long_side)


def build_events(a: argparse.Namespace) -> int:
    run = json.loads(Path(a.events).read_text(encoding="utf-8"))
    settings, start, table = run["settings"], float(run["start"]), run["table"]
    m = re.search(r"seg-(\d+)h(\d+)m(\d+)s", Path(a.video).name)
    offset = a.offset if a.offset is not None else (frame_time(m.group(0)) if m else 0.0)
    events = [e for e in run["events"] if e["kind"] != "cut"]
    files, items, sidecar = {}, [], {}
    card = settings["card_long_frac"] * settings["frame_h"]
    for k, e in enumerate(events):
        t_after = start + e["t"]
        t_before = start + e.get("extra", {}).get("t_before", e["t"] - 3.0)
        # t_before is the last moment the region matched the still table; a hand may already be
        # at its edge, so the picture is taken a moment earlier.
        before, after = grab_frame(a.video, t_before - 0.4), grab_frame(a.video, t_after)
        box = view_to_frame(e["box"], table, settings["width"], after.width, after.height)
        item_id = f"e{k:03d}"
        paths = [f"event/{item_id}-before.jpg", f"event/{item_id}-after.jpg"]
        files[paths[0]] = data_uri(event_view(before, box, 3 * card), a.quality)
        files[paths[1]] = data_uri(event_view(after, box, 3 * card), a.quality)
        kind = e["kind"]
        items.append({"id": item_id, "images": paths, "proposal": {"value": kind, "label": EVENT_OPTIONS[kind]},
                      "alternatives": [{"value": v, "label": lab} for v, lab in EVENT_OPTIONS.items() if v != kind],
                      "note": f"before {hms(offset + t_before)} → after {hms(offset + t_after)} (VOD time)"})
        sidecar[item_id] = {"event": k, "t": e["t"], "t_before": t_before - start, "kind": kind, "box": e["box"]}
    pack_id = a.id or Path(a.out).name.removesuffix(".json").removesuffix(".reviewpack")
    doc = pack_doc(pack_id, "event", "What happened in the purple box? (left: before, right: after)", items, files)
    _write(a.out, doc, {"pack": pack_id, "kind": "event", "events": a.events, "video": a.video, "items": sidecar})
    return 0


# ---------------------------------------------------------------------------------------
# Answers -> labels
# ---------------------------------------------------------------------------------------

def apply_answers(sidecar: dict, answers: dict) -> tuple[list[dict], dict[str, float]]:
    """Label rows from a reviewer's answers, and the model's measured accuracy.

    Identity: every crop of an answered track gets the confirmed or corrected printing; 'unsure'
    gives '?', and 'wrong' without a name gives '!' + the rejected card (known wrong, unknown right).
    A name typed for a card the catalogue lacks (some tokens) gives 'none', with the name kept.
    Event: one row per event with the verdict and the right kind."""
    if answers.get("packId") != sidecar["pack"]:
        raise ValueError(f"answers are for pack {answers.get('packId')!r}, not {sidecar['pack']!r}")
    who = f"review:{answers.get('reviewer') or 'anonymous'}"
    out, tally = [], defaultdict(int)
    for ans in answers["answers"]:
        item = sidecar["items"].get(ans["itemId"])
        if item is None:
            continue
        verdict = ans["verdict"]
        group = "audit" if item.get("audit") else "review"
        tally[f"{group}:{verdict}"] += 1
        if sidecar["kind"] == "identity":
            prop = item["proposal"]
            if verdict == "correct":
                pid, cid = prop["printing_id"], prop["card_id"]
            elif verdict == "wrong" and ans.get("value"):
                pid = ans["value"]
                cid = sidecar.get("card_ids", {}).get(pid, "")
                if not cid:
                    cid = next((x["card_id"] for x in item["alternatives"] if x["printing_id"] == pid), "")
            elif verdict == "wrong" and ans.get("text"):
                pid, cid = "none", "none"  # not in the catalogue; the typed name says what it is
            elif verdict == "wrong":
                pid, cid = "!" + prop["printing_id"], "!" + prop["card_id"]
            else:
                pid, cid = "?", "?"
            for f in item["files"]:
                out.append({"file": f, "track": ans["itemId"], "printing_id": pid, "card_id": cid,
                            "verdict": verdict, "name": ans.get("text", ""), "labeled_by": who})
        else:
            right = item["kind"] if verdict == "correct" else ans.get("value", "?" if verdict == "unsure" else "!" + item["kind"])
            out.append({"event": item["event"], "t": item["t"], "t_before": item["t_before"], "proposed": item["kind"],
                        "verdict": verdict, "kind": right, "labeled_by": who})
    stats: dict[str, float] = {}
    for group in ("review", "audit"):
        c, w = tally[f"{group}:correct"], tally[f"{group}:wrong"]
        if c + w:
            lo, hi = wilson(c, c + w)
            stats[f"{group}_n"] = c + w
            stats[f"{group}_correct"] = c / (c + w)
            stats[f"{group}_lo"], stats[f"{group}_hi"] = lo, hi
        stats[f"{group}_unsure"] = tally[f"{group}:unsure"]
    return out, stats


def cmd_apply(a: argparse.Namespace) -> int:
    pack = Path(a.pack)
    sidecar = json.loads(pack.with_name(pack.name.removesuffix(".json") + ".meta.json").read_text(encoding="utf-8"))
    answers = json.loads(Path(a.answers).read_text(encoding="utf-8"))
    rows, stats = apply_answers(sidecar, answers)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} label rows -> {out}")
    for group in ("review", "audit"):
        if f"{group}_n" in stats:
            print(f"  {group}: model correct {stats[f'{group}_correct']:.1%} of {stats[f'{group}_n']:.0f} "
                  f"(95% CI {stats[f'{group}_lo']:.1%}-{stats[f'{group}_hi']:.1%}), {stats[f'{group}_unsure']:.0f} unsure")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.reviewpack", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("identity", help="a pack of card guesses, one item per track")
    p.add_argument("--crops", required=True, help="folder with crops and crops.json (rifteye_ml.matcrops)")
    p.add_argument("--catalog", required=True, action="append")
    p.add_argument("--cache", required=True)
    p.add_argument("--encoder", default="colorgrid/trim0.03")
    p.add_argument("--gallery-scales", default="120")
    p.add_argument("--embed-cache")
    p.add_argument("--labels", help="labels.csv already made: fits the confidence and skips those tracks")
    p.add_argument("--include-labelled", action="store_true", help="review tracks that already have a label too")
    p.add_argument("--temperature", type=float, default=0.0, help="softmax temperature (0 = fit on --labels)")
    p.add_argument("--max-gap", type=float, default=30.0, help="seconds a track may go unseen")
    p.add_argument("--min-sim", type=float, default=0.9, help="crop-to-crop similarity to continue a track")
    p.add_argument("--max-items", type=int, default=400)
    p.add_argument("--audit", type=float, default=0.1, help="share of random confident items")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=3, help="alternatives per item")
    p.add_argument("--art-px", type=int, default=320)
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--no-context", dest="context", action="store_false")
    p.add_argument("--id")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=build_identity)
    p = sub.add_parser("events", help="a pack of change-gate events")
    p.add_argument("--events", required=True, help="rifteye_ml.changegate --out file")
    p.add_argument("--video", required=True, help="the video the gate ran on")
    p.add_argument("--offset", type=float, help="seconds to add for VOD times (default: from a seg-HHhMMmSSs name)")
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--id")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=build_events)
    p = sub.add_parser("apply", help="exported answers -> labels CSV")
    p.add_argument("--pack", required=True)
    p.add_argument("--answers", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_apply)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""Score detections against a synthetic run's exact annotations, frame by frame.

A card counts as found when a detection of the same class overlaps its full quad with IoU >= 0.75.
The usual 0.5 is too loose here: the cards of a rune column or a fanned pile overlap each other with
IoU 0.5-0.7, so at 0.5 one card's detection could be credited to its neighbour. Matching is one-to-one,
best overlaps first. Recall is split by how much of the card shows, which is the question M1 asks:
are covered cards still found?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .export import CLASSES
from .geometry import canonical_quad, quad_iou

BUCKETS = [("whole", 0.95, 1.01), ("half or more", 0.5, 0.95), ("strip", 0.15, 0.5), ("sliver", 0.08, 0.15)]


def targets(ann: dict, min_visible: float = 0.08) -> list[dict]:
    """The cards a detector should find in a synthetic frame: those showing at least `min_visible`."""
    wx0, wy0, wx1, wy1 = ann["window"]
    out = []
    for c in ann["cards"]:
        if c["kind"] not in CLASSES or c["visible"] < min_visible:
            continue
        q = canonical_quad(c["quad"])
        cx, cy = q.mean(axis=0)
        if not (wx0 <= cx <= wx1 and wy0 <= cy <= wy1):
            continue  # its centre is off the camera window: truncated too far to ask for
        out.append({"cls": c["kind"], "quad": q, "visible": c["visible"], "truncated": c["truncated"], "zone": c["zone"],
                    "long": float(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[1] - q[2])))})
    return out


def match(dets: list[dict], gts: list[dict], iou: float = 0.75) -> list[tuple[int, int, float]]:
    """One-to-one (detection, target, IoU) pairs of the same class, best overlaps first."""
    pairs = []
    for i, d in enumerate(dets):
        dq = np.asarray(d["quad"], np.float64).reshape(4, 2)
        for j, g in enumerate(gts):
            if d["cls"] == g["cls"]:
                v = quad_iou(dq, g["quad"])
                if v >= iou:
                    pairs.append((v, i, j))
    used_d, used_g, out = set(), set(), []
    for v, i, j in sorted(pairs, reverse=True):
        if i not in used_d and j not in used_g:
            used_d.add(i); used_g.add(j); out.append((i, j, v))
    return out


def score_frames(frames: list[tuple[list[dict], list[dict]]], thresholds=(0.3, 0.5), iou: float = 0.75) -> list[dict]:
    """Rows of recall, precision and corner error per threshold, overall and per visibility bucket."""
    rows = []
    for thr in thresholds:
        hit_vis, all_vis, n_det, n_tp, errs, cls_gts = [], [], 0, 0, [], []
        for dets, gts in frames:
            dets = [d for d in dets if d["score"] >= thr]
            m = match(dets, gts, iou)
            n_det += len(dets); n_tp += len(m)
            found = {j for _, j, _ in m}
            for j, g in enumerate(gts):
                all_vis.append(g["visible"]); hit_vis.append(j in found); cls_gts.append(g["cls"])
            for i, j, _ in m:
                q = np.asarray(dets[i]["quad"], np.float64).reshape(4, 2)
                errs.append(float(np.linalg.norm(q - gts[j]["quad"], axis=1).mean() / gts[j]["long"]))
        vis, hit, cls_gts = np.array(all_vis), np.array(hit_vis, bool), np.array(cls_gts)
        base = {"threshold": thr, "precision": round(n_tp / max(n_det, 1), 4),
                "corner_error": round(float(np.median(errs)), 4) if errs else None}
        rows.append({**base, "group": "all", "n": int(len(vis)), "recall": round(float(hit.mean()), 4) if len(vis) else None})
        for name, lo, hi in BUCKETS:
            sel = (vis >= lo) & (vis < hi)
            if sel.any():
                rows.append({**base, "group": name, "n": int(sel.sum()), "recall": round(float(hit[sel].mean()), 4)})
        for c in CLASSES:
            sel = cls_gts == c
            if sel.any():
                rows.append({**base, "group": c, "n": int(sel.sum()), "recall": round(float(hit[sel].mean()), 4)})
    return rows


def box_quad(centre, long_px: float, short_px: float, angle_deg: float) -> np.ndarray:
    """The corners of a `matcrops` box (centre, sides, direction of the long side in image degrees)."""
    a = np.deg2rad(angle_deg)
    u, v = np.array([np.cos(a), np.sin(a)]) * long_px / 2, np.array([-np.sin(a), np.cos(a)]) * short_px / 2
    c = np.asarray(centre, np.float64)
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


def score_real(dets_by_frame: dict[str, list[dict]], crops: list[dict], labels: dict[str, str], iou: float = 0.5) -> list[dict]:
    """Recall of the reviewed real cards of M0, found by the bootstrap mat detector (isolated cards only).

    `crops` are `matcrops` entries (file, frame, box), `labels` maps a crop file to its printing id, or
    `back` for a face-down sleeve. Frames are matched by file name. Only the frames the detector ran on count.
    """
    rows, hits = [], {"card": [], "card_back": []}
    right_class = []
    for c in crops:
        lab = labels.get(c["file"])
        dets = dets_by_frame.get(Path(c["frame"]).name)
        if lab in (None, "?") or dets is None:
            continue
        cls = "card_back" if lab == "back" else "card"
        q = box_quad(c["centre"], c["long_px"], c["short_px"], c["angle_deg"])
        best = max(((quad_iou(np.asarray(d["quad"], np.float64).reshape(4, 2), q), d) for d in dets), default=(0.0, None), key=lambda t: t[0])
        found = best[0] >= iou
        hits[cls].append(found)
        if found:
            right_class.append(best[1]["cls"] == cls)
    for cls, h in hits.items():
        if h:
            rows.append({"group": cls, "n": len(h), "recall": round(float(np.mean(h)), 4)})
    allh = hits["card"] + hits["card_back"]
    if allh:
        rows.insert(0, {"group": "all", "n": len(allh), "recall": round(float(np.mean(allh)), 4),
                        "class_right": round(float(np.mean(right_class)), 4) if right_class else None})
    return rows

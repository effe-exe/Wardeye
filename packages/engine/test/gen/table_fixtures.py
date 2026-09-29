# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""The Python reference for the table modules on real footage: what ml/rifteye_ml computes on the LA final's frames.

    . ml/.venv/bin/activate
    RIFTEYE_DATA=~/rifteye-data python packages/engine/test/gen/table_fixtures.py ~/rifteye-data/m3 \\
        [--detector ~/rifteye-data/models/detector-v0.pth] [--skip-detector]

Reads the JPEGs of m3/frames/la-final and writes m3/fixtures/table/ (private, D-006: frames, crops and detections
never enter the repository):

  frames.json      per raw frame (frames/la-final-rgb): mat_colour, notmat_mask and border_mask of the table window
                   as Recognizer.find uses them, the gate's view of it and its skin, and the gate run on those views
  gate.json        the change gate over all 240 frames, as Recognizer.watch feeds it (crop the table window, resize
                   to 320 px wide, bilinear), at 2 and at 5 frames a second: every event, and the still table's hash
  crops.json/.rgb  real card crops (faces and card backs of the LA final and the Barcelona Regional) and their detail
  autolayout.json  auto_layout on the LA final's first seconds (live/__main__.find_layout: a frame a second, the last
                   five) with the detector replaced by what it said: every call's inputs and outputs are kept

Test: test/table-parity.test.ts (Node, the raw frames) and e2e/table.spec.ts (Chromium decodes the JPEGs as the
extension does). Both need RIFTEYE_M3 and skip without it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from rifteye_ml import changegate as cg
from rifteye_ml import matcrops as mc
from rifteye_ml.live import autolayout as al
from rifteye_ml.live.layouts import LAYOUTS, Layout


def sha(a: np.ndarray, n: int = 16) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:n]


def layout_json(l: Layout) -> dict:
    return {"name": l.name, "title": l.title, "table": list(l.table), "card_long_1080": l.card_long_1080, "split": l.split, "mask": l.mask,
            "mat_tol": l.mat_tol, "mat": list(l.mat) if l.mat else None, "mat_share": l.mat_share}


def event_json(e: cg.ChangeEvent) -> dict:
    return {"t": e.t, "box": list(e.box), "kind": e.kind, "area": e.area, "before_mat": e.before_mat, "after_mat": e.after_mat, "extra": e.extra}


def view_of(frame: np.ndarray, layout: Layout, vw: int = 320) -> np.ndarray:
    """The gate's view of a frame, as live/pipeline.py Recognizer.watch makes it."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = layout.box(w, h)
    tx0, ty0, tx1, ty1 = layout.table
    vh = round(vw * (ty1 - ty0) * 1080 / ((tx1 - tx0) * 1920) / 2) * 2
    return np.asarray(Image.fromarray(frame[y0:y1, x0:x1]).resize((vw, vh), Image.BILINEAR))


def run_gate(views: list[np.ndarray], times: list[float], s: cg.GateSettings, checkpoint_every: int = 10) -> dict:
    gate = cg.ChangeGate(s)
    checkpoints, per_frame = [], []
    for i, (t, v) in enumerate(zip(times, views)):
        new = gate.feed(t, v)
        if new:
            per_frame.append({"i": i, "events": [event_json(e) for e in new]})
        if gate.background is not None and (i % checkpoint_every == 0 or i == len(views) - 1):
            checkpoints.append({"i": i, "background": sha(gate.background), "still": sha(gate.still.astype(np.int32))})
    return {"events": per_frame, "checkpoints": checkpoints, "last_same": sha(gate.last_same), "startup_hand": sha(gate.startup_hand.astype(np.uint8)),
            "off_table": gate.off_table, "mat": [int(v) for v in gate.mat],
            "kinds": [e.kind for e in gate.events]}


def settings_for(layout: Layout, fps: float, **kw) -> cg.GateSettings:
    """What Recognizer makes: GateSettings(fps=fps, card_long_frac=layout.card_long_1080 / 1080)."""
    return cg.GateSettings(fps=fps, card_long_frac=layout.card_long_1080 / 1080, **kw)


def settings_json(s: cg.GateSettings) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.__dict__.items()}


# --- the frames -----------------------------------------------------------------------------------------------------

def load_frames(m3: Path) -> tuple[list[dict], list[np.ndarray]]:
    meta = json.loads((m3 / "frames/la-final/frames.json").read_text())["frames"]
    frames = []
    for m in meta:
        f = np.asarray(Image.open(m3 / "frames/la-final" / m["file"]).convert("RGB"))
        assert hashlib.sha256(f.tobytes()).hexdigest() == m["rgb_sha256"], m["file"]
        frames.append(f)
    return meta, frames


def sec_frames(m3: Path, frames: list[np.ndarray], layout: Layout) -> dict:
    out = {"layout": layout.name, "frames": []}
    raw_names = {0: "f0000", 120: "f0120", 239: "f0239"}
    for idx, name in raw_names.items():
        raw = np.fromfile(m3 / "frames/la-final-rgb" / f"{name}.rgb", np.uint8).reshape(1080, 1920, 3)
        assert np.array_equal(raw, frames[idx]), name
        h, w = raw.shape[:2]
        x0, y0, x1, y1 = layout.box(w, h)
        roi = raw[y0:y1, x0:x1]
        mat = mc.mat_colour(roi)
        mat4 = mc.mat_colour(roi[::4, ::4])  # as Recognizer.find asks for it
        res = {"index": idx, "file": name, "roi": [x0, y0, x1, y1], "mat_colour": [int(v) for v in mat], "mat_colour_step4": [int(v) for v in mat4],
               "mat_colour_frame": [int(v) for v in mc.mat_colour(raw)]}
        masks = {}
        for label, m, tol in [("notmat_estimated", mat, layout.mat_tol), ("notmat_layout_mat", np.array(layout.mat, np.int16), layout.mat_tol),
                              ("notmat_60", mat4, 60), ("notmat_estimated_10", mat, 10)]:
            mask = mc.notmat_mask(roi, m, tol)
            masks[label] = {"mat": [int(v) for v in m], "tol": tol, "count": int(mask.sum()), "sha": sha(mask.astype(np.uint8))}
        border = mc.border_mask(roi)
        masks["border"] = {"count": int(border.sum()), "sha": sha(border.astype(np.uint8))}
        res["masks"] = masks
        view = view_of(raw, layout)
        sk = cg.skin(view)
        res["view"] = {"shape": list(view.shape), "sha": sha(view), "skin_count": int(sk.sum()), "skin_sha": sha(sk.astype(np.uint8))}
        out["frames"].append(res)
    # the gate on a few raw frames held for a few steps each: a cut, then settled changes
    order = [0, 0, 0, 120, 120, 120, 120, 239, 239, 239, 239, 0, 0, 0, 0]
    views = {i: view_of(frames[i], layout) for i in set(order)}
    for fps in (2.0, 5.0):
        s = settings_for(layout, fps)
        out[f"gate_{fps:g}"] = {"order": order, "fps": fps, "settings": settings_json(s),
                                **run_gate([views[i] for i in order], [k / fps for k in range(len(order))], s, checkpoint_every=1)}
    return out


def sec_gate(meta: list[dict], frames: list[np.ndarray], layout: Layout) -> dict:
    views = [view_of(f, layout) for f in frames]
    out = {"layout": layout.name, "view_shape": list(views[0].shape), "views": [sha(v) for v in views], "times": [m["t"] for m in meta], "runs": []}
    for name, s in [("the runner's fps of 2", settings_for(layout, 2.0)), ("the runner's default 5 fps", settings_for(layout, 5.0)),
                    ("a longer settle and a card-sized minimum", settings_for(layout, 2.0, settle_s=1.5, min_area=0.5, diff=32.0)),
                    ("sensitive: a small change, small regions, hands allowed less", settings_for(layout, 2.0, diff=16.0, motion=12.0, min_area=0.15, strong=0.2, hand_share=0.02)),
                    ("the preset's mat colour and two ignored bands", settings_for(layout, 2.0, mat_rgb=layout.mat, ignore=((0.0, 0.0, 1.0, 0.12), (0.0, 0.85, 0.3, 1.0)))),
                    ("fast drift, an early cut, a strict table test", settings_for(layout, 5.0, adapt=0.25, global_cut=0.35, min_mat=0.4, settle_s=0.4))]:
        run = run_gate(views, out["times"], s)
        print(f"  gate {name}: {len(run['kinds'])} events {run['kinds'][:24]}", file=sys.stderr)
        out["runs"].append({"name": name, "settings": settings_json(s), **run})
    return out


# --- real card crops -----------------------------------------------------------------------------------------------

def sec_crops(root: Path, fx: Path) -> dict:
    rng = random.Random(7)
    picks = []
    for d, n_face, n_back in [("la-final", 40, 24), ("bcn-v1", 16, 8)]:
        rows = list(csv.DictReader(open(root / d / "labels.csv")))
        faces = [r for r in rows if r["verdict"] == "correct" and r["card_id"] != "back"]
        backs = [r for r in rows if r["card_id"] == "back"]
        picks += [(d, r["file"], "face") for r in rng.sample(faces, n_face)] + [(d, r["file"], "back") for r in rng.sample(backs, n_back)]
    entries, blob = [], bytearray()
    for d, name, kind in picks:
        im = Image.open(root / d / "crops" / name).convert("RGB")
        a = np.asarray(im)
        entries.append({"set": d, "file": name, "kind": kind, "width": im.width, "height": im.height, "offset": len(blob), "detail": mc.detail(im),
                        "mat_colour": [int(v) for v in mc.mat_colour(a)]})
        blob += a.tobytes()
    (fx / "crops.rgb").write_bytes(bytes(blob))
    print(f"  crops: {len(entries)} ({len(blob) / 1e6:.1f} MB), detail {min(e['detail'] for e in entries):.2f}..{max(e['detail'] for e in entries):.2f}", file=sys.stderr)
    return {"threshold": mc.FACE_DOWN_DETAIL, "crops": entries}


# --- the layout finder ---------------------------------------------------------------------------------------------

class Recording:
    """The detector, asked as auto_layout asks it; every call's inputs and answer are kept."""

    def __init__(self, det, frames: list[np.ndarray]):
        self.det, self.frames, self.calls = det, frames, []

    def __call__(self, f: np.ndarray, box, px: float):
        dets = self.det.detect(Image.fromarray(f), box, px)
        self.calls.append({"frame": next(i for i, g in enumerate(self.frames) if g is f), "box": list(box), "px": px,
                           "out": [{"cls": d["cls"], "score": d["score"], "quad": d["quad"]} for d in dets]})
        return dets


def layout_record(frames: list[np.ndarray], detect) -> dict:
    tw = al.table_window(frames)
    rec = {"borders": list(al.borders(frames)), "table_window": None if tw is None else {"window": list(tw[0]), "mat": list(tw[1]), "share": tw[2]}}
    if detect is None:
        return rec
    recording = Recording(detect, frames)
    lay = al.auto_layout(frames, recording)
    rec["calls"] = recording.calls
    rec["auto_layout"] = None if lay is None else layout_json(lay)
    if tw is not None:
        # card_size on its own, from the recorded answers
        replay = iter(recording.calls)
        rec["card_size"] = al.card_size(lambda f, box, px: [{"score": d["score"], "quad": d["quad"], "cls": d["cls"]} for d in next(replay)["out"]], frames, tw[0])
    print(f"  autolayout: {len(recording.calls)} detector calls, table {rec['table_window'] and rec['table_window']['window']}, layout {rec['auto_layout']}", file=sys.stderr)
    return rec


def sec_autolayout(m3: Path, frames: list[np.ndarray], meta: list[dict], detector: Path | None) -> dict:
    det = None
    if detector is not None:
        import torch
        from rifteye_ml.detect.model import Detector

        torch.set_num_threads(2)
        det = Detector(str(detector), "cpu")
    out = {"first_seconds": [], "raw3": None}
    # live/__main__.find_layout: a look a second, the last five, until a layout is found
    seen, seen_idx, last_t = [], [], None
    for i, (m, f) in enumerate(zip(meta, frames)):
        if last_t is not None and m["t"] - last_t < 1.0:
            continue
        last_t = m["t"]
        seen, seen_idx = (seen + [f])[-5:], (seen_idx + [i])[-5:]
        if len(seen) < 5:
            continue
        print(f"  find_layout attempt at t={m['t']}: frames {seen_idx}", file=sys.stderr)
        rec = {"frames": list(seen_idx), **layout_record(seen, det)}
        out["first_seconds"].append(rec)
        if rec.get("auto_layout") is not None or det is None or len(out["first_seconds"]) >= 4:
            break
    # three frames a minute apart: a Node test can read these without a JPEG decoder
    idx = [0, 120, 239]
    out["raw3"] = {"frames": idx, **layout_record([frames[i] for i in idx], det)}
    return out


def sec_finder(fx: Path, frames: list[np.ndarray]) -> dict:
    """Adds what the layout finder makes of the same frames without a detector (the bootstrap finder), to autolayout.json."""
    doc = json.loads((fx / "autolayout.json").read_text())
    records = doc["first_seconds"] + [doc["raw3"]]
    for rec in records:
        fr = [frames[i] for i in rec["frames"]]
        tw = al.table_window(fr)
        lay = al.auto_layout(fr)
        rec["card_size_finder"] = al.card_size_finder(fr, tw[0], tw[1]) if tw is not None else None
        rec["auto_layout_finder"] = None if lay is None else layout_json(lay)
        print(f"  finder on frames {rec['frames']}: card {rec['card_size_finder']}, layout {rec['auto_layout_finder'] and rec['auto_layout_finder']['card_long_1080']}", file=sys.stderr)
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("m3", type=Path)
    ap.add_argument("--detector", type=Path, default=Path(os.environ.get("RIFTEYE_DATA", Path.home() / "rifteye-data")) / "models" / "detector-v0.pth")
    ap.add_argument("--skip-detector", action="store_true", help="leave autolayout.json as it is")
    ap.add_argument("--only", choices=["frames", "gate", "crops", "autolayout", "finder"], nargs="*")
    a = ap.parse_args()
    fx = a.m3 / "fixtures" / "table"
    fx.mkdir(parents=True, exist_ok=True)
    layout = LAYOUTS["la-rq"]
    only = set(a.only or ["frames", "gate", "crops", "autolayout", "finder"])
    t0 = time.time()
    meta, frames = load_frames(a.m3)
    print(f"{len(frames)} frames decoded in {time.time() - t0:.0f} s", file=sys.stderr)

    def write(name: str, obj: dict) -> None:
        text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
        (fx / name).write_text(text + "\n")
        print(f"{fx / name}: {len(text) / 1000:.0f} KB ({time.time() - t0:.0f} s)", file=sys.stderr)

    if "frames" in only:
        write("frames.json", sec_frames(a.m3, frames, layout))
    if "gate" in only:
        write("gate.json", sec_gate(meta, frames, layout))
    if "crops" in only:
        write("crops.json", sec_crops(Path(os.environ.get("RIFTEYE_DATA", Path.home() / "rifteye-data")) / "real-crops", fx))
    if "autolayout" in only and not a.skip_detector:
        write("autolayout.json", sec_autolayout(a.m3, frames, meta, a.detector))
    if "finder" in only:
        write("autolayout.json", sec_finder(fx, frames))


if __name__ == "__main__":
    main()

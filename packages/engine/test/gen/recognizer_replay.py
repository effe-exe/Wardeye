# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""The Python reference for the recognizer's replay tests (recognizer.replay.test.ts, e2e/recognizer.spec.ts).

live/pipeline.py's Recognizer is run over the 240 shared JPEG frames of the LA final, read with Pillow in order,
built as live/__main__.py builds it: layout la-rq (fixed, not auto) at 2 fps; the finder is detector_boxes over
Detector.detect with detector-v0.onnx in the Detector (ONNX Runtime CPU, float32) and __main__'s --det-score;
the encoder is onnx:embedder-v1.onnx (float32); the gallery is __main__.gallery() at the layout's levels, with its
embedding cache inside the fixtures folder; the temperature is embedder-v1's (EMBEDDER_T); the legend rule is on, as
in __main__ (--no-legend-rule turns it off; meta.json says which). rows.json keeps each row's domains, variant and
tags, which the legend rule reads.

Everything the Recognizer is fed and gives back is written down: per frame the finder's boxes, every embed() call
(each picture's size and the SHA-256 of its RGB bytes, and the rows it gave), per step the state and the events,
and the Recognizer's own bookkeeping after each step (to find where a port first goes its own way). The format
is in the README.md written next to them.

    RIFTEYE_DATA=~/rifteye-data python packages/engine/test/gen/recognizer_replay.py ~/rifteye-data/m3

--scenario cut writes fixtures/recognizer/cut/ instead: what the LA final never shows. Five table frames (f0000 to
f0004), then four frames of another shot (plain grey: the scene says the camera is away), then f0009 to f0011 three
times over with the picture moved 60 px left and 40 px down, as if the camera had been re-framed while away: the
board moves at once, so the Recognizer cuts, finds cards again by name and re-anchors the rest. All of them are
among the frames recognizer_frames.py writes as raw RGB. The unmoved frames' detections are the LA final run's.

--reuse takes the detector's and the embedder's outputs from an earlier run's recording instead of running the
models again (they are the same numbers); the Recognizer itself always runs afresh. Private: the frames come from
a broadcast and the models are trained on card art (D-006), so none of this enters the repository.
"""
from __future__ import annotations

import os

# At most two threads for the heavy parts (the machine is shared). numpy's BLAS gets one: a dot product of more
# than 10000 numbers (the scene's thumbnails) would otherwise be split between threads and summed in their order.
for _k, _v in (("OMP_NUM_THREADS", "2"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "2"), ("HF_HUB_OFFLINE", "1")):
    os.environ.setdefault(_k, _v)

import argparse  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

LAYOUT = "la-rq"
FPS = 2.0
DET_SCORE = 0.4  # live/__main__.py's --det-score default
THREADS = 2
# the cut scenario: (frame file or "grey", how far its picture is moved, or None)
CUT = ([(f"f{k:04d}.jpg", None) for k in range(5)] + [("grey", None)] * 4
       + [(f"f{k:04d}.jpg", (-60, 40)) for k in (9, 10, 11)] * 3)
GREY = 128


def box_json(b) -> dict:
    """A CardBox with what detector_boxes (or smooth) set on it: back, score and vis only when it has them."""
    d = {"centre": [float(b.centre[0]), float(b.centre[1])], "long_px": float(b.long_px), "short_px": float(b.short_px),
         "angle_deg": float(b.angle_deg), "fill": float(b.fill)}
    for k in ("back", "score", "vis"):
        if hasattr(b, k):
            v = getattr(b, k)
            d[k] = bool(v) if k == "back" else float(v)
    return d


def det_json(d: dict) -> dict:
    """One of Detector.detect's cards as it gives it (merge_tiles: cls, score, the quad as 8 numbers, found,
    visible, truncated), as plain JSON."""
    return plain(d)


def rgb_sha256(im: Image.Image) -> str:
    return hashlib.sha256(np.asarray(im.convert("RGB")).tobytes()).hexdigest()


def plain(x):
    """The state and events as JSON takes them (numpy scalars to Python ones)."""
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def snapshot(rec) -> dict:
    """The Recognizer's bookkeeping after a step. A track's probabilities are summed over every card it was ever
    read as (a hundred or more), so only their count, their sum in insertion order and the five best are kept."""
    tracks = []
    for tr in rec.tracks.values():
        prob = list(tr.prob.items())
        best = sorted(prob, key=lambda kv: -kv[1])[:5]
        tracks.append({"id": tr.id, "box": box_json(tr.box), "first": tr.first, "last": tr.last, "hits": tr.hits,
                       "reads": tr.reads, "down": tr.down, "prob_n": len(prob), "prob_sum": float(sum(p for _, p in prob)),
                       "prob_top": [[c, float(p)] for c, p in best],
                       "best_row": [[c, float(tr.best_row[c][0]), int(tr.best_row[c][1])] for c, _ in best if c in tr.best_row],
                       "last_read": tr.last_read, "named": tr.named, "side": tr.side, "kind": tr.kind, "pinned": tr.pinned,
                       "free_since": tr.free_since, "free_at": tr.free_at, "placed": tr.placed})
    sc = rec.scene
    return plain({
        "tracks": tracks, "next_id": rec.next_id, "t0": rec.t0, "last_t": rec.last_t, "away": rec.away, "cut_at": rec.cut_at,
        "pending": [[w, list(b)] for w, b in rec.pending], "plays": [list(p) for p in rec.plays],
        "ghosts": rec.ghosts, "legends": rec.legends, "flashes": rec.flashes,
        "prev_seen": sorted(rec.prev_seen), "before_away": sorted(rec.before_away),
        "anchor_base": sorted(rec.anchor_base), "anchor_pairs": [[list(a), list(b)] for a, b in rec.anchor_pairs],
        "boxes_now": {k: list(v) for k, v in rec.boxes_now.items()},
        "scene": {"n": sc.n, "last_learn": sc.last_learn, "away_since": sc.away_since, "looks": sc.looks,
                  "last_look": sc.last_look},
    })


class Recording:
    """The encoder the Recognizer is given: the real one, with every call's pictures and rows written down."""

    def __init__(self, enc, rows_file: Path, reuse: dict[str, np.ndarray] | None):
        self.enc, self.name, self.dim = enc, enc.name, enc.dim
        self.calls: list[dict] = []
        self.out = open(rows_file, "wb")
        self.n_rows = 0
        self.reuse = reuse or {}
        self.step, self.where, self.todo = -1, "", []
        self.crop_box: dict[int, object] = {}  # id(crop) -> the track box it was cut from, during a read
        self.rec = None

    def embed(self, images):
        sizes = [[im.width, im.height] for im in images]
        hashes = [rgb_sha256(im) for im in images]
        key = hashlib.sha256(json.dumps([sizes, hashes]).encode()).hexdigest()
        x = self.reuse.get(key)
        if x is None:
            x = self.enc.embed(images)
        x = np.ascontiguousarray(x, np.float32)
        self.out.write(x.astype("<f4").tobytes())
        owners = None
        if self.where == "read":  # which track each crop is (identify batches the four turns of a crop together)
            owners = []
            for n in range(0, len(images), 4):
                box = self.crop_box.get(id(images[n]))
                owners.append(next((tr.id for tr in self.rec.tracks.values() if tr.box is box), None))
        self.calls.append({"step": self.step, "call": len(self.calls), "where": self.where, "tracks": owners,
                           "sizes": sizes, "sha256": hashes, "offset": self.n_rows, "count": len(images), "key": key})
        self.n_rows += len(images)
        return x

    def fingerprint(self) -> str:
        return self.enc.fingerprint()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("m3", type=Path, help="the private m3 folder: frames/la-final/ in, fixtures/recognizer/ out")
    ap.add_argument("--frames", type=int, default=0, help="only the first N frames (a quick look)")
    ap.add_argument("--reuse", action="store_true", help="the models' outputs from the last run's recording")
    ap.add_argument("--scenario", choices=["la-final", "cut"], default="la-final")
    ap.add_argument("--no-legend-rule", action="store_true", help="the whole gallery on both sides all along")
    a = ap.parse_args(argv)

    import onnxruntime
    import PIL
    import scipy
    import torch

    torch.set_num_threads(THREADS)
    from rifteye_ml import catalog as cat
    from rifteye_ml.detect.model import Detector
    from rifteye_ml.detect.onnx import OnnxNet
    from rifteye_ml.embed.onnx import Onnx
    from rifteye_ml.live import pipeline
    from rifteye_ml.live.__main__ import DATA, EMBEDDER_T, gallery
    from rifteye_ml.live.layouts import LAYOUTS
    from rifteye_ml.live.pipeline import Recognizer, detector_boxes

    frames_dir = a.m3 / "frames" / "la-final"
    base = a.m3 / "fixtures" / "recognizer"
    out = base if a.scenario == "la-final" else base / a.scenario
    out.mkdir(parents=True, exist_ok=True)
    listing = json.loads((frames_dir / "frames.json").read_text(encoding="utf-8"))
    assert listing["layout"] == LAYOUT and listing["fps"] == FPS
    pillow = {fr["file"]: fr["rgb_sha256"] for fr in listing["frames"]}
    # (key, source, t): the key names the picture (a file, a moved file, or grey)
    if a.scenario == "la-final":
        frames = [(fr["file"], {"file": fr["file"]}, fr["t"]) for fr in listing["frames"][: a.frames or None]]
    else:
        frames = [(f if f == "grey" or sh is None else f"{f}@{sh[0]},{sh[1]}",
                   {"grey": GREY} if f == "grey" else {"file": f, "shift": list(sh) if sh else None}, 0.5 * k)
                  for k, (f, sh) in enumerate(CUT)][: a.frames or None]

    def picture(source: dict) -> np.ndarray:
        if "grey" in source:
            return np.full((listing["size"][1], listing["size"][0], 3), source["grey"], np.uint8)
        im = Image.open(frames_dir / source["file"]).convert("RGB")
        assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest() == pillow[source["file"]], source["file"]
        if source.get("shift"):
            dx, dy = source["shift"]
            im = im.crop((-dx, -dy, im.width - dx, im.height - dy))  # the picture moved by (dx, dy); black where it left
        return np.asarray(im)

    # What an earlier run recorded: the detector's cards per picture, the rows per call. The LA final's detections
    # serve any run for the frames it has as they are (the detector gives the same numbers for the same picture).
    old_dets: dict[str, list[dict]] = {}
    old_rows: dict[str, np.ndarray] = {}
    sources = [(base, False)] if a.scenario != "la-final" else []  # the LA final's detections of its frames as they are
    if a.reuse:
        sources.append((out, True))
    for src, rows_too in sources:
        if not (src / "steps.jsonl").exists():
            continue
        for line in (src / "steps.jsonl").read_text(encoding="utf-8").splitlines():
            s = json.loads(line)
            if s["dets"] is not None:
                old_dets[s["file"]] = s["dets"]
        if rows_too and (src / "embeds.json").exists():
            rows_all = np.fromfile(src / "embeds.bin", "<f4").reshape(-1, 256)
            for c in json.loads((src / "embeds.json").read_text(encoding="utf-8"))["calls"]:
                old_rows[c["key"]] = rows_all[c["offset"]: c["offset"] + c["count"]].copy()
    if old_dets or old_rows:
        print(f"reusing {len(old_dets)} pictures' detections and {len(old_rows)} embed calls", flush=True)

    layout = LAYOUTS[LAYOUT]
    catalog = DATA / "catalog" / "catalog-plus.jsonl"
    cache = DATA / "art"
    # __main__ first makes sure every picture is cached (ensure_catalogue); here nothing is fetched, ever: the rows
    # are the ones whose art is cached, as __main__ keeps them.
    rows = [r for r in cat.read_catalog(catalog) if cat.cache_path(cache, r["image_url"]).exists()]
    enc = Onnx(DATA / "models" / "onnx" / "embedder-v1.onnx", threads=THREADS)
    px = layout.card_px(1080)
    scales = sorted({int(round(px * f / 10) * 10) for f in (0.8, 0.9, 1.0)})
    t0 = time.perf_counter()
    pyr = gallery(enc, rows, cache, base / "embed-cache", scales)
    print(f"gallery {scales} x {pyr.rows} rows in {time.perf_counter() - t0:.0f} s", flush=True)
    (out / "levels").mkdir(exist_ok=True)
    for s, level in pyr.levels.items():
        np.ascontiguousarray(level, "<f4").tofile(out / "levels" / f"{s}.bin")
    # what the Recognizer reads of a row: the legend rule reads domains (a row without them fits any legend), variant
    # and the name (the tokens); tags name a legend's champion, for decklists
    (out / "rows.json").write_text(json.dumps([{"printing_id": r["printing_id"], "card_id": r["card_id"], "name": r["name"],
                                                 "type": r.get("type"),
                                                 **{k: r[k] for k in ("domains", "variant", "tags") if k in r}}
                                                for r in rows], ensure_ascii=False),
                                   encoding="utf-8")

    det = None
    if any(key not in old_dets and "grey" not in source for key, source, _ in frames):
        det = Detector(DATA / "models" / "detector-v0.pth", "cpu")
        det.net = OnnxNet(DATA / "models" / "onnx" / "detector-v0.onnx", det.rf.means, det.rf.stds, threads=THREADS)

    frame_rec: dict = {}

    def finder(t: float, image: np.ndarray, layout=layout) -> list:
        # live/__main__.py's finder, as it is
        h, w = image.shape[:2]
        dets = old_dets.get(frame_rec["file"])
        if dets is None:
            dets = [det_json(d) for d in det.detect(Image.fromarray(image), layout.box(w, h), layout.card_px(h))]
            old_dets[frame_rec["file"]] = dets  # the same picture again gives the same cards
        boxes = detector_boxes(dets, min_score=DET_SCORE)
        frame_rec["dets"], frame_rec["boxes"] = dets, [box_json(b) for b in boxes]
        return boxes

    recording = Recording(enc, out / "embeds.bin", old_rows)
    title = layout.title  # __main__: a.title or LAYOUTS[a.layout].title
    rec = Recognizer(layout, rows, recording, pyr, title=title, fps=FPS, finder=finder, temperature=EMBEDDER_T,
                     legend_rule=not a.no_legend_rule)
    recording.rec = rec

    # Which track each crop is, for the recording only: the crops are cut as the pipeline cuts them.
    card_crop = pipeline.card_crop

    def recorded_crop(frame, box):
        c = card_crop(frame, box)
        recording.crop_box[id(c)] = box
        return c

    pipeline.card_crop = recorded_crop
    read, watch = rec.read, rec.watch

    def reading(t, frame, todo):
        recording.where, recording.todo = "read", [tr.id for tr in todo]
        recording.crop_box.clear()
        try:
            return read(t, frame, todo)
        finally:
            recording.where = ""

    def watching(t, image, frame):
        recording.where = "watch"
        try:
            return watch(t, image, frame)
        finally:
            recording.where = ""

    rec.read, rec.watch = reading, watching

    steps = open(out / "steps.jsonl", "w", encoding="utf-8")
    internals = gzip.open(out / "internals.jsonl.gz", "wt", encoding="utf-8")
    t_run = time.perf_counter()
    for k, (key, source, t) in enumerate(frames):
        image = picture(source)
        frame_rec.clear()
        frame_rec["file"] = key
        recording.step, recording.todo = k, []
        first_call = len(recording.calls)
        tic = time.perf_counter()
        state, events = rec.step(t, image)
        ms = 1000 * (time.perf_counter() - tic)
        steps.write(json.dumps(plain({
            "i": k, "file": key, "source": source, "t": t, "dets": frame_rec.get("dets"), "boxes": frame_rec.get("boxes"),
            "todo": recording.todo, "embeds": list(range(first_call, len(recording.calls))),
            "state": state, "events": events}), ensure_ascii=False) + "\n")
        internals.write(json.dumps(snapshot(rec), ensure_ascii=False) + "\n")
        named = sum(1 for tr in state["tracks"] if tr["state"] == "named")
        print(f"{k:3d} t={t:6.1f} {state['status']:4s} boxes={len(frame_rec.get('boxes') or [])} "
              f"tracks={len(state['tracks'])} named={named} events={len(events)} embeds={len(recording.calls) - first_call} "
              f"{ms:.0f} ms", flush=True)
        for e in events:
            print(f"      {e['kind']}: {e['text']} ({e['track']}, {e['side']})", flush=True)
    steps.close()
    internals.close()
    recording.out.close()
    (out / "embeds.json").write_text(json.dumps({"dim": 256, "calls": recording.calls}), encoding="utf-8")
    meta = {
        "scenario": a.scenario, "frames": "frames/la-final", "steps": len(frames), "layout": asdict(layout), "fps": FPS, "det_score": DET_SCORE,
        "temperature": EMBEDDER_T, "title": title, "legend_rule": rec.legend_rule, "levels": scales, "rows": len(rows), "dim": 256,
        "encoder": enc.name, "embedder": "models/onnx/embedder-v1.onnx", "detector": "models/onnx/detector-v0.onnx",
        "detector_checkpoint": "models/detector-v0.pth", "embed_calls": len(recording.calls), "embed_rows": recording.n_rows,
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")} | {
            "torch": THREADS, "onnxruntime": THREADS},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "pillow": PIL.__version__, "onnxruntime": onnxruntime.__version__, "torch": torch.__version__},
        "run_s": round(time.perf_counter() - t_run, 1), "reused": bool(old_dets or old_rows),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{len(frames)} steps, {len(recording.calls)} embed calls ({recording.n_rows} rows) in {meta['run_s']} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

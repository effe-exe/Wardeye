# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The Python reference for the vision parity tests (test/vision-*.test.ts with RIFTEYE_M3, e2e/vision.spec.ts) and
for the bench's decoded detector check.

    export RIFTEYE_DATA=~/rifteye-data
    python packages/engine/test/gen/vision_parity.py barcelona ~/rifteye-data/m3   # the Barcelona frames
    python packages/engine/test/gen/vision_parity.py detector ~/rifteye-data/m3    # detector and crops
    python packages/engine/test/gen/vision_parity.py check                         # detector-v0.check.detections.json

barcelona: ten table-camera frames of the Barcelona VOD (2854086989, PlusRB's restream, layout plusrb), decoded
  with PyAV and written as the LA frames were: Pillow JPEG quality 90 (4:2:0), with a frames.json like theirs,
  to fixtures/vision/barcelona/.
detector: 21 LA frames and the ten Barcelona ones through Detector.detect as live/__main__.py calls it, with
  detector-v0.onnx as its net (ONNX Runtime CPU, float32, two threads), then detector_boxes at the runner's
  --det-score. Per frame, fixtures/vision/detector/<id>.json holds each tile's SHA-256 (its RGB bytes, and the
  float32 graph input), detect_tiles' output, detect()'s and detector_boxes'; <id>.pred_*.bin the raw outputs of
  the frame's tiles (float32, little-endian, as the graph gives them). Then 40 crops of those boxes
  (live/pipeline.card_crop; no card backs) go to fixtures/vision/crops/ as PNG and raw RGB, with their letterboxed
  input's SHA-256 and embedder-v1.onnx's rows (ONNX Runtime CPU, float32; embed.bin after l2n, embed_raw.bin
  before).
check: the detections Detector.detect_tiles makes of the bench's check outputs (PyTorch fp32 on
  detector-v0.check.input.bin), written next to them as detector-v0.check.detections.json; also what ONNX Runtime
  CPU makes of that tile with the fp32 and fp16 files, to size the check's tolerances.

Private: the frames come from broadcasts and the models are trained on card art (D-006); nothing here enters the
repository.
"""
from __future__ import annotations

import os

# At most two threads for the heavy parts (the machine is shared); numpy's BLAS gets one.
for _k, _v in (("OMP_NUM_THREADS", "2"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "2"), ("HF_HUB_OFFLINE", "1")):
    os.environ.setdefault(_k, _v)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DATA = Path(os.environ.get("RIFTEYE_DATA", "~/rifteye-data")).expanduser()
MODELS = DATA / "models"
ONNX = MODELS / "onnx"
BARCELONA_VOD = DATA / "vods" / "2854086989" / "hq" / "seg-03h29m00s-03h57m30s-1080p60.mp4"
# table-camera times in that segment (seconds from its start, 03:29:00 in the VOD), from the VOD's oh-frames.txt
BARCELONA_T = [170, 330, 490, 650, 820, 980, 1140, 1300, 1460, 1630]
LA_FRAMES = [f"f{i:04d}" for i in range(0, 240, 12)] + ["f0239"]
DET_SCORE = 0.4  # live/__main__.py --det-score


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":")) + "\n", encoding="utf-8")


def barcelona(m3: Path) -> None:
    import av

    out = m3 / "fixtures" / "vision" / "barcelona"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    with av.open(str(BARCELONA_VOD)) as c:
        s = c.streams.video[0]
        for t in BARCELONA_T:
            c.seek(int(t / s.time_base), stream=s)  # the key frame before t, then decoded up to t
            fr = next(f for f in c.decode(s) if f.time is not None and f.time >= t)
            rgb = fr.to_ndarray(format="rgb24")
            name = f"b{t:04d}.jpg"
            Image.fromarray(rgb).save(out / name, quality=90)
            with Image.open(out / name) as im:
                back = np.asarray(im.convert("RGB"))
            frames.append({"file": name, "t": t, "vod_time": fr.time, "rgb_sha256": sha(back.tobytes())})
            print(name, fr.time, flush=True)
    write_json(out / "frames.json", {"clip": "Barcelona Regional, PlusRB restream (VOD 2854086989), segment 03:29:00-03:57:30",
                                     "source": BARCELONA_VOD.name, "layout": "plusrb", "jpeg": "Pillow quality 90, 4:2:0",
                                     "size": [1920, 1080], "note": "t is seconds from the segment's start; rgb_sha256 is Pillow's decode",
                                     "frames": frames})


class Recorder:
    """Detector.detect's insides: the tiles it cuts, the graph's input and outputs, detect_tiles' output."""

    def __init__(self, det):
        self.reset()
        run, detect_tiles = det.net.run, det.detect_tiles

        def rec_run(x):
            out = run(x)
            self.calls.append((np.array(x, copy=True), {k: np.array(v, copy=True) for k, v in out.items()}))
            return out

        def rec_detect_tiles(tiles, threshold=0.3):
            self.tiles += tiles
            out = detect_tiles(tiles, threshold)
            self.per_tile += out
            return out

        det.net.run = rec_run
        det.detect_tiles = rec_detect_tiles

    def reset(self):
        self.calls, self.tiles, self.per_tile = [], [], []


def load_detector():
    import torch

    from rifteye_ml.detect.model import Detector
    from rifteye_ml.detect.onnx import OnnxNet

    torch.set_num_threads(2)
    det = Detector(MODELS / "detector-v0.pth", "cpu")
    det.net = OnnxNet(ONNX / "detector-v0.onnx", det.rf.means, det.rf.stds, threads=2)
    return det


def box_json(b) -> dict:
    return {"centre": [float(b.centre[0]), float(b.centre[1])], "long_px": float(b.long_px), "short_px": float(b.short_px),
            "angle_deg": float(b.angle_deg), "fill": float(b.fill), "back": bool(b.back), "score": float(b.score), "vis": float(b.vis)}


def tile_json(d: dict) -> dict:
    return {"cls": d["cls"], "score": d["score"], "box": d["box"], "quad": np.asarray(d["quad"]).tolist(), "found": d["found"], "visible": d["visible"]}


def detector(m3: Path) -> None:
    from rifteye_ml.detect.export import TILE
    from rifteye_ml.detect.geometry import tile_origins
    from rifteye_ml.embed.onnx import Onnx, batch_of
    from rifteye_ml.encoders import letterbox
    from rifteye_ml.live.layouts import LAYOUTS
    from rifteye_ml.live.pipeline import card_crop, detector_boxes

    vis = m3 / "fixtures" / "vision"
    out = vis / "detector"
    out.mkdir(parents=True, exist_ok=True)
    det = load_detector()
    rec = Recorder(det)
    levels = np.float32(np.arange(256)) / np.float32(255)  # what a tile's byte is as the graph's input
    frames = [("la", n, m3 / "frames" / "la-final" / f"{n}.jpg", "la-rq") for n in LA_FRAMES]
    frames += [("bcn", Path(f["file"]).stem, vis / "barcelona" / f["file"], "plusrb")
               for f in json.loads((vis / "barcelona" / "frames.json").read_text())["frames"]]
    index, crops = [], []
    for clip, name, path, lay in frames:
        fid = f"{clip}-{name}"
        with Image.open(path) as im:
            image = np.asarray(im.convert("RGB"))
        h, w = image.shape[:2]
        layout = LAYOUTS[lay]
        window, card_px = layout.box(w, h), layout.card_px(h)
        rec.reset()
        dets = det.detect(Image.fromarray(image), window, card_px)
        boxes = detector_boxes(dets, min_score=DET_SCORE)
        # detect()'s own arithmetic again, for the tiles' places
        scale = 70.0 / card_px
        sw, sh = max(1, round((window[2] - window[0]) * scale)), max(1, round((window[3] - window[1]) * scale))
        origins = [(tx, ty) for ty in tile_origins(sh, TILE, 0.2) for tx in tile_origins(sw, TILE, 0.2)]
        assert len(rec.calls) == 1 and len(rec.tiles) == len(origins) == len(rec.per_tile)
        x, raw = rec.calls[0]
        tiles = []
        for i, t in enumerate(rec.tiles):
            rgb = np.asarray(t.convert("RGB"))
            assert np.array_equal(x[i], levels[rgb].transpose(2, 0, 1)), "the graph's input is not the tile's bytes / 255"
            tiles.append({"origin": list(origins[i]), "rgb_sha256": sha(rgb.tobytes()), "input_sha256": sha(np.ascontiguousarray(x[i]).tobytes())})
        for k, v in raw.items():
            (out / f"{fid}.{k}.bin").write_bytes(np.ascontiguousarray(v, "<f4").tobytes())
        write_json(out / f"{fid}.json", {
            "id": fid, "file": str(path.relative_to(m3)), "layout": lay, "size": [w, h], "window": list(window), "card_px": card_px,
            "scale": scale, "win_size": [sw, sh], "tiles": tiles, "per_tile": [[tile_json(d) for d in ds] for ds in rec.per_tile],
            "detect": dets, "min_score": DET_SCORE, "boxes": [box_json(b) for b in boxes]})
        index.append(fid)
        faces = [b for b in boxes if not b.back]
        pick = sorted({0, len(faces) // 2}) if len(faces) > 1 else list(range(len(faces)))
        if len(crops) < 40:
            for k in pick[: 40 - len(crops)]:
                crops.append((fid, k, card_crop(Image.fromarray(image), faces[k]), box_json(faces[k])))
        print(fid, len(origins), "tiles", len(dets), "detections", len(boxes), "boxes", flush=True)
    write_json(out / "index.json", {"frames": index, "model": "detector-v0.onnx", "model_sha256": sha((ONNX / "detector-v0.onnx").read_bytes()),
                                    "runtime": "onnxruntime CPU float32, 2 threads", "det_score": DET_SCORE})

    # the embedder on real crops
    cdir = vis / "crops"
    cdir.mkdir(parents=True, exist_ok=True)
    enc = Onnx(ONNX / "embedder-v1.onnx", threads=2)
    images = [c[2] for c in crops]
    rows = enc.embed(images)
    raw = enc.run(batch_of(images, enc.img_size))
    meta = []
    for i, (fid, k, im, box) in enumerate(crops):
        cid = f"c{i:02d}"
        im.save(cdir / f"{cid}.png")
        with Image.open(cdir / f"{cid}.png") as back:
            assert np.array_equal(np.asarray(back.convert("RGB")), np.asarray(im))
        (cdir / f"{cid}.rgb").write_bytes(np.asarray(im).tobytes())
        lb = np.asarray(letterbox(im, enc.img_size))
        meta.append({"id": cid, "frame": fid, "face": k, "box": box, "size": [im.width, im.height], "rgb_sha256": sha(np.asarray(im).tobytes()),
                     "letterbox_sha256": sha(lb.tobytes()), "input_sha256": sha(batch_of([im], enc.img_size).tobytes())})
    (cdir / "embed.bin").write_bytes(np.ascontiguousarray(rows, "<f4").tobytes())
    (cdir / "embed_raw.bin").write_bytes(np.ascontiguousarray(raw, "<f4").tobytes())
    write_json(cdir / "crops.json", {"encoder": enc.name, "img_size": enc.img_size, "dim": enc.dim, "model": "embedder-v1.onnx",
                                     "runtime": "onnxruntime CPU float32, 2 threads", "crops": meta})
    print(len(crops), "crops;", enc.name, flush=True)


def check() -> None:
    import onnxruntime as ort
    import torch

    from rifteye_ml.detect.model import Detector

    torch.set_num_threads(2)
    det = Detector(MODELS / "detector-v0.pth", "cpu")
    x = np.fromfile(ONNX / "detector-v0.check.input.bin", "<f4").reshape(1, 3, 576, 576)
    tile = Image.fromarray(np.round(x[0].transpose(1, 2, 0) * 255).astype(np.uint8))
    shapes = {"pred_logits": (1, 100, 2), "pred_boxes": (1, 100, 4), "pred_keypoints": (1, 100, 8, 8)}

    def decode(outputs: dict) -> list:
        det.net = lambda batch: {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in outputs.items()}
        return [tile_json(d) for d in det.detect_tiles([tile], 0.3)[0]]

    torch_out = {k: np.fromfile(ONNX / f"detector-v0.check.{k}.bin", "<f4").reshape(s) for k, s in shapes.items()}
    expected = decode(torch_out)
    other = {}
    for prec, file in (("fp32", "detector-v0.onnx"), ("fp16", "detector-v0.fp16.onnx")):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        sess = ort.InferenceSession(str(ONNX / file), opts, providers=["CPUExecutionProvider"])
        got = dict(zip(shapes, sess.run(list(shapes), {"tiles": x})))
        other[prec] = decode(got)
    doc = {"note": ("Detector.detect_tiles (RF-DETR's keypoint PostProcess, run twice as detect_tiles runs it) on the PyTorch "
                    "float32 outputs for detector-v0.check.input.bin: the cards the bench's decoded check expects, in tile px "
                    "(576 x 576), every one above detect_tiles' threshold. onnxruntime_cpu holds what ONNX Runtime CPU makes of "
                    "the same tile with each file, for sizing the tolerances. Made by packages/engine/test/gen/vision_parity.py check."),
           "tile": 576, "threshold": 0.3, "source": "pytorch fp32 (detector-v0.check.pred_*.bin)", "detections": expected,
           "onnxruntime_cpu": other}
    write_json(ONNX / "detector-v0.check.detections.json", doc)
    print(len(expected), "expected detections;", {k: len(v) for k, v in other.items()}, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["barcelona", "detector", "check"])
    ap.add_argument("m3", nargs="?", type=Path, default=DATA / "m3")
    a = ap.parse_args(argv)
    if a.what == "barcelona":
        barcelona(a.m3.expanduser())
    elif a.what == "detector":
        detector(a.m3.expanduser())
    else:
        check()
    return 0


if __name__ == "__main__":
    sys.exit(main())

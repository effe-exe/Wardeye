# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
"""The amodal card detector (M1): export synthetic tiles, train, evaluate, run on frames.

    python -m rifteye_ml.detect export --run ~/rifteye-data/synth/v0 --out ~/rifteye-data/detect/v0-tiles
    python -m rifteye_ml.detect train --dataset ~/rifteye-data/detect/v0-tiles --out ~/rifteye-data/detect/v0
    python -m rifteye_ml.detect evaluate --run ~/rifteye-data/synth/v0 --checkpoint .../checkpoint_best_total.pth
    python -m rifteye_ml.detect pack --checkpoint .../checkpoint_best_total.pth --out detector-v0.pth
    python -m rifteye_ml.detect run --frames DIR --table 0.17,0.09,0.86,0.884 --card-px 131 --checkpoint ... --out dets.jsonl
    python -m rifteye_ml.detect onnx detector-v0.pth ~/rifteye-data/models/onnx
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image


def _export(a) -> int:
    from .export import export_run

    t0 = time.time()
    stats = export_run([Path(r) for r in a.run], Path(a.out), target=tuple(float(v) for v in a.target.split(",")),
                       overlap=a.overlap, val_every=a.val_every, quality=a.quality, seed=a.seed, scales=a.scales)
    print(json.dumps(stats), f"in {time.time() - t0:.0f} s")
    return 0


def _train(a) -> int:
    from .model import train

    train(Path(a.dataset), Path(a.out), epochs=a.epochs, batch_size=a.batch_size, grad_accum=a.grad_accum, lr=a.lr,
          device=a.device, num_workers=a.workers, seed=a.seed, resume=a.resume)
    return 0


def _pack(a) -> int:
    from .model import pack

    print(f"{a.out}: {pack(a.checkpoint, a.out) / 1e6:.0f} MB")
    return 0


def _evaluate(a) -> int:
    from .evaluate import score_frames, targets
    from .model import Detector

    det = Detector(a.checkpoint, a.device)
    anns = []
    for r in a.run:
        with open(Path(r) / "annotations.jsonl", encoding="utf-8") as f:
            anns += [(Path(r), x) for x in map(json.loads, f) if x["board"] % a.val_every == a.val_every - 1]
    anns = anns[: a.max_frames or None]
    frames, t0 = [], time.time()
    for k, (run, x) in enumerate(anns):
        with Image.open(run / x["image"]) as im:
            dets = det.detect(im, x["window"], x["card_px"], target=a.target, threshold=min(a.thresholds))
        frames.append((dets, targets(x)))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(anns)} frames in {time.time() - t0:.0f} s")
    rows = score_frames(frames, tuple(a.thresholds))
    for r in rows:
        print(f"  thr {r['threshold']:.2f}  {r['group']:<13} n={r['n']:<6} recall {r['recall']:.3f}  "
              f"precision {r['precision']:.3f}  corner error {r['corner_error']}")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    return 0


def _run(a) -> int:
    from .model import Detector

    det = Detector(a.checkpoint, a.device)
    paths = sorted(p for d in a.frames for p in Path(d).glob("*.jpg")) + sorted(p for d in a.frames for p in Path(d).glob("*.png"))
    if a.only:
        keep = set(Path(a.only).read_text(encoding="utf-8").split())
        paths = [p for p in paths if p.name in keep or str(p) in keep]
    tx0, ty0, tx1, ty1 = (float(v) for v in a.table.split(","))
    t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as f:
        for k, p in enumerate(paths):
            with Image.open(p) as im:
                w, h = im.size
                window = (tx0 * w, ty0 * h, tx1 * w, ty1 * h)
                dets = det.detect(im, window, a.card_px * h / 1080, target=a.target, threshold=a.threshold)
            f.write(json.dumps({"frame": str(p), "width": w, "height": h, "window": [round(v) for v in window], "cards": dets}) + "\n")
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(paths)} frames in {time.time() - t0:.0f} s")
    return 0


def _score_real(a) -> int:
    from .evaluate import score_real

    dets = {}
    for path in a.dets:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                dets[Path(r["frame"]).name] = r["cards"]
    crops = json.loads(Path(a.crops).read_text(encoding="utf-8"))
    with open(a.labels, encoding="utf-8") as f:
        labels = {r["file"]: r["printing_id"] for r in csv.DictReader(f)}
    rows = score_real(dets, crops, labels)
    for r in rows:
        print(f"  {r['group']:<10} n={r['n']:<6} recall {r['recall']:.3f}" + (f"  class right {r['class_right']:.3f}" if r.get("class_right") is not None else ""))
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["group", "n", "recall", "class_right"])
            w.writeheader(); w.writerows(rows)
    return 0


def _onnx(a) -> int:
    import numpy as np
    import torch
    import torchvision.transforms.functional as F

    from .export import TILE
    from .model import Detector
    from .onnx import OUTPUTS, export, session, to_fp16

    det = Detector(a.checkpoint, "cpu")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fp32 = export(det, out / f"{Path(a.checkpoint).stem}.onnx")
    paths = [fp32, to_fp16(fp32, fp32.with_suffix(".fp16.onnx"))]
    tile = torch.randint(0, 256, (1, 3, TILE, TILE), generator=torch.Generator().manual_seed(0)) / 255
    with torch.inference_mode():
        want = det.net(F.normalize(tile, det.rf.means, det.rf.stds))
    # A random tile has no clear best proposals: a copy can rank them otherwise, and those queries' outputs then
    # differ a lot. The queries that agree say more; real frames are the test that counts.
    print("against PyTorch on a random tile (max |diff| per output; queries whose logits agree to 0.05):")
    for p in paths:
        got = dict(zip(OUTPUTS, session(p).run(list(OUTPUTS), {"tiles": tile.numpy()})))
        diff = {k: np.abs(got[k] - want[k].numpy()) for k in OUTPUTS}
        agree = int((diff["pred_logits"].max(-1) < 0.05).sum())
        print(f"  {p.name} ({p.stat().st_size / 1e6:.0f} MB): " + ", ".join(f"{k} {d.max():.2g}" for k, d in diff.items())
              + f"; {agree} of {diff['pred_logits'].shape[1]} queries agree")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.detect", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="a synthetic run -> COCO keypoint tiles (train/valid)")
    e.add_argument("--run", nargs="+", required=True, help="one or more synthetic runs")
    e.add_argument("--out", required=True)
    e.add_argument("--scales", type=int, default=1, help="random scales per frame")
    e.add_argument("--target", default="45,110", help="card long side in the tiles, px (log-uniform range)")
    e.add_argument("--overlap", type=float, default=0.2)
    e.add_argument("--val-every", type=int, default=10, help="every Nth board goes to valid/")
    e.add_argument("--quality", type=int, default=95, help="JPEG quality of the tiles")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(fn=_export)

    t = sub.add_parser("train", help="fine-tune RF-DETR keypoint on exported tiles")
    t.add_argument("--dataset", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--grad-accum", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--device", help="cuda, cpu or mps (default: auto)")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", help="a checkpoint to continue from")
    t.set_defaults(fn=_train)

    k = sub.add_parser("pack", help="a trained checkpoint -> a float16 copy for inference, half the size")
    k.add_argument("--checkpoint", required=True)
    k.add_argument("--out", required=True)
    k.set_defaults(fn=_pack)

    v = sub.add_parser("evaluate", help="recall by visible share, precision and corner error on a synthetic run")
    v.add_argument("--run", nargs="+", required=True, help="the synthetic runs the tiles came from")
    v.add_argument("--checkpoint", required=True)
    v.add_argument("--val-every", type=int, default=10, help="must match the export")
    v.add_argument("--max-frames", type=int)
    v.add_argument("--target", type=float, default=70.0, help="card long side the window is scaled to, px")
    v.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5])
    v.add_argument("--device")
    v.add_argument("--out", help="CSV of the numbers (safe to share)")
    v.set_defaults(fn=_evaluate)

    r = sub.add_parser("run", help="detect cards in real frames -> JSONL (private)")
    r.add_argument("--frames", nargs="+", required=True, help="folders of frames")
    r.add_argument("--only", help="a file listing the frames to use (e.g. overhead-frames.txt)")
    r.add_argument("--table", required=True, help="the camera window as fractions x0,y0,x1,y1")
    r.add_argument("--card-px", type=float, required=True, help="card long side at 1080p, px")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--target", type=float, default=70.0)
    r.add_argument("--threshold", type=float, default=0.3)
    r.add_argument("--device")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=_run)

    q = sub.add_parser("score-real", help="recall of the reviewed isolated cards of an M0 broadcast")
    q.add_argument("--dets", nargs="+", required=True, help="JSONL from `run`")
    q.add_argument("--crops", required=True, help="crops.json written by matcrops")
    q.add_argument("--labels", required=True, help="labels.csv of those crops")
    q.add_argument("--out", help="CSV of the numbers (safe to share)")
    q.set_defaults(fn=_score_real)

    o = sub.add_parser("onnx", help="weights -> <name>.onnx and <name>.fp16.onnx for the browser (private)")
    o.add_argument("checkpoint")
    o.add_argument("out_dir")
    o.set_defaults(fn=_onnx)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

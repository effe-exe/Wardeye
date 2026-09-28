# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""RiftEye live: the cards on the table of a recording or a live stream, named as it plays.

    python -m rifteye_ml.live --source match.mp4 --layout la-rq
    python -m rifteye_ml.live --source https://www.twitch.tv/videos/2885620401 --start 14:54:00 --layout la-rq
    python -m rifteye_ml.live --source twitch.tv/riftbound --layout la-rq

It opens http://127.0.0.1:8765 in the browser: the video with every card it finds boxed and named
(point at one to see it), the cards on each player's side, and the plays as they happen. On the
first run it fetches the public card catalogue and art (about 1.2 GB) into ~/rifteye-data.
Everything stays on this machine. Only the table is looked at: hands and face-down cards are
hidden information and never processed (D-005).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
from PIL import Image

from .. import catalog as cat
from .layouts import LAYOUTS

DATA = Path(os.environ.get("RIFTEYE_DATA", Path.home() / "rifteye-data"))
ENCODER = "colorgrid/trim0.03+dhash/trim0.03"  # colour and structure together (D-019)


def seconds(s: str) -> float:
    """'90', '1:30' or '14:54:00'."""
    parts = [float(p) for p in str(s).split(":")]
    return sum(v * 60 ** k for k, v in enumerate(reversed(parts)))


def ensure_catalogue(catalog: Path | None, cache: Path) -> Path:
    """The catalogue to use: the given one, else the one with the supplement, else the official one,
    fetched from the public card gallery when there is none yet; and every official picture cached."""
    if catalog is None:
        found = [DATA / "catalog" / n for n in ("catalog-plus.jsonl", "catalog.jsonl") if (DATA / "catalog" / n).exists()]
        if found:
            catalog = found[0]
        else:
            print("First run: fetching the public card catalogue into", DATA, flush=True)
            feed, catalog = DATA / "catalog" / "feed", DATA / "catalog" / "catalog.jsonl"
            cat.fetch_feed(feed)
            cat.write_catalog(cat.from_feed(cat.load_feed(feed)), catalog)
    missing = [r for r in cat.read_catalog(catalog)
               if not r["image_url"].startswith(cat.SUPPLEMENT_SCHEME) and not cat.cache_path(cache, r["image_url"]).exists()]
    if missing:
        print(f"Fetching {len(missing)} card pictures from the public card gallery (once, about 1 MB each)", flush=True)
        cat.download_images(missing, cache, workers=8)
    return catalog


def gallery(enc, rows: list[dict], cache: Path, embed_cache: Path, scales: list[int]):
    """The gallery pyramid, from the embedding cache when it has every level (then no art is loaded)."""
    from ..retrieval import Pyramid, at_long_side
    from ..spike import _cached_loader, catalog_key

    key = catalog_key(rows, 512)
    levels: dict[int, np.ndarray] = {}
    todo = []
    for s in scales:
        f = embed_cache / f"{hashlib.sha1(f'{enc.name}|{s}|{key}'.encode()).hexdigest()[:24]}.npy"
        if f.exists():
            levels[s] = np.load(f)
        else:
            todo.append((s, f))
    if todo:
        print(f"Embedding the gallery ({len(rows)} printings, once; later runs start at once)", flush=True)
        load = _cached_loader(str(cache), 512)
        art = [load(r) for r in rows]
        for s, f in todo:
            levels[s] = enc.embed([at_long_side(im, s) for im in art])
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, levels[s])
    return Pyramid(levels)


def best_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def jpeg(image: np.ndarray, width: int = 1280, quality: int = 78) -> bytes:
    im = Image.fromarray(image)
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.live", description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True, help="a video file, a Twitch channel or VOD URL, a YouTube URL, or an HLS URL")
    ap.add_argument("--layout", default="la-rq", choices=sorted(LAYOUTS), help="the broadcast's layout")
    ap.add_argument("--start", default="0", help="where to start in a recording or VOD: seconds or HH:MM:SS")
    ap.add_argument("--fps", type=float, default=5.0, help="frames looked at per second")
    ap.add_argument("--fast", action="store_true", help="a recording as fast as it decodes, not at 1x (tests)")
    ap.add_argument("--catalog", type=Path, help="default: ~/rifteye-data/catalog/catalog-plus.jsonl or catalog.jsonl")
    ap.add_argument("--cache", type=Path, default=DATA / "art", help="the card art cache")
    ap.add_argument("--embed-cache", type=Path, default=DATA / "embed-cache")
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--detector", type=Path, help="trained detector weights (detector-v0.pth); default: the bootstrap finder")
    ap.add_argument("--device", help="for the detector: cuda, mps or cpu (default: the best there is)")
    ap.add_argument("--title", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this much media time (tests)")
    a = ap.parse_args(argv)

    from ..encoders import get_encoder
    from .pipeline import Recognizer
    from .server import LiveServer
    from .source import open_source

    layout = LAYOUTS[a.layout]
    catalog = ensure_catalogue(a.catalog, a.cache)
    rows = [r for r in cat.read_catalog(catalog) if cat.cache_path(a.cache, r["image_url"]).exists()]
    if not rows:
        raise SystemExit(f"no card art cached in {a.cache} for {catalog}")
    by_pid = {r["printing_id"]: r for r in rows}
    enc = get_encoder(a.encoder)
    px = layout.card_px(1080)
    scales = sorted({int(round(px * f / 10) * 10) for f in (0.8, 0.9, 1.0)})
    pyr = gallery(enc, rows, a.cache, a.embed_cache, scales)
    finder = None
    if a.detector:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # a Mac GPU runs what it can, the CPU the rest
        from ..detect.model import Detector
        from .pipeline import detector_boxes

        det = Detector(a.detector, a.device or best_device())

        def finder(t: float, image: np.ndarray) -> list:
            h, w = image.shape[:2]
            return detector_boxes(det.detect(Image.fromarray(image), layout.box(w, h), layout.card_px(h)))
    rec = Recognizer(layout, rows, enc, pyr, title=a.title, fps=a.fps, finder=finder)

    def art(pid: str) -> Path | None:
        r = by_pid.get(pid)
        return cat.cache_path(a.cache, r["image_url"]) if r else None

    server = LiveServer(a.host, a.port, art=art, title=a.title or layout.title)
    url = server.start()
    print(f"RiftEye live: {url}  ({len(rows)} printings, {a.encoder}, layout {layout.name})", flush=True)
    server.publish_state({"t": 0, "status": "starting", "message": f"opening {a.source}", "title": rec.title,
                          "frame": {"width": 1920, "height": 1080}, "fps": {"source": a.fps, "processed": 0},
                          "latency_s": 0, "players": [], "tracks": []})
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()

    rate, last, status, message = 0.0, None, "ended", "the stream ended"
    size = (1920, 1080)
    try:
        with open_source(a.source, fps=a.fps, realtime=not a.fast, start=seconds(a.start), height=1080) as src:
            print(f"{src.kind}: {src.width}x{src.height} at {src.fps:g} fps", flush=True)
            for fr in src:
                state, events = rec.step(fr.t, fr.image)
                size = (fr.image.shape[1], fr.image.shape[0])
                now = time.monotonic()
                if last is not None:
                    rate = 0.8 * rate + 0.2 / max(1e-3, now - last) if rate else 1 / max(1e-3, now - last)
                last = now
                state["fps"] = {"source": src.fps, "processed": round(rate, 1)}
                state["latency_s"] = round(now - fr.wall, 2)
                server.publish_frame(jpeg(fr.image), fr.t, fr.image.shape[1], fr.image.shape[0])
                server.publish_state(state)
                for e in events:
                    server.publish_event(e)
                    print(f"  {time.strftime('%H:%M:%S', time.gmtime(fr.t))} {e['side']:>5}  {e['text']}", flush=True)
                if a.max_seconds and fr.t - seconds(a.start) >= a.max_seconds:
                    message = f"stopped after {a.max_seconds:g} s"
                    break
    except KeyboardInterrupt:
        message = "stopped"
    except Exception as e:  # noqa: BLE001 - shown on the page, then raised
        server.publish_state({"t": 0, "status": "error", "message": str(e), "title": rec.title,
                              "frame": {"width": size[0], "height": size[1]}, "fps": {"source": a.fps, "processed": 0},
                              "latency_s": 0, "players": [], "tracks": []})
        time.sleep(1)
        server.stop()
        raise
    t_end = max((tr.last for tr in rec.tracks.values()), default=0.0)
    server.publish_state({**rec.state(t_end, *size), "status": status, "message": message})
    stay = not a.max_seconds and message != "stopped"  # Ctrl+C quits; a test run (--max-seconds) ends here
    print(message + (" - the page stays up; Ctrl+C to quit" if stay else ""), flush=True)
    try:
        while stay:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

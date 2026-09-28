# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""RiftEye live: the cards on the table of a recording or a live stream, named as it plays.

    python -m rifteye_ml.live --source match.mp4 --layout la-rq
    python -m rifteye_ml.live --source https://www.twitch.tv/videos/2885620401 --start 14:54:00 --layout la-rq
    python -m rifteye_ml.live --source twitch.tv/riftbound --layout la-rq
    python -m rifteye_ml.live --source 'https://www.twitch.tv/videos/2854086989?t=3h40m0s'   # any replay, from its link
    python -m rifteye_ml.live --source browser     # on the Twitch player itself, with the extension (apps/extension)

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
import math
import os
import re
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
# ponytail: the temperature belongs in the packed weights (`embed pack`); until it is there, embedder-v1's
EMBEDDER_T = 0.0347  # embedder-v1, fitted on held-out Barcelona and Los Angeles grand final crops


def seconds(s: str) -> float:
    """'90', '1:30' or '14:54:00'."""
    parts = [float(p) for p in str(s).split(":")]
    return sum(v * 60 ** k for k, v in enumerate(reversed(parts)))


def link_start(link: str) -> float:
    """Where a Twitch or YouTube link says to start: `?t=1h2m3s`, `&t=90s` or `t=90`, as Share, "Copy link at
    current time" gives it; 0 without one."""
    m = re.search(r"[?&#]t=(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?(?=&|$)", link)
    if not m or not any(m.groups()):
        return 0.0
    h, mnt, sec = (int(g or 0) for g in m.groups())
    return float(h * 3600 + mnt * 60 + sec)


def temperature_for(spec: str, enc, given: float | None) -> float | None:
    """How an encoder's scores become confidence: the one given, the one packed with the weights, embedder-v1's
    for a fine-tuned embedder, or None for the pipeline's own (fitted for colour and structure)."""
    if given:
        return given
    meta = getattr(enc, "meta", None) or {}
    if meta.get("temperature"):
        return float(meta["temperature"])
    return EMBEDDER_T if spec.startswith("embedder:") else None


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
        still = [r for r in missing if not cat.cache_path(cache, r["image_url"]).exists()]
        if len(still) > 10:  # a broken link or two is tolerable; a gallery with holes names cards wrongly
            raise SystemExit(f"{len(still)} card pictures could not be fetched (the errors are above). "
                             "Run the same command again to retry: what did arrive is kept.")
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


def find_layout(frames, det, every: float = 1.0, give_up: float = 120.0):
    """`--layout auto`: a layout from the first table shots (`autolayout.py`), one look a second over the
    last five; SystemExit when none shows a table in the first `give_up` s of media."""
    from .autolayout import auto_layout

    detect = (lambda f, box, px: det.detect(Image.fromarray(f), box, px)) if det is not None else None
    seen, last_t, t0 = [], None, None
    for fr in frames:
        t0 = fr.t if t0 is None else t0
        if last_t is not None and fr.t - last_t < every:
            continue
        last_t = fr.t
        seen = (seen + [fr.image])[-5:]
        if len(seen) == 5 and (layout := auto_layout(seen, detect)) is not None:
            return layout
        if fr.t - t0 > give_up:
            break
    raise SystemExit(f"no table found in the first {give_up / 60:g} minutes: name the broadcast's layout with --layout")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rifteye_ml.live", description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True, help="a video file, a Twitch channel or VOD URL, a YouTube URL, an HLS "
                                                    "URL, or browser: the frames the RiftEye extension sends")
    ap.add_argument("--layout", default="auto", choices=sorted(LAYOUTS) + ["auto"],
                    help="the broadcast's layout (default auto: found from its first table shots)")
    ap.add_argument("--start", default="0", help="where to start in a recording or VOD: seconds or HH:MM:SS "
                                                 "(default: the link's own ?t=, else the beginning)")
    ap.add_argument("--fps", type=float, default=5.0, help="frames looked at per second")
    ap.add_argument("--fast", action="store_true", help="a recording as fast as it decodes, not at 1x (tests)")
    ap.add_argument("--catalog", type=Path, help="default: ~/rifteye-data/catalog/catalog-plus.jsonl or catalog.jsonl")
    ap.add_argument("--cache", type=Path, default=DATA / "art", help="the card art cache")
    ap.add_argument("--embed-cache", type=Path, default=DATA / "embed-cache")
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--temperature", type=float,
                    help="how the encoder's scores become confidence (default: fitted for the default encoder; "
                         "another encoder needs its own)")
    ap.add_argument("--detector", type=Path, help="trained detector weights (detector-v0.pth); default: the bootstrap finder")
    ap.add_argument("--device", help="for the detector: cuda, mps or cpu (default: the best there is)")
    ap.add_argument("--det-score", type=float, default=0.4,
                    help="the detector's confidence below which a box is dropped (higher: fewer stray boxes)")
    ap.add_argument("--title", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this much media time (tests)")
    a = ap.parse_args(argv)

    from ..encoders import get_encoder
    from .browser import BrowserSource
    from .pipeline import Recognizer, detector_boxes
    from .server import LiveServer
    from .source import open_source

    catalog = ensure_catalogue(a.catalog, a.cache)
    rows = [r for r in cat.read_catalog(catalog) if cat.cache_path(a.cache, r["image_url"]).exists()]
    if not rows:
        raise SystemExit(f"no card art cached in {a.cache} for {catalog}")
    by_pid = {r["printing_id"]: r for r in rows}
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # a Mac GPU runs what it can, the CPU the rest
    enc = get_encoder(a.encoder)
    temperature = temperature_for(a.encoder, enc, a.temperature)
    det = None
    if a.detector:
        from ..detect.model import Detector

        det = Detector(a.detector, a.device or best_device())

    def art(pid: str) -> Path | None:
        r = by_pid.get(pid)
        return cat.cache_path(a.cache, r["image_url"]) if r else None

    title = a.title or (LAYOUTS[a.layout].title if a.layout in LAYOUTS else "RiftEye live")
    browser = BrowserSource(fps=a.fps) if a.source == "browser" else None
    server = LiveServer(a.host, a.port, art=art, title=title, on_frame=browser.post if browser else None)
    url = server.start()
    print(f"RiftEye live: {url}  ({len(rows)} printings, {a.encoder}, layout {a.layout})", flush=True)

    def status(kind: str, message: str, size: tuple[int, int] = (1920, 1080)) -> None:
        server.publish_state({"t": 0, "status": kind, "message": message, "title": title,
                              "frame": {"width": size[0], "height": size[1]}, "fps": {"source": a.fps, "processed": 0},
                              "latency_s": 0, "players": [], "tracks": []})

    if browser is not None:
        status("starting", "waiting for the RiftEye extension: play a Riftbound video on Twitch")
        print("waiting for the RiftEye extension: play a Riftbound video on Twitch in Chrome", flush=True)
    else:
        status("starting", f"opening {a.source}")
    if not a.no_browser and browser is None:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()

    start = seconds(a.start) or link_start(a.source)
    rate, last, end_status, message = 0.0, None, "ended", "the stream ended"
    size = (1920, 1080)
    rec = None
    try:
        src_ctx = browser if browser is not None else open_source(a.source, fps=a.fps, realtime=not a.fast,
                                                                    start=start, height=1080)
        with src_ctx as src:
            if browser is None:
                print(f"{src.kind}: {src.width}x{src.height} at {src.fps:g} fps", flush=True)
            frames = iter(src)
            layout, video, t_first, last_t = None, None, None, None
            while True:  # once per broadcast: the extension's viewer can open another video
                if layout is None or a.layout == "auto" and video is not None:
                    if a.layout == "auto":
                        status("starting", "finding the table and the size of a card")
                        layout = find_layout(frames, det, give_up=math.inf if browser is not None else 120.0)
                        print(f"layout found: table {layout.table}, cards {layout.card_long_1080:g} px long at 1080p", flush=True)
                    else:
                        layout = LAYOUTS[a.layout]
                    px = layout.card_px(1080)
                    status("starting", "getting the cards ready for this table (a minute or two the first time)")
                    pyr = gallery(enc, rows, a.cache, a.embed_cache, sorted({int(round(px * f / 10) * 10) for f in (0.8, 0.9, 1.0)}))
                finder = None
                if det is not None:
                    def finder(t: float, image: np.ndarray, layout=layout) -> list:
                        h, w = image.shape[:2]
                        return detector_boxes(det.detect(Image.fromarray(image), layout.box(w, h), layout.card_px(h)),
                                              min_score=a.det_score)
                rec = Recognizer(layout, rows, enc, pyr, title=title, fps=a.fps, finder=finder,
                                 **({"temperature": temperature} if temperature else {}))
                switched = False
                for fr in frames:
                    key = getattr(fr, "video", None)
                    if browser is not None and video is not None and key != video:
                        video, switched = key, True  # another video: its own table, its own board
                        break
                    video = key
                    if last_t is not None and not -1.0 <= fr.t - last_t <= 15.0:
                        rec = Recognizer(layout, rows, enc, pyr, title=title, fps=a.fps, finder=finder,
                                         **({"temperature": temperature} if temperature else {}))  # a jump: a new board
                    last_t = fr.t
                    t_first = fr.t if t_first is None else t_first
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
                    if a.max_seconds and fr.t - t_first >= a.max_seconds:
                        message = f"stopped after {a.max_seconds:g} s"
                        break
                if not switched:
                    break
    except KeyboardInterrupt:
        message = "stopped"
    except (Exception, SystemExit) as e:  # noqa: BLE001 - shown on the page, then raised
        status("error", str(e), size)
        time.sleep(1)
        server.stop()
        raise
    if rec is not None:
        t_end = max((tr.last for tr in rec.tracks.values()), default=0.0)
        server.publish_state({**rec.state(t_end, *size), "status": end_status, "message": message})
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

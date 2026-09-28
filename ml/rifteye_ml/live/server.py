# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The local web server for the live runner.

`LiveServer` shows the recognition loop's work at http://127.0.0.1:8765 while it runs. It knows
nothing about cards or Riftbound: it just holds the newest video frame, the newest state snapshot
and a capped feed of events, and serves them to whatever browser tabs are open.

`publish_*` only swaps a reference and wakes any client thread waiting on it, so each call is O(1)
and can never be slowed by a client. Every connection gets its own thread (`ThreadingHTTPServer`);
a tab that stops reading blocks only the thread serving that tab, never the recognition loop or any
other viewer, because publishers never touch a client socket and reader threads never hold the lock
while writing to one.

    server = LiveServer(port=0, art=lambda printing_id: art_path_for(printing_id))
    url = server.start()                # e.g. "http://127.0.0.1:51234/"
    ...
    server.publish_frame(jpeg_bytes, t=12.3, width=1920, height=1080)
    server.publish_state({...})         # schema: see live/static/app.js and the live runner's README
    server.publish_event({...})
    server.stop()
"""
from __future__ import annotations

import errno
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlsplit

from PIL import Image

STATIC_DIR = Path(__file__).resolve().parent / "static"
# request path -> (file in STATIC_DIR, Content-Type)
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
_EVENTS_KEPT = 200      # publish_event: how much feed history a newly connecting client is sent
_SSE_POLL_S = 0.2       # also the state throttle: at most 5 state events per client per second
_SSE_KEEPALIVE_S = 15.0
_ART_LONG_PX = 360
_MJPEG_BOUNDARY = "rifteyeframe"
_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _starting_state(title: str) -> dict:
    """What /state.json and /events show before the recognition loop has published anything."""
    return {
        "t": 0.0, "status": "starting", "message": "", "title": title,
        "frame": {"width": 0, "height": 0}, "fps": {"source": 0.0, "processed": 0.0},
        "latency_s": 0.0, "players": [], "tracks": [],
    }


def _shrink_to_long_side(im: Image.Image, long_side: int) -> Image.Image:
    """Downscale-only resize so the longer side is at most `long_side` px (never enlarges small art)."""
    scale = long_side / max(im.size)
    if scale >= 1:
        return im
    size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
    return im.resize(size, Image.LANCZOS)


class LiveServer:
    """Runs the live page's HTTP server in a background thread.

    Holds only the *latest* frame, state and a short event feed, never a per-client queue, so a
    slow or absent browser tab can only fall behind, never hold up the recognition loop or another
    tab. State and event dicts are treated as immutable once published: callers should hand over a
    fresh dict each time rather than mutate one they already published.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765,
                 art: Callable[[str], Path | None] | None = None, title: str = "RiftEye live") -> None:
        self._host = host
        self._port = port
        self._art = art
        self._title = title
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

        # One lock/condition guards the frame, state and event feed: publish_* holds it only long
        # enough to swap a reference, readers wait on it and release it before touching a socket.
        self._cond = threading.Condition()
        self._closed = False
        self._frame: tuple[bytes, float, int, int] | None = None
        self._frame_seq = 0
        self._state: dict = _starting_state(title)
        self._state_seq = 0
        self._events: deque[tuple[int, dict]] = deque(maxlen=_EVENTS_KEPT)
        self._event_seq = 0

        # Separate from `_cond`: resizing art can take a few ms and must never hold up publish_*.
        self._art_lock = threading.Lock()
        self._art_cache: dict[str, bytes] = {}
        self._static_cache: dict[str, bytes] = {}

    # -- lifecycle ------------------------------------------------------------------------------

    def start(self) -> str:
        """Serves the page in a daemon thread and returns its URL. port=0 picks a free port; a port another
        program holds gives way to the next free one of the nine after it."""
        if self._httpd is not None:
            raise RuntimeError("LiveServer is already started")
        for port in [self._port] if self._port == 0 else range(self._port, self._port + 10):
            try:
                httpd = ThreadingHTTPServer((self._host, port), _Handler)
                break
            except OSError as e:
                if e.errno != errno.EADDRINUSE or port == self._port + 9:
                    raise
        httpd.daemon_threads = True
        httpd.live = self  # type: ignore[attr-defined]  # the handler's only way back to this object
        self._httpd = httpd
        self._port = httpd.server_address[1]
        self._thread = threading.Thread(target=httpd.serve_forever, name="rifteye-live", daemon=True)
        self._thread.start()
        return f"http://{self._host}:{self._port}/"

    def stop(self) -> None:
        """Stops serving and frees the port. Safe to call more than once."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()  # wake any client thread blocked waiting for new data
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # -- publish_* (called from the recognition loop; must never block) -------------------------

    def publish_frame(self, jpeg: bytes, t: float, width: int, height: int) -> None:
        """The newest video frame. Replaces whatever frame was published before it."""
        with self._cond:
            self._frame = (jpeg, t, width, height)
            self._frame_seq += 1
            self._cond.notify_all()

    def publish_state(self, state: dict) -> None:
        """The newest snapshot (schema: live/static/app.js). Replaces the previous one."""
        with self._cond:
            self._state = state
            self._state_seq += 1
            self._cond.notify_all()

    def publish_event(self, event: dict) -> None:
        """Appends to the feed. Only the most recent `_EVENTS_KEPT` are kept."""
        with self._cond:
            self._event_seq += 1
            self._events.append((self._event_seq, event))
            self._cond.notify_all()

    # -- read side, used only by the request handler below ---------------------------------------

    def state_snapshot(self) -> dict:
        with self._cond:
            return self._state

    def wait_frame(self, last_seq: int) -> tuple[bytes | None, int]:
        """Blocks until a frame newer than `last_seq` is published, or the server stops (then
        returns `(None, last_seq)`). A long poll rather than a queue: a client that calls this in
        a loop always gets the newest frame, never a backlog of stale ones."""
        with self._cond:
            while not self._closed and (self._frame is None or self._frame_seq == last_seq):
                self._cond.wait(timeout=1.0)
            if self._closed:
                return None, last_seq
            jpeg, _t, _w, _h = self._frame  # type: ignore[misc]  # the loop above guarantees this
            return jpeg, self._frame_seq

    def sse_connect(self) -> tuple[dict, int, list[dict], int]:
        """What a newly connected /events client is sent right away: the state and the recent feed."""
        with self._cond:
            return self._state, self._state_seq, [e for _, e in self._events], self._event_seq

    def sse_wait(self, seen_state_seq: int, seen_event_seq: int,
                 timeout: float) -> tuple[dict | None, int, list[dict], int, bool]:
        """Waits up to `timeout` s for anything newer than what this client has already seen.
        Returns (state-or-None, state_seq, new events, event_seq, closed)."""
        with self._cond:
            deadline = time.monotonic() + timeout
            while (self._state_seq == seen_state_seq and self._event_seq == seen_event_seq
                   and not self._closed):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)
            state = self._state if self._state_seq != seen_state_seq else None
            events = [e for seq, e in self._events if seq > seen_event_seq]
            return state, self._state_seq, events, self._event_seq, self._closed

    def art_jpeg(self, printing_id: str) -> bytes | None:
        """The card's art as a JPEG no longer than 360 px, from the `art` callback, cached by
        printing id. Uses its own lock, never `_cond`'s: a resize must never delay publish_*."""
        with self._art_lock:
            cached = self._art_cache.get(printing_id)
        if cached is not None:
            return cached
        if self._art is None:
            return None
        try:
            path = self._art(printing_id)
            if path is None:
                return None
            with Image.open(Path(path)) as im:
                im = _shrink_to_long_side(im.convert("RGB"), _ART_LONG_PX)
                buf = BytesIO()
                im.save(buf, "JPEG", quality=85)
                data = buf.getvalue()
        except Exception:
            return None  # a missing, unreadable or corrupt file is a 404, not a crashed thread
        with self._art_lock:
            self._art_cache[printing_id] = data
        return data

    def host_ok(self, host: str) -> bool:
        """Serving this machine only, a request must name this machine: a page on another site that
        points its own domain at 127.0.0.1 (DNS rebinding) gets nothing. On a LAN address, any name."""
        host = host.strip().lower()
        if self._host not in _LOOPBACK or not host:  # a browser always sends one; a bare HTTP/1.0 client is local
            return True
        name = host[1:host.find("]")] if host.startswith("[") else host.rsplit(":", 1)[0]
        return name in _LOOPBACK

    def static_bytes(self, name: str) -> bytes:
        if name not in self._static_cache:
            self._static_cache[name] = (STATIC_DIR / name).read_bytes()
        return self._static_cache[name]


class _Handler(BaseHTTPRequestHandler):
    """One instance per connection (ThreadingHTTPServer); `self.server.live` is the LiveServer."""

    server_version = "RiftEyeLive/1"

    def log_message(self, format: str, *args) -> None:
        pass  # a local dev tool: keep stdout/stderr quiet

    def do_GET(self) -> None:
        live: LiveServer = self.server.live  # type: ignore[attr-defined]
        path = urlsplit(self.path).path
        if not live.host_ok(self.headers.get("Host", "")):
            self.send_error(403)
            return
        try:
            if path in _STATIC_FILES:
                self._static(live, path)
            elif path == "/state.json":
                self._json(live.state_snapshot())
            elif path == "/stream.mjpg":
                self._mjpeg(live)
            elif path == "/events":
                self._sse(live)
            elif path.startswith("/art/"):
                self._art(live, path)
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # the client went away mid-response; nothing left to do

    def _static(self, live: LiveServer, path: str) -> None:
        name, content_type = _STATIC_FILES[path]
        try:
            body = live.static_bytes(name)
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _art(self, live: LiveServer, path: str) -> None:
        rest = path[len("/art/"):]
        if not rest.endswith(".jpg"):
            self.send_error(404)
            return
        printing_id = unquote(rest[: -len(".jpg")])
        body = live.art_jpeg(printing_id) if printing_id else None
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mjpeg(self, live: LiveServer) -> None:
        self.send_response(200)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}")
        self.end_headers()
        last_seq = 0
        while True:
            jpeg, last_seq = live.wait_frame(last_seq)
            if jpeg is None:
                return  # the server is stopping
            self.wfile.write(f"--{_MJPEG_BOUNDARY}\r\n"
                             f"Content-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
            self.wfile.write(jpeg)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

    def _sse(self, live: LiveServer) -> None:
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.end_headers()
        state, state_seq, events, event_seq = live.sse_connect()
        self._sse_send("state", state)
        for e in events:
            self._sse_send("feed", e)
        self.wfile.flush()
        pending_state: dict | None = None
        last_state_sent = 0.0
        last_activity = time.monotonic()
        while True:
            wait = _SSE_POLL_S
            if pending_state is not None:
                wait = max(0.0, min(wait, _SSE_POLL_S - (time.monotonic() - last_state_sent)))
            new_state, state_seq, new_events, event_seq, closed = live.sse_wait(state_seq, event_seq, wait)
            if closed:
                return
            if new_state is not None:
                pending_state = new_state  # coalesce: only the latest state matters once we do send
            for e in new_events:
                self._sse_send("feed", e)
                last_activity = time.monotonic()
            now = time.monotonic()
            if pending_state is not None and now - last_state_sent >= _SSE_POLL_S:
                self._sse_send("state", pending_state)
                pending_state = None
                last_state_sent = now
                last_activity = now
            if now - last_activity >= _SSE_KEEPALIVE_S:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                last_activity = now

    def _sse_send(self, event: str, data: dict) -> None:
        self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
        self.wfile.flush()

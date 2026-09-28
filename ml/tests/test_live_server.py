import json
import os
import socket
import time
import urllib.error
from io import BytesIO
from urllib.parse import urlsplit
from urllib.request import urlopen

import pytest
from PIL import Image

from rifteye_ml.live.server import LiveServer

SAMPLE_STATE = {
    "t": 12.5, "status": "live", "message": "", "title": "test match",
    "frame": {"width": 1920, "height": 1080}, "fps": {"source": 5.0, "processed": 3.1},
    "latency_s": 0.2,
    "players": [{"side": "left", "label": "Player 1", "legend": None},
                {"side": "right", "label": "Player 2", "legend": None}],
    "tracks": [],
}
EVENT_1 = {"t": 10.0, "kind": "played", "text": "Blade Dancer played",
           "printing_id": "SFD-195a", "track": "t1", "side": "left"}
EVENT_2 = {"t": 11.0, "kind": "left", "text": "A card left the table",
           "printing_id": None, "track": "t1", "side": "left"}


def _jpeg(size=(64, 48), color=(40, 60, 200)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def _noisy_jpeg(size=(320, 240)) -> bytes:
    # incompressible-ish content, so the encoded bytes are big enough to fill a tiny TCP window
    im = Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _port_of(url: str) -> int:
    return urlsplit(url).port


def _recv_until(sock, needle: bytes, limit: int = 500_000) -> bytes:
    """Accumulates raw bytes from a streaming response until `needle` shows up. A plain socket
    recv() (unlike a buffered http.client response's .read(n)) returns whatever is available
    rather than blocking to fill the requested size, which matters here: the server may go idle
    (waiting for the next publish) well before `limit` bytes ever arrive."""
    buf = b""
    while needle not in buf and len(buf) < limit:
        chunk = sock.recv(8192)
        if not chunk:
            break
        buf += chunk
    return buf


def _read_sse_messages(f, count):
    """Reads `count` complete "event: ...\\ndata: ...\\n\\n" messages from an SSE file object,
    skipping keep-alive comment lines."""
    out = []
    event, data_lines = None, []
    while len(out) < count:
        line = f.readline()
        if not line:
            raise AssertionError("connection closed before enough SSE messages arrived")
        line = line.decode("utf-8").rstrip("\n").rstrip("\r")
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        elif line == "":
            if event is not None:
                out.append((event, "".join(data_lines)))
            event, data_lines = None, []
    return out


@pytest.fixture
def server():
    srv = LiveServer(host="127.0.0.1", port=0)
    url = srv.start()
    try:
        yield srv, url
    finally:
        srv.stop()


# --------------------------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------------------------

def test_start_returns_a_working_url_and_stop_frees_the_port(server):
    srv, url = server
    assert url == f"http://127.0.0.1:{_port_of(url)}/"
    with urlopen(url, timeout=5) as resp:
        assert resp.status == 200

    port = _port_of(url)
    srv.stop()
    # the exact same port can be bound again right away
    srv2 = LiveServer(host="127.0.0.1", port=port)
    url2 = srv2.start()
    try:
        assert _port_of(url2) == port
        with urlopen(url2, timeout=5) as resp:
            assert resp.status == 200
    finally:
        srv2.stop()


def test_stop_is_safe_to_call_twice():
    srv = LiveServer(port=0)
    srv.start()
    srv.stop()
    srv.stop()  # must not raise


# --------------------------------------------------------------------------------------------
# /state.json
# --------------------------------------------------------------------------------------------

def test_state_json_round_trips(server):
    srv, url = server
    srv.publish_state(SAMPLE_STATE)
    with urlopen(url + "state.json", timeout=5) as resp:
        assert resp.headers["Content-Type"].startswith("application/json")
        data = json.loads(resp.read())
    assert data == SAMPLE_STATE


def test_state_json_has_a_sensible_default_before_any_publish(server):
    _srv, url = server
    with urlopen(url + "state.json", timeout=5) as resp:
        data = json.loads(resp.read())
    assert data["status"] == "starting"
    assert data["tracks"] == [] and data["players"] == []


# --------------------------------------------------------------------------------------------
# static files
# --------------------------------------------------------------------------------------------

def test_static_files_are_served_with_sensible_content_types(server):
    _srv, url = server
    checks = {"": "text/html", "app.js": "text/javascript", "style.css": "text/css"}
    for path, want in checks.items():
        with urlopen(url + path, timeout=5) as resp:
            assert resp.status == 200
            assert want in resp.headers["Content-Type"]
            assert len(resp.read()) > 0


# --------------------------------------------------------------------------------------------
# /art/<printing_id>.jpg
# --------------------------------------------------------------------------------------------

def test_art_is_resized_and_cached(tmp_path):
    img_path = tmp_path / "art.jpg"
    Image.new("RGB", (900, 500), (10, 200, 90)).save(img_path, "JPEG")
    calls = []

    def art(printing_id):
        calls.append(printing_id)
        return img_path

    srv = LiveServer(port=0, art=art)
    url = srv.start()
    try:
        with urlopen(url + "art/SFD-195a.jpg", timeout=5) as resp:
            assert resp.headers["Content-Type"] == "image/jpeg"
            body = resp.read()
        im = Image.open(BytesIO(body))
        assert max(im.size) <= 360

        with urlopen(url + "art/SFD-195a.jpg", timeout=5) as resp2:
            resp2.read()
        assert calls == ["SFD-195a"]  # the second request was served from the in-memory cache
    finally:
        srv.stop()


def test_art_ids_are_url_decoded(tmp_path):
    img_path = tmp_path / "art.jpg"
    Image.new("RGB", (100, 140), (5, 5, 5)).save(img_path, "JPEG")
    seen = []

    def art(printing_id):
        seen.append(printing_id)
        return img_path

    srv = LiveServer(port=0, art=art)
    url = srv.start()
    try:
        for encoded in ("SFD-T02.jpg", "VEN-189%2A.jpg", "VEN-189*.jpg"):
            with urlopen(url + "art/" + encoded, timeout=5) as resp:
                resp.read()
        # both spellings decode to the same id, so the cache means the callback only sees it once
        assert seen == ["SFD-T02", "VEN-189*"]
    finally:
        srv.stop()


def test_art_404_for_unknown_id_and_with_no_callback(tmp_path):
    img_path = tmp_path / "art.jpg"
    Image.new("RGB", (100, 100)).save(img_path, "JPEG")
    srv = LiveServer(port=0, art=lambda pid: img_path if pid == "SFD-195a" else None)
    url = srv.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urlopen(url + "art/UNKNOWN-001.jpg", timeout=5)
        assert exc.value.code == 404
    finally:
        srv.stop()

    srv2 = LiveServer(port=0)  # no art callback at all
    url2 = srv2.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urlopen(url2 + "art/SFD-195a.jpg", timeout=5)
        assert exc.value.code == 404
    finally:
        srv2.stop()


# --------------------------------------------------------------------------------------------
# /stream.mjpg
# --------------------------------------------------------------------------------------------

def test_stream_mjpg_sends_the_published_jpeg(server):
    srv, url = server
    jpeg = _jpeg()
    srv.publish_frame(jpeg, t=1.0, width=64, height=48)

    sock = socket.create_connection(("127.0.0.1", _port_of(url)), timeout=5)
    sock.sendall(b"GET /stream.mjpg HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    body = _recv_until(sock, jpeg)
    header, _, _ = body.partition(b"\r\n\r\n")
    assert header.startswith(b"HTTP/1.0 200") or header.startswith(b"HTTP/1.1 200")
    assert b"Content-Type: multipart/x-mixed-replace" in header
    assert jpeg in body
    sock.close()


# --------------------------------------------------------------------------------------------
# /events
# --------------------------------------------------------------------------------------------

def test_events_sends_state_and_feed_on_connect_then_a_new_state(server):
    srv, url = server
    srv.publish_state(SAMPLE_STATE)
    srv.publish_event(EVENT_1)
    srv.publish_event(EVENT_2)

    sock = socket.create_connection(("127.0.0.1", _port_of(url)), timeout=5)
    sock.settimeout(5)
    sock.sendall(b"GET /events HTTP/1.0\r\n\r\n")
    f = sock.makefile("rb")
    while True:  # skip the HTTP status line and headers
        line = f.readline()
        if line in (b"\r\n", b""):
            break

    msgs = _read_sse_messages(f, count=3)
    assert msgs[0] == ("state", json.dumps(SAMPLE_STATE))
    feed_texts = {json.loads(data)["text"] for kind, data in msgs[1:] if kind == "feed"}
    assert feed_texts == {EVENT_1["text"], EVENT_2["text"]}

    new_state = dict(SAMPLE_STATE, t=99.0, status="ended", message="done")
    srv.publish_state(new_state)
    more = _read_sse_messages(f, count=1)
    assert more == [("state", json.dumps(new_state))]

    sock.close()


# --------------------------------------------------------------------------------------------
# publish_* must never block on a slow or absent client
# --------------------------------------------------------------------------------------------

def test_publish_frame_never_blocks_on_a_client_that_never_reads(server):
    srv, url = server
    sock = socket.create_connection(("127.0.0.1", _port_of(url)), timeout=5)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
    sock.sendall(b"GET /stream.mjpg HTTP/1.0\r\n\r\n")
    # deliberately never read from this socket again

    frames = [_noisy_jpeg() for _ in range(50)]
    t0 = time.monotonic()
    for jpeg in frames:
        srv.publish_frame(jpeg, t=0.0, width=320, height=240)
    elapsed = time.monotonic() - t0

    sock.close()
    assert elapsed < 2.0, f"publish_frame took {elapsed:.2f}s with a non-reading client"


def test_publish_state_and_event_never_block_on_a_client_that_never_reads(server):
    srv, url = server
    sock = socket.create_connection(("127.0.0.1", _port_of(url)), timeout=5)
    sock.sendall(b"GET /events HTTP/1.0\r\n\r\n")
    # deliberately never read the response

    t0 = time.monotonic()
    for i in range(50):
        srv.publish_state(dict(SAMPLE_STATE, t=float(i)))
        srv.publish_event(dict(EVENT_1, t=float(i)))
    elapsed = time.monotonic() - t0

    sock.close()
    assert elapsed < 2.0, f"publish_state/publish_event took {elapsed:.2f}s with a non-reading client"


def test_only_this_machine_s_names_are_served_on_loopback(server):
    srv, url = server
    port = _port_of(url)

    def status(host):
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.sendall(f"GET /state.json HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode())
            return _recv_until(sock, b"\r\n").split(b" ")[1]

    assert status(f"127.0.0.1:{port}") == b"200" and status(f"localhost:{port}") == b"200"
    assert status(f"evil.example:{port}") == b"403"  # a rebound domain reaching this port
    assert LiveServer(host="0.0.0.0").host_ok("192.168.1.20:8765")  # on the LAN any name will do

import subprocess
import sys
import time

import numpy as np
import pytest

from rifteye_ml.live import source as live_source
from rifteye_ml.live.source import Frame, classify, open_source


def _clip(tmp_path, seconds: float = 3.0, size: str = "320x180", rate: int = 30, name: str = "clip.mp4") -> str:
    """A tiny synthetic H.264 clip (ffmpeg's lavfi testsrc), so tests need no real footage."""
    import imageio_ffmpeg

    path = tmp_path / name
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
           "-i", f"testsrc=size={size}:rate={rate}:duration={seconds}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return str(path)


def test_classify_every_spec_form(tmp_path):
    f = tmp_path / "video.mp4"
    f.write_bytes(b"not a real video, just needs to exist")
    assert classify(str(f)) == "file"
    assert classify(str(tmp_path / "missing.mp4")) == "file"

    assert classify("https://example.com/stream.m3u8") == "url"
    assert classify("http://example.com/clip.mp4") == "url"

    assert classify("twitch.tv/riftbound") == "twitch-live"
    assert classify("https://www.twitch.tv/riftbound") == "twitch-live"
    assert classify("https://twitch.tv/riftbound") == "twitch-live"

    assert classify("https://www.twitch.tv/videos/2885620401") == "twitch-vod"
    assert classify("twitch.tv/videos/2885620401") == "twitch-vod"
    assert classify("www.twitch.tv/videos/2885620401") == "twitch-vod"

    assert classify("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "youtube"
    assert classify("youtu.be/dQw4w9WgXcQ") == "youtube"
    assert classify("https://www.youtube.com/live/dQw4w9WgXcQ") == "youtube"

    # existing on disk wins over looking like a URL
    tricky = tmp_path / "twitch.tv"
    tricky.write_bytes(b"x")
    assert classify(str(tricky)) == "file"


def test_decodes_at_the_requested_fps_with_shape_dtype_and_t(tmp_path):
    clip = _clip(tmp_path, seconds=3.0)
    with open_source(clip, fps=5.0, realtime=False, height=None) as src:
        assert src.kind == "file" and src.live is False
        assert (src.width, src.height) == (320, 180)
        assert src.fps == 5.0
        frames = list(src)
    assert len(frames) == 15  # 3 s at 5 fps
    assert src.dropped == 0
    for i, fr in enumerate(frames):
        assert isinstance(fr, Frame)
        assert fr.index == i
        assert fr.image.shape == (180, 320, 3)
        assert fr.image.dtype == np.uint8
        assert fr.t == pytest.approx(i / 5.0, abs=1e-6)
        assert fr.wall > 0


def test_start_offset(tmp_path):
    clip = _clip(tmp_path, seconds=3.0)
    with open_source(clip, fps=5.0, realtime=False, start=1.0, height=None) as src:
        frames = list(src)
    assert [f.index for f in frames] == list(range(10))  # 2 s left of a 3 s clip, at 5 fps
    assert frames[0].t == pytest.approx(1.0, abs=1e-6)
    assert frames[-1].t == pytest.approx(1.0 + 9 / 5.0, abs=1e-6)


def test_height_scaling(tmp_path):
    clip = _clip(tmp_path, seconds=1.0)
    with open_source(clip, fps=5.0, realtime=False, height=90) as src:
        assert (src.width, src.height) == (160, 90)
        frames = list(src)
    assert frames and all(f.image.shape == (90, 160, 3) for f in frames)


def test_realtime_paces_a_clip_to_about_its_duration(tmp_path):
    clip = _clip(tmp_path, seconds=3.0)
    t0 = time.monotonic()
    with open_source(clip, fps=5.0, realtime=True, height=None) as src:
        frames = list(src)
    dt = time.monotonic() - t0
    assert len(frames) == 15
    # ffmpeg's own -re has some fixed startup slack (observed ~0.5-0.6 s for a 3 s clip); the band is
    # wide so this does not flake on a slower machine or a different ffmpeg build, while still catching
    # "not actually realtime" (that finishes in well under a second, the way realtime=False does).
    assert 1.0 <= dt <= 8.0


def test_slow_consumer_drops_frames_and_stays_recent(tmp_path):
    clip = _clip(tmp_path, seconds=3.0)
    t0 = time.monotonic()
    seen = []
    with open_source(clip, fps=10.0, realtime=True, height=None) as src:
        for i, fr in enumerate(src):
            seen.append((fr.index, fr.t, time.monotonic() - t0))
            time.sleep(0.5)
            if i >= 3:
                break
        dropped = src.dropped
    assert dropped > 0
    indices = [i for i, _, _ in seen]
    assert any(b - a > 1 for a, b in zip(indices, indices[1:]))  # a gap: some decoded frames were skipped
    for _, t, elapsed in seen:
        assert abs(t - elapsed) < 1.5  # the delivered frame tracks "now", not a stale one


def test_close_stops_ffmpeg(tmp_path):
    clip = _clip(tmp_path, seconds=3.0)
    src = open_source(clip, fps=5.0, realtime=True, height=None)
    fr = next(iter(src))
    assert fr.index == 0
    proc = src._proc
    assert proc.poll() is None  # still decoding; most of the 3 s clip is left
    src.close()
    assert proc.poll() is not None  # terminate() (or kill(), after 2 s) reaped it


def test_missing_streamlink_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlink", None)
    with pytest.raises(RuntimeError, match="pip install streamlink"):
        live_source._twitch_url("twitch.tv/riftbound", "best")


def test_missing_ytdlp_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    with pytest.raises(RuntimeError, match="pip install yt-dlp"):
        live_source._youtube_url("youtu.be/dQw4w9WgXcQ", 1080)


def test_a_live_source_reconnects_and_gives_up_after_three_tries_per_outage(tmp_path, monkeypatch):
    clip = _clip(tmp_path, seconds=1.0)
    calls = []

    def resolve(spec, kind, quality, height):
        calls.append(spec)
        if len(calls) == 2:
            raise RuntimeError("offline for a moment")
        return clip, True

    monkeypatch.setattr(live_source, "_resolve", resolve)
    monkeypatch.setattr(live_source, "_BACKOFF", (0.0,))
    frames = []
    with open_source("twitch.tv/riftbound", fps=5.0, realtime=False, height=None) as src:
        assert src.live
        with pytest.raises(RuntimeError, match="gave up after 3 reconnect attempts"):
            for fr in src:
                frames.append(fr.index)
    assert frames == list(range(15)) and len(calls) == 4  # the first session, then 1 failed and 2 good reconnects


def test_a_good_run_between_outages_gets_a_fresh_budget(tmp_path, monkeypatch):
    clip = _clip(tmp_path, seconds=1.0)
    monkeypatch.setattr(live_source, "_resolve", lambda *a: (clip, True))
    monkeypatch.setattr(live_source, "_BACKOFF", (0.0,))
    monkeypatch.setattr(live_source, "_FRESH_AFTER", 0.5)  # a 1 s session counts as a good run
    with open_source("twitch.tv/riftbound", fps=5.0, realtime=False, height=None) as src:
        frames = [fr.index for _, fr in zip(range(30), src)]
    assert frames == list(range(30))  # six sessions, twice the budget, and it never gave up


def test_a_replay_link_copied_at_the_current_time_starts_there():
    from rifteye_ml.live.__main__ import link_start

    assert link_start("https://www.twitch.tv/videos/2854086989?t=3h40m0s") == 3 * 3600 + 40 * 60
    assert link_start("https://www.twitch.tv/videos/2854086989?t=01h02m03s") == 3723
    assert link_start("https://www.youtube.com/watch?v=abc&t=90s") == 90 and link_start("https://youtu.be/abc?t=95") == 95
    assert link_start("https://www.twitch.tv/videos/2854086989") == 0 and link_start("https://www.twitch.tv/tfue?tt=5") == 0


def test_each_encoder_reads_with_its_own_temperature():
    from rifteye_ml.live.__main__ import EMBEDDER_T, temperature_for

    class Packed:
        meta = {"temperature": 0.05}

    assert temperature_for("colorgrid/trim0.03+dhash/trim0.03", object(), None) is None  # the pipeline's own
    assert temperature_for("embedder:/x/embedder-v1.pth", object(), None) == EMBEDDER_T
    assert temperature_for("embedder:/x/embedder-v2.pth", Packed(), None) == 0.05
    assert temperature_for("embedder:/x/embedder-v1.pth", Packed(), 0.02) == 0.02

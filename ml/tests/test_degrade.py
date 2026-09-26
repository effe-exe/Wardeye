import numpy as np

from rifteye_ml.degrade import StreamSettings, h264_roundtrip, psnr, simulate
from rifteye_ml.fixtures import synthetic_card


def _texture_frames(n, w=640, h=360, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(h // 8, w // 8, 3), dtype=np.uint8)
    base = np.kron(base, np.ones((8, 8, 1), dtype=np.uint8))  # blocky detail the codec must spend bits on
    return [np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8) for _ in range(n)], base


def test_h264_roundtrip_keeps_shape_and_costs_quality_at_low_bitrate():
    frames, base = _texture_frames(10)
    lo = h264_roundtrip(iter(frames), 640, 360, StreamSettings(bitrate_kbps=150, fps=30), keep={9})
    hi = h264_roundtrip(iter(frames), 640, 360, StreamSettings(bitrate_kbps=8000, fps=30), keep={9})
    assert lo[9].shape == hi[9].shape == (360, 640, 3)
    assert psnr(lo[9], base) < psnr(hi[9], base)


def test_simulate_returns_one_crop_per_card_at_the_requested_height():
    cards = [synthetic_card(i) for i in range(10)]
    s = StreamSettings(frame_w=640, frame_h=360, card_h=60, bitrate_kbps=2000, frames_per_board=6, seed=3)
    crops = simulate(cards, s)
    assert [c.card_index for c in crops] == list(range(10))
    for c in crops:
        assert c.rotation in (0, 90, 180, 270)
        assert max(c.image.size) == 60
        up = c.upright()
        assert up.height > up.width  # portrait again once the known rotation is undone


def test_realism_levels():
    import pytest

    from rifteye_ml.degrade import REALISM, with_realism

    base = StreamSettings()
    assert with_realism(base, "codec") == base
    cam = with_realism(base, "camera")
    assert cam.occlusion_prob > 0 and cam.box_jitter > 0 and cam.card_h == base.card_h
    assert set(REALISM) == {"codec", "camera", "foil"}
    with pytest.raises(ValueError):
        with_realism(base, "cinematic")


def test_perspective_coeffs_identity_and_camera_geometry():
    from rifteye_ml.degrade import _camera_geometry, _perspective_coeffs

    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float64)
    assert np.allclose(_perspective_coeffs(square, square), [1, 0, 0, 0, 1, 0, 0, 0], atol=1e-9)
    card = np.full((88, 63, 3), 200, np.uint8)
    s = StreamSettings(tilt_deg=15, keystone=0.03, angle_jitter_deg=4)
    out = _camera_geometry(card, s, np.random.default_rng(1))
    assert out.shape[2] == 4 and out[..., 3].max() == 255
    assert out[0, 0, 3] < 255 or out[-1, -1, 3] < 255  # table shows through at a corner


def test_camera_level_changes_the_crops_but_keeps_one_per_card():
    from rifteye_ml.degrade import with_realism

    cards = [synthetic_card(i) for i in range(8)]
    s = StreamSettings(frame_w=640, frame_h=360, card_h=60, bitrate_kbps=2000, frames_per_board=6, seed=5)
    codec, again = simulate(cards, s), simulate(cards, s)
    camera = simulate(cards, with_realism(s, "camera"))
    assert all(np.array_equal(np.asarray(a.image), np.asarray(b.image)) for a, b in zip(codec, again))  # deterministic
    assert [c.card_index for c in camera] == list(range(8))
    assert all(min(c.image.size) >= 2 for c in camera)
    assert any(a.image.size != b.image.size or not np.array_equal(np.asarray(a.image), np.asarray(b.image))
               for a, b in zip(codec, camera))


def test_foil_level_adds_colour_shifts_without_touching_other_levels():
    import numpy as np
    from rifteye_ml.degrade import REALISM, StreamSettings, _add_foil, with_realism

    assert with_realism(StreamSettings(), "camera").foil_prob == 0.0
    assert with_realism(StreamSettings(), "foil").foil_prob == 0.2
    assert REALISM["foil"]["tilt_deg"] == REALISM["camera"]["tilt_deg"]
    card = np.full((131, 95, 3), 120, np.uint8)
    a = _add_foil(card, np.random.default_rng(3))
    b = _add_foil(card, np.random.default_rng(3))
    assert np.array_equal(a, b)  # seeded
    hue_spread = a.reshape(-1, 3).astype(int).std(axis=0)
    assert hue_spread.max() > 5  # colours now vary across a flat grey card

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

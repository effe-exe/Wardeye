import numpy as np
import pytest
from PIL import Image, ImageDraw

from rifteye_ml.fixtures import load_fixture_image, synthetic_catalog

MAT = (30, 40, 55)


def _broadcast(cards, seed=0):
    """A 960 x 540 frame: busy side panels drawn with a gold border at 20% and 80% of the width, a mat
    between them, and each (image, x, y) pasted as a 56 x 78 card."""
    rng = np.random.default_rng(seed)
    im = Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8))
    im.paste(Image.new("RGB", (576, 540), MAT), (192, 0))
    d = ImageDraw.Draw(im)
    d.line([(191, 0), (191, 539)], fill=(212, 175, 55), width=3)
    d.line([(768, 0), (768, 539)], fill=(212, 175, 55), width=3)
    for card, x, y in cards:
        im.paste(card.resize((56, 78), Image.BOX), (x, y))
    return np.asarray(im)


def test_the_table_window_and_card_size_come_from_the_footage():
    pytest.importorskip("scipy")
    from rifteye_ml.live.autolayout import auto_layout

    art = [load_fixture_image(r) for r in synthetic_catalog(4, seed=3)]
    spots = [(260, 80), (420, 200), (600, 90), (330, 380)]
    frames = [_broadcast([(a, x, y) for a, (x, y) in zip(art, spots)], seed=k) for k in range(5)]
    layout = auto_layout(frames)
    x0, y0, x1, y1 = layout.table
    assert abs(x0 - 0.2) < 0.02 and abs(x1 - 0.8) < 0.02 and y0 < 0.02 and y1 > 0.98  # the panels stay outside
    assert layout.card_long_1080 == pytest.approx(156, rel=0.08)  # 78 px cards in a 540 px frame
    assert max(abs(c - m) for c, m in zip(layout.mat, MAT)) <= 8


def test_no_table_no_layout():
    pytest.importorskip("scipy")
    from rifteye_ml.live.autolayout import auto_layout

    rng = np.random.default_rng(1)
    player_cam = [rng.integers(0, 255, (540, 960, 3), dtype=np.uint8) for _ in range(5)]
    assert auto_layout(player_cam) is None

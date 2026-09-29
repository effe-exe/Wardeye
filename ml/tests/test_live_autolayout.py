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


def test_a_webcam_over_a_border_does_not_hide_it():
    # a co-stream: the co-streamer's webcam lies over the right border for the bottom 35% of the frame, so
    # that border runs only 65% of the height; the left one, whole, vouches for its mirror image
    pytest.importorskip("scipy")
    from rifteye_ml.live.autolayout import borders, table_window

    art = [load_fixture_image(r) for r in synthetic_catalog(4, seed=3)]
    spots = [(260, 80), (420, 200), (600, 90), (330, 380)]
    frames = []
    for k in range(5):
        f = _broadcast([(a, x, y) for a, (x, y) in zip(art, spots)], seed=k).copy()
        f[350:, 700:840] = (90 + 10 * k, 60, 50)  # the webcam: a face that moves, no edge where the border was
        frames.append(f)
    assert abs(borders(frames)[2] - 0.8) < 0.01
    x0, y0, x1, y1 = table_window(frames)[0]
    assert abs(x0 - 0.2) < 0.02 and abs(x1 - 0.8) < 0.02  # the right panel stays outside


def test_a_framed_picture_between_bars_keeps_its_sides():
    # Stockholm's layout: bars above and below the picture, a thin frame line around it that runs only
    # between the bars (88% of the height), and side panels whose lower half is as dark as the mat
    pytest.importorskip("scipy")
    from rifteye_ml.live.autolayout import borders, table_window

    rng = np.random.default_rng(4)
    art = [load_fixture_image(r) for r in synthetic_catalog(4, seed=3)]
    frames = []
    for k in range(5):
        im = Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8))  # the player cams
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 959, 31], fill=(20, 28, 60))       # the bar above, with a logo
        d.rectangle([0, 508, 959, 539], fill=(20, 28, 60))    # and below
        d.rectangle([0, 0, 100, 31], fill=(200, 120, 40))
        d.rectangle([0, 300, 159, 507], fill=(26, 36, 58))    # a card back in each panel, near the mat's colour
        d.rectangle([801, 300, 959, 507], fill=(26, 36, 58))
        d.rectangle([160, 32, 800, 507], fill=MAT)
        d.rectangle([158, 32, 160, 507], fill=(30, 80, 250))  # the frame line, between the bars only
        d.rectangle([800, 32, 802, 507], fill=(30, 80, 250))
        d.rectangle([158, 30, 802, 32], fill=(30, 80, 250))
        d.rectangle([158, 507, 802, 509], fill=(30, 80, 250))
        for a, (x, y) in zip(art, [(260, 80), (420, 200), (600, 90), (330, 380)]):
            im.paste(a.resize((56, 78), Image.BOX), (x, y))
        frames.append(np.asarray(im))
    x0, y0, x1, y1 = borders(frames)
    assert abs(x0 - 161 / 960) < 0.01 and abs(x1 - 800 / 960) < 0.01
    wx0, wy0, wx1, wy1 = table_window(frames)[0]
    assert abs(wx0 - 161 / 960) < 0.02 and abs(wx1 - 800 / 960) < 0.02  # the panels stay outside
    assert wy0 > 0.05 and wy1 < 0.95                                      # and the bars


def test_no_table_no_layout():
    pytest.importorskip("scipy")
    from rifteye_ml.live.autolayout import auto_layout

    rng = np.random.default_rng(1)
    player_cam = [rng.integers(0, 255, (540, 960, 3), dtype=np.uint8) for _ in range(5)]
    assert auto_layout(player_cam) is None

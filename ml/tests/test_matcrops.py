import numpy as np
from PIL import Image, ImageDraw

from rifteye_ml.matcrops import find_cards, upright_crop


def _card(w=94, h=131, seed=0):
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (w, h), (20, 20, 24))  # dark sleeve / border
    d = ImageDraw.Draw(im)
    d.rectangle([5, 5, w - 6, h - 6], fill=tuple(int(c) for c in rng.integers(120, 255, 3)))
    d.rectangle([10, 10, w - 11, h // 2], fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    return im


def _paste(frame, card, cx, cy, angle):
    rgba = card.convert("RGBA").rotate(angle, expand=True, resample=Image.BICUBIC)
    frame.paste(rgba, (int(cx - rgba.width / 2), int(cy - rgba.height / 2)), rgba)


def test_finds_isolated_cards_and_rejects_stacks():
    frame = Image.new("RGB", (900, 500), (190, 25, 45))  # red mat
    for i, (x, y, a) in enumerate([(150, 150, 0), (400, 160, 12), (650, 170, 90)]):
        _paste(frame, _card(seed=i), x, y, a)
    _paste(frame, _card(seed=7), 300, 380, 0)   # a stack: two overlapping cards merge into one blob
    _paste(frame, _card(seed=8), 360, 400, 0)
    ImageDraw.Draw(frame).ellipse([780, 400, 800, 420], fill=(10, 10, 10))  # a die: too small
    boxes = find_cards(np.asarray(frame), long_px=131)
    assert len(boxes) == 3
    for b in boxes:
        assert abs(b.long_px - 131) < 5 and abs(b.short_px - 94) < 5 and b.fill > 0.9
    tilted = min(boxes, key=lambda b: abs(b.centre[0] - 400))
    assert abs(abs(tilted.angle_deg - 90) - 12) < 3  # 12° off vertical
    crop = upright_crop(frame, tilted)
    assert crop.height > crop.width and abs(crop.height - 131) <= 2
    # Straightened, not skewed further: the crop matches the card itself, upright or upside down.
    ref = np.asarray(_card(seed=1), np.float32)
    got = np.asarray(crop.resize((94, 131)), np.float32)
    assert min(np.abs(got - ref).mean(), np.abs(got[::-1, ::-1] - ref).mean()) < 25


def test_notmat_mask_finds_light_bordered_cards_on_any_mat():
    from rifteye_ml.matcrops import mat_colour, notmat_mask

    rng = np.random.default_rng(0)
    mat = np.array([30, 60, 140], np.uint8)  # a blue mat
    frame = np.clip(mat + rng.normal(0, 4, (360, 640, 3)), 0, 255).astype(np.uint8)
    frame[100:231, 200:295] = (235, 235, 230)   # a white-bordered card, 131 x 95
    frame[110:221, 210:285] = (120, 40, 60)     # its art
    assert np.abs(mat_colour(frame) - mat.astype(np.int16)).max() <= 8
    assert find_cards(frame, 131) == []          # the dark-border rule misses it
    boxes = find_cards(frame, 131, mask=notmat_mask(frame, mat_colour(frame)))
    assert len(boxes) == 1 and abs(boxes[0].long_px - 131) < 4

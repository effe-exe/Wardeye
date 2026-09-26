import numpy as np
import pytest
from PIL import Image

from rifteye_ml.demo import attach_events, box_view, corners, looks_face_down


def test_corners_of_an_upright_card():
    c = corners({"centre": (100.0, 200.0), "long_px": 130.0, "short_px": 94.0, "angle_deg": 90.0})
    xs, ys = [p[0] for p in c], [p[1] for p in c]
    assert max(xs) - min(xs) == pytest.approx(94) and max(ys) - min(ys) == pytest.approx(130)
    assert np.mean(xs) == pytest.approx(100) and np.mean(ys) == pytest.approx(200)


def test_face_down_is_a_plain_sleeve():
    assert looks_face_down(Image.new("RGB", (95, 131), (30, 45, 160)))
    face = np.zeros((131, 95, 3), np.uint8)
    face[:60] = (200, 60, 40)
    face[60:] = (235, 235, 230)  # art above, text box below
    assert not looks_face_down(Image.fromarray(face))


def test_box_view_turns_landscape_regions_upright():
    frame = Image.new("RGB", (1920, 1080))
    assert box_view(frame, (0.1, 0.1, 0.2, 0.15)).height > box_view(frame, (0.1, 0.1, 0.2, 0.15)).width
    tall = box_view(frame, (0.1, 0.1, 0.13, 0.2))
    assert tall.height > tall.width


def test_events_take_the_card_seen_in_their_box():
    card = {"id": "k0", "samples": [[5.0, 0.5, 0.5, 0.1, 0.07, 90], [6.0, 0.5, 0.5, 0.1, 0.07, 90], [9.0, 0.5, 0.5, 0.1, 0.07, 90]]}
    back = {"id": "k1", "faceDown": True, "samples": [[5.0, 0.2, 0.2, 0.1, 0.07, 90]]}
    events = [
        {"t": 4.8, "tBefore": 3.0, "kind": "appeared", "box": [0.45, 0.45, 0.55, 0.55]},   # played: seen just after
        {"t": 5.5, "tBefore": 5.0, "kind": "appeared", "box": [0.45, 0.45, 0.55, 0.55]},   # the same change twice
        {"t": 10.0, "tBefore": 9.5, "kind": "disappeared", "box": [0.45, 0.45, 0.55, 0.55]},  # left: seen just before
        {"t": 5.0, "tBefore": 4.0, "kind": "appeared", "box": [0.15, 0.15, 0.25, 0.25]},   # only a face-down card there
        {"t": 30.0, "tBefore": 29.0, "kind": "changed", "box": [0.45, 0.45, 0.55, 0.55]},  # nothing seen then
    ]
    got = attach_events(events, [card, back])
    assert [(e["t"], e["kind"], e["track"]) for e in got] == [(4.8, "played", "k0"), (10.0, "left", "k0")]

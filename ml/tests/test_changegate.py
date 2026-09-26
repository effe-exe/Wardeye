import numpy as np

from rifteye_ml.changegate import ChangeGate, GateSettings

MAT = (190, 25, 45)


def _frame(cards=(), hand=None, w=320, h=180):
    f = np.zeros((h, w, 3), np.uint8)
    f[:] = MAT
    for x, y, colour in cards:  # a card is about 16 x 22 px at this scale
        f[y:y + 22, x:x + 16] = colour
    if hand is not None:
        x, y = hand
        f[y:y + 30, x:x + 40] = (225, 180, 150)
    return f


def test_card_played_moved_and_hand_ignored():
    s = GateSettings(fps=5, width=320, mat_rgb=MAT)
    gate = ChangeGate(s)
    t = 0.0
    seq = [_frame()] * 3
    seq += [_frame(hand=(40 + 12 * i, 60)) for i in range(5)]           # a hand sweeps across: no event
    seq += [_frame()] * 5
    seq += [_frame(cards=[(100, 60, (30, 30, 200))], hand=(95, 50))] * 2  # card placed, hand still on it
    seq += [_frame(cards=[(100, 60, (30, 30, 200))])] * 6                # hand gone, card settles
    seq += [_frame(cards=[(200, 90, (30, 30, 200))])] * 6                # the card moves
    kinds = []
    for f in seq:
        kinds += [e.kind for e in gate.feed(t, f)]
        t += 1 / s.fps
    assert kinds.count("appeared") == 2 and kinds.count("disappeared") == 1, kinds
    first = next(e for e in gate.events if e.kind == "appeared")
    x0, y0, x1, y1 = first.box
    assert 95 <= x0 <= 101 and 15 <= x1 - x0 <= 18 and 20 <= y1 - y0 <= 24
    assert abs(first.extra["t_before"] - 12 / s.fps) < 1e-6  # the last empty frame before the hand came back
    assert len(gate.snapshots) == len(gate.events)


def test_camera_reframing_is_one_cut_event():
    s = GateSettings(fps=5, width=320, mat_rgb=MAT)
    gate = ChangeGate(s)
    reframed = _frame()
    reframed[:, :200] = np.random.default_rng(0).integers(0, 120, size=(180, 200, 3))  # most of the view differs
    kinds = []
    for i, f in enumerate([_frame()] * 3 + [reframed] * 5):
        kinds += [e.kind for e in gate.feed(i / 5, f)]
    assert kinds == ["cut"]


def test_cutaway_frames_are_skipped_and_changes_meanwhile_found():
    s = GateSettings(fps=5, width=320, mat_rgb=MAT)
    gate = ChangeGate(s)
    cam = np.full((180, 320, 3), 90, np.uint8)  # a player cam: no playmat in view
    seq = [_frame()] * 3 + [cam] * 10 + [_frame(cards=[(150, 80, (240, 200, 40))])] * 6
    kinds = []
    for i, f in enumerate(seq):
        kinds += [e.kind for e in gate.feed(i / 5, f)]
    assert kinds == ["appeared"]  # the card played during the cutaway, and no event for the cam itself


def test_a_hand_in_the_first_frame_leaving_is_not_a_change():
    s = GateSettings(fps=5, width=320, mat_rgb=MAT)
    gate = ChangeGate(s)
    seq = [_frame(hand=(100, 60))] * 2 + [_frame()] * 8          # the window starts with a hand on the table
    seq += [_frame(cards=[(200, 90, (30, 30, 200))])] * 8         # then a real card is played
    t = 0.0
    for f in seq:
        gate.feed(t, f)
        t += 1 / s.fps
    assert [e.kind for e in gate.events] == ["appeared"]

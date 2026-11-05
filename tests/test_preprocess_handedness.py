import numpy as np

from preprocessing.hand_recon.handedness import assign_hands, hand_boxes


def _dets(tracks):
    """{track: (frames, x, cls)} -> flat frame, track, x, cls arrays."""
    f, tr, x, c = [], [], [], []
    for k, (frames, xs, cls) in tracks.items():
        f += list(frames)
        tr += [k] * len(frames)
        x += list(np.broadcast_to(xs, len(frames)))
        c += [cls] * len(frames)
    return np.array(f), np.array(tr), np.array(x, float), np.array(c, float)


def test_pair_order_beats_wrong_detector_class():
    # both hands labelled "right" by the detector; the image-left one is the left hand
    f, tr, x, c = _dets({1: (range(50), 300.0, 1), 2: (range(50), 900.0, 1)})
    hand = assign_hands(f, tr, x, c)
    assert (hand[tr == 1] == 0).all() and (hand[tr == 2] == 1).all()


def test_mono_single_hand_follows_detector():
    f, tr, x, c = _dets({1: (range(30), 900.0, 0)})
    assert (assign_hands(f, tr, x, c) == 0).all()
    f, tr, x, c = _dets({1: (range(30), 300.0, 1)})
    assert (assign_hands(f, tr, x, c) == 1).all()


def test_lone_tracklet_is_decided_by_the_stereo_twin():
    # this view: both hands in 0-19, then only a new tracklet (3) on the left hand,
    # mislabelled "right"; the twin (shifted by 60 px disparity) still sees both hands
    f, tr, x, c = _dets({1: (range(20), 300.0, 0), 2: (range(20), 900.0, 1),
                         3: (range(20, 60), 320.0, 1)})
    tf, _, tx, _ = _dets({1: (range(60), 240.0, 0), 2: (range(60), 840.0, 1)})
    assert (assign_hands(f, tr, x, c)[tr == 3] == 1).all()          # mono: detector wins
    hand = assign_hands(f, tr, x, c, twin=(tf, tx))
    assert (hand[tr == 3] == 0).all() and (hand[tr == 1] == 0).all()


def test_simultaneous_detections_never_share_a_hand():
    # a long, confidently "left" tracklet meets a short one that is also called left
    f, tr, x, c = _dets({1: (range(100), 300.0, 0), 2: (range(10), 100.0, 0)})
    hand = assign_hands(f, tr, x, c)
    both = np.isin(f, range(10))
    assert (hand[both & (tr == 2)] == 0).all() and (hand[both & (tr == 1)] == 1).all()


def test_hand_boxes_bridge_within_a_tracklet_only():
    box = np.array([[0, 0, 10, 10, 0.9], [20, 0, 30, 10, 0.8], [100, 0, 110, 10, 0.7]])
    frame, track, hand = np.array([0, 4, 8]), np.array([1, 1, 2]), np.array([1, 1, 1])
    boxes, has, real = hand_boxes(frame, track, box, hand, 1, 10)
    assert has[:5].all() and not has[5:8].any() and has[8]          # no bridge 4 -> 8
    assert real.sum() == 3 and np.allclose(boxes[2, :4], [10, 0, 20, 10]) and boxes[2, 4] == 0
    assert not hand_boxes(frame, track, box, hand, 0, 10)[1].any()

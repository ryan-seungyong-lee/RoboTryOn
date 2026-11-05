"""Hand-state captions (robotryon.captions)."""
import numpy as np

from robotryon.captions import (NOT_VISIBLE, HandStateCuts, all_prompts, hand_fields,
                                hand_state_arrays, hand_state_prompt, sample_prompt, words_at)


def _flat_hand(z=0.5, flip=False):
    """Straight fingers in the z = const plane, in front of the camera."""
    j = np.zeros((21, 3))
    j[:, 2] = z
    for f, x0 in enumerate((-0.3, -0.1, 0.1, 0.3)):       # fingers radial from the wrist
        d = np.array([x0, 1.0]) / np.hypot(x0, 1.0)
        for k in range(4):
            j[5 + 4 * f + k, :2] = d * (0.08 + 0.025 * k)
    for k in range(4):
        j[1 + k, :2] = np.array([-0.7, 0.7]) * (0.03 + 0.015 * k)
    if flip:
        j[:, 0] *= -1
    return j


def test_open_hand_closure_and_facing():
    c, d = hand_fields(_flat_hand(), "left")
    assert abs(c) < 1e-6                      # straight fingers: no flexion
    assert d < 30 or d > 150                  # knuckle plane roughly faces the camera
    c2, d2 = hand_fields(_flat_hand(flip=True), "left")
    assert abs(d + d2 - 180.0) < 1e-6         # mirrored hand shows the other side


def test_words_and_prompt():
    cuts = HandStateCuts()
    assert cuts.words(0.1, 10.0) == "open, palm toward camera"
    assert cuts.words(2.0, 150.0) == "closed fist, back of hand toward camera"
    assert cuts.words(float("nan"), 10.0) == NOT_VISIBLE
    p = hand_state_prompt("open, edge-on", NOT_VISIBLE)
    assert p == "Match the hand pose to the drawn skeleton. Left hand: open, edge-on. Right hand: not visible."


def test_words_at_untracked_hand():
    j3 = np.zeros((3, 2, 21, 3)); j3[:, 0] = _flat_hand()
    valid = np.zeros((3, 2), bool); valid[:, 0] = True
    left, right = words_at([hand_state_arrays(j3, valid)], 1, HandStateCuts())
    assert left.startswith("open") and right == NOT_VISIBLE


def test_vocabulary_covers_samples():
    vocab = set(all_prompts())
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert sample_prompt(("open, edge-on", NOT_VISIBLE), rng, 0.1, 0.2) in vocab

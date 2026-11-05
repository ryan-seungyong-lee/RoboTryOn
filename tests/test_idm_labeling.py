import numpy as np
import pytest

from robotryon.idm.labeling import ChunkAccumulator, smooth_actions, state_from_action


def test_state_is_delayed_action():
    a = np.arange(30, dtype=np.float32).reshape(10, 3)
    q = state_from_action(a, 4)
    assert q.shape == a.shape and q.dtype == a.dtype
    np.testing.assert_array_equal(q[:4], np.repeat(a[:1], 4, 0))
    np.testing.assert_array_equal(q[4:], a[:-4])
    np.testing.assert_array_equal(state_from_action(a, 0), a)
    np.testing.assert_array_equal(state_from_action(a[:2], 4), np.repeat(a[:1], 2, 0))


def test_chunk_averaging():
    L, H = 7, 3
    acc = ChunkAccumulator(L, 1, H)
    for s in range(L):                       # chunk starting at s predicts value s everywhere
        acc.add(s, np.full((H, 1), float(s)))
    got = acc.result()[:, 0]
    want = [np.mean([s for s in range(max(0, u - H + 1), u + 1)]) for u in range(L)]
    np.testing.assert_allclose(got, want)


def test_chunk_averaging_requires_coverage():
    acc = ChunkAccumulator(5, 1, 2)
    acc.add(0, np.zeros((2, 1)))
    with pytest.raises(RuntimeError):
        acc.result()


def test_savgol_groups():
    t = np.arange(40, dtype=np.float64)
    rng = np.random.default_rng(0)
    quad = 0.01 * t ** 2 - t + 3
    a = np.stack([quad, quad + rng.normal(0, 1, 40), rng.normal(0, 1, 40)], 1).astype(np.float32)
    groups = {"arm": slice(0, 2), "hand": slice(2, 3)}
    out = smooth_actions(a, groups, {"hand"}, window=9, hand_window=5)
    np.testing.assert_allclose(out[4:-4, 0], quad[4:-4], atol=1e-3)   # order-2 SG keeps a parabola
    assert np.std(np.diff(out[:, 1] - quad)) < np.std(np.diff(a[:, 1] - quad))
    from scipy.signal import savgol_filter
    np.testing.assert_allclose(out[:, 2], savgol_filter(a[:, 2], 5, 2, mode="nearest"), atol=1e-6)
    short = smooth_actions(a[:4], groups, {"hand"}, window=9, hand_window=5)
    np.testing.assert_array_equal(short, a[:4])                   # windows longer than the clip

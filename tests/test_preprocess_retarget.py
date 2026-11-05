import numpy as np

from preprocessing.retime import resample_keypoints, time_map
from robotryon.embodiment.retarget import (CHAINS17, CHAINS21, VECTORS17, bone_match, human17,
                                           local_vectors, place_on_human, to21)


def _hand(rng):
    j = np.zeros((21, 3))
    for f, chain in CHAINS21.items():
        base = rng.normal([0.08, 0, 0], 0.02)
        d = rng.normal([1, 0, 0], 0.2)
        for k, i in enumerate(chain):
            j[i] = base + (k * 0.025 + 0.0) * d
    j[CHAINS21["little"][0]] += [0, 0.04, 0]
    j[CHAINS21["index"][0]] += [0, -0.04, 0]
    return j


def test_bone_match_lengths_and_directions():
    rng = np.random.default_rng(0)
    human = _hand(rng)
    robot = rng.normal(0, 0.05, (17, 3))
    robot[0] = 0
    out = bone_match(robot, human)
    for f, (k, distal) in CHAINS17.items():
        hm, ht = CHAINS21[f][0], CHAINS21[f][-1]
        assert np.isclose(np.linalg.norm(out[k]), np.linalg.norm(human[hm]))
        assert np.isclose(np.linalg.norm(out[distal[-1]] - out[k]),
                          np.linalg.norm(human[ht] - human[hm]))
        # directions of the robot segments are kept
        u = out[distal[-1]] - out[k]
        v = robot[distal[-1]] - robot[k]
        assert np.isclose(u @ v / np.linalg.norm(u) / np.linalg.norm(v), 1.0)


def test_place_on_human_identity():
    rng = np.random.default_rng(1)
    human = _hand(rng)
    robot17 = to21(human)[list((0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20))]
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R *= np.sign(np.linalg.det(R))
    moved = robot17 @ R.T + 1.0                      # robot hand elsewhere, rotated
    placed = place_on_human(moved, human, match_bones=False)
    assert np.allclose(placed, robot17, atol=1e-9)


def test_local_vectors_rotation_invariant():
    rng = np.random.default_rng(2)
    h = human17(_hand(rng))
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R *= np.sign(np.linalg.det(R))
    v1, v2 = local_vectors(h), local_vectors(h @ R.T + 0.3)
    assert v1.shape == (len(VECTORS17), 3) and np.allclose(v1, v2, atol=1e-9)


def test_retime_keyframes_exact_and_speed():
    T = 200
    speed = np.full(T, 0.3)
    src, where = time_map(speed, [0, 50, 120, 199], [0, 50, 120, 199], 0.15, 1.0)
    # human twice as fast as the target: output is about twice as long
    assert abs(len(src) - 2 * T) < 4
    for k, o in where.items():
        assert src[o] == k
    assert np.all(np.diff(src) > 0)
    kp = np.random.default_rng(0).normal(size=(T, 2, 17, 2)).astype(np.float32)
    vis = np.ones((T, 2, 17), bool)
    rk, rv = resample_keypoints(kp, vis, src)
    assert rv.all() and np.allclose(rk[where[50]], kp[50])

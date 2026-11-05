import numpy as np

from preprocessing.keyframes.select_keyframes import select_keyframes
from robotryon.common.config import Config

CFG = Config(smooth_sigma_s=0.05, minimum_order=3, disp_quantile=0.5, max_gap=145,
             fallback_conf_quantile=0.9, max_pairs=400)


def _reach_and_rest(stops, T=600, fps=20.0):
    """Both wrists move between rest positions; returns (T,2,3) positions."""
    t = np.arange(T)
    x = np.zeros(T)
    for a, b, d in stops:          # move by d between frames a and b (smooth ramp)
        s = np.clip((t - a) / (b - a), 0, 1)
        x += d * (3 * s ** 2 - 2 * s ** 3)
    pos = np.zeros((T, 2, 3))
    pos[:, 0, 0] = x
    pos[:, 1, 0] = 0.5 * x
    pos[:, :, 2] = 0.5
    return pos


def test_first_last_and_rests():
    pos = _reach_and_rest([(20, 60, 0.3), (100, 140, -0.2), (200, 240, 0.4)], T=300)
    out = select_keyframes(pos, np.ones((300, 2), bool), np.ones((300, 2)), 20.0, CFG)
    k = out["indices"]
    assert k[0] == 0 and k[-1] == 299
    assert np.all(np.diff(k) > 0)
    # keyframes fall in the rest periods after each reach, not during motion
    for a, b in ((60, 100), (140, 200)):
        assert any(a - 5 <= x <= b + 5 for x in k), (k, a, b)


def test_cap_respected_and_fallback():
    T = 1000
    pos = np.zeros((T, 2, 3))
    pos[:, :, 2] = 0.5
    pos[:, 0, 0] = np.linspace(0, 1, T)               # constant speed: no local minima
    conf = np.full((T, 2), 0.5)
    conf[300, :] = 0.95                               # one very confident frame
    out = select_keyframes(pos, np.ones((T, 2), bool), conf, 20.0, CFG)
    k = out["indices"]
    assert k[0] == 0 and k[-1] == T - 1
    assert np.all(np.diff(k) <= 145)
    assert "fallback_confidence" in out["reasons"]


def test_cap_when_no_confidence():
    T = 400
    pos = np.zeros((T, 2, 3))
    out = select_keyframes(pos, np.zeros((T, 2), bool), np.zeros((T, 2)), 20.0, CFG)
    assert out["indices"] == [0, 145, 290, 399]
    assert out["reasons"] == ["first", "cap", "cap", "last"]


def test_short_clip_and_single_frame():
    pos = np.zeros((1, 2, 3))
    out = select_keyframes(pos, np.ones((1, 2), bool), np.ones((1, 2)), 20.0, CFG)
    assert out["indices"] == [0]

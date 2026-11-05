import numpy as np

from robotryon.skeleton import SkeletonAugment, draw_skeleton
from robotryon.skeleton.topology import segment_length_per_keypoint


def _hand(rng):
    kp = rng.uniform(100, 400, (2, 17, 2)).astype(np.float32)
    return kp, np.ones((2, 17), bool)


def test_drop_rates():
    rng = np.random.default_rng(0)
    aug = SkeletonAugment()
    all_dropped, kp_kept = 0, []
    for _ in range(4000):
        kp, vis = _hand(rng)
        _, v = aug(kp, vis, rng)
        if not v.any():
            all_dropped += 1
        else:
            kp_kept.append(v.mean())
    assert abs(all_dropped / 4000 - 0.2) < 0.03
    assert abs(np.mean(kp_kept) - 0.8) < 0.02


def test_shift_within_half_segment():
    rng = np.random.default_rng(1)
    aug = SkeletonAugment(p_drop_all=0, p_drop_keypoint=0)
    for _ in range(200):
        kp, vis = _hand(rng)
        out, _ = aug(kp, vis, rng)
        seg = segment_length_per_keypoint(kp, vis)
        assert (np.linalg.norm(out - kp, axis=-1) <= 0.5 * seg + 1e-3).all()


def test_temporal_shift_is_constant():
    rng = np.random.default_rng(2)
    kp = np.repeat(_hand(rng)[0][None], 10, 0) + np.arange(10)[:, None, None, None]
    vis = np.ones(kp.shape[:-1], bool)
    out, _ = SkeletonAugment(p_drop_all=0, p_drop_keypoint=0)(kp, vis, rng)
    d = out - kp
    assert np.allclose(d, d[:1], atol=1e-3)


def test_render_shape():
    rng = np.random.default_rng(3)
    kp, vis = _hand(rng)
    img = draw_skeleton(kp, vis, 512, 288)
    assert img.shape == (288, 512, 3) and img.dtype == np.uint8 and img.any()

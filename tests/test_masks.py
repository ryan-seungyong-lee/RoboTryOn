import cv2
import numpy as np
from PIL import Image

from robotryon.masks import MaskAug, arm_hulls, fill_hulls, fov_augment, inference_mask, scale_hulls


def _two_arms(h=180, w=320):
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (20, 100), (60, 179), 1, -1)
    cv2.line(m, (60, 140), (90, 170), 1, 1)            # thin cable touching the arm
    cv2.circle(m, (240, 130), 30, 1, -1)
    m[5, 5] = 1                                          # noise speck
    return m


def test_hull_per_arm_and_noise_removed():
    polys = arm_hulls(_two_arms())
    assert len(polys) == 2
    filled = fill_hulls(polys, 320, 180)
    assert all(f.sum() > 0 for f in filled)
    assert filled[0][5, 5] == 0 and filled[1][5, 5] == 0


def test_scale_hulls():
    p = [np.array([[10, 20], [30, 20], [30, 40]], np.float32)]
    q = scale_hulls(p, 2.0, 0.5, (10, 20))
    np.testing.assert_allclose(q[0], [[0, 0], [40, 0], [40, 10]])


def test_train_mask_dilation_range_and_bonus():
    aug = MaskAug()
    rng = np.random.default_rng(0)
    d = [aug.sample_dilate(rng) for _ in range(2000)]
    db = [aug.sample_dilate(rng, bonus=True) for _ in range(2000)]
    assert min(d) >= 3 and max(d) <= 71 and all(x % 2 for x in d)
    assert min(db) >= 3 + 21 and max(db) <= 71
    assert aug.infer_dilate() == 58


def test_train_mask_object_cut_rate():
    aug = MaskAug()
    polys = arm_hulls(_two_arms())
    obj = np.zeros((180, 320), np.uint8)
    cv2.circle(obj, (240, 130), 12, 1, -1)
    rng = np.random.default_rng(1)
    cuts = [aug.train_mask(polys, 320, 180, rng, object_mask=obj)[1] > 0 for _ in range(400)]
    assert 0.4 < np.mean(cuts) < 0.6
    m, e = aug.train_mask(polys, 320, 180, rng, object_mask=None)
    assert e == 0 and m.max() == 1


def test_inference_mask_subtracts_object():
    polys = arm_hulls(_two_arms())
    obj = np.zeros((180, 320), np.uint8)
    cv2.circle(obj, (240, 130), 20, 1, -1)
    full = inference_mask(polys, 320, 180, 58)
    cut = inference_mask(polys, 320, 180, 58, obj, 9)
    assert full[130, 240] == 1 and cut[130, 240] == 0
    assert cut.sum() < full.sum()


def test_fov_augment_shared_and_shape():
    rng = np.random.default_rng(0)
    a = Image.fromarray(np.full((64, 96, 3), 200, np.uint8))
    m = Image.fromarray(np.full((64, 96), 255, np.uint8))
    for _ in range(20):
        (a2, m2), s = fov_augment([a, m], rng, 0.85, 1.15)
        assert a2.size == (96, 64) and m2.size == (96, 64)
        assert 0.85 <= s <= 1.15
        if s > 1.01:   # padded: black border on both streams
            assert np.asarray(m2).min() < 128 and np.asarray(a2).min() < 128


def test_robot_only_reference_is_white():
    import numpy as np
    from robotryon.masks import robot_only
    img = np.full((40, 60, 3), 90, np.uint8)
    mask = np.zeros((40, 60), bool)
    mask[10:30, 20:40] = True
    out = robot_only(img, mask, erode=3)
    assert (out[0, 0] == 255).all() and (out[20, 30] == 90).all()

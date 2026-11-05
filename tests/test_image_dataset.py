"""Image-model dataset on a tiny synthetic episode tree (CPU only)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from robotryon.common.config import load_config

CFG = Path(__file__).resolve().parents[1] / "configs" / "image" / "klein9b_fft.yaml"
W, H, T, FPS = 320, 180, 160, 20


def _write_video(path, frames):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()


def make_episode(root: Path, name: str, embodiment: str, views=("left", "right"), obj=True):
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 31)
    rgb, msk, om, kp = [], [], [], np.zeros((T, 2, 17, 2), np.float32)
    for t in range(T):
        img = np.full((H, W, 3), 90, np.uint8)
        x = 60 + t // 2
        m = np.zeros((H, W), np.uint8)
        cv2.rectangle(m, (x, 90), (x + 30, H - 1), 255, -1)
        img[m > 0] = (200, 120, 60)
        o = np.zeros((H, W), np.uint8)
        cv2.circle(o, (x + 15, 95), 10, 255, -1)
        rgb.append(img), msk.append(m), om.append(o)
        kp[t, 1] = np.stack([x + 15 + rng.uniform(-8, 8, 17), 90 + rng.uniform(-8, 8, 17)], -1)
    vis = np.zeros((T, 2, 17), bool)
    vis[:, 1] = True
    for v in views:
        _write_video(d / f"video_{v}.mp4", rgb)
        _write_video(d / f"mask_{v}.mp4", msk)
        if obj:
            _write_video(d / f"object_mask_{v}.mp4", om)
        np.savez(d / f"keypoints_{v}.npz", kp=kp, vis=vis)
    (d / "meta.json").write_text(json.dumps({"fps": FPS, "views": list(views), "num_frames": T,
                                             "caption": "pick up the cup", "embodiment": embodiment}))
    return d


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("eps")
    make_episode(root, "ep0", "robot_a")
    make_episode(root, "ep1", "robot_a", obj=False)
    make_episode(root, "ep2", "robot_a", views=("main",))
    return root


def _cfg(root, *extra):
    return load_config(CFG, [f"data.roots=[{root}]", "image.long_edge=320", *extra])


def test_reference_window(tree):
    from robotryon.image.dataset import KeyframeDataset
    ds = KeyframeDataset(_cfg(tree), None)
    ep = ds.episodes[0]
    rng = np.random.default_rng(0)
    refs = [ds.reference_index(ep, 150, rng) for _ in range(500)]
    assert min(refs) >= 150 - 5 * FPS and max(refs) <= 150 - 2 * FPS
    assert ds.reference_index(ep, 30, rng) == 0           # too early -> frame 0


def test_getitem_shapes_stereo_and_mono(tree):
    from robotryon.image.dataset import FillBatchSampler, KeyframeDataset, collate
    ds = KeyframeDataset(_cfg(tree), None)
    s = ds[(0, 120, (0, 1), 7)]
    assert s["target"].shape == (2, 3, 176, 320)
    assert s["mask"].shape == (2, 1, 176, 320)
    assert s["mask"].max() > 0.5 and s["target"].min() >= -1 and s["target"].max() <= 1
    m = ds[(2, 50, (0,), 3)]
    assert m["target"].shape == (1, 3, 176, 320)
    sampler = FillBatchSampler(ds, 3, stereo_p=0.75, seed=0)
    it = iter(sampler)
    n_views = []
    for _ in range(200):
        b = next(it)
        assert len({len(x[2]) for x in b}) == 1
        n_views.append(len(b[0][2]))
    assert 0.6 < np.mean(np.array(n_views) == 2) < 0.9
    batch = collate([ds[x] for x in next(it)])
    assert batch["cond"].shape[0] == 3


def test_robot_only_reference(tree):
    from robotryon.image.dataset import KeyframeDataset
    ds = KeyframeDataset(_cfg(tree, "data.p_robot_only=1.0"), None)
    s = ds[(0, 100, (0, 1), 11)]
    assert s["robot_only"]
    ref = (s["reference"][0].numpy() + 1) * 127.5          # mostly white background
    assert (ref.min(axis=0) > 245).mean() > 0.7


def test_prompts(tree):
    from robotryon.captions import INSTRUCTION
    from robotryon.image.dataset import KeyframeDataset
    ds = KeyframeDataset(_cfg(tree), None)
    prompts = ds.all_prompts()
    assert len(prompts) == 10 * 10 + 2 and "" in prompts
    assert INSTRUCTION in prompts
    # no hand_state files in this tree: both hands read "not visible"
    assert ds[(0, 100, (0, 1), 11)]["prompt"] in prompts


def test_reference_skeleton_probability(tree):
    from robotryon.image.dataset import KeyframeDataset
    base = ("skeleton.augment.p_drop_all=0.0", "skeleton.augment.p_drop_keypoint=0.0", "data.p_robot_only=0.0")
    on = KeyframeDataset(_cfg(tree, "data.ref_skeleton_p=1.0", *base), None)[(0, 120, (0, 1), 5)]
    off = KeyframeDataset(_cfg(tree, "data.ref_skeleton_p=0.0", *base), None)[(0, 120, (0, 1), 5)]
    assert on["ref_skeleton"] and not off["ref_skeleton"]
    # identical except where the skeleton is painted
    d = (on["reference"] - off["reference"]).abs().amax(1)
    assert 0 < (d > 0.1).float().mean() < 0.1
    assert (on["target"] == off["target"]).all()


def test_ar_reference_modes():
    import torch
    from robotryon.image.ar_keyframes import KeyframeTranslator, TranslatedKeyframe

    class Codec:
        def encode(self, img):                      # (H, W, 3) -> (1, 128, H/16, W/16)
            m = float(np.asarray(img).mean())
            return torch.full((1, 128, img.shape[0] // 16, img.shape[1] // 16), m)

    tr = KeyframeTranslator.__new__(KeyframeTranslator)
    tr.codec = Codec()
    skel = np.zeros((32, 48, 3), np.uint8)
    skel[5, 5] = 255
    ref = {"left": np.full((64, 96, 3), 100, np.uint8)}
    prev_tokens = torch.zeros(1, 6, 128)
    prev = {"left": TranslatedKeyframe(np.full((32, 48, 3), 7, np.uint8), None, None, prev_tokens)}
    tr.ref_skeleton = False
    assert tr.reference("left", prev, ref, skel) is prev_tokens            # latent reuse
    assert tr.reference("left", {}, ref, skel)[0, 0, 0] == 100              # first keyframe, clean
    tr.ref_skeleton = True
    painted = tr.reference("left", prev, ref, skel)[0, 0, 0]
    assert 7 < painted < 8                                                  # decoded image + skeleton
    assert tr.reference("left", {}, ref, skel)[0, 0, 0] > 100               # first keyframe painted

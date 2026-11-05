"""A training item from a tiny synthetic stereo episode in the shared format."""
import json

import cv2
import numpy as np

from robotryon.skeleton import SkeletonAugment
from robotryon.video.dataset import VideoClipDataset, load_episodes


def make_episode(d, T=40, W=128, H=72):
    d.mkdir()
    rng = np.random.default_rng(0)
    for v in ("left", "right"):
        wr = cv2.VideoWriter(str(d / f"video_{v}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
        for t in range(T):
            wr.write(np.full((H, W, 3), t * 5, np.uint8))
        wr.release()
        kp = (rng.random((T, 2, 17, 2)) * [W, H]).astype(np.float32)
        np.savez(d / f"keypoints_{v}.npz", kp=kp, vis=np.ones((T, 2, 17), bool))
    (d / "meta.json").write_text(json.dumps({"fps": 20, "views": ["left", "right"],
                                             "num_frames": T, "caption": "pick up the cup",
                                             "embodiment": "robot"}))


def test_item(tmp_path):
    make_episode(tmp_path / "ep0")
    eps, w = load_episodes([str(tmp_path)], 9, 20.0, "a {embodiment}: {caption}")
    assert len(eps) == 1 and w == [1.0] and eps[0].caption == "a robot: pick up the cup"
    for mono_p, V in ((0.0, 2), (1.0, 1)):
        ds = VideoClipDataset(eps, (64, 32), (32, 16), (9, 33), 0.9, 0.5, mono_p,
                              SkeletonAugment(p_drop_all=0.0), {"thickness": 1, "radius": 1})
        for epoch in range(3):
            ds.set_epoch(epoch)
            it = ds[0]
            F = it["num_frames"]
            assert (F - 1) % 8 == 0 and 9 <= F <= 33
            assert it["video"].shape == (V, 3, F, 32, 64)
            assert it["condition"].shape == (V, 3, F, 16, 32)
            assert it["video"].min() >= -1 and it["video"].max() <= 1
            assert it["condition"].max() > -1          # skeleton was drawn

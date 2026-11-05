#!/usr/bin/env python
"""Hand skeleton of a human clip: reconstructed hands -> robot hand -> 17 keypoints per view.

For a person the keypoints come from the reconstructed hand retargeted to the target robot
hand (``Embodiment.retarget_hand``), so the skeleton has the robot's proportions and finger
coupling (Appendix B.2.2): the AnyTeleop vector retargeting of
``robotryon.embodiment.retarget``, run over the clip in order so each frame starts from the
previous one. The human video is recorded from the position of the robot's head
camera, so the robot camera model projects it. ``--no_retarget`` instead draws the human
hand itself (MediaPipe 21 -> 17 keypoints).

Writes ``keypoints_<view>.npz`` (``kp`` float32 (T, 2, 17, 2), ``vis`` bool (T, 2, 17)).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from preprocessing.episode_io import parse, parser, save_keypoints
from robotryon.embodiment import Embodiment, load_embodiment
from robotryon.skeleton.topology import MEDIAPIPE21_TO_17


def human_keypoints(emb: Embodiment, joints3d: np.ndarray, valid: np.ndarray, view: str,
                    image_size, retarget: bool = True):
    """(T, 2, 21, 3) camera-frame joints of one view -> kp (T,2,17,2), vis (T,2,17)."""
    T = len(joints3d)
    kp = np.zeros((T, 2, 17, 2), np.float32)
    vis = np.zeros((T, 2, 17), bool)
    prev: dict = {}
    for t in range(T):
        if not valid[t].any():
            prev = {}
            continue
        if retarget:
            pts, q = emb.retarget_hand(joints3d[t], valid=valid[t], prev=prev)
            # a hand that drops out restarts from the open hand when it comes back
            prev = q
        else:
            pts = np.where(valid[t, :, None, None], joints3d[t][:, MEDIAPIPE21_TO_17], np.nan)
        kp[t], vis[t] = emb.keypoints_2d(pts, view, image_size, frame="camera")
    return kp, vis


def main():
    p = parser("hands.npz -> keypoints_<view>.npz on the target robot's hand")
    p.add_argument("--embodiment", required=True)
    p.add_argument("--hands", required=True, help="hands.npz from hand_recon/fuse_views.py")
    p.add_argument("--out", required=True, help="episode directory")
    p.add_argument("--no_retarget", action="store_true", help="draw the human hand itself")
    a, _ = parse(p)
    emb = load_embodiment(a.embodiment)
    z = np.load(a.hands)
    for i, view in enumerate(z["views"]):
        view = str(view)
        size = tuple(int(x) for x in z["image_wh"][i])
        kp, vis = human_keypoints(emb, z["joints3d"][i], z["valid"][i], view, size,
                                  retarget=not a.no_retarget)
        save_keypoints(Path(a.out) / f"keypoints_{view}.npz", kp, vis)
        print(f"{view}: T={len(kp)} visible={vis.mean():.3f}")


if __name__ == "__main__":
    main()

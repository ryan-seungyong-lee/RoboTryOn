#!/usr/bin/env python
"""Hand skeleton of a robot episode: recorded joint state -> FK -> 17 keypoints per view.

For a robot frame the keypoints come from forward kinematics of the recorded joint state,
projected into each camera view (Appendix B.2.2). Writes ``keypoints_<view>.npz``
(``kp`` float32 (T, 2, 17, 2) pixels at the video resolution, ``vis`` bool (T, 2, 17)) and
``hand_state_<view>.npz`` (closure / dorsumness of each hand in that camera, robotryon.captions).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from preprocessing.episode_io import LeRobotEpisode, parse, parser, save_keypoints, video_info
from robotryon.captions import hand_state_arrays, save_hand_state
from robotryon.embodiment import Embodiment, load_embodiment


def robot_hand_state(emb: Embodiment, states: np.ndarray, view: str):
    """(T, J) joint vectors -> closure, dorsumness (T, 2) of the FK hands in ``view``'s camera frame."""
    pts, valid = [], []
    for q in states:
        T_cb = np.linalg.inv(emb.camera_pose(q, view))            # base -> camera
        kp = emb.hand_keypoints(q)                                  # (2, 17, 3) base frame
        pts.append(kp @ T_cb[:3, :3].T + T_cb[:3, 3])
        valid.append(np.isfinite(kp).all(axis=(-1, -2)))
    return hand_state_arrays(np.stack(pts), np.stack(valid))


def robot_keypoints(emb: Embodiment, states: np.ndarray, view: str, image_size):
    """(T, J) joint vectors -> kp (T, 2, 17, 2) float32, vis (T, 2, 17) bool."""
    out = [emb.keypoints_in_view(q, view, image_size) for q in states]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def main():
    p = parser("Robot episode -> keypoints_<view>.npz")
    p.add_argument("--embodiment", required=True)
    p.add_argument("--lerobot_root", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--out", required=True, help="episode directory")
    p.add_argument("--column", default="observation.state")
    p.add_argument("--views", nargs="*", default=None, help="default: every camera in the YAML")
    a, _ = parse(p)
    emb = load_embodiment(a.embodiment)
    ep = LeRobotEpisode(a.lerobot_root, a.episode)
    states = ep.column(a.column)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for view in a.views or emb.views:
        _, W, H, n = video_info(ep.video(emb.spec.cameras[view].video_key))
        T = min(n, len(states))
        kp, vis = robot_keypoints(emb, states[:T], view, (W, H))
        save_keypoints(Path(a.out) / f"keypoints_{view}.npz", kp, vis)
        save_hand_state(Path(a.out) / f"hand_state_{view}.npz", *robot_hand_state(emb, states[:T], view))
        print(f"{view}: T={T} {W}x{H} visible={vis.mean():.3f}")


if __name__ == "__main__":
    main()

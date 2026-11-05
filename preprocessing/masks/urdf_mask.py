#!/usr/bin/env python
"""Robot arm and hand mask from the rendered model and SAM3 (Appendix B.2.3).

For a robot frame the mask is built from a rendering of the robot at the recorded joint state
and a SAM3 segmentation, which are complementary: the render cannot lose a link but carries
calibration error, SAM3 follows the real outline but misses parts and fires on other arms in
view. A SAM3 component is kept only if enough of it lies on the (dilated) render, and the
mask is the union of the render and the kept SAM3 components. Without a SAM3 mask the render
alone is used. The mask is saved tight; enlargement happens in ``robotryon.masks``.

Writes a gray mask mp4 (>127 = robot), e.g. ``mask_<view>.mp4``.
"""

from __future__ import annotations

import cv2
import numpy as np

from preprocessing.episode_io import (LeRobotEpisode, VideoWriter, iter_frames, parse, parser,
                                      video_info)


def filter_components(mask: np.ndarray, render: np.ndarray, dilate_frac: float = 0.03,
                      min_overlap: float = 0.2, min_px: int = 40) -> np.ndarray:
    """Keep the 8-connected components of ``mask`` with at least ``min_overlap`` of their area
    on ``render`` dilated by ``dilate_frac`` of the image width (and at least ``min_px``)."""
    m = mask.astype(np.uint8)
    if not m.any():
        return mask.astype(bool)
    k = max(3, int(round(dilate_frac * mask.shape[1])) | 1)
    near = cv2.dilate(render.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    on = np.bincount(lab[near > 0].ravel(), minlength=n)
    area = stats[:, cv2.CC_STAT_AREA]
    keep = (on >= min_overlap * area) & (area >= min_px)
    keep[0] = False
    return keep[lab]


def robot_mask(render: np.ndarray, sam3: np.ndarray | None, cfg) -> np.ndarray:
    """Render ∪ the SAM3 components that agree with it."""
    if sam3 is None:
        return render
    return render | filter_components(sam3, render, cfg.dilate_frac, cfg.min_overlap, cfg.min_px)


def write_robot_mask(emb, states: np.ndarray, view: str, out, fps: float, size, cfg,
                     sam3_video=None) -> None:
    """Stream the robot mask of every frame of one view to ``out``."""
    sam = iter_frames(sam3_video) if sam3_video else None
    with VideoWriter(out, size[0], size[1], fps, gray=True) as w:
        for q in states:
            render = emb.silhouette(q, view, size, scale=cfg.render_scale)
            s = None
            if sam is not None:
                f = next(sam, None)
                s = None if f is None else f[..., 0] > 127
            w.write(robot_mask(render, s, cfg))


def main():
    p = parser("Robot mask = render ∪ agreeing SAM3 parts", "configs/preprocess/masks.yaml")
    p.add_argument("--embodiment", required=True)
    p.add_argument("--lerobot_root", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--view", required=True)
    p.add_argument("--sam3", default=None, help="SAM3 robot mask mp4 (sam3_robot.py)")
    p.add_argument("--out", required=True)
    a, cfg = parse(p)
    from robotryon.embodiment import load_embodiment
    emb = load_embodiment(a.embodiment)
    ep = LeRobotEpisode(a.lerobot_root, a.episode)
    fps, W, H, n = video_info(ep.video(emb.spec.cameras[a.view].video_key))
    states = ep.column("observation.state")[:n]
    write_robot_mask(emb, states, a.view, a.out, fps, (W, H), cfg.robot_mask, a.sam3)


if __name__ == "__main__":
    main()

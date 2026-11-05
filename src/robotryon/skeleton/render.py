"""Draw hand skeletons as 2D images.

The same renderer is used for every embodiment and for humans, so the skeleton carries pose
only and never reveals what kind of hand produced it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .topology import COLORS, EDGES, GROUP_OF


def draw_skeleton(kp: np.ndarray, vis: np.ndarray, width: int, height: int,
                  thickness: int = 2, radius: int = 3, supersample: int = 2) -> np.ndarray:
    """Render one frame of hand skeletons on a black canvas.

    Args:
        kp: ``(H, 17, 2)`` pixel coordinates.
        vis: ``(H, 17)`` visibility. Invisible or non-finite keypoints are skipped together with
            their edges.
        width, height: canvas size in pixels.

    Returns:
        ``(height, width, 3)`` uint8 RGB image.
    """
    s = max(1, int(supersample))
    canvas = np.zeros((height * s, width * s, 3), np.uint8)
    # Same stroke convention as the original renderer: sizes are given on the supersampled
    # canvas at half scale, so thickness=2, radius=3 is the production look.
    lw, dr = max(1, thickness * s // 2), max(1, radius * s // 2)
    kp = np.asarray(kp, np.float64)
    vis = np.asarray(vis, bool) & np.isfinite(kp).all(-1)
    for h in range(kp.shape[0]):
        pts = np.round(kp[h] * s).astype(np.int64) if vis[h].any() else None
        if pts is None:
            continue
        for a, b, g in EDGES:
            if vis[h, a] and vis[h, b]:
                cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), COLORS[g], lw, cv2.LINE_AA)
        for j in range(kp.shape[1]):
            if vis[h, j]:
                cv2.circle(canvas, tuple(pts[j]), dr, COLORS[GROUP_OF[j]], -1, cv2.LINE_AA)
    if s > 1:
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    return canvas


def draw_skeleton_video(kp: np.ndarray, vis: np.ndarray, width: int, height: int,
                        **kw) -> np.ndarray:
    """``(T, H, 17, 2)`` keypoints -> ``(T, height, width, 3)`` uint8 skeleton video."""
    return np.stack([draw_skeleton(kp[t], vis[t], width, height, **kw)
                     for t in range(kp.shape[0])])


def paint(frame: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """Overlay a rendered skeleton on a frame; skeleton pixels win wherever drawn."""
    out = frame.copy()
    lit = skeleton.max(axis=-1) > 0
    out[lit] = skeleton[lit]
    return out


def condition_frame(frame: np.ndarray, mask: np.ndarray | None,
                    skeleton: np.ndarray) -> np.ndarray:
    """Phi(o; M, S) = ((1 - M) * o) (+) S: black out the mask region, then draw the skeleton.

    The skeleton is drawn even slightly outside the mask so fingertips are never clipped.
    ``mask=None`` leaves the frame unmasked (mask-free inference).
    """
    out = frame.copy()
    if mask is not None:
        out[np.asarray(mask).astype(bool)] = 0
    return paint(out, skeleton)

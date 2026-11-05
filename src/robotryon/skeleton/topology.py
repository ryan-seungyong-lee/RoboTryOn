"""Hand keypoint topology shared by every embodiment.

Each hand has 17 keypoints in MediaPipe order with the four distal (DIP) joints removed:

    0           wrist
    1, 2, 3, 4  thumb: base (CMC), MCP, IP, tip
    5, 6, 7     index: knuckle (MCP), middle (PIP), tip
    8, 9, 10    middle
    11, 12, 13  ring
    14, 15, 16  little

Robot hands map their links onto the same 17 points, so hands with different degrees of
freedom share one skeleton. Arrays of keypoints use shape ``(..., 2, 17, D)``: hand 0 is the
left hand, hand 1 is the right hand.
"""

from __future__ import annotations

import numpy as np

NUM_KEYPOINTS = 17
NUM_HANDS = 2

KEYPOINT_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_tip",
    "middle_mcp", "middle_pip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_tip",
    "little_mcp", "little_pip", "little_tip",
)

# MediaPipe-21 index of each of the 17 keypoints (the DIP joints 7/11/15/19 are dropped).
MEDIAPIPE21_TO_17 = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20)

GROUP_OF = (
    "palm",
    "thumb", "thumb", "thumb", "thumb",
    "index", "index", "index",
    "middle", "middle", "middle",
    "ring", "ring", "ring",
    "little", "little", "little",
)

# (parent, child, group). The palm connects the wrist to the thumb base and the little knuckle
# and links the knuckles to one another; each finger is a chain from its knuckle to its tip.
EDGES: tuple[tuple[int, int, str], ...] = (
    (0, 1, "palm"), (0, 5, "palm"), (5, 8, "palm"), (8, 11, "palm"), (11, 14, "palm"),
    (0, 14, "palm"),
    (1, 2, "thumb"), (2, 3, "thumb"), (3, 4, "thumb"),
    (5, 6, "index"), (6, 7, "index"),
    (8, 9, "middle"), (9, 10, "middle"),
    (11, 12, "ring"), (12, 13, "ring"),
    (14, 15, "little"), (15, 16, "little"),
)

COLORS = {
    "palm": (255, 170, 60), "thumb": (255, 232, 70), "index": (110, 235, 120),
    "middle": (95, 205, 255), "ring": (185, 150, 255), "little": (255, 125, 205),
}

# Finger segments a keypoint belongs to, used to scale the training-time perturbation.
# A keypoint is displaced by at most half the length of the shortest finger segment it touches.
_FINGER_EDGES = tuple((a, b) for a, b, g in EDGES if g != "palm")


def segment_length_per_keypoint(kp: np.ndarray, vis: np.ndarray | None = None) -> np.ndarray:
    """Length of the finger segment each keypoint belongs to.

    Args:
        kp: ``(..., 17, D)`` keypoints of one or more hands.
        vis: optional ``(..., 17)`` visibility; invisible endpoints do not contribute.

    Returns:
        ``(..., 17)`` lengths. The wrist uses its distance to the nearest knuckle. A keypoint
        without any valid segment gets 0, so it is never displaced.
    """
    kp = np.asarray(kp, dtype=np.float64)
    if vis is None:
        vis = np.isfinite(kp).all(-1)
    out = np.full(kp.shape[:-1], np.inf)
    edges = _FINGER_EDGES + ((0, 5), (0, 1))
    for a, b in edges:
        ok = vis[..., a] & vis[..., b]
        d = np.linalg.norm(np.nan_to_num(kp[..., a, :] - kp[..., b, :]), axis=-1)
        d = np.where(ok, d, np.inf)
        out[..., a] = np.minimum(out[..., a], d)
        out[..., b] = np.minimum(out[..., b], d)
    out[~np.isfinite(out)] = 0.0
    return out

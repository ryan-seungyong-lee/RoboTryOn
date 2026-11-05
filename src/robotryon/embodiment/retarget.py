"""Retarget a reconstructed human hand onto a robot hand (AnyTeleop vector retargeting).

The skeleton that conditions the generative models must look like the target robot's hand,
so a human hand is not drawn directly. Instead (Appendix B.2.2):

1. Both hands are expressed in a wrist frame (origin at the wrist, x toward the knuckle
   centroid, z along the knuckle-plane normal) built identically from the human joints and
   from the robot's 17 keypoints.
2. Following the vector retargeting of AnyTeleop (Qin et al., 2023), the robot hand joints are
   optimized so that the robot's wrist-to-fingertip and wrist-to-middle-joint vectors match the
   human's, scaled by the ratio of the palm sizes, within the joint limits of the robot model,
   with a small penalty on the change from the previous frame (``VECTORS17``,
   ``MujocoEmbodiment.retarget_hand``).
3. Robot forward kinematics at those joints gives the robot hand's 17 keypoints. Bone
   matching stretches each robot finger to the human's knuckle distance and finger length
   while keeping the robot's directions (the robot's own curl), so the fingertips reach what
   the human touches.
4. The result is placed at the human wrist frame, in the human's camera frame.

The frame helpers and the vector set live here; the optimization needs FK and is done by the
embodiment.
"""

from __future__ import annotations

import numpy as np

from robotryon.skeleton.topology import MEDIAPIPE21_TO_17

# MediaPipe / MANO 21-joint layout: wrist, then four joints per finger, proximal to distal.
CHAINS21 = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16), "little": (17, 18, 19, 20)}
# 17-keypoint layout (robotryon.skeleton.topology): (knuckle, distal points) per finger.
CHAINS17 = {"thumb": (1, (2, 3, 4)), "index": (5, (6, 7)), "middle": (8, (9, 10)),
            "ring": (11, (12, 13)), "little": (14, (15, 16))}
FINGERS4 = ("index", "middle", "ring", "little")

# Robot wrist frame from the 17-keypoint layout: wrist, index knuckle, little knuckle.
WRIST17, INDEX17, LITTLE17 = 0, 5, 14

# Retargeting vectors as (origin, target) pairs of the 17-keypoint layout: wrist to every
# fingertip and wrist to the joint before it (thumb distal, index/middle/ring/little middle).
TIPS17 = tuple(distal[-1] for _, distal in CHAINS17.values())
MIDS17 = (3, 6, 9, 12, 15)
VECTORS17 = tuple((WRIST17, t) for t in TIPS17) + tuple((WRIST17, m) for m in MIDS17)


def to21(kp: np.ndarray) -> np.ndarray:
    """``(..., 17|21, 3)`` -> 21 joints. Missing DIP joints are put midway between PIP and
    tip, which leaves every finger length and direction used below unchanged."""
    kp = np.asarray(kp, np.float64)
    if kp.shape[-2] == 21:
        return kp
    out = np.zeros(kp.shape[:-2] + (21, 3))
    out[..., MEDIAPIPE21_TO_17, :] = kp
    for pip, dip, tip in ((6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20)):
        out[..., dip, :] = 0.5 * (out[..., pip, :] + out[..., tip, :])
    return out


def wrist_frame(w, imcp, lmcp):
    """Right-handed frame (columns = axes) at the wrist, or None when degenerate."""
    x = (imcp + lmcp) * 0.5 - w
    n = np.cross(imcp - w, lmcp - w)
    nx, nn = np.linalg.norm(x), np.linalg.norm(n)
    if nx < 1e-9 or nn < 1e-9:
        return None
    x, n = x / nx, n / nn
    return np.stack([x, np.cross(n, x), n], axis=1)


def palm_size(p17: np.ndarray) -> float:
    """Mean wrist-to-knuckle distance of the four fingers (17-keypoint layout)."""
    return float(np.mean([np.linalg.norm(p17[CHAINS17[f][0]] - p17[WRIST17]) for f in FINGERS4]))


def local_vectors(p17: np.ndarray):
    """``(17, 3)`` keypoints -> ``(len(VECTORS17), 3)`` vectors in the hand's wrist frame,
    or None when the frame is degenerate."""
    R = wrist_frame(p17[WRIST17], p17[INDEX17], p17[LITTLE17])
    if R is None:
        return None
    o, t = zip(*VECTORS17)
    return (p17[list(t)] - p17[list(o)]) @ R


def human17(j21: np.ndarray) -> np.ndarray:
    """``(21, 3)`` human joints -> the 17-keypoint layout."""
    return np.asarray(j21, np.float64)[list(MEDIAPIPE21_TO_17)]


def bone_match(local17: np.ndarray, j21: np.ndarray) -> np.ndarray:
    """Stretch robot finger segments (wrist-frame 17 keypoints) to the human's lengths.

    Each knuckle is moved along its own direction to the human wrist-to-knuckle distance, then
    the rest of the finger is scaled about the knuckle so knuckle-to-tip matches the human's.
    Directions stay the robot's.
    """
    out = local17.copy()
    for f, (k, distal) in CHAINS17.items():
        hm, ht = CHAINS21[f][0], CHAINS21[f][-1]
        r_palm, h_palm = np.linalg.norm(local17[k]), np.linalg.norm(j21[hm] - j21[0])
        if r_palm > 1e-6 and h_palm > 1e-6:
            out[k] = local17[k] * (h_palm / r_palm)
        r_fin = np.linalg.norm(local17[distal[-1]] - local17[k])
        h_fin = np.linalg.norm(j21[ht] - j21[hm])
        s = h_fin / r_fin if (r_fin > 1e-6 and h_fin > 1e-6) else 1.0
        for i in distal:
            out[i] = out[k] + s * (local17[i] - local17[k])
    return out


def place_on_human(robot17: np.ndarray, j21: np.ndarray, match_bones: bool = True):
    """Move robot keypoints (any frame) onto the human wrist frame; None if degenerate."""
    Rr = wrist_frame(robot17[WRIST17], robot17[INDEX17], robot17[LITTLE17])
    Rh = wrist_frame(j21[0], j21[5], j21[17])
    if Rr is None or Rh is None:
        return None
    local = (robot17 - robot17[WRIST17]) @ Rr
    if match_bones:
        local = bone_match(local, j21)
    return j21[0] + local @ Rh.T

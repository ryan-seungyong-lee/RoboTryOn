"""Robot-only reference image: the robot's pixels on a plain white background.

The first keyframe of a demonstration has no translated predecessor, so its reference is a real
frame of the target robot with everything but the robot removed; this selects the embodiment.
Training uses the same construction for its robot-only exemplars.
"""

from __future__ import annotations

import cv2
import numpy as np

WHITE = 255


def robot_only(image: np.ndarray, mask: np.ndarray, erode: int = 7,
               background: int = WHITE) -> np.ndarray:
    """Keep the robot pixels of ``image`` (H, W, 3) where ``mask`` (H, W) is set.

    The mask is eroded by ``erode`` px so no background leaks in at the silhouette; everything
    else is filled with ``background``.
    """
    m = np.asarray(mask).astype(bool)
    if erode > 0:
        k = int(erode) | 1
        m = cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    out = np.full_like(image, background)
    out[m] = image[m]
    return out

"""Convex hull per arm.

A segmentation silhouette gives the embodiment away (a robot arm has cables and a different
outline than a human arm). Each connected component of the mask is therefore replaced by its
convex hull. Small gaps are closed before the component split, so a limb and a thin cable that
pinches to a one-pixel thread stay one component instead of producing a separate cable-shaped
polygon.
"""
from __future__ import annotations

import cv2
import numpy as np

PRE_DILATE = 9            # closing kernel before the component split (px at mask resolution)
MIN_COMPONENT_PX = 256    # smaller components are segmentation noise, not an arm


def arm_hulls(mask: np.ndarray, pre_dilate: int = PRE_DILATE,
              min_component_px: int = MIN_COMPONENT_PX) -> list[np.ndarray]:
    """Binary mask ``(H, W)`` -> list of ``(K, 2)`` float32 hull polygons in mask pixels."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() == 0:
        return []
    if pre_dilate > 0:
        k = pre_dilate | 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab = cv2.connectedComponents(m)
    polys = []
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        if ys.size < min_component_px:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        polys.append(cv2.convexHull(pts).reshape(-1, 2).astype(np.float32))
    return polys


def scale_hulls(polys: list[np.ndarray], sx: float, sy: float,
                offset: tuple[float, float] = (0.0, 0.0)) -> list[np.ndarray]:
    """Map polygons from mask pixels to output pixels: ``(p - offset) * (sx, sy)``."""
    off = np.asarray(offset, np.float32)
    s = np.array([sx, sy], np.float32)
    return [(p - off) * s for p in polys]


def fill_hulls(polys: list[np.ndarray], width: int, height: int) -> list[np.ndarray]:
    """Rasterise each polygon separately -> list of ``(H, W)`` uint8 {0, 1} masks."""
    out = []
    for p in polys:
        one = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(one, p.astype(np.int32), 1)
        out.append(one)
    return out

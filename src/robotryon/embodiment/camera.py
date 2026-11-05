"""Camera intrinsics: projection of camera-frame points to pixels and back to rays.

A camera is calibrated at one resolution (``width`` x ``height``); every method takes the
image size it should work at and rescales the intrinsics, so the same calibration serves a
video stored at any resolution with the same aspect. Points are in the OpenCV optical frame
(x right, y down, z forward). Two lens models are supported:

* ``pinhole``: OpenCV plumb-bob distortion ``[k1, k2, p1, p2, k3]`` (empty = rectified).
* ``fisheye``: OpenCV equidistant model ``[k1, k2, k3, k4]`` (zeros = ideal equidistant).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "pinhole"
    dist: tuple = ()
    # Additive pixel offset applied after projection (a fitted principal-point residual),
    # given at the calibration resolution.
    pp_offset: tuple = (0.0, 0.0)
    # Points closer than this to the lens are inside the robot's own head and are not drawn.
    min_depth: float = 0.15
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.model not in ("pinhole", "fisheye"):
            raise ValueError(f"unknown camera model {self.model!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        known = {"width", "height", "fx", "fy", "cx", "cy", "model", "dist", "pp_offset",
                 "min_depth"}
        kw = {k: d[k] for k in known if k in d}
        kw["dist"] = tuple(float(x) for x in d.get("dist", ()) or ())
        kw["pp_offset"] = tuple(float(x) for x in d.get("pp_offset", (0.0, 0.0)))
        return cls(**kw, extra={k: v for k, v in d.items() if k not in known})

    # --- intrinsics at a given resolution -------------------------------------------------
    def _scale(self, image_size):
        if image_size is None:
            return 1.0, 1.0
        return image_size[0] / self.width, image_size[1] / self.height

    def K(self, image_size=None) -> np.ndarray:
        sx, sy = self._scale(image_size)
        return np.array([[self.fx * sx, 0.0, self.cx * sx],
                         [0.0, self.fy * sy, self.cy * sy],
                         [0.0, 0.0, 1.0]])

    def D(self) -> np.ndarray:
        n = 4 if self.model == "fisheye" else 5
        d = np.zeros(n)
        d[:len(self.dist)] = self.dist
        return d

    def offset(self, image_size=None) -> np.ndarray:
        sx, sy = self._scale(image_size)
        return np.array([self.pp_offset[0] * sx, self.pp_offset[1] * sy])

    # --- projection ------------------------------------------------------------------------
    def project(self, pts_cam: np.ndarray, image_size=None):
        """``(..., 3)`` camera-frame points -> ``(..., 2)`` pixels and ``(...)`` depth-ok mask.

        Points behind ``min_depth`` get NaN pixels.
        """
        pts = np.asarray(pts_cam, np.float64)
        shape = pts.shape[:-1]
        flat = pts.reshape(-1, 3)
        ok = np.isfinite(flat).all(1) & (flat[:, 2] > self.min_depth)
        px = np.full((flat.shape[0], 2), np.nan)
        if ok.any():
            obj = flat[ok].reshape(-1, 1, 3)
            K, D = self.K(image_size), self.D()
            if self.model == "fisheye":
                uv, _ = cv2.fisheye.projectPoints(obj, np.zeros(3), np.zeros(3), K, D)
            else:
                uv, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, D)
            px[ok] = uv.reshape(-1, 2) + self.offset(image_size)
        return px.reshape(shape + (2,)), ok.reshape(shape)

    def normalized(self, pixels: np.ndarray, image_size=None) -> np.ndarray:
        """``(..., 2)`` pixels -> ``(..., 2)`` undistorted normalized coordinates (x/z, y/z)."""
        uv = np.asarray(pixels, np.float64)
        shape = uv.shape[:-1]
        uv = (uv.reshape(-1, 2) - self.offset(image_size)).reshape(-1, 1, 2)
        K, D = self.K(image_size), self.D()
        if self.model == "fisheye":
            n = cv2.fisheye.undistortPoints(uv, K, D)
        else:
            n = cv2.undistortPoints(uv, K, D)
        return n.reshape(shape + (2,))

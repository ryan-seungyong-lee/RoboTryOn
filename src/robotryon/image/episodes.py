"""Episode directories in the shared RoboTryOn data layout, and random frame access.

::

    <root>/<episode_id>/
      meta.json              {"fps", "views", "num_frames", "caption", "embodiment"}
      video_<view>.mp4       RGB
      keypoints_<view>.npz   kp (T, 2, 17, 2) px at video resolution, vis (T, 2, 17)
      mask_<view>.mp4        tight arm + hand mask (gray, > 127 = masked)
      object_mask_<view>.mp4 optional manipulated-object mask
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class Episode:
    path: Path
    views: list[str]
    num_frames: int
    fps: float
    caption: str = ""
    embodiment: str = ""
    root: int = 0
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, root: int = 0) -> "Episode":
        path = Path(path)
        m = json.loads((path / "meta.json").read_text())
        return cls(path=path, views=list(m["views"]), num_frames=int(m["num_frames"]),
                   fps=float(m["fps"]), caption=str(m.get("caption", "")),
                   embodiment=str(m.get("embodiment", "")), root=root, extra=m)

    def video(self, view: str) -> Path:
        return self.path / f"video_{view}.mp4"

    def mask(self, view: str) -> Path:
        return self.path / f"mask_{view}.mp4"

    def object_mask(self, view: str) -> Path | None:
        p = self.path / f"object_mask_{view}.mp4"
        return p if p.exists() else None

    def keypoints(self, view: str) -> Path:
        return self.path / f"keypoints_{view}.npz"


def scan_root(root: str | Path, root_index: int = 0) -> list[Episode]:
    return [Episode.load(p.parent, root_index) for p in sorted(Path(root).glob("*/meta.json"))]


class FrameReader:
    """Small LRU of open ``cv2.VideoCapture`` objects and keypoint files (one per worker)."""

    def __init__(self, max_open: int = 16):
        self._caps: OrderedDict[str, cv2.VideoCapture] = OrderedDict()
        self._kps: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.max_open = max_open

    def _cap(self, path: Path) -> cv2.VideoCapture:
        key = str(path)
        if key in self._caps:
            self._caps.move_to_end(key)
            return self._caps[key]
        cap = cv2.VideoCapture(key)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video {key}")
        self._caps[key] = cap
        while len(self._caps) > self.max_open:
            self._caps.popitem(last=False)[1].release()
        return cap

    def rgb(self, path: Path, t: int) -> np.ndarray:
        cap = self._cap(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"cannot decode frame {t} of {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def binary(self, path: Path, t: int) -> np.ndarray:
        """Gray mask frame -> bool ``(H, W)`` (> 127)."""
        return self.rgb(path, t).max(axis=-1) > 127

    def keypoints(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        key = str(path)
        if key not in self._kps:
            z = np.load(key)
            self._kps[key] = (z["kp"].astype(np.float32), z["vis"].astype(bool))
            while len(self._kps) > 4 * self.max_open:
                self._kps.popitem(last=False)
        return self._kps[key]


def output_geometry(src_w: int, src_h: int, long_edge: int, aspect: tuple[int, int] | None,
                    anchor: str = "bottom") -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Crop box ``(x0, y0, w, h)`` in source pixels and output size ``(W, H)``.

    A source with a different aspect ratio than ``aspect`` is cropped to it first ("bottom" keeps
    the bottom edge, where the arms enter; "center" crops symmetrically). The long edge is then
    resized to ``long_edge`` and both sides snapped to multiples of 16 (VAE /8 and 2x2 packing).
    """
    x0, y0, w, h = 0, 0, src_w, src_h
    if aspect is not None and abs(src_w / src_h - aspect[0] / aspect[1]) > 1e-3:
        if src_w / src_h < aspect[0] / aspect[1]:          # too tall: crop height
            h = int(round(src_w * aspect[1] / aspect[0]))
            y0 = src_h - h if anchor == "bottom" else (src_h - h) // 2
        else:                                                # too wide: crop width, centred
            w = int(round(src_h * aspect[0] / aspect[1]))
            x0 = (src_w - w) // 2
    s = long_edge / max(w, h)
    snap = lambda v: max(16, int(round(v / 16) * 16))      # noqa: E731
    return (x0, y0, w, h), (snap(w * s), snap(h * s))


def crop_resize(img: np.ndarray, box, size, src_size, nearest: bool = False) -> np.ndarray:
    """Crop ``box`` (source-frame pixels) out of ``img`` and resize to ``size``.

    ``img`` may have a different resolution than the source frame ``src_size`` (e.g. a mask video
    stored smaller); the box is rescaled accordingly. RGB uses PIL Lanczos (as the original
    pipeline did), masks use nearest neighbour.
    """
    fx, fy = img.shape[1] / src_size[0], img.shape[0] / src_size[1]
    x0, y0 = int(round(box[0] * fx)), int(round(box[1] * fy))
    x1, y1 = int(round((box[0] + box[2]) * fx)), int(round((box[1] + box[3]) * fy))
    sub = img[y0:y1, x0:x1]
    if nearest:
        return cv2.resize(sub.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    pil = Image.fromarray(sub)
    return np.asarray(pil if pil.size == tuple(size) else pil.resize(size, Image.LANCZOS))


@dataclass
class ViewFrame:
    """One camera frame mapped to the model resolution.

    image: uint8 ``(H, W, 3)``; hulls: per-arm convex hulls in output pixels; kp/vis: hand
    keypoints in output pixels; object_mask: uint8 ``(H, W)`` or None.
    """
    image: np.ndarray
    hulls: list
    kp: np.ndarray
    vis: np.ndarray
    object_mask: np.ndarray | None


def load_view_frame(reader: FrameReader, ep: Episode, view: str, t: int, long_edge: int,
                    aspect=None, anchor: str = "bottom", with_object: bool = True) -> ViewFrame:
    from ..masks import arm_hulls, scale_hulls
    rgb = reader.rgb(ep.video(view), t)
    src = (rgb.shape[1], rgb.shape[0])
    box, size = output_geometry(src[0], src[1], long_edge, aspect, anchor)
    sx, sy = size[0] / box[2], size[1] / box[3]
    image = crop_resize(rgb, box, size, src)
    m = reader.binary(ep.mask(view), t)
    mx, my = src[0] / m.shape[1], src[1] / m.shape[0]          # mask px -> source px
    hulls = scale_hulls(arm_hulls(m), mx * sx, my * sy, (box[0] / mx, box[1] / my))
    kp, vis = reader.keypoints(ep.keypoints(view))
    kp_t = (kp[t] - np.array(box[:2], np.float32)) * np.array([sx, sy], np.float32)
    om = None
    if with_object and ep.object_mask(view) is not None:
        om = crop_resize(reader.binary(ep.object_mask(view), t).astype(np.uint8), box, size, src,
                         nearest=True)
    return ViewFrame(image, hulls, kp_t, vis[t].copy(), om)

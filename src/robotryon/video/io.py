"""Minimal mp4 / png I/O for the video model (RGB uint8 arrays, OpenCV underneath)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_frames(path: Path, start: int, count: int, size=None) -> np.ndarray:
    """Frames ``[start, start + count)`` as ``(count, H, W, 3)`` RGB, optionally resized
    (INTER_AREA) to ``size = (W, H)``."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, f = cap.read()
        if not ok:
            break
        if size is not None:
            f = cv2.resize(f, tuple(size), interpolation=cv2.INTER_AREA)
        frames.append(f[..., ::-1])
    cap.release()
    if len(frames) < count:
        raise RuntimeError(f"{path}: wanted {count} frames from {start}, got {len(frames)}")
    return np.ascontiguousarray(np.stack(frames))


def num_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_image(path: Path) -> np.ndarray:
    """PNG -> ``(H, W, 3)`` RGB uint8 at its own resolution."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(img[..., ::-1])


def write_video(frames: np.ndarray, path: Path, fps: float) -> None:
    """``(F, H, W, 3)`` RGB uint8 -> mp4 (mp4v)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        wr.write(np.ascontiguousarray(f[..., ::-1]))
    wr.release()

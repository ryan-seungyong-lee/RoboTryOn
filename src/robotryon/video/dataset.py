"""Training clips for the video model: robot video, its skeleton flow, and its caption.

Episodes follow the shared format (``<root>/<episode>/meta.json``, ``video_<view>.mp4``,
``keypoints_<view>.npz``). One item is a random window of one episode:

* length ``F`` drawn uniformly from the ``8k + 1`` values in ``frames_range`` that fit,
  start drawn uniformly;
* stereo episodes give both views (``V = 2``), except with probability ``mono_p`` where one
  random view is used (``V = 1``), so one model learns stereo and mono generation;
* one crop rectangle (bottom-anchored, scale in ``[min_scale, 1]``) and one horizontal flip
  for every view and for the skeleton. A mirrored stereo pair also swaps its views, since the
  reflected right image is what a left camera would see;
* the skeleton flow ``S_{1:T}`` is rendered on the fly from the keypoints, after the paper's
  augmentation (``robotryon.skeleton.SkeletonAugment``: shift every keypoint by up to half its
  finger segment, drop the whole skeleton with p=0.2, each keypoint with p=0.2), held fixed
  over the clip. The views of a stereo clip are augmented jointly (the hands of both views
  share one drop-all decision).

The skeleton is drawn directly at ``skeleton_size`` (the reference resolution), the same
resolution inference renders at.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..skeleton import SkeletonAugment, draw_skeleton_video
from .chunks import allowed_frame_counts
from .io import read_frames


@dataclass
class Episode:
    path: Path
    views: tuple[str, ...]
    num_frames: int
    caption: str
    root_index: int


def caption_text(template: str, meta: dict, embodiment: str | None = None) -> str:
    """The text prompt: ``template`` filled with the episode's caption (and embodiment)."""
    return template.format(caption=meta.get("caption", ""),
                           embodiment=embodiment or meta.get("embodiment", ""))


def load_episodes(roots: list, min_frames: int, fps: float,
                  caption_template: str = "{caption}") -> tuple[list[Episode], list[float]]:
    """All usable episodes under ``roots`` and their sampling weights.

    ``roots`` items are paths or ``{path, weight}``. A root's weight is split over its
    episodes in proportion to their length.
    """
    episodes, weights = [], []
    for ri, r in enumerate(roots):
        path, w = (r, 1.0) if isinstance(r, str) else (r["path"], float(r.get("weight", 1.0)))
        found = []
        for m in sorted(Path(path).glob("*/meta.json")):
            meta = json.loads(m.read_text())
            if abs(float(meta["fps"]) - fps) > 1e-6:
                raise ValueError(f"{m}: fps {meta['fps']} != {fps}; resample the episode first")
            if int(meta["num_frames"]) < min_frames:
                continue
            found.append(Episode(m.parent, tuple(meta["views"]), int(meta["num_frames"]),
                                 caption_text(caption_template, meta), ri))
        if not found:
            raise FileNotFoundError(f"no episode with >= {min_frames} frames under {path}")
        total = sum(e.num_frames for e in found)
        episodes += found
        weights += [w * e.num_frames / total for e in found]
    return episodes, weights


def crop_rect(W: int, H: int, aspect: float, scale: float, rng: random.Random):
    """A crop of the given aspect, bottom-anchored (arms enter from the bottom edge)."""
    bw, bh = (int(round(H * aspect)), H) if W / H > aspect else (W, int(round(W / aspect)))
    w = max(8, int(round(bw * scale)))
    h = min(bh, max(8, int(round(w / aspect))))
    return rng.randint(0, max(0, W - w)), H - h, w, h


def crop_frames(frames: np.ndarray, rect, flip: bool, size) -> np.ndarray:
    x, y, w, h = rect
    out = np.empty((len(frames), size[1], size[0], 3), np.uint8)
    for i, f in enumerate(frames):
        c = f[y:y + h, x:x + w]
        out[i] = cv2.resize(c[:, ::-1] if flip else c, size, interpolation=cv2.INTER_AREA)
    return out


def crop_keypoints(kp: np.ndarray, rect, flip: bool, size) -> np.ndarray:
    """Map pixel keypoints through the same crop, flip and resize as the frames."""
    x, y, w, h = rect
    out = np.array(kp, np.float32, copy=True)
    out[..., 0] = (out[..., 0] - x + 0.5) * size[0] / w - 0.5
    out[..., 1] = (out[..., 1] - y + 0.5) * size[1] / h - 0.5
    if flip:
        out[..., 0] = size[0] - 1 - out[..., 0]
    return out


def to_tensor(a: np.ndarray) -> torch.Tensor:
    """``(V, F, H, W, 3)`` uint8 RGB -> ``(V, 3, F, H, W)`` float in [-1, 1]."""
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 4, 1, 2, 3).float() / 127.5 - 1


class VideoClipDataset(Dataset):
    """Random training windows. Items are unbatched dicts (clip lengths differ)."""

    def __init__(self, episodes: list[Episode], out_size, skeleton_size, frames_range,
                 min_scale: float, flip_p: float, mono_p: float, augment: SkeletonAugment,
                 draw_kw: dict | None = None, fps: float = 20.0, seed: int = 0):
        self.episodes = episodes
        self.out_size, self.skeleton_size = tuple(out_size), tuple(skeleton_size)
        self.choices = allowed_frame_counts(*frames_range)
        self.min_scale, self.flip_p, self.mono_p = min_scale, flip_p, mono_p
        self.augment, self.draw_kw = augment, dict(draw_kw or {})
        self.fps, self.seed, self.epoch = fps, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]
        rng = random.Random(hash((self.seed, self.epoch, idx)) & 0xFFFFFFFF)
        nrng = np.random.default_rng(rng.getrandbits(64))
        F = rng.choice([f for f in self.choices if f <= ep.num_frames])
        start = rng.randint(0, ep.num_frames - F)
        views = list(ep.views)
        if len(views) > 1 and rng.random() < self.mono_p:
            views = [rng.choice(views)]

        frames = [read_frames(ep.path / f"video_{v}.mp4", start, F) for v in views]
        kps, vis = [], []
        for v in views:
            z = np.load(ep.path / f"keypoints_{v}.npz")
            kps.append(z["kp"][start:start + F])
            vis.append(z["vis"][start:start + F].astype(bool))

        H, W = frames[0].shape[1:3]
        rect = crop_rect(W, H, self.out_size[0] / self.out_size[1],
                         rng.uniform(self.min_scale, 1.0), rng)
        flip = rng.random() < self.flip_p
        video = np.stack([crop_frames(f, rect, flip, self.out_size) for f in frames])
        kp = np.concatenate([crop_keypoints(k, rect, flip, self.skeleton_size) for k in kps], 1)
        kp, vs = self.augment(kp, np.concatenate(vis, 1), nrng, temporal=True)
        sw, sh = self.skeleton_size
        skel = np.stack([draw_skeleton_video(kp[:, 2 * i:2 * i + 2], vs[:, 2 * i:2 * i + 2],
                                             sw, sh, **self.draw_kw)
                         for i in range(len(views))])
        if flip and len(views) == 2:                  # a mirrored stereo pair swaps cameras
            video, skel = video[::-1], skel[::-1]
        return {"video": to_tensor(video), "condition": to_tensor(skel),
                "caption": ep.caption, "num_frames": F, "fps": self.fps,
                "meta": {"episode": ep.path.name, "start": start, "views": len(views),
                         "flip": flip}}

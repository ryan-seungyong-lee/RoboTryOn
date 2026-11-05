"""Readers and writers for the episode directory every model trains and infers on.

    <episode>/
      meta.json               {"fps", "views", "num_frames", "caption", "embodiment"}
      video_<view>.mp4        RGB, H.264
      keypoints_<view>.npz    kp float32 (T, 2, 17, 2) pixels at video resolution (0 where
                              invisible); vis bool (T, 2, 17); hands ordered (left, right)
      mask_<view>.mp4         tight arm+hand mask, gray, >127 = masked (lossless H.264)
      object_mask_<view>.mp4  manipulated object, same encoding (optional)
      keyframes.json          {"indices": [...]} (human clips)

Also holds the small CLI helper shared by the preprocessing scripts: I/O paths are ordinary
flags, hyperparameters come from ``--config`` plus dotted ``key=value`` overrides.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from robotryon.common.config import Config, load_config, merge
from robotryon.skeleton.topology import NUM_HANDS, NUM_KEYPOINTS


# --- CLI -------------------------------------------------------------------------------
def parser(description: str, default_config: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=default_config, help="hyperparameter YAML")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. max_gap=120")
    return p


def parse(p: argparse.ArgumentParser, argv=None) -> tuple[argparse.Namespace, Config]:
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.overrides) if a.config else Config()
    return a, cfg


def stage_config(cfg: Config, config_path, name: str) -> Config:
    """Config of one stage of a driver: the file named in ``cfg.stages[name]`` (relative to
    the driver config) with ``cfg[name]`` overrides merged on top."""
    base = load_config(Path(config_path).parent / cfg.stages[name]).to_dict()
    over = cfg[name].to_dict() if isinstance(cfg.get(name), Config) else {}
    return _as_config(merge(base, over))


def _as_config(x):
    if isinstance(x, dict):
        return Config({k: _as_config(v) for k, v in x.items()})
    return [_as_config(v) for v in x] if isinstance(x, list) else x


# --- video -----------------------------------------------------------------------------
def video_info(path) -> tuple[float, int, int, int]:
    """(fps, width, height, num_frames) of a video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, W, H, n


def iter_frames(path):
    """Yield RGB uint8 frames."""
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()


class VideoWriter:
    """Stream frames to ffmpeg. ``gray=True`` writes a lossless single-channel mask video."""

    def __init__(self, path, width: int, height: int, fps: float, gray: bool = False,
                 crf: int = 18):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path, self.gray = str(path), gray
        enc = (["-qp", "0", "-preset", "veryfast"] if gray
               else ["-crf", str(crf), "-preset", "medium"])
        self.proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "gray" if gray else "rgb24", "-s", f"{width}x{height}",
             "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", *enc, "-pix_fmt", "yuv420p",
             "-an", self.path], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame)
        if frame.dtype == bool:
            frame = frame.astype(np.uint8) * 255
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_video(path, frames, fps: float, gray: bool = False) -> None:
    frames = list(frames)
    H, W = frames[0].shape[:2]
    with VideoWriter(path, W, H, fps, gray=gray) as w:
        for f in frames:
            w.write(f)


def read_mask_video(path) -> np.ndarray:
    """Mask mp4 -> bool (T, H, W)."""
    return np.stack([f[..., 0] > 127 for f in iter_frames(path)])


def transcode(src, dst, fps: float | None = None, max_frames: int | None = None) -> None:
    """Re-encode ``src`` as H.264, optionally at a constant ``fps`` (frames are dropped or
    duplicated, not blended) and cut to ``max_frames``."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    vf = ["-vf", f"fps={fps:g}"] if fps else []
    cut = ["-frames:v", str(max_frames)] if max_frames else []
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src), *vf,
                    *cut, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
                    str(dst)], check=True)


# --- episode files ---------------------------------------------------------------------
def save_keypoints(path, kp: np.ndarray, vis: np.ndarray) -> None:
    kp, vis = np.asarray(kp, np.float32), np.asarray(vis, bool)
    T = kp.shape[0]
    if kp.shape != (T, NUM_HANDS, NUM_KEYPOINTS, 2) or vis.shape != (T, NUM_HANDS, NUM_KEYPOINTS):
        raise ValueError(f"bad keypoint shapes {kp.shape} {vis.shape}")
    kp = np.where(vis[..., None] & np.isfinite(kp), kp, 0.0).astype(np.float32)
    np.savez_compressed(path, kp=kp, vis=vis)


def load_keypoints(path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["kp"], z["vis"]


def save_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1))


def save_meta(episode_dir, fps: float, views, num_frames: int, caption: str = "",
              embodiment: str = "") -> None:
    save_json(Path(episode_dir) / "meta.json",
              {"fps": float(fps), "views": list(views), "num_frames": int(num_frames),
               "caption": caption, "embodiment": embodiment})


# --- LeRobot v2.1 ----------------------------------------------------------------------
class LeRobotEpisode:
    """One episode of a LeRobot v2.1 dataset: joint vectors, per-view video, task text."""

    def __init__(self, root, episode: int):
        self.root, self.episode = Path(root), int(episode)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = float(self.info["fps"])
        chunk = self.episode // int(self.info.get("chunks_size", 1000))
        self._fmt = dict(episode_chunk=chunk, episode_index=self.episode)

    def column(self, name: str = "observation.state") -> np.ndarray:
        import pyarrow.parquet as pq
        p = self.root / self.info["data_path"].format(**self._fmt)
        t = pq.read_table(p, columns=[name])
        return np.asarray(t.column(name).to_pylist(), np.float64)

    def video(self, video_key: str) -> Path:
        return self.root / self.info["video_path"].format(video_key=video_key, **self._fmt)

    def task(self) -> str:
        tasks = {}
        p = self.root / "meta" / "tasks.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    tasks[int(d["task_index"])] = d["task"]
        try:
            idx = int(self.column("task_index")[0])
        except Exception:
            return next(iter(tasks.values()), "")
        return tasks.get(idx, "")

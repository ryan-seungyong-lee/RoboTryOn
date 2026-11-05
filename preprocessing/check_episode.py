#!/usr/bin/env python
"""Validate an episode directory against the shared format (see ``episode_io``).

Checks meta.json fields, that every per-view file exists with the documented shape and dtype,
that all of them have ``num_frames`` frames at one resolution, and that keyframes (if
present) are sorted, in range and include the first and last frame. Prints one line per file
and exits non-zero if any check fails.

    python -m preprocessing.check_episode <episode_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from preprocessing.episode_io import iter_frames, video_info


def check_episode(ep) -> list[str]:
    ep, errs = Path(ep), []
    meta = json.loads((ep / "meta.json").read_text())
    for k, t in (("fps", (int, float)), ("views", list), ("num_frames", int), ("caption", str),
                 ("embodiment", str)):
        if not isinstance(meta.get(k), t):
            errs.append(f"meta.json: {k!r} missing or not {t}")
    T = meta["num_frames"]
    for v in meta["views"]:
        fps, W, H, n = video_info(ep / f"video_{v}.mp4")
        print(f"video_{v}.mp4: {n} frames {W}x{H} @ {fps:g} fps")
        if n != T:
            errs.append(f"video_{v}.mp4 has {n} frames, meta says {T}")
        if abs(fps - meta["fps"]) > 1e-3:
            errs.append(f"video_{v}.mp4 fps {fps} != meta fps {meta['fps']}")
        z = np.load(ep / f"keypoints_{v}.npz")
        kp, vis = z["kp"], z["vis"]
        print(f"keypoints_{v}.npz: kp {kp.dtype}{kp.shape} vis {vis.dtype}{vis.shape} "
              f"visible {vis.mean():.3f}")
        if kp.dtype != np.float32 or kp.shape != (T, 2, 17, 2):
            errs.append(f"keypoints_{v}.npz kp is {kp.dtype}{kp.shape}, want float32{(T, 2, 17, 2)}")
        if vis.dtype != bool or vis.shape != (T, 2, 17):
            errs.append(f"keypoints_{v}.npz vis is {vis.dtype}{vis.shape}, want bool{(T, 2, 17)}")
        if not np.isfinite(kp).all():
            errs.append(f"keypoints_{v}.npz has non-finite values")
        for name, required in (("mask", True), ("object_mask", False)):
            f = ep / f"{name}_{v}.mp4"
            if not f.exists():
                if required:
                    errs.append(f"{f.name} missing")
                continue
            _, mw, mh, mn = video_info(f)
            vals = np.unique(np.concatenate([np.unique(fr[..., 0]) for fr in iter_frames(f)]))
            binary = np.all((vals <= 5) | (vals >= 250))
            print(f"{f.name}: {mn} frames {mw}x{mh} values {'binary' if binary else vals[:8]}")
            if (mn, mw, mh) != (T, W, H):
                errs.append(f"{f.name} is {mn}x{mw}x{mh}, video is {T}x{W}x{H}")
            if not binary:
                errs.append(f"{f.name} is not a binary mask")
    kf = ep / "keyframes.json"
    if kf.exists():
        idx = json.loads(kf.read_text())["indices"]
        print(f"keyframes.json: {idx}")
        if idx[0] != 0 or idx[-1] != T - 1 or np.any(np.diff(idx) <= 0):
            errs.append(f"keyframes {idx} must be increasing from 0 to {T - 1}")
    return errs


def main():
    errs = check_episode(sys.argv[1])
    for e in errs:
        print("ERROR", e)
    print("OK" if not errs else f"{len(errs)} problem(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()

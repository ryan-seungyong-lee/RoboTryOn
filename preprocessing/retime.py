#!/usr/bin/env python
"""Retime a human episode phase by phase to a target speed (Appendix B.2.5, optional).

By default a human video is synthesized as recorded. For the retiming study each episode is
divided into phases (approach, grasp, lift, place; by default the intervals between
keyframes) and each phase is resampled in time so its mean wrist speed matches the target
speed of that phase (the mean action speed of teleoperated episodes in the same phase) times
a chosen multiple. The keyframe images are kept: keyframes land exactly on integer output
frames, and the skeleton flow between them is resampled, so the video model synthesizes the
same keyframes at new temporal positions.

Reads an episode directory (+ its ``hands.npz``) and writes a retimed episode directory:
keypoints are linearly interpolated, video and masks take the nearest source frame,
``keyframes.json`` gets the new indices (``source_indices`` keeps the old ones) and
``timemap.json`` the fractional source frame of every output frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from preprocessing.episode_io import (VideoWriter, iter_frames, load_keypoints, parse, parser,
                                      save_json, save_keypoints, save_meta, video_info)
from preprocessing.keyframes.select_keyframes import smooth_tracks, wrist_speed


def time_map(speed: np.ndarray, anchors: list[int], phases: list[int], target, multiple: float):
    """Fractional source frame per output frame.

    Args:
        speed: (T,) human wrist speed (m/s). anchors: frames that must land on integer output
            frames (keyframes, first and last frame). phases: boundaries of the phases.
        target: target speed (m/s), one value or one per phase.
    Returns:
        (M,) source frame of each output frame, and the output index of every anchor.
    """
    T = len(speed)
    phases = sorted(set([0, T - 1, *phases]))
    targets = np.broadcast_to(np.asarray(target, np.float64), (len(phases) - 1,))
    rate = np.ones(T)                                       # output frames per source frame
    for (a, b), v in zip(zip(phases[:-1], phases[1:]), targets):
        human = float(speed[a:b + 1].mean())
        if human > 1e-6 and v > 0:
            rate[a:b + 1] = human / (v * multiple)
    t_out = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]))])
    anchors = sorted(set([0, T - 1, *anchors]))
    k = [int(round(t_out[a])) for a in anchors]
    for i in range(1, len(k)):                              # keep anchors strictly increasing
        k[i] = max(k[i], k[i - 1] + 1)
    # rescale output time piecewise so every anchor lands exactly on its integer frame
    tau = np.zeros(T)
    for (a, b), (ka, kb) in zip(zip(anchors[:-1], anchors[1:]), zip(k[:-1], k[1:])):
        seg = (t_out[a:b + 1] - t_out[a]) / max(t_out[b] - t_out[a], 1e-9)
        tau[a:b + 1] = ka + seg * (kb - ka)
    src = np.interp(np.arange(k[-1] + 1), tau, np.arange(T, dtype=np.float64))
    src[k] = anchors
    return src, dict(zip(anchors, k))


def resample_keypoints(kp: np.ndarray, vis: np.ndarray, src: np.ndarray):
    """Linear interpolation between neighbouring frames where both are visible, else nearest."""
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, len(kp) - 1)
    w = (src - i0)[:, None, None, None]
    near = np.round(src).astype(int)
    both = vis[i0] & vis[i1]
    lin = (1 - w) * kp[i0] + w * kp[i1]
    out_kp = np.where(both[..., None], lin, kp[near])
    return out_kp.astype(np.float32), both | vis[near]


def resample_video(src_path, dst_path, src: np.ndarray, gray: bool = False) -> None:
    """Nearest-source-frame resampling of a video or mask video."""
    fps, W, H, _ = video_info(src_path)
    near = np.round(src).astype(int)
    frames = iter_frames(src_path)
    cur, cur_i = None, -1
    with VideoWriter(dst_path, W, H, fps, gray=gray) as w:
        for j in near:
            while cur_i < j:
                cur, cur_i = next(frames), cur_i + 1
            w.write(cur[..., 0] if gray else cur)


def main():
    p = parser("Retime a human episode phase by phase", "configs/preprocess/retime.yaml")
    p.add_argument("--episode", required=True, help="episode directory (with hands.npz)")
    p.add_argument("--out", required=True, help="retimed episode directory")
    p.add_argument("--phases", type=int, nargs="*", default=None,
                   help="phase boundary frames (default: the keyframes)")
    a, cfg = parse(p)
    ep, out = Path(a.episode), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ep / "meta.json").read_text())
    keys = json.loads((ep / "keyframes.json").read_text())["indices"]
    z = np.load(ep / "hands.npz")
    T = int(meta["num_frames"])
    valid = z["valid"][0][:T]
    pos = smooth_tracks(z["joints3d"][0][:T, :, 0], valid, cfg.smooth_sigma_s * meta["fps"])
    speed = wrist_speed(pos, valid.sum(0) >= 2, meta["fps"])
    phases = a.phases if a.phases is not None else keys
    src, where = time_map(speed, keys, phases, cfg.target_speed, cfg.multiple)

    for view in meta["views"]:
        kp, vis = load_keypoints(ep / f"keypoints_{view}.npz")
        save_keypoints(out / f"keypoints_{view}.npz", *resample_keypoints(kp[:T], vis[:T], src))
        for name, gray in (("video", False), ("mask", True), ("object_mask", True)):
            f = ep / f"{name}_{view}.mp4"
            if f.exists():
                resample_video(f, out / f.name, src, gray)
    save_json(out / "keyframes.json", {"indices": [where[k] for k in keys],
                                       "source_indices": [int(k) for k in keys]})
    save_json(out / "timemap.json", {"source_frame": src.round(4).tolist(),
                                     "target_speed": cfg.target_speed, "multiple": cfg.multiple})
    save_meta(out, meta["fps"], meta["views"], len(src), meta.get("caption", ""),
              meta.get("embodiment", ""))
    print(f"{out}: {T} -> {len(src)} frames, keyframes {[where[k] for k in keys]}")


if __name__ == "__main__":
    main()

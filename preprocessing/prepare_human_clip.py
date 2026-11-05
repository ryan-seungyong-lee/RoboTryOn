#!/usr/bin/env python
"""Human demonstration video -> episode directory for RoboTryOn inference.

Stages (each is also a standalone script):

1. ``video_<view>.mp4``: every view resampled to ``fps`` and cut to a common length T.
2. ``hands.npz``: HaWoR on all views at once (``hand_recon/run_hawor.py``, run in HaWoR's
   environment through ``--hawor_python``; the views are cross-checked to tell the hands
   apart), then stereo fusion (``hand_recon/fuse_views.py``). Pass ``--hands`` to reuse an
   existing reconstruction.
3. ``keyframes.json``: ``keyframes/select_keyframes.py``.
4. ``keypoints_<view>.npz``: the hands retargeted to the target robot hand and projected with
   the robot's camera model (``skeleton/human_skeleton.py``).
5. ``mask_<view>.mp4`` (SAM3 human arm + hand) and, with ``--object_prompt``,
   ``object_mask_<view>.mp4`` (``masks/``). ``--skip_masks`` leaves both out.
6. ``meta.json``.

Files written (T = frames after resampling, H x W = video size):
    meta.json             {"fps": float, "views": [str], "num_frames": T, "caption": str,
                           "embodiment": str}
    video_<view>.mp4      RGB H.264, T frames, fps
    keypoints_<view>.npz  kp float32 (T, 2, 17, 2) px at W x H; vis bool (T, 2, 17)
    hand_state_<view>.npz closure, dorsumness float32 (T, 2): hand state for the image prompt
    mask_<view>.mp4       gray lossless H.264, T frames, >127 = human arm/hand
    object_mask_<view>.mp4  same encoding, >127 = object (optional)
    keyframes.json        {"indices": [int], "reasons": [str], "threshold": float,
                           "candidates": [int]}
    hands.npz             see hand_recon/fuse_views.py (kept for retiming and wrist correction)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from preprocessing.episode_io import (parse, parser, save_json, save_keypoints, save_meta,
                                      stage_config, transcode, video_info)
from preprocessing.hand_recon.fuse_views import fuse
from preprocessing.keyframes.select_keyframes import select_keyframes
from preprocessing.skeleton.human_skeleton import human_keypoints
from robotryon.captions import hand_state_arrays, save_hand_state
from robotryon.embodiment import load_embodiment

RUN_HAWOR = Path(__file__).resolve().parent / "hand_recon" / "run_hawor.py"


def prepare_videos(views: dict[str, str], out: Path, fps: float) -> int:
    for v, src in views.items():
        transcode(src, out / f"video_{v}.mp4", fps=fps)
    T = min(video_info(out / f"video_{v}.mp4")[3] for v in views)
    cut_videos(views, out, T)
    return T


def cut_videos(views, out: Path, T: int) -> None:
    """Cut every view to T frames so all per-frame files of the episode agree."""
    for v in views:
        if video_info(out / f"video_{v}.mp4")[3] != T:
            tmp = out / f"video_{v}.tmp.mp4"
            transcode(out / f"video_{v}.mp4", tmp, max_frames=T)
            tmp.replace(out / f"video_{v}.mp4")


def reconstruct_hands(emb, views, out: Path, args, cfg) -> None:
    work = Path(tempfile.mkdtemp(prefix="hawor_", dir=out))
    cmd = [args.hawor_python, str(RUN_HAWOR), "--hawor_root", args.hawor_root,
           "--work_dir", str(work)]
    for v in views:
        _, W, H, _ = video_info(out / f"video_{v}.mp4")
        cmd += ["--video", str(out / f"video_{v}.mp4"), "--out", str(work / f"hawor_{v}.npz"),
                "--img_focal", f"{emb.camera(v).K((W, H))[0, 0]:.4f}"]
    subprocess.run(cmd, check=True)
    per_view = {v: dict(np.load(work / f"hawor_{v}.npz")) for v in views}
    cams = {v: emb.camera(v) for v in views}
    v0 = next(iter(views))
    T_ref = {v: emb.stereo_extrinsic(v0, v) for v in views}
    res = fuse(per_view, cams, T_ref, cfg)
    np.savez_compressed(out / "hands.npz", fps=np.float32(per_view[v0]["fps"]), **res)
    shutil.rmtree(work, ignore_errors=True)


def match_fps(path: Path, fps: float) -> dict:
    """Load hands.npz and bring it to the episode frame rate (nearest frame in time) if needed,
    so every per-frame file of the episode, hands included, shares one time base."""
    z = dict(np.load(path))
    src = float(z["fps"])
    if abs(src - fps) < 1e-3:
        return z
    n_src = z["valid"].shape[1]
    n = int(round((n_src - 1) / src * fps)) + 1
    idx = np.clip(np.round(np.arange(n) / fps * src).astype(int), 0, n_src - 1)
    for k in ("joints3d", "valid", "conf", "source"):
        if k in z:
            z[k] = z[k][:, idx]
    for k in ("wrist", "confidence"):
        if k in z:
            z[k] = z[k][idx]
    z["fps"] = np.float32(fps)
    np.savez_compressed(path, **z)
    print(f"{path}: resampled {n_src} frames at {src:g} fps -> {n} at {fps:g} fps")
    return z


def make_masks(views, out: Path, cfg, object_prompts) -> None:
    from preprocessing.masks.sam3_video import Sam3Video, segment_to_video
    sam = Sam3Video()
    try:
        for v in views:
            video = str(out / f"video_{v}.mp4")
            segment_to_video(video, str(out / f"mask_{v}.mp4"), cfg.human, sam=sam)
            if object_prompts:
                segment_to_video(video, str(out / f"object_mask_{v}.mp4"), cfg.object,
                                 prompts=object_prompts, sam=sam)
    finally:
        sam.shutdown()


def main():
    p = parser(__doc__.split("\n")[0], "configs/preprocess/prepare.yaml")
    p.add_argument("--embodiment", required=True, help="target robot YAML")
    p.add_argument("--video", nargs=2, action="append", required=True, metavar=("VIEW", "MP4"),
                   help="view name (as in the embodiment cameras) and its video")
    p.add_argument("--out", required=True, help="episode directory")
    p.add_argument("--caption", default="", help="task instruction")
    p.add_argument("--hands", default=None, help="existing hands.npz (skips HaWoR)")
    p.add_argument("--hawor_root", default=None, help="HaWoR checkout")
    p.add_argument("--hawor_python", default=sys.executable, help="python of HaWoR's env")
    p.add_argument("--object_prompt", action="append", default=[])
    p.add_argument("--skip_masks", action="store_true")
    a, cfg = parse(p)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    emb = load_embodiment(a.embodiment)
    views = dict(a.video)
    for v in views:
        if v not in emb.views:
            p.error(f"view {v!r} is not a camera of {a.embodiment}: {emb.views}")

    T = prepare_videos(views, out, float(cfg.fps))
    if a.hands:
        shutil.copyfile(a.hands, out / "hands.npz")
    else:
        if not a.hawor_root:
            p.error("--hawor_root is required unless --hands is given")
        reconstruct_hands(emb, views, out, a, stage_config(cfg, a.config, "hand_recon"))
    hands = match_fps(out / "hands.npz", float(cfg.fps))
    order = [str(v) for v in hands["views"]]
    if hands["valid"].shape[1] < T:
        T = hands["valid"].shape[1]
        cut_videos(views, out, T)

    kcfg = stage_config(cfg, a.config, "keyframes")
    i0 = int(kcfg.get("view_index", 0))
    keys = select_keyframes(hands["joints3d"][i0, :T, :, 0], hands["valid"][i0, :T],
                            hands["conf"][i0, :T], float(cfg.fps), kcfg)
    save_json(out / "keyframes.json", keys)

    for v in views:
        i = order.index(v)
        _, W, H, _ = video_info(out / f"video_{v}.mp4")
        kp, vis = human_keypoints(emb, hands["joints3d"][i, :T], hands["valid"][i, :T], v,
                                  (W, H), retarget=bool(cfg.retarget))
        save_keypoints(out / f"keypoints_{v}.npz", kp, vis)
        save_hand_state(out / f"hand_state_{v}.npz",
                        *hand_state_arrays(hands["joints3d"][i, :T], hands["valid"][i, :T].astype(bool)))

    if not a.skip_masks:
        make_masks(views, out, stage_config(cfg, a.config, "masks"), a.object_prompt)
    save_meta(out, cfg.fps, list(views), T, a.caption, emb.name)
    print(f"{out}: T={T} views={list(views)} keyframes={keys['indices']}")


if __name__ == "__main__":
    main()

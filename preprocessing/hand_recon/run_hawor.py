#!/usr/bin/env python
"""Reconstruct both hands of one or two videos with HaWoR (Zhang et al., 2025) as 21 MANO joints.

HaWoR is installed separately (https://github.com/ThunderVVV/HaWoR) together with its
weights and the license-gated MANO models (``_DATA/data/mano/MANO_{LEFT,RIGHT}.pkl``); run
this script in HaWoR's own environment with a GPU and pass its checkout as ``--hawor_root``.

Per video: frames are extracted at 30 fps, hands are detected and tracked with HaWoR's
detector (at most two per frame, overlapping boxes merged), and every tracklet is assigned to
the left or right hand BEFORE its pose is regressed (``handedness.py``: two simultaneous hands
are ordered left-to-right in the image, the views of a stereo pair are cross-checked, and the
detector's own left/right class only decides tracklets without such evidence). Give the two
views of a stereo camera in one call so that they are checked against each other.

HaWoR's pose regressor is trained on right hands, so a left hand is regressed on a mirrored
crop and un-mirrored by negating the y and z components of its axis-angle rotations; it then
goes through MANO_LEFT. Both hands come out in the same layout (MediaPipe order: wrist, then
four joints per finger), in the camera frame of a pinhole with focal ``--img_focal`` and the
principal point at the image centre. Pass the real camera's fx (at the video's resolution) as
``--img_focal`` so the 3D joints are metric in that camera.

Output, one ``--out`` npz per ``--video``:
    joints3d (T, 2, 21, 3) float32  camera frame, metres; hands (left, right)
    joints2d (T, 2, 21, 2) float32  pixels of the same pinhole
    valid    (T, 2) bool            frames inside a track (gaps of a track are bridged by
                                    interpolated boxes)
    conf     (T, 2) float32         hand-detector confidence; 0 on bridged frames
    img_focal, img_wh, fps

The results are resampled in time from the 30 fps frames to the input video's own frames
(T = its frame count, ``fps`` = its frame rate).
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handedness import assign_hands, hand_boxes  # noqa: E402  (runs outside the package)

HAWOR_FPS = 30.0          # frame rate HaWoR's video pipeline works at


def load_model(hawor_root: str, checkpoint: str):
    sys.path.insert(0, hawor_root)
    os.chdir(hawor_root)                      # HaWoR resolves _DATA/ relative to the cwd
    from pathlib import Path

    import lib.models.hawor as hm
    from hawor.configs import get_config
    cfg = get_config(str(Path(checkpoint).parent.parent / "model_config.yaml"),
                     update_cachedir=True)
    if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
        cfg.defrost()
        cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.freeze()
    return hm.HAWOR.load_from_checkpoint(checkpoint, strict=False, cfg=cfg)


def extract_frames(video: str, out_dir: str) -> np.ndarray:
    """JPEG frames of ``video`` at ``HAWOR_FPS`` (reused if already extracted)."""
    os.makedirs(out_dir, exist_ok=True)
    if not glob.glob(os.path.join(out_dir, "*.jpg")):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", video, "-vf", f"fps={HAWOR_FPS:g}",
                        "-start_number", "0", os.path.join(out_dir, "%06d.jpg")], check=True)
    return np.array(sorted(glob.glob(os.path.join(out_dir, "*.jpg"))))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / max(union, 1e-9)


def detect_hands(imgs, weights: str, thresh: float = 0.2, max_iou: float = 0.5) -> dict:
    """HaWoR's hand detector and tracker on every frame, keeping at most two hands per frame.

    HaWoR's own ``detect_track`` keeps one box per predicted class, so it drops a hand whenever
    both get the same class; here boxes are merged regardless of class (IoU > ``max_iou``) and
    the two most confident are kept. Untracked boxes get a tracklet of their own.

    Returns flat arrays: frame (N,), track (N,), box (N, 5) xyxy + score, cls (N,) 1 = right.
    """
    import cv2
    import torch
    from ultralytics import YOLO

    model = YOLO(weights)
    rows, fresh = [], -1
    for t, f in enumerate(imgs):
        with torch.no_grad():
            b = model.track(cv2.imread(f), conf=thresh, persist=True, verbose=False)[0].boxes
        xyxy, conf, cls = (v.cpu().numpy() for v in (b.xyxy, b.conf, b.cls))
        ids = b.id.cpu().numpy() if b.id is not None else np.full(len(conf), np.nan)
        keep = []
        for i in np.argsort(-conf):
            if all(_iou(xyxy[i], xyxy[j]) <= max_iou for j in keep):
                keep.append(i)
        for i in keep[:2]:
            if np.isnan(ids[i]):
                ids[i], fresh = fresh, fresh - 1
            rows.append((t, int(ids[i]), *xyxy[i], conf[i], cls[i]))
    r = np.array(rows, np.float64).reshape(-1, 8)
    return dict(frame=r[:, 0].astype(int), track=r[:, 1].astype(int), box=r[:, 2:7],
                cls=r[:, 7])


def regress(model, imgs, det: dict, img_focal: float, device: str = "cuda") -> dict:
    """HaWoR pose + MANO joints of both hands of one view, given detections with ``hand``."""
    import cv2
    from hawor.utils.process import run_mano, run_mano_left
    from hawor.utils.rotation import rotation_matrix_to_angle_axis
    from lib.pipeline.tools import parse_chunks

    T = len(imgs)
    H, W = cv2.imread(imgs[0]).shape[:2]
    center = np.array([W / 2.0, H / 2.0])
    j3 = np.zeros((T, 2, 21, 3), np.float32)
    j2 = np.zeros((T, 2, 21, 2), np.float32)
    valid = np.zeros((T, 2), bool)
    conf = np.zeros((T, 2), np.float32)
    for h in (0, 1):
        boxes, has, real = hand_boxes(det["frame"], det["track"], det["box"], det["hand"], h, T)
        if real.sum() < 2:
            continue
        conf[:, h] = np.where(real, boxes[:, 4], 0.0)
        flip = h == 0
        frames = np.flatnonzero(has)
        for fr, bx in zip(*parse_chunks(frames, boxes[frames], min_len=1)):
            res = model.inference(imgs[fr], bx, img_focal=img_focal, img_center=center,
                                  do_flip=flip, device=device)
            rot, trans, betas = res["pred_rotmat"], res["pred_trans"], res["pred_shape"]
            trans = trans[:, 0] if trans.ndim == 3 else trans
            root = rotation_matrix_to_angle_axis(rot[None, :, 0])
            pose = rotation_matrix_to_angle_axis(rot[None, :, 1:])
            if flip:
                for a in (root, pose):
                    a[..., 1] *= -1
                    a[..., 2] *= -1
            out = (run_mano_left if flip else run_mano)(
                trans[None].float(), root.float(), pose.float(), betas=betas[None].float(),
                use_cuda=device == "cuda")
            p = out["joints"][0].detach().cpu().numpy()[:, :21].astype(np.float64)
            j3[fr, h] = p
            j2[fr, h] = p[..., :2] / np.clip(p[..., 2:3], 1e-6, None) * img_focal + center
            valid[fr, h] = True
    return dict(joints3d=j3, joints2d=j2, valid=valid, conf=conf,
                img_focal=np.float32(img_focal), img_wh=np.array([W, H]))


def resample_to_video(res: dict, n_frames: int, fps: float, src_fps: float = HAWOR_FPS) -> dict:
    """Resample per-frame results from HaWoR's frame times (k / src_fps) to the video's (t / fps).

    Joints are interpolated linearly between the two neighbouring HaWoR frames when both are
    valid, otherwise taken from the nearest one; ``valid`` and ``conf`` come from the nearest.
    """
    T = len(res["valid"])
    x = np.clip(np.arange(n_frames) / fps * src_fps, 0, T - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (x - lo)[:, None]
    near = np.where(w[:, 0] < 0.5, lo, hi)
    out = dict(res)
    out["valid"] = res["valid"][near]
    out["conf"] = res["conf"][near]
    for k in ("joints3d", "joints2d"):
        a = res[k]
        both = (res["valid"][lo] & res["valid"][hi])[:, :, None, None]
        lin = a[lo] * (1 - w[:, :, None, None]) + a[hi] * w[:, :, None, None]
        out[k] = np.where(both, lin, a[near]).astype(np.float32)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", action="append", required=True,
                   help="input video; give twice for the two views of a stereo camera")
    p.add_argument("--out", action="append", required=True, help="output .npz, one per --video")
    p.add_argument("--img_focal", type=float, action="append", required=True,
                   help="camera fx in pixels at the video's resolution (once, or per --video)")
    p.add_argument("--hawor_root", required=True, help="HaWoR checkout (with _DATA/, weights/)")
    p.add_argument("--checkpoint", default=None,
                   help="default: <hawor_root>/weights/hawor/checkpoints/hawor.ckpt")
    p.add_argument("--work_dir", default=None, help="scratch for extracted frames")
    a = p.parse_args()
    n = len(a.video)
    if len(a.out) != n or len(a.img_focal) not in (1, n) or n > 2:
        p.error("give one or two --video, one --out per video, and one --img_focal or one each")
    videos = [os.path.abspath(v) for v in a.video]
    outs = [os.path.abspath(o) for o in a.out]
    focals = a.img_focal * n if len(a.img_focal) == 1 else a.img_focal
    work = os.path.abspath(a.work_dir or os.path.splitext(outs[0])[0] + "_work")
    root = os.path.abspath(a.hawor_root)
    ckpt = a.checkpoint or os.path.join(root, "weights/hawor/checkpoints/hawor.ckpt")

    import cv2
    imgs = [extract_frames(v, os.path.join(work, f"view{i}")) for i, v in enumerate(videos)]
    dets = [detect_hands(im, os.path.join(root, "weights/external/detector.pt")) for im in imgs]
    for i, d in enumerate(dets):
        twin = dets[1 - i] if n == 2 else None
        d["hand"] = assign_hands(d["frame"], d["track"], 0.5 * (d["box"][:, 0] + d["box"][:, 2]),
                                 d["cls"], None if twin is None else
                                 (twin["frame"], 0.5 * (twin["box"][:, 0] + twin["box"][:, 2])))
        print(f"{videos[i]}: {len(np.unique(d['track']))} tracklets, "
              f"{(d['hand'] != (d['cls'] > 0.5)).sum()}/{len(d['hand'])} detections "
              f"assigned against the detector's class")
    model = load_model(root, ckpt)
    for video, out, focal, im, d in zip(videos, outs, focals, imgs, dets):
        cap = cv2.VideoCapture(video)
        fps, n_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        res = resample_to_video(regress(model, im, d, focal), n_frames, fps)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, fps=np.float32(fps), **res)
        print(f"{out}: T={len(res['valid'])} valid L={res['valid'][:, 0].sum()} "
              f"R={res['valid'][:, 1].sum()}")


if __name__ == "__main__":
    main()

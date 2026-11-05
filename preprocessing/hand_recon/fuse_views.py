#!/usr/bin/env python
"""Fuse per-view HaWoR reconstructions into ``hands.npz`` (Appendix B.2.2).

A monocular reconstruction places the hand at roughly the right image position but its depth
is only as good as HaWoR's scale guess. With a stereo camera two corrections are made:

1. **Wrist depth by triangulation.** For each hand and frame seen in both views, the two 2D
   wrist positions are triangulated with the stereo calibration (taken from the embodiment's
   camera models, since the human video is recorded from the robot's head camera). Each
   view's hand is then translated along its wrist ray so the wrist sits at the triangulated
   depth: the hand keeps its reconstructed shape and its wrist keeps its image position. The
   per-frame depth ratio is gated by reprojection error, interpolated over frames without a
   triangulation and median-filtered in time.
2. **Complementing views.** A hand missing in one view (out of frame or misdetected) is
   filled from the other view's (depth-corrected) reconstruction, transformed by the stereo
   extrinsic. A copy whose wrist would land within ``complement_min_sep_px`` of that view's
   other hand is skipped: the two views disagree about which hand that is, and copying would
   draw one hand twice.

Output ``hands.npz``:
    views     (V,) str
    joints3d  (V, T, 2, 21, 3) float32  per view, that view's camera frame, metres
    valid     (V, T, 2) bool
    conf      (V, T, 2) float32         reconstruction confidence (detector score)
    source    (V, T, 2) int8            view index the hand came from, -1 if none
    image_wh  (V, 2) int32
    wrist     (T, 2, 3) float32         wrists in the FIRST view's camera frame (= joints3d[0, :, :, 0])
    confidence (T, 2) float32           conf of the first view, 0 where the hand is missing
    fps       float
A single view is passed through unchanged (V = 1).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from preprocessing.episode_io import parse, parser
from robotryon.embodiment import load_spec


def triangulate(xa: np.ndarray, xb: np.ndarray, T_a_b: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation from normalized image coordinates.

    Args:
        xa, xb: ``(N, 2)`` undistorted normalized coordinates in cameras a and b.
        T_a_b: 4x4 transform from camera b's frame to camera a's.
    Returns:
        ``(N, 3)`` points in camera a's frame.
    """
    Pa = np.eye(4)[:3]
    Pb = np.linalg.inv(T_a_b)[:3]
    out = np.zeros((len(xa), 3))
    for i, (a, b) in enumerate(zip(xa, xb)):
        A = np.stack([a[0] * Pa[2] - Pa[0], a[1] * Pa[2] - Pa[1],
                      b[0] * Pb[2] - Pb[0], b[1] * Pb[2] - Pb[1]])
        X = np.linalg.svd(A)[2][-1]
        out[i] = X[:3] / X[3]
    return out


def _depth_log_ratio(r: np.ndarray, ok: np.ndarray, smooth: int) -> np.ndarray:
    """Fill gaps by interpolation and median-filter a per-frame log depth ratio."""
    if not ok.any():
        return np.zeros_like(r)
    t = np.arange(len(r))
    out = np.interp(t, t[ok], r[ok])
    return median_filter(out, size=max(1, smooth), mode="nearest") if smooth > 1 else out


def fuse(per_view: dict[str, dict], cams: dict, T_ref: dict, cfg) -> dict:
    """Depth-correct and complement per-view reconstructions.

    Args:
        per_view: view -> run_hawor output (joints3d (T,2,21,3), valid, conf, joints2d, img_wh).
        cams: view -> CameraModel of the embodiment.
        T_ref: view -> 4x4 transform from that view's camera frame to the first view's.
    """
    views = list(per_view)
    T = min(len(per_view[v]["valid"]) for v in views)
    j3 = np.stack([per_view[v]["joints3d"][:T].astype(np.float64) for v in views])
    valid = np.stack([per_view[v]["valid"][:T] for v in views])
    conf = np.stack([per_view[v]["conf"][:T] for v in views]).astype(np.float32)
    source = np.where(valid, np.arange(len(views))[:, None, None], -1).astype(np.int8)
    wh = np.stack([np.asarray(per_view[v]["img_wh"]) for v in views]).astype(np.int32)

    if len(views) == 2 and cfg.get("triangulate_wrist", True):
        a, b = 0, 1
        T_a_b = np.linalg.inv(T_ref[views[a]]) @ T_ref[views[b]]
        for h in range(2):
            both = valid[a, :, h] & valid[b, :, h]
            idx = np.where(both)[0]
            if len(idx) == 0:
                continue
            uv = [per_view[v]["joints2d"][:T][idx, h, 0] for v in views]
            xa = cams[views[a]].normalized(uv[0], tuple(wh[a]))
            xb = cams[views[b]].normalized(uv[1], tuple(wh[b]))
            Xa = triangulate(xa, xb, T_a_b)
            Xb = Xa @ np.linalg.inv(T_a_b)[:3, :3].T + np.linalg.inv(T_a_b)[:3, 3]
            for v, X, x in ((a, Xa, xa), (b, Xb, xb)):
                z_tri, z_rec = X[:, 2], j3[v, idx, h, 0, 2]
                reproj = np.linalg.norm(X[:, :2] / np.maximum(X[:, 2:], 1e-9) - x, axis=1)
                reproj_px = reproj * cams[views[v]].K(tuple(wh[v]))[0, 0]
                ok = ((reproj_px < cfg.get("max_reproj_px", 8.0))
                      & (z_tri > cfg.get("min_depth", 0.1)) & (z_tri < cfg.get("max_depth", 2.0))
                      & (z_rec > 1e-6))
                r = np.zeros(T)
                okT = np.zeros(T, bool)
                r[idx[ok]] = np.log(z_tri[ok] / z_rec[ok])
                okT[idx[ok]] = True
                ratio = np.exp(_depth_log_ratio(r, okT, int(cfg.get("depth_smooth", 9))))
                w = j3[v, :, h, 0]                                   # (T, 3) wrist
                shift = w * (ratio - 1.0)[:, None]                   # move along the wrist ray
                j3[v, :, h] += np.where(valid[v, :, h, None], shift, 0.0)[:, None, :]

    if len(views) == 2 and cfg.get("complement", True):
        uv = lambda x: x[..., :2] / np.maximum(x[..., 2:], 1e-9)    # noqa: E731
        for v, o in ((0, 1), (1, 0)):
            T_v_o = np.linalg.inv(T_ref[views[v]]) @ T_ref[views[o]]
            moved = j3[o] @ T_v_o[:3, :3].T + T_v_o[:3, 3]
            sep = np.linalg.norm(uv(moved[:, :, 0]) - uv(j3[v, :, ::-1, 0]), axis=-1)
            clash = valid[v, :, ::-1] & (sep * cams[views[v]].K(tuple(wh[v]))[0, 0]
                                         < cfg.get("complement_min_sep_px", 50.0))
            fill = ~valid[v] & valid[o] & ~clash                     # (T, 2)
            j3[v] = np.where(fill[..., None, None], moved, j3[v])
            conf[v] = np.where(fill, conf[o], conf[v])
            source[v] = np.where(fill, o, source[v])
        valid = source >= 0

    return dict(views=np.array(views), joints3d=j3.astype(np.float32), valid=valid,
                conf=conf, source=source, image_wh=wh,
                wrist=j3[0, :, :, 0].astype(np.float32),
                confidence=np.where(valid[0], conf[0], 0.0).astype(np.float32))


def main():
    p = parser("Fuse per-view HaWoR outputs into hands.npz", "configs/preprocess/hand_recon.yaml")
    p.add_argument("--embodiment", required=True, help="embodiment YAML (camera calibration)")
    p.add_argument("--view", nargs=2, action="append", metavar=("NAME", "NPZ"), required=True,
                   help="view name and its run_hawor.py output; give once per view")
    p.add_argument("--out", required=True)
    a, cfg = parse(p)
    spec = load_spec(a.embodiment)
    per_view = {v: dict(np.load(f)) for v, f in a.view}
    cams = {v: spec.cameras[v].intrinsics for v in per_view}
    T_ref = {v: np.eye(4) for v in per_view}
    if len(per_view) == 2:
        from robotryon.embodiment import load_embodiment
        emb = load_embodiment(a.embodiment)
        v0 = a.view[0][0]
        T_ref = {v: emb.stereo_extrinsic(v0, v) for v in per_view}
    out = fuse(per_view, cams, T_ref, cfg)
    fps = float(next(iter(per_view.values()))["fps"])
    np.savez_compressed(a.out, fps=np.float32(fps), **out)
    print(f"{a.out}: views={list(out['views'])} T={out['valid'].shape[1]} "
          f"valid={out['valid'].mean(axis=1).round(3).tolist()}")


if __name__ == "__main__":
    main()

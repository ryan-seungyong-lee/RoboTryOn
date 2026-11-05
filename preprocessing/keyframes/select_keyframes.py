#!/usr/bin/env python
"""Select keyframes from the 3D wrist trajectories of the reconstructed hands (Appendix B.2.1).

* **Candidates** are local minima of the wrist speed. Speed is ``|d/dt smooth(p)|`` of each
  wrist (positions are smoothed before differentiation), combined over both hands by the max,
  so a candidate is a frame where both wrists are slow: the hands are unblurred there and a
  hand has typically arrived somewhere.
* Walking forward from the previous keyframe, the **next keyframe** is the first candidate
  whose wrist positions differ from those at the previous keyframe by more than a threshold.
  The displacement between two frames is the largest wrist movement over the hands seen in
  both; the threshold is the ``disp_quantile`` quantile of this displacement over all pairs of
  frames of the episode, so keyframes mark changes of location rather than fixed intervals.
* Consecutive keyframes are at most ``max_gap`` frames apart (145 at 20 fps, about 7 s). When
  no candidate qualifies within the cap, the frames of the window with the highest
  reconstruction confidence (at least the ``fallback_conf_quantile`` quantile of the window)
  are considered and the one with the largest displacement is taken; failing that (no
  confident frame), the frame at the cap.
* The first and last frames are always keyframes.

Output ``keyframes.json``: ``{"indices": [...], "reasons": [...], "threshold": float, ...}``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmin

from preprocessing.episode_io import parse, parser, save_json


def smooth_tracks(pos: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """(T, H, 3) wrist positions -> gap-filled, Gaussian-smoothed positions."""
    out = np.zeros_like(pos, dtype=np.float64)
    t = np.arange(len(pos))
    for h in range(pos.shape[1]):
        ok = valid[:, h]
        if ok.sum() < 2:
            continue
        p = np.stack([np.interp(t, t[ok], pos[ok, h, a]) for a in range(3)], 1)
        out[:, h] = gaussian_filter1d(p, sigma, axis=0, mode="nearest") if sigma > 0 else p
    return out


def wrist_speed(pos: np.ndarray, hands: np.ndarray, fps: float) -> np.ndarray:
    """(T,) combined speed: max over the tracked hands of |d/dt p|."""
    if not hands.any():
        return np.zeros(len(pos))
    v = np.linalg.norm(np.gradient(pos[:, hands], axis=0), axis=-1) * fps
    return v.max(axis=1)


def displacement(pos: np.ndarray, valid: np.ndarray, i, j) -> np.ndarray:
    """Largest wrist displacement between frames i and j over hands seen in both (0 if none)."""
    d = np.linalg.norm(pos[i] - pos[j], axis=-1)
    return np.where(valid[i] & valid[j], d, 0.0).max(axis=-1)


def select_keyframes(wrist: np.ndarray, valid: np.ndarray, conf: np.ndarray, fps: float,
                     cfg) -> dict:
    """Run the keyframe rule.

    Args:
        wrist: ``(T, 2, 3)`` wrist positions (any metric frame); valid, conf: ``(T, 2)``.
        cfg: ``smooth_sigma_s, minimum_order, disp_quantile, max_gap, fallback_conf_quantile,
            max_pairs``.
    Returns:
        ``{"indices", "reasons", "threshold", "candidates"}``.
    """
    T = len(wrist)
    valid = np.asarray(valid, bool)
    conf = np.where(valid, np.asarray(conf, np.float64), 0.0)
    hands = valid.sum(0) >= 2
    pos = smooth_tracks(np.asarray(wrist, np.float64), valid, cfg.smooth_sigma_s * fps)
    speed = wrist_speed(pos, hands, fps)
    cand = argrelmin(speed, order=int(cfg.minimum_order))[0] if T > 2 else np.array([], int)
    cand = cand[(cand > 0) & (cand < T - 1)]

    # threshold from the episode's distribution of pairwise wrist displacements
    sub = np.unique(np.linspace(0, T - 1, min(T, int(cfg.max_pairs))).astype(int))
    ii, jj = np.triu_indices(len(sub), 1)
    pair = displacement(pos, valid, sub[ii], sub[jj]) if len(ii) else np.zeros(1)
    thr = float(np.quantile(pair, cfg.disp_quantile))

    # a frame is as confident as its least confident tracked hand
    frame_conf = conf[:, hands].min(axis=1) if hands.any() else np.zeros(T)
    gap = int(cfg.max_gap)
    keys, reasons = [0], ["first"]
    while T - 1 - keys[-1] > 0:
        prev = keys[-1]
        if T - 1 - prev <= gap:
            hit = [c for c in cand if c > prev and displacement(pos, valid, c, prev) > thr]
            keys.append(hit[0] if hit else T - 1)
            reasons.append("candidate" if hit else "last")
            continue
        end = prev + gap
        hit = [c for c in cand if prev < c <= end and displacement(pos, valid, c, prev) > thr]
        if hit:
            keys.append(hit[0])
            reasons.append("candidate")
            continue
        win = np.arange(prev + 1, end + 1)
        wc = frame_conf[win]
        if wc.max() > 0:
            top = win[(wc >= np.quantile(wc, cfg.fallback_conf_quantile)) & (wc > 0)]
            d = displacement(pos, valid, top, prev)
            keys.append(int(top[np.argmax(d)]))
            reasons.append("fallback_confidence")
        else:
            keys.append(end)
            reasons.append("cap")
    return {"indices": [int(k) for k in keys], "reasons": reasons, "threshold": thr,
            "candidates": [int(c) for c in cand]}


def main():
    p = parser("Select keyframes from hands.npz", "configs/preprocess/keyframes.yaml")
    p.add_argument("--hands", required=True, help="hands.npz from hand_recon/fuse_views.py")
    p.add_argument("--out", required=True, help="keyframes.json")
    a, cfg = parse(p)
    z = np.load(a.hands)
    v = int(cfg.get("view_index", 0))
    out = select_keyframes(z["joints3d"][v][:, :, 0], z["valid"][v], z["conf"][v],
                           float(z["fps"]), cfg)
    save_json(a.out, out)
    print(f"{a.out}: {len(out['indices'])} keyframes {out['indices']}")


if __name__ == "__main__":
    main()

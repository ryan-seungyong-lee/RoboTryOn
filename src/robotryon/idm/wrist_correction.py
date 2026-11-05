"""Wrist correction of IDM action labels (paper Sec. 3.4, App. B.2.6).

The IDM reads joint angles from pixels and places the wrist less accurately than the 3D hand
reconstruction of the human source video. For each hand we therefore move the wrist, and only
the wrist, toward the reconstruction:

1. The reconstructed human wrist (camera frame) is mapped to the robot base frame with the
   head-camera extrinsics T_base_cam.
2. The robot hand implied by the IDM joints (FK) is scaled by ``hand_scale``
   about its wrist; the target wrist is the one at which the scaled hand's end-effector keypoint
   coincides with the reconstructed wrist.
3. Targets are kept only on high-confidence frames. The offset d = target - IDM wrist is
   cleaned: excursion rejection (hysteresis around a running median of accepted offsets),
   linear fill of short gaps, zero-phase 2nd-order Butterworth low-pass per usable segment.
4. Damped least squares IK on the arm joints of that side: wrist position to the IDM wrist + d,
   wrist orientation held at the FK orientation of the IDM arm joints (weighted rotation
   residual), damping toward the IDM arm joints, numerical Jacobian of FK, warm-started from
   the previous frame of the segment.
5. The solution is blended into the IDM joints with weights ramping 0 -> 1 over ``ramp`` frames
   at segment edges (not at the clip ends), then the arm tracks get SG smoothing. Hand joints
   and all other joints are left untouched.

Kinematics interface required (minimal protocol; ``robotryon.embodiment.Embodiment`` is
expected to satisfy it)::

    class Kinematics(Protocol):
        arm_joint_slices: dict[str, slice]   # {"left": slice, "right": slice} into the action
        def fk(self, q: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
            # {"left": {"wrist_pos": (3,), "wrist_rot": (3, 3), "keypoints": (17, 3)}, "right": ...}
            # robot base frame, metres; keypoints use robotryon.skeleton order (0 = wrist)

Hands are indexed 0 = left, 1 = right, as in ``robotryon.skeleton``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

SIDES = ("left", "right")

DEFAULTS = dict(
    hand_scale=1.0, ee_keypoint=0, min_confidence=0.5,
    reject_enter_mm=150.0, reject_exit_mm=80.0, reject_hard_mm=500.0, reject_ref_n=15,
    gap_fill=6, cutoff_hz=2.0, fps=20.0, ramp=8, smooth_window=9, smooth_order=2,
    ik_iters=30, w_rot=2000.0, damping=0.5, step_cap=0.08, fd_eps=1e-3,
)


# ----------------------------------------------------------------------------- track cleaning
def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [a, b) runs of True."""
    out, a = [], None
    for i, m in enumerate(mask):
        if m and a is None:
            a = i
        if not m and a is not None:
            out.append((a, i))
            a = None
    if a is not None:
        out.append((a, len(mask)))
    return out


def reject_excursions(d, ok, enter, exit_, hard, ref_n=15):
    """Refuse frames whose offset leaves the level it has been holding (running median of the
    last ``ref_n`` accepted offsets); a refusal lasts until the offset is back within ``exit_``.
    ``hard`` caps |d| before a level is established."""
    good, buf, ref = ok.copy(), [], None
    for t in range(len(d)):
        if not ok[t]:
            continue
        if np.linalg.norm(d[t]) > hard:
            good[t] = False
            continue
        if ref is None:
            buf.append(d[t])
            if len(buf) >= 5:
                ref = np.median(np.array(buf[-ref_n:]), axis=0)
            continue
        dev = float(np.linalg.norm(d[t] - ref))
        if dev > enter or (dev > exit_ and not good[max(t - 1, 0)]):
            good[t] = False
        else:
            buf.append(d[t])
            ref = np.median(np.array(buf[-ref_n:]), axis=0)
    return good


def fill_short_gaps(track, ok, max_gap):
    """Linearly interpolate refused runs of <= max_gap frames between usable frames (in place);
    returns the new usable mask."""
    ok = ok.copy()
    for a, b in segments(~ok):
        if a == 0 or b == len(ok) or b - a > max_gap:
            continue
        for k in range(track.shape[1]):
            track[a:b, k] = np.interp(np.arange(a, b), [a - 1, b], [track[a - 1, k], track[b, k]])
        ok[a:b] = True
    return ok


def lowpass(track, ok, cutoff_hz, fps):
    """Zero-phase Butterworth per usable segment (segments too short for filtfilt are kept)."""
    b, a = butter(2, cutoff_hz / (fps / 2.0), "low")
    out = track.copy()
    for lo, hi in segments(ok):
        if hi - lo > 3 * max(len(a), len(b)) + 2:
            out[lo:hi] = np.stack([filtfilt(b, a, track[lo:hi, c]) for c in range(track.shape[1])], 1)
    return out


def ramp_weights(ok, ramp):
    """1 inside usable segments, ramping linearly at segment edges that are not clip ends."""
    T, w = len(ok), np.zeros(len(ok))
    for a, b in segments(ok):
        for t in range(a, b):
            up = 1.0 if a == 0 else (t - a + 1) / ramp
            dn = 1.0 if b == T else (b - t) / ramp
            w[t] = min(1.0, up, dn)
    return w


# ----------------------------------------------------------------------------- IK
def rotvec(R_cur, R_ref):
    """Axis-angle of R_cur R_ref^T."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R_cur @ R_ref.T).as_rotvec()


def ik_wrist(fk: Callable, q, side, sl, target_mm, R_ref, q_ref, iters=30, w_rot=2000.0,
             damping=0.5, step_cap=0.08, fd_eps=1e-3):
    """Damped least squares on the joints ``sl``: wrist position (mm) to ``target_mm``,
    orientation pinned to ``R_ref`` (``w_rot`` mm per rad), damping toward ``q_ref``.
    Returns (q, position residual in mm before the last step)."""
    s = np.array(q, dtype=np.float64)
    n = sl.stop - sl.start

    def res(x):
        h = fk(x)[side]
        return np.concatenate([np.asarray(h["wrist_pos"]) * 1000.0 - target_mm,
                               w_rot * rotvec(np.asarray(h["wrist_rot"]), R_ref)])

    lam, last = damping * 1e4, np.inf
    for _ in range(iters):
        r = res(s)
        if not np.all(np.isfinite(r)):
            return np.array(q, dtype=np.float64), np.inf
        last = float(np.linalg.norm(r[:3]))
        J = np.zeros((6, n))
        for j in range(n):
            s2 = s.copy()
            s2[sl.start + j] += fd_eps
            J[:, j] = (res(s2) - r) / fd_eps
        if not np.all(np.isfinite(J)):
            return np.array(q, dtype=np.float64), np.inf
        dq = np.linalg.solve(J.T @ J + (1.0 + lam) * np.eye(n), -J.T @ r - lam * (s[sl] - q_ref[sl]))
        norm = np.linalg.norm(dq)
        if norm > step_cap:
            dq *= step_cap / norm
        s[sl] += dq
        if norm < 1e-4:
            break
    return s, last


# ----------------------------------------------------------------------------- main entry
def camera_to_base(points_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """(T, 2, 3) camera-frame points -> base frame; T_base_cam is (4, 4) or (T, 4, 4)."""
    Tm = np.broadcast_to(np.asarray(T_base_cam, np.float64), (len(points_cam), 4, 4))
    return np.einsum("tij,thj->thi", Tm[:, :3, :3], points_cam) + Tm[:, None, :3, 3]


def correct_wrists(actions: np.ndarray, kin, wrist_base: np.ndarray, confidence: np.ndarray,
                   cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    """Correct the arm joints of ``actions`` (T, D) so the wrists follow ``wrist_base``.

    wrist_base: (T, 2, 3) reconstructed human wrists in the robot base frame (metres, NaN when
    missing); confidence: (T, 2). Returns the corrected actions and per-side statistics.
    """
    c = {**DEFAULTS, **(cfg or {})}
    act = np.asarray(actions, np.float64)
    T = len(act)
    out = act.copy()
    fk_idm = [kin.fk(act[t]) for t in range(T)]
    info = {}
    for hi, side in enumerate(SIDES):
        sl = kin.arm_joint_slices[side]
        sl = sl if isinstance(sl, slice) else slice(*sl)
        p_idm = np.stack([np.asarray(f[side]["wrist_pos"]) for f in fk_idm]) * 1000.0
        R_idm = [np.asarray(f[side]["wrist_rot"]) for f in fk_idm]
        # end effector of the scaled IDM hand, relative to its wrist
        ee = np.stack([np.asarray(f[side]["keypoints"])[c["ee_keypoint"]] for f in fk_idm]) * 1000.0
        target = wrist_base[:, hi] * 1000.0 - c["hand_scale"] * (ee - p_idm)
        d = target - p_idm
        ok = (confidence[:, hi] >= c["min_confidence"]) & np.isfinite(d).all(1)
        n_conf = int(ok.sum())
        d = np.where(ok[:, None], d, 0.0)
        ok = reject_excursions(d, ok, c["reject_enter_mm"], c["reject_exit_mm"],
                               c["reject_hard_mm"], c["reject_ref_n"])
        n_rej = n_conf - int(ok.sum())
        ok = fill_short_gaps(d, ok, c["gap_fill"])
        d = lowpass(d, ok, c["cutoff_hz"], c["fps"])
        q_ik, prev, resid = act[:, sl].copy(), None, []
        for t in range(T):
            if not ok[t]:
                prev = None
                continue
            cur = act[t].copy()
            if prev is not None:
                cur[sl] = prev
            new, r = ik_wrist(kin.fk, cur, side, sl, p_idm[t] + d[t], R_idm[t], act[t],
                              c["ik_iters"], c["w_rot"], c["damping"], c["step_cap"], c["fd_eps"])
            if not np.isfinite(r):
                ok[t], prev = False, None
                continue
            q_ik[t] = prev = new[sl]
            resid.append(r)
        w = ramp_weights(ok, c["ramp"])
        blended = act[:, sl] + w[:, None] * (q_ik - act[:, sl])
        if T > c["smooth_window"]:
            blended = savgol_filter(blended, c["smooth_window"], c["smooth_order"], axis=0, mode="nearest")
        out[:, sl] = blended
        info[side] = dict(confident_frames=n_conf, rejected=n_rej, corrected_frames=int((w > 0).sum()),
                          ik_residual_mm_median=float(np.median(resid)) if resid else None)
    return out.astype(np.asarray(actions).dtype), info

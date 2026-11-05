"""MuJoCo implementation of :class:`Embodiment`, generic over the YAML description.

The robot model (MJCF, or URDF which MuJoCo also parses) is supplied by the user; nothing
robot-specific is coded here. Forward kinematics is ``mj_kinematics`` on a pose built from
the recorded joint vector: unlisted joints stay at ``qpos0``, coupled passive joints are
filled from their sources and ``fixed_joints`` are applied last. A head camera mounted on a
model frame therefore follows the recorded head joints.

Pin ``mujoco==3.11.0``: with 3.12 site-based camera poses silently collapse (every point
lands at camera depth ~0), so a different version raises unless
``ROBOTRYON_ALLOW_ANY_MUJOCO=1``.

Rendering (``silhouette``) needs an OpenGL backend; set ``MUJOCO_GL=egl`` (GPU) or
``MUJOCO_GL=osmesa`` (CPU) before the first import of mujoco.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from . import retarget as RT
from .base import Embodiment, empty_hands
from .spec import HANDS, EmbodimentSpec, load_spec

MUJOCO_PIN = "3.11.0"
_SIL_HIDDEN_GROUP = 5


def _import_mujoco():
    import mujoco
    if mujoco.__version__ != MUJOCO_PIN and os.environ.get("ROBOTRYON_ALLOW_ANY_MUJOCO") != "1":
        raise RuntimeError(
            f"mujoco {mujoco.__version__} != {MUJOCO_PIN}: newer versions break site-based "
            f"camera poses without an error. Install mujoco=={MUJOCO_PIN} or set "
            f"ROBOTRYON_ALLOW_ANY_MUJOCO=1.")
    return mujoco


class MujocoEmbodiment(Embodiment):
    def __init__(self, spec: EmbodimentSpec | str):
        self.spec = load_spec(spec) if isinstance(spec, (str, os.PathLike)) else spec
        mj = self._mj = _import_mujoco()
        if not os.path.exists(self.spec.model_path):
            raise FileNotFoundError(f"robot model not found: {self.spec.model_path}")
        self.model = mj.MjModel.from_xml_path(self.spec.model_path)
        self.data = mj.MjData(self.model)

        names = set(n for n in self.spec.joints if n) | set(self.spec.fixed_joints)
        names |= {c[0] for c in self.spec.coupling} | {c[2] for c in self.spec.coupling}
        names |= {j for js in self.spec.retarget.joints.values() for j in js}
        self._adr = {n: self._joint_adr(n) for n in names}
        self._rt_bounds = {h: self._joint_limits(js) for h, js in self.spec.retarget.joints.items()}
        self._rt_palm: dict[str, float] = {}
        self._kp = {h: [(*self._frame(n), off) for n, off in
                        zip(self.spec.keypoints[h], self.spec.keypoint_offsets[h])]
                    for h in self.spec.hands}
        # the wrist pose: the explicit wrist_frame, else keypoint 0 (with its offset)
        self._wrist = {h: (*self._frame(self.spec.wrist_frame[h]),
                           self.spec.keypoint_offsets[h][0]
                           if self.spec.wrist_frame[h] == self.spec.keypoints[h][0] else None)
                       for h in self.spec.hands}
        self._mount = {v: (self._frame(c.mount.frame) if c.mount.frame else None)
                       for v, c in self.spec.cameras.items()}
        self._renderers: dict = {}

    # --- name resolution -----------------------------------------------------------------
    def _joint_adr(self, name: str) -> int:
        jid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} not in {self.spec.model_path}")
        return int(self.model.jnt_qposadr[jid])

    def _joint_limits(self, names) -> tuple[np.ndarray, np.ndarray]:
        """``(lo, hi)`` of each joint: the model's limits ([-pi, pi] when unlimited), narrowed
        by ``retarget.limits``."""
        lo, hi = [], []
        for n in names:
            jid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, n)
            r = self.model.jnt_range[jid]
            ok = bool(self.model.jnt_limited[jid]) and r[1] > r[0]
            a, b = (float(r[0]), float(r[1])) if ok else (-np.pi, np.pi)
            if n in self.spec.retarget.limits:
                ua, ub = self.spec.retarget.limits[n]
                a, b = max(a, ua), min(b, ub)
                if b <= a:
                    raise ValueError(f"retarget.limits for {n} [{ua}, {ub}] leaves no range "
                                     f"inside the model limits")
            lo.append(a)
            hi.append(b)
        return np.asarray(lo), np.asarray(hi)

    def _frame(self, name: str) -> tuple[str, int]:
        """``site:x`` / ``body:x`` / ``x`` (site first, then body) -> (kind, id)."""
        kind, _, bare = name.rpartition(":")
        T = self._mj.mjtObj
        for k in ([kind] if kind else ["site", "body"]):
            oid = self._mj.mj_name2id(self.model, T.mjOBJ_SITE if k == "site" else T.mjOBJ_BODY,
                                      bare)
            if oid >= 0:
                return k, oid
        raise ValueError(f"frame {name!r} not in {self.spec.model_path}")

    def _pos(self, f):
        p = (self.data.site_xpos if f[0] == "site" else self.data.xpos)[f[1]].copy()
        if len(f) > 2 and f[2] is not None:                  # point fixed in the frame
            p = p + self._rot(f) @ f[2]
        return p

    def _rot(self, f):
        m = self.data.site_xmat if f[0] == "site" else self.data.xmat
        return m[f[1]].reshape(3, 3).copy()

    # --- kinematics ----------------------------------------------------------------------
    def joint_dict(self, q) -> dict[str, float]:
        """Joint vector (``spec.joints`` order) or dict -> complete ``{joint: value}``."""
        if q is None:
            d = {}
        elif isinstance(q, dict):
            d = {k: float(v) for k, v in q.items()}
        else:
            q = np.asarray(q, np.float64).reshape(-1)
            if len(q) != len(self.spec.joints):
                raise ValueError(f"expected {len(self.spec.joints)} joints, got {len(q)}")
            d = {n: float(v) * self.spec.joint_scale.get(n, 1.0) for n, v in zip(self.spec.joints, q) if n}
        for joint, ratio, src in self.spec.coupling:
            if src in d:
                d[joint] = ratio * d[src]
        d.update(self.spec.fixed_joints)
        return d

    def set_state(self, q) -> None:
        self.data.qpos[:] = self.model.qpos0
        for n, v in self.joint_dict(q).items():
            adr = self._adr.get(n)
            if adr is None:
                adr = self._adr[n] = self._joint_adr(n)
            self.data.qpos[adr] = v
        self._mj.mj_kinematics(self.model, self.data)

    def _hand_points(self) -> np.ndarray:
        """Keypoints of the current state, base frame, with the per-finger length scale.

        The scale pulls each finger's distal points toward its knuckle
        (``p <- p_knuckle + s * (p - p_knuckle)``), a fitted correction for model fingers that
        render longer than the real hand.
        """
        out = empty_hands()
        for hi, h in enumerate(HANDS):
            if h not in self._kp:
                continue
            p = np.stack([self._pos(f) for f in self._kp[h]])
            for finger, (k, distal) in RT.CHAINS17.items():
                s = self.spec.finger_scale.get(finger)
                if s is not None:
                    p[list(distal)] = p[k] + s * (p[list(distal)] - p[k])
            out[hi] = p
        return out

    def fk(self, q) -> dict[str, dict[str, np.ndarray]]:
        self.set_state(q)
        kp = self._hand_points()
        return {h: {"wrist_pos": self._pos(self._wrist[h]), "wrist_rot": self._rot(self._wrist[h]),
                    "keypoints": kp[HANDS.index(h)]}
                for h in HANDS if h in self._wrist}

    def hand_keypoints(self, q) -> np.ndarray:
        self.set_state(q)
        return self._hand_points()

    def _camera_pose_now(self, view: str) -> np.ndarray:
        m = self.spec.cameras[view].mount
        if m.extrinsic is not None:
            return m.extrinsic.copy()
        f = self._mount[view]
        R_frame = self._rot(f)
        T = np.eye(4)
        T[:3, :3] = R_frame @ m.rotation.T
        T[:3, 3] = self._pos(f) + R_frame @ m.translation
        return T

    def camera_pose(self, q, view: str) -> np.ndarray:
        self.set_state(q)
        return self._camera_pose_now(view)

    def keypoints_in_view(self, q, view: str, image_size):
        """One FK pass -> projected skeleton ``kp (2,17,2)``, ``vis (2,17)`` of a robot frame."""
        self.set_state(q)
        T = np.linalg.inv(self._camera_pose_now(view))
        pts = self._hand_points() @ T[:3, :3].T + T[:3, 3]
        return self.keypoints_2d(pts, view, image_size, frame="camera")

    # --- rendering -----------------------------------------------------------------------
    def _renderer(self, view: str, image_size, scale: float):
        key = (view, tuple(image_size), scale)
        if key in self._renderers:
            return self._renderers[key]
        mj, model = self._mj, self.model
        inc, exc = tuple(self.spec.silhouette_bodies), tuple(self.spec.silhouette_exclude)
        for g in range(model.ngeom):
            body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
            show = body.startswith(inc) and not (exc and body.startswith(exc))
            model.geom_group[g] = 0 if show else _SIL_HIDDEN_GROUP
        # Render a symmetric wide pinhole, then remap each output pixel through the real
        # camera's (distorted, off-centre) ray so the silhouette matches the projection.
        W, H = image_size
        w, h = max(8, round(W * scale)), max(8, round(H * scale))
        cam = self.camera(view)
        u, v = np.meshgrid((np.arange(w) + 0.5) * W / w - 0.5, (np.arange(h) + 0.5) * H / h - 0.5)
        n = cam.normalized(np.stack([u, v], -1), image_size)
        f = cam.K(image_size)[1, 1] * h / H
        ext = np.abs(n.reshape(-1, 2)).max(0) * 1.03
        Wp, Hp = int(2 * np.ceil(ext[0] * f)) + 2, int(2 * np.ceil(ext[1] * f)) + 2
        mx = (n[..., 0] * f + Wp / 2.0 - 0.5).astype(np.float32)
        my = (n[..., 1] * f + Hp / 2.0 - 0.5).astype(np.float32)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), Wp)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), Hp)
        model.vis.map.znear = 0.001
        model.vis.quality.offsamples = 0          # MSAA would blend segmentation ids
        r = mj.Renderer(model, height=Hp, width=Wp)
        r.enable_segmentation_rendering()
        opt = mj.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[0] = 1
        opt.sitegroup[:] = 0
        mcam = mj.MjvCamera()
        mj.mjv_defaultCamera(mcam)
        self._renderers[key] = (r, opt, mcam, mx, my, f, Hp)
        return self._renderers[key]

    def silhouette(self, q, view: str, image_size=None, scale: float = 0.5) -> np.ndarray:
        cam = self.camera(view)
        image_size = tuple(image_size or (cam.width, cam.height))
        r, opt, mcam, mx, my, f, Hp = self._renderer(view, image_size, scale)
        self.set_state(q)
        T = self._camera_pose_now(view)
        r.update_scene(self.data, camera=mcam, scene_option=opt)
        near, far = 0.02, 6.0
        half_h = near * (Hp / 2.0) / f
        for gc in r.scene.camera:
            gc.pos[:] = T[:3, 3]
            gc.forward[:] = T[:3, 2]
            gc.up[:] = -T[:3, 1]
            gc.frustum_center = 0.0
            gc.frustum_bottom, gc.frustum_top = -half_h, half_h
            gc.frustum_near, gc.frustum_far = near, far
            gc.orthographic = 0
        seg = r.render()[:, :, 0]
        sil = cv2.remap((seg >= 0).astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return cv2.resize(sil, image_size, interpolation=cv2.INTER_NEAREST) > 0

    # --- retargeting ---------------------------------------------------------------------
    def _hand_vectors(self, hand: str, x: np.ndarray):
        """Retargeting vectors (wrist frame) of one robot hand at hand-joint values ``x``."""
        self.set_state(dict(zip(self.spec.retarget.joints[hand], x)))
        return RT.local_vectors(self._hand_points()[HANDS.index(hand)])

    def _palm(self, hand: str) -> float:
        if hand not in self._rt_palm:
            lo, hi = self._rt_bounds[hand]
            self.set_state(dict(zip(self.spec.retarget.joints[hand], np.clip(0.0, lo, hi))))
            self._rt_palm[hand] = RT.palm_size(self._hand_points()[HANDS.index(hand)])
        return self._rt_palm[hand]

    def _solve_hand(self, hand: str, h17: np.ndarray, x_prev=None):
        """AnyTeleop vector retargeting of one hand: hand joints (model order) or None."""
        from scipy.optimize import least_squares

        rt = self.spec.retarget
        target = RT.local_vectors(h17)
        if target is None:
            return None
        scale = rt.scaling if rt.scaling is not None else self._palm(hand) / max(RT.palm_size(h17), 1e-6)
        target = target * scale
        lo, hi = self._rt_bounds[hand]
        x0 = np.clip(np.zeros(len(lo)) if x_prev is None else np.asarray(x_prev, np.float64), lo, hi)
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        w = np.sqrt(rt.smoothing)

        def residual(x):
            v = self._hand_vectors(hand, x)
            r = np.zeros(target.size) if v is None else (v - target).ravel()
            return r if x_prev is None else np.concatenate([r, w * (x - x0)])

        sol = least_squares(residual, x0, bounds=(lo, hi), loss="huber", f_scale=rt.huber_delta,
                            diff_step=1e-4, max_nfev=200)
        return sol.x

    def retarget_hand(self, human_kp3d, valid=None, prev=None):
        j21 = RT.to21(human_kp3d)
        if valid is None:
            valid = np.isfinite(j21).all(axis=(-1, -2))
        rt = self.spec.retarget
        q = {}
        for hi, h in enumerate(HANDS):
            joints = rt.joints.get(h)
            if not valid[hi] or not joints:
                continue
            x_prev = None
            if prev and all(n in prev for n in joints):
                x_prev = [prev[n] for n in joints]
            x = self._solve_hand(h, RT.human17(j21[hi]), x_prev)
            if x is not None:
                q.update({n: float(v) for n, v in zip(joints, x)})
        out = empty_hands()
        if not q:
            return out, q
        robot = self.hand_keypoints(q)
        for hi, h in enumerate(HANDS):
            if valid[hi] and all(n in q for n in rt.joints.get(h) or [None]):
                placed = RT.place_on_human(robot[hi], j21[hi], rt.bone_match)
                if placed is not None:
                    out[hi] = placed
        return out, q


__all__ = ["MujocoEmbodiment", "MUJOCO_PIN"]

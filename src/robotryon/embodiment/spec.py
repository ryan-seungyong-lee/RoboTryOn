"""Embodiment description parsed from YAML (see ``configs/embodiments/gr1.yaml``).

Everything that differs between robots lives here as data: the kinematic model file, the
layout of the recorded joint vector, passive joints coupled to actuated ones, the link that
stands for each of the 17 hand keypoints, the cameras, the links that make up the robot's
silhouette, and which hand joints the human-hand retargeting moves. Parsing is pure Python
so configs can be validated without a physics engine.

Strings may contain ``{s}``, which is replaced by each hand's side token (``hands.sides``), so
a mirror-symmetric robot is described once for both hands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robotryon.skeleton.topology import NUM_KEYPOINTS

from .camera import CameraModel

HANDS = ("left", "right")


@dataclass
class CameraMount:
    """Where a camera sits: on a model frame that moves with the robot, or fixed in the base.

    ``frame`` is ``site:<name>``/``body:<name>`` (or a bare name, resolved site first);
    ``rotation`` maps the frame's axes to the optical frame (``p_opt = R @ p_frame``);
    ``translation`` offsets the lens origin in the mount frame (e.g. a stereo baseline).
    ``extrinsic`` instead gives a fixed 4x4 camera-to-base pose.
    """
    frame: str | None = None
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    extrinsic: np.ndarray | None = None


@dataclass
class CameraSpec:
    intrinsics: CameraModel
    mount: CameraMount
    video_key: str | None = None


@dataclass
class RetargetSpec:
    """Vector retargeting of a human hand onto the robot hand (``robotryon.embodiment.retarget``).

    ``joints``: per hand, the actuated hand joints the optimization moves; coupled joints follow
    their source through ``coupling`` and the limits come from the robot model.
    ``scaling``: factor on the human vectors, or None for the ratio of the palm sizes.
    ``smoothing``: weight of the squared change from the previous frame's joints.
    ``huber_delta``: Huber threshold (m) of the vector residuals.
    ``limits``: ``{joint: (lo, hi)}`` narrowing the model's joint limits for the retargeting, e.g.
    to keep finger joints out of hyperextension the model allows but the real hand never uses.
    """
    joints: dict[str, list[str]] = field(default_factory=dict)
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    scaling: float | None = None
    smoothing: float = 4e-3
    huber_delta: float = 0.02
    bone_match: bool = True


@dataclass
class EmbodimentSpec:
    name: str
    model_path: str
    joints: list[str | None]
    joint_scale: dict[str, float]                     # joint -> model value / recorded value
    arm_joints: dict[str, tuple[int, int]]            # hand -> [start, stop) in the joint vector
    coupling: list[tuple[str, float, str]]            # (joint, ratio, source joint)
    fixed_joints: dict[str, float]
    keypoints: dict[str, list[str]]                   # hand -> 17 frame names
    keypoint_offsets: dict[str, list]                 # hand -> 17 offsets (m, in the frame) or None
    wrist_frame: dict[str, str]                       # hand -> frame giving the wrist pose
    finger_scale: dict[str, float]
    cameras: dict[str, CameraSpec]
    silhouette_bodies: list[str]
    silhouette_exclude: list[str]
    retarget: RetargetSpec = field(default_factory=RetargetSpec)
    raw: dict = field(default_factory=dict)

    @property
    def hands(self) -> list[str]:
        return [h for h in HANDS if h in self.keypoints]

    @property
    def views(self) -> list[str]:
        return list(self.cameras)


def _fmt(x: Any, s: str) -> Any:
    return x.replace("{s}", s) if isinstance(x, str) else x


def _per_hand(value, hand: str):
    """``value`` or ``value[hand]`` when it is given per hand."""
    if isinstance(value, dict) and hand in value:
        return value[hand]
    return value


def parse_spec(cfg: dict, base_dir: str | Path | None = None) -> EmbodimentSpec:
    hands_cfg = cfg.get("hands", {})
    sides = hands_cfg.get("sides", {"left": "left", "right": "right"})

    keypoints, keypoint_offsets, wrist_frame = {}, {}, {}
    for hand, tok in sides.items():
        if hand not in HANDS:
            raise ValueError(f"hands.sides keys must be left/right, got {hand!r}")
        entries = (hands_cfg.get(hand) or {}).get("keypoints") or hands_cfg.get("keypoints")
        if entries is None:
            raise ValueError(f"no keypoints given for the {hand} hand")
        if len(entries) != NUM_KEYPOINTS:
            raise ValueError(f"{hand} hand needs {NUM_KEYPOINTS} keypoints, got {len(entries)}")
        # an entry is a frame name, or {frame: name, offset: [x, y, z]} for a point fixed in it
        names = [_fmt(e["frame"] if isinstance(e, dict) else e, tok) for e in entries]
        keypoint_offsets[hand] = [np.asarray(_per_hand(e["offset"], hand), np.float64).reshape(3)
                                  if isinstance(e, dict) and e.get("offset") is not None else None
                                  for e in entries]
        keypoints[hand] = names
        wf = (hands_cfg.get(hand) or {}).get("wrist_frame") or hands_cfg.get("wrist_frame")
        wrist_frame[hand] = _fmt(wf, tok) if wf else names[0]

    coupling = []
    for c in cfg.get("coupling", []) or []:
        templated = "{s}" in c["joint"] or "{s}" in c["source"]
        for tok in (sides.values() if templated else [""]):
            coupling.append((_fmt(c["joint"], tok), float(c["ratio"]), _fmt(c["source"], tok)))

    cameras = {}
    for view, c in (cfg.get("cameras") or {}).items():
        m = c.get("mount", {}) or {}
        ext = c.get("extrinsic")
        mount = CameraMount(
            frame=m.get("frame"),
            rotation=np.asarray(m.get("rotation", np.eye(3)), np.float64).reshape(3, 3),
            translation=np.asarray(m.get("translation", [0, 0, 0]), np.float64).reshape(3),
            extrinsic=None if ext is None else np.asarray(ext, np.float64).reshape(4, 4))
        if mount.frame is None and mount.extrinsic is None:
            raise ValueError(f"camera {view!r} needs mount.frame or extrinsic")
        cameras[view] = CameraSpec(CameraModel.from_dict(c["intrinsics"]), mount,
                                   c.get("video_key"))

    rt = cfg.get("retarget", {}) or {}
    if "drivers" in rt:
        raise ValueError("retarget.drivers (percentile retargeting) was replaced by vector "
                         "retargeting: list the hand joints under retarget.joints instead")
    rt_joints = {hand: [_fmt(j, tok) for j in (_per_hand(rt.get("joints") or [], hand) or [])]
                 for hand, tok in sides.items()}
    limits = {}
    for j, r in (rt.get("limits") or {}).items():
        for hand, tok in sides.items():
            rr = _per_hand(r, hand)
            limits[_fmt(j, tok)] = (float(rr[0]), float(rr[1]))
    scaling = rt.get("scaling", "auto")
    retarget = RetargetSpec(
        joints=rt_joints,
        limits=limits,
        scaling=None if scaling in (None, "auto") else float(scaling),
        smoothing=float(rt.get("smoothing", 4e-3)),
        huber_delta=float(rt.get("huber_delta", 0.02)),
        bone_match=bool(rt.get("bone_match", True)))

    model_path = str(cfg["model_path"])
    if base_dir is not None and not Path(model_path).is_absolute():
        model_path = str(Path(base_dir) / model_path)
    sil = cfg.get("silhouette", {}) or {}
    toks = list(sides.values())
    expand = lambda xs: sorted({_fmt(x, t) for x in xs or [] for t in toks})  # noqa: E731
    return EmbodimentSpec(
        name=str(cfg.get("name", "robot")),
        model_path=model_path,
        joints=[j["name"] if isinstance(j, dict) else j for j in cfg.get("joints", [])],
        joint_scale={j["name"]: float(j["scale"]) for j in cfg.get("joints", [])
                     if isinstance(j, dict) and "scale" in j},
        arm_joints={h: (int(r[0]), int(r[1])) for h, r in (cfg.get("arm_joints") or {}).items()},
        coupling=coupling,
        fixed_joints={k: float(v) for k, v in (cfg.get("fixed_joints") or {}).items()},
        keypoints=keypoints,
        keypoint_offsets=keypoint_offsets,
        wrist_frame=wrist_frame,
        finger_scale={k: float(v) for k, v in (hands_cfg.get("finger_scale") or {}).items()},
        cameras=cameras,
        silhouette_bodies=expand(sil.get("bodies")),
        silhouette_exclude=expand(sil.get("exclude")),
        retarget=retarget,
        raw=cfg,
    )


def load_spec(path: str | Path) -> EmbodimentSpec:
    path = Path(path)
    return parse_spec(yaml.safe_load(path.read_text()), base_dir=path.parent)

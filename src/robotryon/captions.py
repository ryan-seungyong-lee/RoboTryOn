"""Text prompts: per-hand state words and the embodiment phrase.

Image model. A keyframe's prompt states two facts per hand that a 2D skeleton conveys weakly at
the sizes hands occupy in the frame -- how closed the hand is and which side faces the camera::

    Match the hand pose to the drawn skeleton.
    Left hand: open, palm toward camera. Right hand: closed fist, back of hand toward camera.

``closure`` is the MCP + PIP flexion averaged over the four fingers (radians);
``dorsumness`` is 180 minus the angle between the knuckle-plane normal and the ray from the hand
to the camera (degrees, high = back of the hand toward the lens). Both are read from the
3D hand in the camera frame -- the reconstructed human hand, or the robot hand from forward
kinematics, with the same formula -- and cut into words (``HandStateCuts``). A hand that is not
tracked is ``not visible``.

Each episode stores both values per view and frame in ``hand_state_<view>.npz`` (``closure``,
``dorsumness``, ``(T, 2)`` float32, NaN where a hand is not tracked), written by the preprocessing
for human clips (reconstructed hands) and robot episodes (forward kinematics), so training and
inference read the same quantity.

Video model. The prompt is the task caption (``text.template``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INSTRUCTION = "Match the hand pose to the drawn skeleton."
CLOSURE = ("open", "half-closed", "closed fist")
FACING = ("palm toward camera", "edge-on", "back of hand toward camera")
NOT_VISIBLE = "not visible"

# MediaPipe / MANO 21-joint layout: (MCP, PIP, DIP, tip) per finger.
_CHAINS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
           "little": (17, 18, 19, 20)}


def _angle(a, b, c) -> float:
    """Flexion at b: 0 when a-b-c is straight."""
    u, v = a - b, c - b
    u = u / max(np.linalg.norm(u), 1e-9)
    v = v / max(np.linalg.norm(v), 1e-9)
    return float(np.pi - np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))


def hand_fields(j21: np.ndarray, hand: str) -> tuple[float, float]:
    """``(21, 3)`` joints of one hand in the camera frame (camera at the origin, looking down +z)
    -> ``(closure, dorsumness)``; NaN when the hand is not finite."""
    j = np.asarray(j21, np.float64)
    if not np.isfinite(j).all():
        return float("nan"), float("nan")
    w = j[0]
    closure = np.mean([_angle(w, j[a], j[b]) + _angle(j[a], j[b], j[c])
                       for a, b, c, _ in _CHAINS.values()])
    n = np.cross(j[5] - w, j[17] - w)
    ray = -j.mean(0)                                   # hand -> camera
    nn, nr = np.linalg.norm(n), np.linalg.norm(ray)
    if nn < 1e-9 or nr < 1e-9:
        return float(closure), float("nan")
    d = 180.0 - float(np.degrees(np.arccos(np.clip(np.dot(n, ray) / (nn * nr), -1.0, 1.0))))
    # the normal leaves through the palm on the right hand and the back on the left
    return float(closure), (d if hand == "left" else 180.0 - d)


@dataclass
class HandStateCuts:
    """Edges that turn ``closure`` / ``dorsumness`` into words (lower, upper)."""
    closure: tuple[float, float] = (0.95, 1.15)
    facing: tuple[float, float] = (49.0, 121.0)

    @classmethod
    def from_config(cls, cfg) -> "HandStateCuts":
        cfg = cfg or {}
        return cls(tuple(float(x) for x in cfg.get("closure", cls.closure)),
                   tuple(float(x) for x in cfg.get("facing", cls.facing)))

    def words(self, closure: float, dorsumness: float) -> str:
        if not (np.isfinite(closure) and np.isfinite(dorsumness)):
            return NOT_VISIBLE
        ci = int(closure >= self.closure[0]) + int(closure >= self.closure[1])
        fi = int(dorsumness >= self.facing[0]) + int(dorsumness >= self.facing[1])
        return f"{CLOSURE[ci]}, {FACING[fi]}"


def hand_state_prompt(left: str, right: str) -> str:
    """The image prompt from the two hands' words."""
    return f"{INSTRUCTION} Left hand: {left}. Right hand: {right}."


def all_prompts() -> list[str]:
    """Every prompt training can produce: each pair of hand descriptions, the hand-dropout prompt
    (the instruction alone) and the empty (unconditional) prompt."""
    cells = [f"{c}, {f}" for c in CLOSURE for f in FACING] + [NOT_VISIBLE]
    return [hand_state_prompt(a, b) for a in cells for b in cells] + [INSTRUCTION, ""]


def sample_prompt(words: tuple[str, str], rng: np.random.Generator, p_uncond: float,
                  p_drop: float) -> str:
    """Training prompt: empty with ``p_uncond``, hand words dropped with ``p_drop``."""
    u = float(rng.random())
    if u < p_uncond:
        return ""
    if u < p_uncond + p_drop:
        return INSTRUCTION
    return hand_state_prompt(*words)


def hand_state_arrays(joints3d: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(T, 2, 21|17, 3)`` camera-frame joints and ``(T, 2)`` validity -> ``closure`` and
    ``dorsumness`` ``(T, 2)`` float32 (NaN where invalid)."""
    from .embodiment.retarget import to21
    j = to21(np.asarray(joints3d, np.float64))
    T = j.shape[0]
    clo = np.full((T, 2), np.nan, np.float32)
    dor = np.full((T, 2), np.nan, np.float32)
    for t in range(T):
        for h, name in enumerate(("left", "right")):
            if valid[t, h]:
                clo[t, h], dor[t, h] = hand_fields(j[t, h], name)
    return clo, dor


def save_hand_state(path, closure: np.ndarray, dorsumness: np.ndarray) -> None:
    np.savez_compressed(path, closure=closure.astype(np.float32), dorsumness=dorsumness.astype(np.float32))


def load_hand_state(path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["closure"], z["dorsumness"]


def words_at(states: list[tuple[np.ndarray, np.ndarray]], t: int,
             cuts: HandStateCuts) -> tuple[str, str]:
    """Words of both hands at frame ``t`` from one or more views' hand states (averaged)."""
    out = []
    for h in range(2):
        vals = [(c[t, h], d[t, h]) for c, d in states
                if t < len(c) and np.isfinite(c[t, h]) and np.isfinite(d[t, h])]
        if not vals:
            out.append(NOT_VISIBLE)
            continue
        cl, dn = np.mean(np.asarray(vals, np.float64), axis=0)
        out.append(cuts.words(float(cl), float(dn)))
    return out[0], out[1]

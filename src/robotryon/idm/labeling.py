"""Action labels for a generated robot video (paper Sec. 3.4, App. B.2.6).

1. Sliding window, stride 1: at every frame t the IDM reads frames (t, t+K) (clamped to the
   clip) and predicts the chunk a_t .. a_{t+H-1}.
2. Chunk averaging: frame u appears in the H chunks that start at u-H+1 .. u; its label is the
   mean of those predictions (fewer at the clip ends).
3. Savitzky-Golay smoothing (polynomial order 2) per joint group: a long window on the arm-like
   groups, a short one on the hand groups so that grasp onsets stay sharp.
4. Optional wrist correction (``wrist_correction.py``).
5. Proprioceptive state from the actions: q_t = a_{t-d} (the robot reaches a commanded pose
   d frames later); the first d states repeat a_0.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import savgol_filter

from .data import collate, model_inputs, preprocess_frames, unnormalize


class ChunkAccumulator:
    """Average every chunk prediction that covers a frame."""

    def __init__(self, length: int, action_dim: int, horizon: int):
        self.L, self.H = int(length), int(horizon)
        self._sum = np.zeros((self.L, action_dim), dtype=np.float64)
        self._cnt = np.zeros((self.L, 1), dtype=np.float64)

    def add(self, start: int, chunk: np.ndarray) -> None:
        n = min(self.H, len(chunk), self.L - start)
        if n > 0:
            self._sum[start:start + n] += chunk[:n]
            self._cnt[start:start + n] += 1.0

    def result(self) -> np.ndarray:
        if (self._cnt == 0).any():
            raise RuntimeError("some frames are covered by no chunk")
        return (self._sum / self._cnt).astype(np.float32)


def smooth_actions(actions: np.ndarray, groups: dict[str, slice], hand_groups, window: int,
                   hand_window: int, order: int = 2) -> np.ndarray:
    """SG filter per group along time; hand groups use ``hand_window``. A window longer than the
    clip (or not above the order) leaves the group unchanged."""
    out = np.array(actions, dtype=np.float32, copy=True)
    for name, sl in groups.items():
        w = int(hand_window if name in hand_groups else window) | 1
        if order < w <= len(out):
            out[:, sl] = savgol_filter(out[:, sl], w, order, axis=0, mode="nearest")
    return out


def state_from_action(actions: np.ndarray, delay: int) -> np.ndarray:
    """q_t = a_{t-delay}; q_t = a_0 for t < delay."""
    a = np.asarray(actions)
    d = min(int(delay), len(a))
    return np.concatenate([np.repeat(a[:1], d, axis=0), a[: len(a) - d]], axis=0)


def group_slices(spec: dict, widths: dict[str, int]) -> dict[str, slice]:
    out, s = {}, 0
    for g in spec["action_groups"]:
        out[g] = slice(s, s + widths[g])
        s += widths[g]
    return out


@torch.no_grad()
def predict_actions(model, views: list[np.ndarray], spec: dict, stats: dict, frame_gap: int,
                    batch_size: int, generator: torch.Generator | None = None) -> np.ndarray:
    """Chunk-averaged raw actions (T, native_dim) for one clip.

    ``views``: one (T, H, W, 3) uint8 RGB array per view in ``spec["views"]`` order, already at
    the embodiment's ``video_size``. ``stats``: per-group action statistics of this embodiment.
    """
    H, A = model.cfg["action_horizon"], model.cfg["action_dim"]
    groups = list(spec["action_groups"])
    q01 = torch.tensor(np.concatenate([stats[g]["q01"] for g in groups]).tolist())
    q99 = torch.tensor(np.concatenate([stats[g]["q99"] for g in groups]).tolist())
    n_native = len(q01)
    T = min(len(v) for v in views)
    device = next(model.parameters()).device
    acc = ChunkAccumulator(T, n_native, H)

    def sample(t):
        obs = np.clip(np.array([t, t + frame_gap]), 0, T - 1)
        frames = np.stack([v[obs] for v in views], 1).reshape(-1, *views[0].shape[1:])
        return model_inputs(preprocess_frames(frames), spec, H, A)

    for s in range(0, T, batch_size):
        batch = collate([sample(t) for t in range(s, min(s + batch_size, T))])
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model.get_action(batch, generator=generator).cpu()[..., :n_native]
        raw = unnormalize(pred, q01, q99, spec.get("norm", "q99")).numpy()
        for b, chunk in enumerate(raw):
            acc.add(s + b, chunk)
    return acc.result()

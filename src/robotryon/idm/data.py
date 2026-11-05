# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from GR00T-Dreams (Apache-2.0): gr00t/data/dataset.py (LeRobot v2.1 reader),
# gr00t/data/transform/{video,state_action,mirror}.py, gr00t/model/transforms_idm.py and the
# multi-embodiment mixture. Rewritten as one compact pipeline; the eval-mode image path and
# the q99 normalisation are numerically identical to the original.
"""IDM training data: LeRobot v2.1 datasets of robot videos with joint actions.

Each embodiment is declared in the config (no robot list in code)::

    name: openarm_rh56f1           # also the key of its statistics in the checkpoint metadata
    embodiment_id: 29              # row of the per-embodiment action encoder/decoder
    root: /path/to/lerobot_dataset
    views: [camera_ego_left, camera_ego_right]     # LeRobot video keys; one key = mono
    view_ids: [0, 1, 2, 3]         # view-embedding id of [t0 v0, t0 v1, t1 v0, t1 v1]
    action_groups: [neck_joints, left_arm_joints, ...]   # modality.json groups, raw order
    hand_groups: [left_hand_joints, right_hand_joints]   # smoothed with the short SG window
    norm: q99                      # or q99_const (near-constant joints -> 0)
    video_size: [512, 288]         # dataset frame size; labelled videos are resized to it
    mirror: {prob, view_swap, swap, sign}      # optional left/right reflection augmentation

A sample is frames (t, t+K) of every view and the chunk a_{t:t+K}, normalised to [-1, 1] with
the dataset's q01/q99 and zero-padded to the model's action_dim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONST_EPS = 1e-3


# ----------------------------------------------------------------------------- normalisation
def normalize(x: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor, mode: str) -> torch.Tensor:
    q01, q99 = q01.to(x.dtype), q99.to(x.dtype)
    if mode == "q99":                       # constant dims (q01 == q99) pass through
        live = q01 != q99
        out = x.clone()
    elif mode == "q99_const":               # near-constant dims are mapped to 0
        live = (q99 - q01) > CONST_EPS
        out = torch.zeros_like(x)
    else:
        raise ValueError(f"unknown norm mode {mode!r}")
    out[..., live] = 2 * (x[..., live] - q01[live]) / (q99[live] - q01[live]) - 1
    return out.clamp(-1, 1)


def unnormalize(x: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor, mode: str) -> torch.Tensor:
    q01, q99 = q01.to(x.dtype), q99.to(x.dtype)
    out = (x + 1) / 2 * (q99 - q01) + q01
    if mode == "q99_const":
        out = torch.where((q99 - q01) > CONST_EPS, out, ((q01 + q99) / 2).expand_as(x))
    return out


def group_stats(stats: dict, groups: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate the per-group q01/q99 (metadata["statistics"]["action"]) in group order."""
    q01 = np.concatenate([stats[g]["q01"] for g in groups])
    q99 = np.concatenate([stats[g]["q99"] for g in groups])
    return torch.tensor(q01.tolist()), torch.tensor(q99.tolist())


# ----------------------------------------------------------------------------- images
def preprocess_frames(frames: np.ndarray, train: bool = False, fov_range=None,
                      color_jitter=None) -> np.ndarray:
    """(N, H, W, 3) uint8 -> (N, 3, 224, 224) float32, ImageNet-normalised.

    All N frames (every view and both timesteps of one sample) share one random draw, so the
    stereo geometry and the temporal pair stay consistent. Train-time augmentation: field of
    view (random zoom-in crop or zoom-out black padding at a random offset) and colour jitter.
    """
    import torchvision.transforms.v2 as T
    import torchvision.transforms.v2.functional as TF

    x = torch.from_numpy(np.ascontiguousarray(frames)).to(torch.float32).div(255.0).permute(0, 3, 1, 2)
    if train and fov_range is not None:
        x = _random_fov(x, *fov_range)
    x = TF.resize(x, [IMAGE_SIZE, IMAGE_SIZE], interpolation=T.InterpolationMode.BILINEAR,
                  antialias=True)
    if train and color_jitter is not None:
        x = T.ColorJitter(*color_jitter)(x)
    u8 = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).numpy()
    # Channels-last memory (a transposed view, no copy): cuDNN picks its convolution kernel by
    # layout, and this is the layout the IDM was trained and evaluated with.
    return ((u8.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(0, 3, 1, 2)


def _random_fov(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    import torchvision.transforms.v2.functional as TF

    H, W = x.shape[-2:]
    s = float(torch.empty(1).uniform_(lo, hi))
    if s <= 1.0:
        h, w = max(1, round(H * s)), max(1, round(W * s))
        top, left = int(torch.randint(0, H - h + 1, (1,))), int(torch.randint(0, W - w + 1, (1,)))
        return x[..., top:top + h, left:left + w]
    h, w = round(H * s), round(W * s)
    top, left = int(torch.randint(0, h - H + 1, (1,))), int(torch.randint(0, w - W + 1, (1,)))
    return TF.pad(x, [left, top, w - W - left, h - H - top], fill=0.0)


def model_inputs(images: np.ndarray, spec: dict, horizon: int, action_dim: int) -> dict:
    """The per-sample fields every batch needs besides the action target."""
    n_native = native_dim(spec)
    mask = np.zeros((horizon, action_dim), dtype=bool)
    mask[:, :n_native] = True
    view_ids = list(spec["view_ids"])
    assert len(view_ids) == len(images), f"view_ids {view_ids} for {len(images)} images"
    return dict(images=images, view_ids=np.asarray(view_ids, np.int64), n_images=len(images),
                embodiment_id=int(spec["embodiment_id"]), actions_mask=mask)


def collate(samples: list[dict]) -> dict:
    out = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        cat = k in ("images", "view_ids")
        out[k] = torch.from_numpy(np.concatenate(vals) if cat else np.stack(vals)) \
            if isinstance(vals[0], np.ndarray) else torch.tensor(vals)
    return out


# ----------------------------------------------------------------------------- LeRobot reader
class LeRobotMeta:
    """meta/{info,modality,episodes,stats} of a LeRobot v2.1 dataset, reduced to what the IDM
    needs: per-group column slices of the action, video paths and episode lengths."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        m = self.root / "meta"
        self.info = json.loads((m / "info.json").read_text())
        self.modality = json.loads((m / "modality.json").read_text())
        self.episodes = [json.loads(l) for l in (m / "episodes.jsonl").read_text().splitlines() if l.strip()]
        self.stats = json.loads((m / "stats.json").read_text())

    def action_slice(self, group: str) -> tuple[str, int, int]:
        g = self.modality["action"][group]
        return g.get("original_key") or "action", g["start"], g["end"]

    def action_statistics(self, groups: list[str]) -> dict:
        out = {}
        for g in groups:
            col, a, b = self.action_slice(g)
            out[g] = {k: list(np.asarray(v)[a:b]) for k, v in self.stats[col].items()}
        return out

    def parquet(self, ep: int) -> Path:
        return self.root / self.info["data_path"].format(
            episode_chunk=ep // self.info["chunks_size"], episode_index=ep)

    def video(self, ep: int, view: str) -> Path:
        key = self.modality["video"][view].get("original_key") or view
        return self.root / self.info["video_path"].format(
            episode_chunk=ep // self.info["chunks_size"], episode_index=ep, video_key=key)


def native_dim(spec: dict) -> int:
    return int(spec["action_dim"])


def frames_at(path: Path, timestamps: np.ndarray) -> np.ndarray:
    """Frames nearest to the given timestamps (seconds), (N, H, W, 3) uint8 RGB."""
    import decord

    vr = decord.VideoReader(str(path))
    ts = vr.get_frame_timestamp(range(len(vr)))[:, :1]
    return vr.get_batch(np.abs(ts - timestamps).argmin(axis=0)).asnumpy()


class Mirror:
    """Left/right reflection: flip the frames, exchange the two eyes, swap left/right joint
    groups (with a per-joint sign) and negate the joints listed in ``sign``. Declared per
    embodiment because the sign map depends on the kinematics."""

    def __init__(self, cfg: dict, groups: list[str], views: list[str]):
        self.prob = float(cfg.get("prob", 0.0))
        self.view_perm = list(range(len(views)))
        for a, b in cfg.get("view_swap", []):
            ia, ib = views.index(a), views.index(b)
            self.view_perm[ia], self.view_perm[ib] = ib, ia
        self.swaps = [(groups.index(a), groups.index(b), s) for a, b, s in cfg.get("swap", [])]
        self.signs = {groups.index(g): np.asarray(s) for g, s in (cfg.get("sign") or {}).items()}

    def __call__(self, frames: list[np.ndarray], parts: list[np.ndarray]):
        if self.prob <= 0 or float(torch.rand(())) >= self.prob:
            return frames, parts
        frames = [np.ascontiguousarray(frames[i][..., ::-1, :]) for i in self.view_perm]
        parts = list(parts)
        for ia, ib, s in self.swaps:
            s = 1.0 if s is None else np.asarray(s)
            parts[ia], parts[ib] = parts[ib] * s, parts[ia] * s
        for i, s in self.signs.items():
            parts[i] = parts[i] * s
        return frames, parts


class LeRobotIDMDataset(Dataset):
    """Every frame t of every episode is a sample: frames (t, t+K) of all views (clamped to the
    episode), and actions a_t .. a_{t+H-1} (the last action repeats past the end)."""

    def __init__(self, spec: dict, model_cfg: dict, data_cfg: dict, train: bool = True):
        self.spec, self.train = spec, train
        self.meta = LeRobotMeta(spec["root"])
        self.groups = list(spec["action_groups"])
        self.H, self.A = int(model_cfg["action_horizon"]), int(model_cfg["action_dim"])
        self.K = int(data_cfg["frame_gap"])
        eps = self.meta.episodes[: data_cfg.get("max_episodes") or None]
        self.steps = [(e["episode_index"], t, e["length"]) for e in eps for t in range(e["length"])]
        self.statistics = self.meta.action_statistics(self.groups)
        self.q01, self.q99 = group_stats(self.statistics, self.groups)
        widths = [self.meta.action_slice(g)[2] - self.meta.action_slice(g)[1] for g in self.groups]
        assert sum(widths) == native_dim(spec) <= self.A, (spec["name"], widths, self.A)
        self.mirror = Mirror(spec.get("mirror") or {}, self.groups, list(spec["views"]))
        self.fov = spec.get("fov_range", data_cfg.get("fov_range"))
        self.jitter = data_cfg.get("color_jitter")
        self._cache = (None, None)

    def __len__(self):
        return len(self.steps)

    def metadata(self) -> dict:
        return {"statistics": {"action": self.statistics}, "spec": dict(self.spec)}

    def _episode(self, ep: int):
        if self._cache[0] != ep:
            import pandas as pd
            self._cache = (ep, pd.read_parquet(self.meta.parquet(ep)))
        return self._cache[1]

    def __getitem__(self, i: int) -> dict:
        ep, t, L = self.steps[i]
        df = self._episode(ep)
        obs = np.clip(np.array([t, t + self.K]), 0, L - 1)
        ts = df["timestamp"].to_numpy()[obs]
        frames = [frames_at(self.meta.video(ep, v), ts) for v in self.spec["views"]]
        rows = np.minimum(np.arange(t, t + self.H), L - 1)
        parts = []
        for g in self.groups:
            col, a, b = self.meta.action_slice(g)
            parts.append(np.stack(df[col].to_numpy()[rows])[:, a:b].astype(np.float64))
        if self.train:
            frames, parts = self.mirror(frames, parts)
        # (v, t) -> images ordered [t0 v0, t0 v1, t1 v0, t1 v1]
        stacked = np.stack(frames, 1).reshape(-1, *frames[0].shape[1:])
        images = preprocess_frames(stacked, self.train, self.fov, self.jitter)
        act = normalize(torch.from_numpy(np.concatenate(parts, 1)), self.q01, self.q99,
                        self.spec.get("norm", "q99")).numpy()
        sample = model_inputs(images, self.spec, self.H, self.A)
        sample["actions"] = np.pad(act, ((0, 0), (0, self.A - act.shape[1]))).astype(np.float32)
        return sample


class MixtureDataset(Dataset):
    """Weighted mixture of embodiments. Each epoch, embodiment k fills round(w_k * N) of the
    N = sum(len) slots with a fresh random permutation of its samples (repeated if up-weighted)."""

    def __init__(self, datasets: list[LeRobotIDMDataset], weights: list[float], seed: int = 42):
        self.datasets = datasets
        w = np.asarray(weights, np.float64)
        self.weights = w / w.sum()
        self.seed, self.N = seed, sum(len(d) for d in datasets)
        self.epoch, self.table = None, None
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if epoch == self.epoch:
            return
        self.epoch = epoch
        rng = np.random.default_rng(self.seed + epoch)
        raw = self.weights * self.N
        slots = np.floor(raw).astype(int)
        for i in np.argsort(-(raw - slots))[: self.N - slots.sum()]:
            slots[i] += 1
        rows = []
        for k, (ds, n_slots) in enumerate(zip(self.datasets, slots)):
            reps, rem = divmod(int(n_slots), len(ds))
            local = np.concatenate([rng.permutation(len(ds)) for _ in range(reps)]
                                   + [rng.permutation(len(ds))[:rem]])
            rows.append(np.stack([np.full(n_slots, k), local], 1))
        self.table = np.concatenate(rows).astype(np.int64)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        k, j = self.table[i]
        return self.datasets[k][j]


def embodiment_specs(data_cfg: dict) -> dict[str, dict]:
    """Embodiments declared in the config, keyed by name, with ``data.roots`` overrides applied."""
    roots = data_cfg.get("roots") or {}
    specs = {}
    for e in data_cfg["embodiments"]:
        spec = dict(e)
        spec["root"] = roots.get(spec["name"], spec.get("root"))
        specs[spec["name"]] = spec
    return specs

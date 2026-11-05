"""Training samples for the keyframe-translation model (paper Sec. 3.2, App. B.2.3).

A sample is a robot frame o_t of one episode, seen by one camera (mono) or both (stereo):

* target   o_t
* cond     Phi(o_t; M_t, S_t): the randomised shape-agnostic mask blacked out, the augmented
           hand skeleton (rendered on the fly from keypoints) drawn on top
* mask     M_t
* reference o_t' with t' drawn uniformly from [t - 5 s, t - 2 s] of the same episode (frame 0 if
           the target is too early). With probability ``p_robot_only`` the reference is instead a
           "robot-only" exemplar: a frame of ANOTHER episode of the same embodiment with
           everything but the robot (its tight mask, eroded) filled with white. This is
           the reference the first keyframe gets at inference.
* prompt   the hand-state prompt of the target frame (each hand's closure / facing words from the
           episode's ``hand_state_<view>.npz``; see ``robotryon.captions``), as a cached
           embedding.

With probability ``ref_skeleton_p`` the target's (augmented) skeleton is also painted on the
reference, sequential or robot-only alike; otherwise the reference is left clean.

The two views of a stereo sample share every random draw (mask dilations, skeleton corruption,
reference frame, field-of-view crop): they are two views of one instant.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..masks import MaskAug, fov_augment, robot_only
from ..skeleton import SkeletonAugment, condition_frame, draw_skeleton, paint
from .episodes import Episode, FrameReader, crop_resize, load_view_frame, output_geometry, scan_root
from ..captions import HandStateCuts, all_prompts, load_hand_state, sample_prompt, words_at
from .text import TextEmbedCache


def _to_tensor(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        return torch.from_numpy(a / 255.0)[None]
    return torch.from_numpy(a / 127.5 - 1.0).permute(2, 0, 1).contiguous()


class KeyframeDataset(torch.utils.data.Dataset):
    """Indexed by ``(episode_index, frame, view_indices, seed)`` tuples from :class:`FillBatchSampler`."""

    def __init__(self, cfg, text_cache: TextEmbedCache | None):
        d = cfg.data
        roots = d.roots if isinstance(d.roots, (list, tuple)) else [d.roots]
        self.root_weights = []
        self.episodes: list[Episode] = []
        for i, r in enumerate(roots):
            path, w = (r.path, float(r.get("weight", 1.0))) if isinstance(r, dict) else (r, 1.0)
            eps = scan_root(path, i)
            if not eps:
                raise FileNotFoundError(f"no episodes (*/meta.json) under {path}")
            self.episodes += eps
            self.root_weights.append(w)
        self.long_edge = int(cfg.image.long_edge)
        self.aspect = tuple(cfg.image.aspect) if cfg.image.get("aspect") else None
        self.anchor = cfg.image.get("crop_anchor", "bottom")
        self.ref_window = tuple(d.ref_window_sec)
        self.p_robot_only = float(d.p_robot_only)
        self.ref_skeleton_p = float(d.ref_skeleton_p)
        self.robot_only_erode = int(d.robot_only.erode)
        self.robot_only_bg = int(d.robot_only.background)
        self.robot_only_min_area = float(d.robot_only.min_area)
        self.bonus_tags = set(cfg.masks.get("bonus_embodiments", []) or [])
        self.mask_aug = MaskAug.from_config(cfg.masks)
        self.fov = cfg.masks.fov
        self.skel_aug = SkeletonAugment.from_config(cfg.skeleton.augment)
        self.thickness = tuple(cfg.skeleton.thickness)
        self.cap = cfg.caption
        self.text_cache = text_cache
        self.reader: FrameReader | None = None
        self.by_embodiment: dict[str, list[int]] = {}
        for i, ep in enumerate(self.episodes):
            self.by_embodiment.setdefault(ep.embodiment, []).append(i)

    def __len__(self) -> int:
        return len(self.episodes)

    def hand_states(self, ep: Episode, views: list[str]) -> list:
        return [load_hand_state(ep.path / f"hand_state_{v}.npz") for v in views
                if (ep.path / f"hand_state_{v}.npz").exists()]

    def all_prompts(self) -> set[str]:
        return set(all_prompts())

    # -- reference ------------------------------------------------------------------------------
    def reference_index(self, ep: Episode, t: int, rng: np.random.Generator) -> int:
        lo, hi = self.ref_window
        newest = t - int(round(lo * ep.fps))
        oldest = max(0, t - int(round(hi * ep.fps)))
        if newest < oldest:
            return 0
        return int(rng.integers(oldest, newest + 1))

    def draw_exemplar(self, ep_idx: int, views: list[str], rng) -> tuple[int, int] | None:
        """(episode, frame) of another episode of the same embodiment with a visible robot."""
        ep = self.episodes[ep_idx]
        pool = [i for i in self.by_embodiment.get(ep.embodiment, [])
                if i != ep_idx and all(v in self.episodes[i].views for v in views)]
        if not pool:
            return None
        ex = int(pool[int(rng.integers(0, len(pool)))])
        ex_ep = self.episodes[ex]
        for _ in range(8):
            t = int(rng.integers(0, ex_ep.num_frames))
            m = self.reader.binary(ex_ep.mask(views[0]), t)
            if m.mean() >= self.robot_only_min_area:
                return ex, t
        return None

    def robot_only(self, ex: int, t: int, view: str, size) -> np.ndarray:
        ep = self.episodes[ex]
        rgb = self.reader.rgb(ep.video(view), t)
        src = (rgb.shape[1], rgb.shape[0])
        box, _ = output_geometry(src[0], src[1], self.long_edge, self.aspect, self.anchor)
        img = crop_resize(rgb, box, size, src)
        m = crop_resize(self.reader.binary(ep.mask(view), t).astype(np.uint8) * 255, box, size, src)
        return robot_only(img, m > 127, self.robot_only_erode, self.robot_only_bg)

    # -- one view -------------------------------------------------------------------------------
    def view_streams(self, ep: Episode, view: str, t: int, ref_t: int, exemplar, ref_skeleton: bool,
                     rng) -> dict:
        f = load_view_frame(self.reader, ep, view, t, self.long_edge, self.aspect, self.anchor)
        H, W = f.image.shape[:2]
        mask, _ = self.mask_aug.train_mask(f.hulls, W, H, rng, bonus=ep.embodiment in self.bonus_tags,
                                           object_mask=f.object_mask)
        kp, vis = self.skel_aug(f.kp, f.vis, rng)
        th = int(rng.integers(self.thickness[0], self.thickness[1] + 1))
        skel = draw_skeleton(kp, vis, W, H, thickness=th, radius=max(1, int(round(th * 1.5))))
        cond = condition_frame(f.image, mask, skel)
        if exemplar is not None:
            ref = self.robot_only(exemplar[0], exemplar[1], view, (W, H))
        else:
            rgb = self.reader.rgb(ep.video(view), ref_t)
            src = (rgb.shape[1], rgb.shape[0])
            box, _ = output_geometry(src[0], src[1], self.long_edge, self.aspect, self.anchor)
            ref = crop_resize(rgb, box, (W, H), src)
        if ref_skeleton:
            ref = paint(ref, skel)
        return {"target": f.image, "cond": cond, "mask": mask * 255, "reference": ref,
                "has_hull": bool(f.hulls)}

    def __getitem__(self, item) -> dict:
        ep_idx, t, view_idx, seed = item
        if self.reader is None:
            self.reader = FrameReader()
        ep = self.episodes[ep_idx]
        views = [ep.views[i] for i in view_idx]
        rng = np.random.default_rng([int(seed), 0])
        exemplar = None
        if float(rng.random()) < self.p_robot_only:
            exemplar = self.draw_exemplar(ep_idx, views, rng)
        ref_skeleton = float(rng.random()) < self.ref_skeleton_p
        for attempt in range(8):           # prefer frames where the arm is visible
            ref_t = self.reference_index(ep, t, rng)
            out = [self.view_streams(ep, v, t, ref_t, exemplar, ref_skeleton,
                                     np.random.default_rng([int(seed), 1, attempt])) for v in views]
            if out[0]["has_hull"]:
                break
            t = int(rng.integers(0, ep.num_frames))
        keys = ("target", "cond", "mask", "reference")
        imgs = [Image.fromarray(np.asarray(o[k], np.uint8)) for o in out for k in keys]
        imgs, _ = fov_augment(imgs, rng, float(self.fov.crop_min), float(self.fov.scale_max),
                              float(self.fov.get("flip_prob", 0.0)))
        n = len(keys)
        sample = {k: torch.stack([_to_tensor(imgs[v * n + j]) for v in range(len(views))])
                  for j, k in enumerate(keys)}
        words = words_at(self.hand_states(ep, views), t, HandStateCuts.from_config(self.cap.get("cuts")))
        prompt = sample_prompt(words, rng, float(self.cap.p_uncond), float(self.cap.p_drop_hands))
        sample["prompt"] = prompt
        if self.text_cache is not None:
            sample["prompt_embeds"] = self.text_cache.load(prompt)
        sample["robot_only"] = exemplar is not None
        sample["ref_skeleton"] = ref_skeleton
        return sample


def collate(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in ("target", "cond", "mask", "reference")}
    if "prompt_embeds" in batch[0]:
        out["prompt_embeds"] = torch.stack([b["prompt_embeds"] for b in batch])
    out["prompt"] = [b["prompt"] for b in batch]
    out["robot_only"] = [b["robot_only"] for b in batch]
    return out


class FillBatchSampler(torch.utils.data.Sampler):
    """Endless stream of homogeneous batches for one rank.

    A batch is all-stereo or all-mono (the token sequence length depends on it): stereo with
    probability ``stereo_p`` (paper: stereo : mono = 3 : 1) when stereo episodes exist. Items pick
    a data root by weight, an episode uniformly within it, a frame uniformly, and for mono a view
    uniformly. Each rank draws its own stream from ``(seed, rank)``.
    """

    def __init__(self, dataset: KeyframeDataset, batch_size: int, stereo_p: float, seed: int,
                 rank: int = 0):
        self.ds, self.bs, self.stereo_p = dataset, int(batch_size), float(stereo_p)
        self.rng = np.random.default_rng([int(seed), int(rank)])
        eps = dataset.episodes
        n_roots = len(dataset.root_weights)
        self.pools = {s: [[i for i, e in enumerate(eps) if e.root == r and (len(e.views) >= 2 or not s)]
                          for r in range(n_roots)] for s in (True, False)}
        self.has_stereo = any(self.pools[True])

    def _root_probs(self, stereo: bool) -> np.ndarray:
        w = np.array([wt if self.pools[stereo][r] else 0.0
                      for r, wt in enumerate(self.ds.root_weights)], np.float64)
        return w / w.sum()

    def __iter__(self):
        while True:
            stereo = self.has_stereo and float(self.rng.random()) < self.stereo_p
            p = self._root_probs(stereo)
            batch = []
            for r in self.rng.choice(len(p), size=self.bs, p=p):
                pool = self.pools[stereo][int(r)]
                e = pool[int(self.rng.integers(0, len(pool)))]
                ep = self.ds.episodes[e]
                t = int(self.rng.integers(0, ep.num_frames))
                views = (0, 1) if stereo else (int(self.rng.integers(0, len(ep.views))),)
                batch.append((e, t, views, int(self.rng.integers(0, 2 ** 31))))
            yield batch

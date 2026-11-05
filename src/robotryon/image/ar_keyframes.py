"""Autoregressive keyframe translation (paper Eq. 3.2, App. B.2.3).

Human keyframes are translated in temporal order. Keyframe t_j is conditioned on

* the conditioned human frame Phi(o^h; M^h, S^h): the human arm mask (convex hull per arm,
  fixed dilation, eroded object removed) blacked out and the retargeted robot-hand skeleton drawn
  on top, plus the mask itself as an input channel;
* the reference: for the first keyframe, the encoded robot-only reference image supplied by the
  user (it selects the embodiment); for every later keyframe, the DENOISED LATENT TOKENS of the
  previous keyframe, reused directly as reference tokens (no decode / re-encode).

``ref_skeleton: true`` replaces latent reuse by a pixel-space reference: the previous keyframe is
decoded, the current keyframe's skeleton is painted on it, and the result is re-encoded (the first
keyframe's reference image is painted the same way).

Stereo episodes are translated jointly (both views in one token sequence) by default; ``views:
mono`` translates each view as its own chain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..masks import MaskAug, inference_mask
from ..skeleton import condition_frame, draw_skeleton, paint
from .episodes import Episode, FrameReader, load_view_frame
from .fill_channels import pack
from .sampler import LatentCodec, ViewCondition, denoise
from ..captions import HandStateCuts, hand_state_arrays, hand_state_prompt, load_hand_state, words_at


@dataclass
class TranslatedKeyframe:
    image: np.ndarray        # uint8 (H, W, 3) robot keyframe
    cond: np.ndarray         # uint8 (H, W, 3) conditioned human frame
    mask: np.ndarray         # uint8 (H, W) {0, 1} inference mask
    tokens: torch.Tensor     # (1, h*w, 128) denoised latent tokens (next keyframe's reference)


class KeyframeTranslator:
    def __init__(self, pipe, cfg, device="cuda"):
        self.pipe, self.cfg, self.device = pipe, cfg, device
        self.codec = LatentCodec(pipe.vae, device)
        self.reader = FrameReader()
        self.dilate = MaskAug.from_config(cfg.masks).infer_dilate()
        self.ref_skeleton = bool(cfg.get("ref_skeleton", False))
        self._prompts: dict[str, tuple] = {}

    def prompt_fn(self, ep: Episode, episode_dir: Path):
        """``prompt(t)``: the instruction and each hand's closure / facing words at keyframe t
        (``robotryon.captions``), read from the episode's ``hand_state_<view>.npz`` (or computed
        from ``hands.npz`` when absent)."""
        cuts = HandStateCuts.from_config(self.cfg.caption.get("cuts"))
        states = []
        for i, v in enumerate(ep.views):
            f = episode_dir / f"hand_state_{v}.npz"
            if f.exists():
                states.append(load_hand_state(f))
        if not states:
            z = np.load(episode_dir / "hands.npz")
            states = [hand_state_arrays(z["joints3d"][i], z["valid"][i].astype(bool))
                      for i in range(z["joints3d"].shape[0])]
        return lambda t: hand_state_prompt(*words_at(states, int(t), cuts))

    def encode_prompt(self, text: str):
        if text not in self._prompts:
            self._prompts[text] = self.pipe.encode_prompt(prompt=text, device=self.device)
        return self._prompts[text]

    def condition(self, ep: Episode, view: str, t: int):
        """(frame, conditioned frame, mask, skeleton image) of one human keyframe at model resolution."""
        c = self.cfg
        f = load_view_frame(self.reader, ep, view, t, int(c.image.long_edge),
                            tuple(c.image.aspect) if c.image.get("aspect") else None,
                            c.image.get("crop_anchor", "bottom"))
        H, W = f.image.shape[:2]
        mask = inference_mask(f.hulls, W, H, self.dilate, f.object_mask, int(c.masks.obj_erode))
        skel = draw_skeleton(f.kp, f.vis, W, H)
        cond = condition_frame(f.image, mask, skel)
        return f.image, cond, mask, skel

    def reference(self, view: str, prev: dict, references: dict, skel: np.ndarray) -> torch.Tensor:
        """Reference tokens of one view for the current keyframe."""
        if view in prev and not self.ref_skeleton:
            return prev[view].tokens                       # latent reuse
        if view in prev:
            img = prev[view].image
        else:
            H, W = skel.shape[:2]
            img = np.asarray(Image.fromarray(references[view]).resize((W, H), Image.LANCZOS))
        return pack(self.codec.encode(paint(img, skel) if self.ref_skeleton else img))

    @torch.no_grad()
    def translate_chain(self, views: list[str], keyframes: list[int], references: dict[str, np.ndarray],
                        prompt, conditions) -> dict:
        """One chain over ``keyframes``; ``views`` are denoised jointly.

        ``prompt`` is a string or ``prompt(t)`` giving each keyframe its own prompt.
        ``conditions(view, t)`` returns ``(frame, conditioned frame, mask, skeleton image)``.
        Returns ``{view: {t: TranslatedKeyframe}}``.
        """
        c = self.cfg
        prompt_at = prompt if callable(prompt) else (lambda t: prompt)
        neg, neg_ids = self.encode_prompt("") if float(c.guidance_scale) > 1.0 else (None, None)
        prev: dict[str, TranslatedKeyframe] = {}
        out: dict[str, dict] = {v: {} for v in views}
        for t in keyframes:
            conds, frames = [], []
            for v in views:
                _, cond, mask, skel = conditions(v, t)
                mask_t = torch.from_numpy(mask.astype(np.float32))[None, None].to(self.device)
                conds.append(ViewCondition(self.codec.encode(cond), mask_t,
                                           self.reference(v, prev, references, skel)))
                frames.append((cond, mask))
            pos, pos_ids = self.encode_prompt(prompt_at(t))
            gen = torch.Generator(device=self.device).manual_seed(int(c.seed) + int(t))
            tokens = denoise(self.pipe.transformer, self.pipe.scheduler, conds, pos, pos_ids, neg, neg_ids,
                             steps=int(c.steps), guidance_scale=float(c.guidance_scale), generator=gen)
            for v, tok, (cond, mask) in zip(views, tokens, frames):
                H, W = mask.shape
                img = self.codec.decode_tokens(tok, H // 16, W // 16)
                out[v][t] = prev[v] = TranslatedKeyframe(img, cond, mask, tok)
        return out

    def translate_episode(self, episode_dir: str | Path, references: dict[str, np.ndarray],
                          out_dir: str | Path) -> Path:
        """Translate every keyframe of a preprocessed human episode and write the results."""
        c = self.cfg
        ep = Episode.load(episode_dir)
        kf = json.loads((Path(episode_dir) / "keyframes.json").read_text())["indices"]
        keyframes = sorted({int(i) for i in kf} | {0})
        stereo = len(ep.views) >= 2 and c.views in ("auto", "stereo")
        if c.views == "stereo" and not stereo:
            raise ValueError(f"{episode_dir}: views=stereo needs two camera views, found {ep.views}")
        chains = [ep.views[:2]] if stereo else [[v] for v in ep.views]
        missing = [v for v in ep.views if v not in references]
        if missing:
            raise ValueError(f"no reference image for view(s) {missing}")
        prompt_at = self.prompt_fn(ep, Path(episode_dir))
        prompts = {int(t): prompt_at(t) for t in keyframes}
        out = Path(out_dir) / ep.path.name
        out.mkdir(parents=True, exist_ok=True)
        for views in chains:
            res = self.translate_chain(views, keyframes, references, prompt_at,
                                       lambda v, t: self.condition(ep, v, t))
            for v, per_t in res.items():
                for t, k in per_t.items():
                    Image.fromarray(k.image).save(out / f"keyframe_{v}_{t:06d}.png")
                    Image.fromarray(k.cond).save(out / f"cond_{v}_{t:06d}.png")
                    Image.fromarray(k.mask * 255).save(out / f"mask_{v}_{t:06d}.png")
        (out / "keyframes.json").write_text(json.dumps({
            "indices": keyframes, "views": ep.views, "joint_stereo": stereo, "prompts": prompts,
            "ref_skeleton": self.ref_skeleton, "steps": int(c.steps),
            "guidance_scale": float(c.guidance_scale),
            "seed": int(c.seed)}, indent=1))
        return out

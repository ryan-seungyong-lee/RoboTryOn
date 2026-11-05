"""Video synthesis with the in-context LoRA (paper Eq. 3.4 and App. B.4).

``StereoIcLoraPipeline`` extends diffusers' ``LTX2InContextPipeline`` with two things:

* **stereo**: ``conditions`` and ``reference_conditions`` are per-view lists. Each view is
  prepared by the parent and the token groups are interleaved by role,
  ``[base(L) base(R) | kf(L) kf(R) | ref(L) ref(R)]``, so the reference tokens stay at the end
  of the sequence where the parent's attention mask expects them. A fourth RoPE axis
  (rope4d.py) tells the views apart; mono uses the pipeline's own coordinates.
* **exact keyframe times**: diffusers places a keyframe at a latent index, which snaps it to
  pixel frame ``8k - 7``. Training places keyframes at their exact pixel frame ``[f, f+1)``,
  so the appended keyframe coordinates are rewritten to the requested pixel frames. The
  keyframe at frame 0 replaces latent frame 0 and needs no rewrite.

``generate`` runs the two-stage recipe on one clip. Both stages use the full LTX-2.5
transformer with the IC-LoRA and Lightricks' distilled LoRA, the published distilled sigma
schedules (8 steps, then 3 refinement steps from sigma 0.909) and neutral guidance. Both stages
are conditioned on the same keyframes and skeleton flow. ``two_stage.mode``:

* ``same_res``: stage 1 already at the output resolution; stage 2 refines at that resolution.
* ``upsample``: stage 1 at half the output resolution, the stage-1 latents are upsampled x2 by
  the LTX-2.5 spatial latent upsampler, and stage 2 refines at the output resolution.

The skeleton flow is always given at one ``skeleton_size`` (512x288); each stage encodes it
with ``reference_downscale = stage width / skeleton width`` (1 at 512x288, 2 at 1024x576),
exactly as the two training stages did.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from diffusers.pipelines.ltx2 import LTX2InContextPipeline

from .chunks import latent_index
from .rope4d import add_view_coord, install_view_axis_rope, set_view_coord

MAX_SEQ = 256                       # text tokens; training used the same
DISTILLED_LORA = "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"


def snapshot_dir(repo_or_path: str) -> str:
    """A local directory for a Hugging Face repo id (newest cached snapshot) or a path."""
    if os.path.isdir(repo_or_path):
        return repo_or_path
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    slug = "models--" + repo_or_path.replace("/", "--")
    hits = sorted(glob.glob(os.path.join(HF_HUB_CACHE, slug, "snapshots", "*")))
    return hits[-1] if hits else snapshot_download(repo_or_path)


class StereoIcLoraPipeline(LTX2InContextPipeline):
    """``LTX2InContextPipeline`` with a view axis and exact-frame keyframes."""

    num_views = 1
    exact_kf_px: list[int] | None = None       # pixel frames of the appended keyframes

    def _unpack_latents(self, latents, num_frames, height, width, patch_size=1,
                        patch_size_t=1):
        """Unpack the views' base tokens into a batch ``(V, C, F, H, W)`` for decoding."""
        per_view = num_frames * height * width
        views = latents.shape[1] // per_view
        if views < 2:
            return super()._unpack_latents(latents, num_frames, height, width, patch_size,
                                           patch_size_t)
        return torch.cat([super(StereoIcLoraPipeline, self)._unpack_latents(
            latents[:, v * per_view:(v + 1) * per_view], num_frames, height, width,
            patch_size, patch_size_t) for v in range(views)], dim=0)

    def _retime_keyframes(self, kf_coords, n_kf, kw):
        """Move the appended keyframe blocks to their exact pixel frames ``[f, f+1) / fps``."""
        if not self.exact_kf_px or not n_kf:
            return kf_coords
        fps = float(kw["frame_rate"])
        tpf = (int(kw["height"]) // 32) * (int(kw["width"]) // 32)
        if n_kf != len(self.exact_kf_px) * tpf:
            raise ValueError(f"{n_kf} appended keyframe tokens do not match "
                             f"{len(self.exact_kf_px)} keyframes of {tpf} tokens")
        out = kf_coords.clone()
        for j, f in enumerate(self.exact_kf_px):
            sl = slice(j * tpf, (j + 1) * tpf)
            out[:, 0, sl, 0] = float(f) / fps
            out[:, 0, sl, 1] = float(f + 1) / fps
        return out

    def prepare_latents(self, conditions=None, reference_conditions=None, **kw):
        views = self.num_views
        conditions = conditions or [None] * views
        reference_conditions = reference_conditions or [None] * views
        lat_in = kw.pop("latents", None)       # stage 2: stage-1 (upsampled) latents per view

        def view_latents(v):
            if lat_in is None:
                return None
            if lat_in.dim() == 5 and lat_in.shape[0] == views:
                return lat_in[v:v + 1]
            raise ValueError(f"latents {tuple(lat_in.shape)} do not split into {views} views")

        self._single_view_coords = True        # reference coords from the parent stay 3-axis
        try:
            per = [super(StereoIcLoraPipeline, self).prepare_latents(
                       conditions=conditions[v], reference_conditions=reference_conditions[v],
                       latents=view_latents(v), **kw) for v in range(views)]
        finally:
            self._single_view_coords = False

        groups = {k: [[] for _ in range(3)] for k in ("lat", "mask", "clean")}
        kf_co, ref_co, ref_masks, ref_counts = [], [], [], []
        for v, (lat, mask, clean, acoords, nref, rmask, rcounts) in enumerate(per):
            n_app = 0 if acoords is None else acoords.shape[2]
            n_kf, n_base = n_app - nref, lat.shape[1] - n_app
            bounds = (slice(0, n_base), slice(n_base, n_base + n_kf),
                      slice(n_base + n_kf, lat.shape[1]))
            for name, src in (("lat", lat), ("mask", mask), ("clean", clean)):
                for g, sl in enumerate(bounds):
                    groups[name][g].append(src[:, sl])
            if acoords is not None:
                a4 = set_view_coord(acoords, v) if views > 1 else acoords
                kf_co.append(self._retime_keyframes(a4[:, :, :n_kf], n_kf, kw))
                ref_co.append(a4[:, :, n_kf:])
            if rmask is not None:
                ref_masks.append(rmask)
            ref_counts.extend(rcounts or [])

        def flat(g):
            return torch.cat(g[0] + g[1] + g[2], dim=1)
        coords = torch.cat(kf_co + ref_co, dim=2) if (kf_co or ref_co) else None
        return (flat(groups["lat"]), flat(groups["mask"]), flat(groups["clean"]), coords,
                sum(x.shape[1] for x in groups["lat"][2]),
                torch.cat(ref_masks, dim=1) if ref_masks else None, ref_counts)


def _install_stereo(pipe: StereoIcLoraPipeline) -> None:
    """View-axis RoPE, and a base grid that covers every view (view axis stamped per view)."""
    install_view_axis_rope(pipe.transformer)
    pipe._single_view_coords = False
    rope = pipe.transformer.rope
    orig = rope.prepare_video_coords

    def widened(*args, **kwargs):
        plain = orig(*args, **kwargs)
        if pipe._single_view_coords:
            return plain
        return torch.cat([add_view_coord(plain, v) for v in range(pipe.num_views)], dim=2)

    rope.prepare_video_coords = widened


def build_pipeline(lora: str, *, base: str = "Lightricks/LTX-2.5-Diffusers",
                   transformer_subfolder: str = "transformer_full", views: int = 1,
                   attention_backend: str | None = "native", upsample: bool = False,
                   device: str = "cuda") -> StereoIcLoraPipeline:
    """Load LTX-2.5 + IC-LoRA + distilled LoRA, without the text encoder.

    Args:
        lora: checkpoint directory holding ``pytorch_lora_weights.safetensors`` (or the file).
        views: views generated jointly in one sequence (2 = stereo, 1 = mono).
        upsample: also load the spatial latent upsampler (``two_stage.mode: upsample``).
    """
    from diffusers import LTX2VideoTransformer3DModel

    root = snapshot_dir(base)
    pipe = StereoIcLoraPipeline.from_pretrained(
        root, transformer=LTX2VideoTransformer3DModel.from_pretrained(
            root, subfolder=transformer_subfolder, dtype=torch.bfloat16),
        text_encoder=None, tokenizer=None, dtype=torch.bfloat16)
    pipe.num_views = int(views)
    if views > 1:
        _install_stereo(pipe)
    lora_dir, name = ((os.path.dirname(lora), os.path.basename(lora)) if os.path.isfile(lora)
                      else (lora, "pytorch_lora_weights.safetensors"))
    pipe.load_lora_weights(lora_dir, weight_name=name, adapter_name="iclora")
    if not any("lora" in k for k, _ in pipe.transformer.named_parameters()):
        raise RuntimeError(f"no LoRA weights attached from {lora}")
    pipe.load_lora_weights(root, weight_name=DISTILLED_LORA, adapter_name="distilled")
    pipe.set_adapters(["iclora", "distilled"], [1.0, 1.0])
    if attention_backend:
        pipe.transformer.set_attention_backend(attention_backend)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.to(device)
    pipe.upsampler = None
    if upsample:
        from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        model = LTX2LatentUpsamplerModel.from_pretrained(
            root, subfolder="latent_upsampler", dtype=torch.bfloat16).to(device)
        pipe.upsampler = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=model)
    return pipe


def _u8(img: torch.Tensor) -> np.ndarray:
    """``(3, H, W)`` in [-1, 1] -> ``(1, H, W, 3)`` uint8 (keyframe convention)."""
    return ((img.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0)[None].cpu().numpy()


def _f01(video: torch.Tensor) -> np.ndarray:
    """``(3, F, H, W)`` in [-1, 1] -> ``(F, H, W, 3)`` float in [0, 1] (reference convention)."""
    return ((video.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).float().cpu().numpy()


def _to_uint8(out: torch.Tensor, views: int) -> torch.Tensor:
    """Pipeline ``pt`` output in [0, 1], ``(V, F, 3, H, W)`` -> ``(V, F, H, W, 3)`` uint8."""
    v = out if out.shape[0] == views else out[0]
    frames = [x if x.shape[0] == 3 else x.permute(1, 0, 2, 3) for x in v]   # -> (3, F, H, W)
    return torch.stack([(x.float().clamp(0, 1) * 255).byte().permute(1, 2, 3, 0).cpu()
                        for x in frames])


@torch.no_grad()
def generate(pipe: StereoIcLoraPipeline, *, skeleton: torch.Tensor,
             keyframes: list[tuple[int, torch.Tensor]], text: tuple, num_frames: int,
             size: tuple[int, int], fps: float, seed: int = 0,
             mode: str = "same_res") -> torch.Tensor:
    """Synthesize one clip.

    Args:
        skeleton: ``(V, 3, F, h, w)`` skeleton flow in [-1, 1] at the reference resolution.
        keyframes: ``[(pixel frame, (V, 3, H, W) in [-1, 1])]`` sorted, first at frame 0. Any
            resolution: the pipeline resizes them for each stage.
        text: ``(prompt_embeds, mask, negative_embeds, negative_mask)`` (see text.py).
        size: output ``(width, height)`` of one view.
        mode: ``same_res`` or ``upsample``.

    Returns:
        ``(V, F, height, width, 3)`` uint8 RGB.
    """
    from diffusers.pipelines.ltx2 import LTX2ReferenceCondition, LTX2VideoCondition
    from diffusers.pipelines.ltx2.utils import (DISTILLED_SIGMA_VALUES,
                                                STAGE_2_DISTILLED_SIGMA_VALUES)
    if mode not in ("same_res", "upsample"):
        raise ValueError(f"two_stage.mode must be same_res or upsample, got {mode!r}")
    views = int(skeleton.shape[0])
    if views != pipe.num_views:
        raise ValueError(f"{views} views given to a {pipe.num_views}-view pipeline")
    px = [p for p, _ in keyframes]
    lat = [latent_index(p, num_frames) for p in px]
    if px != sorted(set(px)) or len(set(lat)) != len(lat):
        raise ValueError(f"keyframes must be sorted and in distinct latent frames: {px}")
    pipe.exact_kf_px = [p for p, li in zip(px, lat) if li != 0]

    W, H = size
    skel_w = skeleton.shape[-1]
    W1, H1 = (W // 2, H // 2) if mode == "upsample" else (W, H)
    pe, pm, npe, npm = (x.cuda() for x in text)
    kw = dict(prompt_embeds=pe, prompt_attention_mask=pm, negative_prompt_embeds=npe,
              negative_prompt_attention_mask=npm,
              conditions=[[LTX2VideoCondition(frames=_u8(img[v]), index=li, crf=0)
                           for li, (_, img) in zip(lat, keyframes)] for v in range(views)],
              reference_conditions=[[LTX2ReferenceCondition(frames=_f01(skeleton[v]))]
                                    for v in range(views)],
              reference_downscale_factor=W1 // skel_w, max_sequence_length=MAX_SEQ,
              height=H1, width=W1, num_frames=num_frames, frame_rate=fps,
              generator=torch.Generator("cuda").manual_seed(seed), return_dict=False,
              audio_latents=torch.zeros(1, pipe.audio_latent_channels, 1,
                                        pipe.audio_mel_bins // pipe.audio_vae_mel_compression_ratio,
                                        device="cuda", dtype=torch.float32),
              guidance_scale=1.0, stg_scale=0.0, modality_scale=1.0, audio_guidance_scale=1.0,
              audio_stg_scale=0.0, audio_modality_scale=1.0)

    lat1 = pipe(sigmas=list(DISTILLED_SIGMA_VALUES), output_type="latent", **kw)[0]
    if mode == "upsample":
        lat1 = pipe.upsampler(latents=lat1, output_type="latent", return_dict=False)[0]
    s2 = list(STAGE_2_DISTILLED_SIGMA_VALUES)
    kw.update(height=H, width=W, reference_downscale_factor=W // skel_w)
    out = pipe(sigmas=s2, latents=lat1, noise_scale=s2[0], output_type="pt", **kw)[0]
    return _to_uint8(out, views)

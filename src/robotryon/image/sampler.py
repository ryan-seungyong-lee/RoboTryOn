"""Inpainting sampler for keyframe translation (true classifier-free guidance, Euler flow matching).

``Flux2KleinPipeline.__call__`` cannot express channel-concatenated conditioning, so the denoising
loop is written out here. The token layout must match training exactly (see ``fill_channels`` and
``stereo_ids``)::

    [ target_v (T=0, L=v): noisy | conditioned frame | mask ]  for every view v
    [ reference_v (T=10, L=v): 128 latent channels, zero-padded ]  for every view v

The schedule is the pipeline's: ``sigmas = linspace(1, 1/steps, steps)`` shifted by the empirical
``mu`` of the total target token count (both views for stereo). FLUX.2 [klein] Base has no guidance
embedding, so guidance is true CFG against the empty prompt:
``v = v_uncond + s * (v_cond - v_uncond)``.

The sampler returns the final latent TOKENS. In autoregressive inference these tokens are fed
directly as the next keyframe's reference tokens, without decoding and re-encoding (App. B.2.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .fill_channels import BASE_IN, FILL_IN, fold_mask_to_latent, pack, pad_condition_tokens, unpack
from .stereo_ids import fill_ids


class LatentCodec:
    """Pixels <-> normalised, 2x2-patchified FLUX.2 VAE latents ``(1, 128, H/16, W/16)``."""

    def __init__(self, vae, device):
        self.vae, self.device = vae, device
        self.mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device)
        self.std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device)

    @staticmethod
    def to_tensor(img: np.ndarray) -> torch.Tensor:
        """uint8 ``(H, W, 3)`` -> float ``(1, 3, H, W)`` in [-1, 1]."""
        return torch.from_numpy(np.asarray(img).astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]

    @staticmethod
    def patchify(lat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = lat.shape
        lat = lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
        return lat.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def unpatchify(lat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = lat.shape
        lat = lat.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
        return lat.reshape(b, c // 4, h * 2, w * 2)

    def encode_tensor(self, px: torch.Tensor) -> torch.Tensor:
        lat = self.vae.encode(px.to(self.device, self.vae.dtype)).latent_dist.mode()
        return (self.patchify(lat) - self.mean) / self.std

    def encode(self, img: np.ndarray) -> torch.Tensor:
        return self.encode_tensor(self.to_tensor(img))

    def decode_tokens(self, tokens: torch.Tensor, h: int, w: int) -> np.ndarray:
        """``(1, h*w, 128)`` tokens -> uint8 ``(16h, 16w, 3)`` image."""
        lat = unpack(tokens, h, w) * self.std + self.mean
        img = self.vae.decode(self.unpatchify(lat).to(self.vae.dtype), return_dict=False)[0]
        return ((img[0].float().permute(1, 2, 0).clamp(-1, 1).cpu().numpy() + 1) * 127.5).astype(np.uint8)


@dataclass
class ViewCondition:
    """Conditioning of one camera view.

    cond_latent: ``(1, 128, h, w)`` latent of the conditioned frame Phi(o; M, S).
    mask: ``(1, 1, 16h, 16w)`` pixel mask in [0, 1].
    reference_tokens: ``(1, h*w, 128)`` reference tokens (an encoded image, or the previous
        keyframe's denoised tokens).
    """
    cond_latent: torch.Tensor
    mask: torch.Tensor
    reference_tokens: torch.Tensor


@torch.no_grad()
def denoise(transformer, scheduler, views: list[ViewCondition], prompt_embeds, text_ids,
            neg_embeds=None, neg_text_ids=None, steps: int = 30, guidance_scale: float = 3.5,
            generator: torch.Generator | None = None) -> list[torch.Tensor]:
    """Jointly denoise one or more views. Returns one ``(1, h*w, 128)`` token tensor per view."""
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
    device, dtype = prompt_embeds.device, transformer.dtype
    n = len(views)
    _, _, h, w = views[0].cond_latent.shape
    S = h * w
    cond = [torch.cat([pack(v.cond_latent.float()),
                       pack(fold_mask_to_latent(v.mask.to(device, torch.float32)))], dim=-1).to(dtype)
            for v in views]
    refs = [pad_condition_tokens(v.reference_tokens.to(dtype), FILL_IN) for v in views]
    ids = fill_ids(h, w, n).to(device)
    lats = [torch.randn((1, S, BASE_IN), generator=generator, device=device, dtype=dtype)
            for _ in range(n)]

    use_cfg = guidance_scale is not None and guidance_scale > 1.0 and neg_embeds is not None
    sigmas = np.linspace(1.0, 1 / steps, steps)
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=compute_empirical_mu(S * n, steps))
    scheduler.set_begin_index(0)
    for t in scheduler.timesteps:
        hidden = torch.cat([torch.cat([lats[i], cond[i]], dim=-1) for i in range(n)] + refs, dim=1)
        tt = t.expand(1).to(dtype) / 1000
        pred = transformer(hidden_states=hidden, timestep=tt, guidance=None,
                           encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=ids,
                           return_dict=False)[0]
        if use_cfg:
            neg = transformer(hidden_states=hidden, timestep=tt, guidance=None,
                              encoder_hidden_states=neg_embeds, txt_ids=neg_text_ids, img_ids=ids,
                              return_dict=False)[0]
            pred = neg + guidance_scale * (pred - neg)
        # One Euler step per view against the same t: rewind the scheduler's step counter.
        idx = scheduler.step_index
        for i in range(n):
            scheduler._step_index = idx
            lats[i] = scheduler.step(pred[:, i * S:(i + 1) * S], t, lats[i],
                                     return_dict=False)[0].to(dtype)
    return lats

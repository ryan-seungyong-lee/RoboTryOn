"""Channel-concatenated inpainting conditioning for the FLUX.2 [klein] transformer.

The conditioned frame and the mask are pixel-aligned with the target, so they ride the CHANNEL
axis of the target tokens (FLUX-Fill style)::

    target token = [ noisy latent (128) | conditioned-frame latent (128) | folded mask (256) ]

The patch embedding ``x_embedder`` is ``Linear(128, inner_dim)`` in the base model and is applied
once to the whole token sequence, so it is widened to 512 input features. The new columns are
zero-initialised: before training the widened model reproduces the base model exactly, and a
conditioning stream it cannot use degrades to "ignored" rather than to noise. The widened
embedding is always trained in full (FFT and LoRA).

Reference tokens are spatially unaligned with the target and ride the SEQUENCE axis instead; they
carry only the 128 real latent channels and are zero-padded to 512, which contributes exactly
what the base model would because the new columns start at zero.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_IN = 128          # 32 VAE latent channels x 2x2 patchify
MASK_IN = 256          # 8x8 pixel fold -> 64 channels at H/8, then 2x2 patchify
FILL_IN = BASE_IN + BASE_IN + MASK_IN      # 512


def widen_x_embedder(transformer, new_in: int = FILL_IN) -> nn.Linear:
    """Grow ``transformer.x_embedder`` in place (old columns kept, new ones zero). Idempotent.

    Also records the new width in the model config (``in_channels=512, out_channels=128``) so a
    full checkpoint written with ``save_pretrained`` reloads with ``from_pretrained`` directly.
    """
    old: nn.Linear = transformer.x_embedder
    if old.in_features == new_in:
        return old
    if old.in_features != BASE_IN:
        raise ValueError(f"x_embedder has {old.in_features} inputs, expected {BASE_IN} or {new_in}")
    new = nn.Linear(new_in, old.out_features, bias=old.bias is not None,
                    device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :BASE_IN].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    transformer.x_embedder = new
    out_ch = transformer.config.out_channels or transformer.config.in_channels
    transformer.register_to_config(in_channels=new_in, out_channels=out_ch)
    return new


def fold_mask_to_latent(mask: torch.Tensor, vae_scale: int = 8) -> torch.Tensor:
    """``(B, 1, H, W)`` mask in [0, 1] -> ``(B, 256, H/16, W/16)``.

    The mask is not interpolated: each 8x8 pixel block is folded into 64 channels (as in FLUX
    Fill) so sub-latent edge precision survives, then the same 2x2 patchify as the latents.
    """
    b, c, h, w = mask.shape
    if c != 1:
        raise ValueError(f"expected a single-channel mask, got {c}")
    if h % (vae_scale * 2) or w % (vae_scale * 2):
        raise ValueError(f"mask {h}x{w} must be a multiple of {vae_scale * 2}")
    lh, lw = h // vae_scale, w // vae_scale
    m = mask.view(b, lh, vae_scale, lw, vae_scale)
    m = m.permute(0, 2, 4, 1, 3).reshape(b, vae_scale * vae_scale, lh, lw)
    m = m.view(b, vae_scale * vae_scale, lh // 2, 2, lw // 2, 2)
    return m.permute(0, 1, 3, 5, 2, 4).reshape(b, vae_scale * vae_scale * 4, lh // 2, lw // 2)


def pad_condition_tokens(packed: torch.Tensor, new_in: int = FILL_IN) -> torch.Tensor:
    """Zero-pad ``(B, N, 128)`` reference tokens to the widened embedding width."""
    if packed.shape[-1] == new_in:
        return packed
    return F.pad(packed, (0, new_in - packed.shape[-1]))


def pack(latents: torch.Tensor) -> torch.Tensor:
    """``(B, C, h, w)`` -> ``(B, h*w, C)`` tokens (same as ``Flux2KleinPipeline._pack_latents``)."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1)


def unpack(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Inverse of :func:`pack`."""
    b, n, c = tokens.shape
    return tokens.permute(0, 2, 1).reshape(b, c, h, w)


def fill_tokens(noisy: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Target tokens ``[noisy | cond | mask]`` from latents ``(B,128,h,w)`` and pixel mask ``(B,1,H,W)``."""
    return torch.cat([pack(noisy), pack(cond), pack(fold_mask_to_latent(mask.to(cond.dtype)))], dim=-1)

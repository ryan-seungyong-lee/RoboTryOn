"""Assembling one training step of the in-context video model g_phi (paper Eq. 3.3).

A training sequence holds three groups of tokens, in the order of diffusers'
``LTX2InContextPipeline``::

    [ target tokens | keyframe tokens | skeleton-flow reference tokens ]

* **target**: the clip ``w = E_vid(o_{1:T})``, noised by flow matching to ``w_tau``. The model
  predicts the velocity ``eps - w`` on these tokens.
* **keyframes** ``o_T``: each keyframe is one pixel frame, VAE-encoded on its own (exactly what
  inference can build from a PNG), clean (timestep 0), and placed at its true time by RoPE
  coordinates ``[f, f+1) / fps``. The keyframe at frame 0 instead *replaces* latent frame 0 of
  the target, which encodes exactly that one frame; it is excluded from the loss.
* **reference**: the rendered skeleton flow ``S_{1:T}``, VAE-encoded like the clip, clean, at
  ``1 / ref_downscale`` resolution with its spatial coordinates scaled back onto the target
  grid.

Stereo clips hold both views in every group, each view encoded separately and laid end to end
inside the group; a fourth RoPE axis (rope4d.py) tells the views apart. Mono clips use plain
three-axis coordinates. The transformer is bidirectional, so a keyframe appended at the end
of the sequence conditions every frame.
"""

from __future__ import annotations

import torch

AUDIO_TOKENS = 1        # LTX-2 is audio-video; training feeds a one-token silent audio stream


def sample_keyframe_indices(num_frames: int, count_weights: dict[int, float],
                            include_last_p: float, generator: torch.Generator | None = None
                            ) -> list[int]:
    """Keyframes of one training clip (paper App. B.2.4): 1 to 4, the first frame always.

    The count is drawn from ``count_weights``; the remaining keyframes sit at random frames.
    With probability ``include_last_p`` one of them is the clip's last frame, which is where
    inference always places the shared boundary keyframe of a chunk.
    """
    counts = sorted(count_weights)
    probs = torch.tensor([float(count_weights[c]) for c in counts])
    n = min(counts[int(torch.multinomial(probs, 1, generator=generator))], num_frames)
    idx = {0}
    if n >= 2 and torch.rand((), generator=generator) < include_last_p:
        idx.add(num_frames - 1)
    pool = torch.randperm(num_frames - 1, generator=generator) + 1     # frames 1..F-1
    for f in pool.tolist():
        if len(idx) >= n:
            break
        idx.add(f)
    return sorted(idx)


def _per_view(coords: torch.Tensor, num_views: int) -> list[torch.Tensor]:
    if num_views == 1:
        return [coords]                       # mono: no view axis, pretrained RoPE untouched
    from .rope4d import add_view_coord
    return [add_view_coord(coords, v) for v in range(num_views)]


def build_video_coords(rope, num_latent_frames: int, height: int, width: int,
                       ref_downscale: int, device, fps: float, keyframe_indices: list[int],
                       num_views: int = 1) -> torch.Tensor:
    """RoPE coordinates for ``[target | keyframes | reference]``, matching ``build_step``.

    ``height``/``width`` are one view's latent grid. Keyframe tokens are one pixel frame wide
    at ``[f, f+1)`` (already divided by fps, like ``prepare_video_coords``).
    """
    tgt = rope.prepare_video_coords(1, num_latent_frames, height, width, device=device, fps=fps)
    parts = _per_view(tgt, num_views)
    if keyframe_indices:
        one = rope.prepare_video_coords(1, 1, height, width, device=device, fps=fps)
        for f in keyframe_indices:
            c = one.clone()
            c[:, 0, :, 0] = float(f) / fps
            c[:, 0, :, 1] = float(f + 1) / fps
            parts += _per_view(c, num_views)
    if ref_downscale == 1:
        ref = tgt.clone()
    else:
        ref = rope.prepare_video_coords(1, num_latent_frames, height // ref_downscale,
                                        width // ref_downscale, device=device, fps=fps)
        ref[:, 1:3] *= ref_downscale
    parts += _per_view(ref, num_views)
    return torch.cat(parts, dim=2)


def build_step(target: torch.Tensor, reference: torch.Tensor,
               keyframes: dict[int, torch.Tensor], sigma: torch.Tensor,
               tokens_per_frame: int, num_views: int = 1):
    """Noise the target and lay out one sequence.

    Args:
        target: ``(1, V * seq, C)`` packed clean target tokens, view-major.
        reference: ``(1, V * ref_seq, C)`` packed skeleton-flow tokens.
        keyframes: ``{pixel frame: (1, V * tokens_per_frame, C)}``, one-frame encodes.
        sigma: ``(1,)`` flow-matching noise level in [0, 1].
        tokens_per_frame: spatial tokens of one view's latent frame.

    Returns:
        ``(hidden, timestep, velocity, pinned, appended)``: the input tokens, per-token
        timesteps (``sigma * 1000`` on noised tokens, 0 on clean ones), the velocity target
        for the first ``V * seq`` outputs, a mask of target tokens replaced by the frame-0
        keyframe, and the appended keyframe indices in sequence order.
    """
    b, seq, _ = target.shape
    per_view = seq // num_views
    noise = torch.randn_like(target)
    s = sigma.view(-1, 1, 1).to(target.dtype)
    noisy = (1.0 - s) * target + s * noise
    pinned = torch.zeros(b, seq, dtype=torch.bool, device=target.device)
    if 0 in keyframes:                         # frame 0 replaces latent frame 0 of each view
        noisy = noisy.clone()
        for v in range(num_views):
            dst, src = v * per_view, v * tokens_per_frame
            noisy[:, dst:dst + tokens_per_frame] = keyframes[0][:, src:src + tokens_per_frame]
            pinned[:, dst:dst + tokens_per_frame] = True
    appended = sorted(f for f in keyframes if f != 0)
    hidden = torch.cat([noisy] + [keyframes[f] for f in appended] + [reference], dim=1)
    ts = (sigma.view(-1, 1) * 1000.0).to(target.dtype).expand(b, hidden.shape[1]).clone()
    ts[:, :seq] = torch.where(pinned, torch.zeros_like(ts[:, :seq]), ts[:, :seq])
    ts[:, seq:] = 0.0
    return hidden, ts, noise - target, pinned, appended


def flow_matching_loss(pred: torch.Tensor, velocity: torch.Tensor,
                       pinned: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and true velocity over the noised target tokens."""
    keep = (~pinned).unsqueeze(-1)
    diff = (pred.float() - velocity.float()) ** 2 * keep
    return diff.sum() / keep.expand_as(diff).sum().clamp(min=1)


def shifted_logit_normal_sigma(seq_len: int, device=None, std: float = 1.0, eps: float = 1e-3,
                               uniform_prob: float = 0.1) -> torch.Tensor:
    """Noise level for one sample, as LTX-2's trainer draws it.

    Adapted from LTX-2 ``ltx_trainer.timestep_samplers.ShiftedLogitNormalTimestepSampler``
    (Apache-2.0): a logit-normal whose mean shifts with the token count, stretched between its
    0.5th and 99.9th percentiles, mixed with 10% uniform samples.
    """
    m = (2.05 - 0.95) / (4096 - 1024)
    mu = m * seq_len + (0.95 - m * 1024)
    x = torch.sigmoid(torch.randn((1,), device=device) * std + mu)
    p999 = torch.sigmoid(torch.tensor(mu + 3.0902 * std, device=device))
    p005 = torch.sigmoid(torch.tensor(mu - 2.5758 * std, device=device))
    z = (x - p005) / (p999 - p005)
    z = torch.clamp(torch.where(z >= eps, z, 2 * eps - z), 0, 1)
    uniform = (1 - eps) * torch.rand((1,), device=device) + eps
    return torch.where(torch.rand((1,), device=device) > uniform_prob, z, uniform)

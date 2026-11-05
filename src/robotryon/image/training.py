"""Flow-matching objective of the keyframe-translation model (paper Eq. 3.1).

With ``z = E_img(o_t)`` and ``z_tau = (1 - tau) z + tau eps``, the transformer predicts
``eps - z`` from the target tokens ``[z_tau | E_img(Phi(o_t; M_t, S_t)) | fold(M_t)]`` and the
reference tokens ``E_img(o_t')``. ``tau`` is drawn uniformly from the base model's training
sigma grid, one value per sample shared by its views. For stereo samples the loss is the mean
over the two views.
"""
from __future__ import annotations

import torch

from .fill_channels import FILL_IN, fill_tokens, pack, pad_condition_tokens, unpack
from .sampler import LatentCodec
from .stereo_ids import fill_ids
from .text import text_ids_for


def flow_matching_loss(transformer, codec: LatentCodec, sched_sigmas: torch.Tensor,
                       sched_timesteps: torch.Tensor, batch: dict, device) -> torch.Tensor:
    B, V = batch["target"].shape[:2]
    with torch.no_grad():
        enc = [{k: codec.encode_tensor(batch[k][:, v].to(device))
                for k in ("target", "cond", "reference")} for v in range(V)]
    z0 = enc[0]["target"]
    _, _, h, w = z0.shape
    n = sched_timesteps.numel()
    idx = (torch.rand(B) * n).long().clamp(max=n - 1)
    sigma = sched_sigmas[idx].to(device=device, dtype=z0.dtype).view(B, 1, 1, 1)
    timestep = sched_timesteps[idx].to(device)

    targets, tokens = [], []
    for v, e in enumerate(enc):
        noise = torch.randn_like(e["target"])
        noisy = (1.0 - sigma) * e["target"] + sigma * noise
        tokens.append(fill_tokens(noisy, e["cond"], batch["mask"][:, v].to(device)))
        targets.append(noise - e["target"])
    tokens += [pad_condition_tokens(pack(e["reference"]), FILL_IN) for e in enc]
    hidden = torch.cat(tokens, dim=1).to(torch.bfloat16)
    ids = fill_ids(h, w, V).to(device).expand(B, -1, -1)
    prompt_embeds = batch["prompt_embeds"].to(device)
    pred = transformer(hidden_states=hidden, timestep=timestep / 1000, guidance=None,
                       encoder_hidden_states=prompt_embeds, txt_ids=text_ids_for(prompt_embeds),
                       img_ids=ids, return_dict=False)[0]
    S = h * w
    losses = [((unpack(pred[:, v * S:(v + 1) * S], h, w).float() - targets[v].float()) ** 2)
              .reshape(B, -1).mean(1).mean() for v in range(V)]
    return sum(losses) / V

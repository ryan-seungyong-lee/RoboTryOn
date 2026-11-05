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
# Adapted from GR00T-Dreams (Apache-2.0): gr00t/model/idm.py,
# gr00t/model/action_head/flow_matching_action_head_idm.py, multimodal_projector.py.
# Changes: Depth Anything V2 vision tower instead of SigLIP, no hydra, plain safetensors I/O,
# image tokens placed from a per-sample image count instead of token-id arrays.
"""Inverse dynamics model h_psi (paper Sec. 3.4, App. B.2.6).

Given two frames o_t and o_{t+K} (K = 20, 1 s at 20 fps; one or two camera views each), the IDM
predicts the action chunk a_{t:t+K} by flow matching:

    images --DINOv2 trunk of Depth Anything V2-L--> 16x16 patch tokens (CLS dropped)
           --linear + 2x(2x2 pool MLP)--> 16 tokens per image, + view embedding
    tokens --self-attention transformer--> visual context (left zero-padded to 112 tokens)
    noisy chunk a^tau --per-embodiment action encoder--> DiT (AdaLN on tau, cross-attention to
           the context) --per-embodiment decoder--> velocity v;  loss ||v - (a - eps)||^2

One model is shared by all embodiments. Actions are zero-padded to ``action_dim`` (the widest
robot) and the padded dimensions are masked out of the input and the loss; the action encoder
and decoder have separate weights per embodiment id. At inference the chunk is integrated
from Gaussian noise with ``num_inference_timesteps`` Euler steps.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from .dit import DiT, SelfAttentionTransformer

DEFAULT_MODEL_CFG = dict(
    vision_tower="depth-anything/Depth-Anything-V2-Large-hf",
    vision_hidden_size=1024, hidden_size=1024,
    action_dim=66, action_horizon=20,
    max_num_embodiments=32, max_num_views=6, max_seq_len=1024,
    max_vl_tokens=112, tokens_per_image=16,
    frozen_vision_layers=[11],
    num_inference_timesteps=16, num_timestep_buckets=1000,
    noise_beta_alpha=1.5, noise_beta_beta=1.0, noise_s=0.999,
    dit=dict(num_layers=8, num_attention_heads=16, attention_head_dim=64, dropout=0.2,
             output_dim=1024, interleave_self_attention=True),
    vl_self_attention=dict(num_layers=4, num_attention_heads=16, attention_head_dim=64,
                           dropout=0.2),
)


# ----------------------------------------------------------------------------- vision tower
class DepthAnythingTower(nn.Module):
    """The DINOv2 trunk of Depth Anything V2 (the DPT neck and depth head are discarded).
    Input: ImageNet-normalised 224x224 images. Output: (N, 256, D) post-layernorm patch tokens
    of the last layer, CLS token removed."""

    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation

        if pretrained:
            model = DepthAnythingForDepthEstimation.from_pretrained(name)
        else:
            model = DepthAnythingForDepthEstimation._from_config(DepthAnythingConfig.from_pretrained(name))
        self.backbone = model.backbone
        # Only read with bool_masked_pos, which the backbone never passes: no gradient (DDP).
        self.backbone.embeddings.mask_token.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values).feature_maps[-1][:, 1:]


class _DownSample(nn.Module):
    """Concatenate 2x2 neighbouring tokens of a square grid: (N, h*w, C) -> (N, h*w/4, 4C)."""

    def forward(self, x):
        n, L, c = x.shape
        h = w = int(L ** 0.5)
        x = x.reshape(n, h, w, c)
        if w % 2 or h % 2:
            x = F.pad(x, (0, 0, 0, h % 2, 0, w % 2))
            n, w, h, c = x.shape
        x = x.view(n, w, h // 2, c * 2).permute(0, 2, 1, 3).contiguous()
        return x.view(n, h // 2, w // 2, c * 4).reshape(n, -1, c * 4)


class MultimodalProjector(nn.Module):
    """``mlp_doubledownsample``: 256 patch tokens -> 64 -> 16 tokens."""

    def __init__(self, mm_hidden: int, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(
            _DownSample(), nn.LayerNorm(mm_hidden * 4), nn.Linear(mm_hidden * 4, mm_hidden),
            nn.GELU(), nn.Linear(mm_hidden, mm_hidden),
            _DownSample(), nn.LayerNorm(mm_hidden * 4), nn.Linear(mm_hidden * 4, hidden),
            nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, x):
        return self.layers(x)


# ----------------------------------------------------------------------------- action head
class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        return torch.bmm(x, self.W[cat_ids]) + self.b[cat_ids].unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        return self.layer2(F.relu(self.layer1(x, cat_ids)), cat_ids)


def sinusoidal(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """(B, T) -> (B, T, dim) sin/cos encoding."""
    timesteps = timesteps.float()
    half = dim // 2
    exponent = -torch.arange(half, dtype=torch.float, device=timesteps.device) * (
        torch.log(torch.tensor(10000.0)) / half)
    freqs = timesteps.unsqueeze(-1) * exponent.exp()
    return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        B, T, _ = actions.shape
        a_emb = self.W1(actions, cat_ids)
        tau = sinusoidal(timesteps.unsqueeze(1).expand(-1, T), self.hidden_size).to(a_emb.dtype)
        x = self.W2(torch.cat([a_emb, tau], dim=-1), cat_ids)
        return self.W3(x * torch.sigmoid(x), cat_ids)


class IDM(nn.Module):
    """Flow-matching inverse dynamics model. See the module docstring.

    Batch (see ``data.collate``): ``images`` (N, 3, 224, 224) for all samples concatenated in
    order [t0 view0, t0 view1, t1 view0, ...]; ``view_ids`` (N,); ``n_images`` (B,);
    ``embodiment_id`` (B,); ``actions_mask`` (B, H, A) bool, True on the robot's native
    dimensions; ``actions`` (B, H, A) normalised targets (training only).
    """

    def __init__(self, cfg: dict | None = None, pretrained_vision: bool = True):
        super().__init__()
        c = {**DEFAULT_MODEL_CFG, **(cfg or {})}
        self.cfg = c
        D = c["hidden_size"]
        self.vision = DepthAnythingTower(c["vision_tower"], pretrained=pretrained_vision)
        self.vision_projector = nn.Linear(c["vision_hidden_size"], D)
        self.mm_projector = MultimodalProjector(c["vision_hidden_size"], D)
        self.view_embedding = nn.Embedding(c["max_num_views"], D)
        self.position_embedding = nn.Embedding(c["max_seq_len"], D)
        nn.init.normal_(self.view_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        self.vl_self_attention = SelfAttentionTransformer(**c["vl_self_attention"])
        self.dit = DiT(**c["dit"])
        self.action_encoder = MultiEmbodimentActionEncoder(c["action_dim"], D, c["max_num_embodiments"])
        self.action_decoder = CategorySpecificMLP(c["max_num_embodiments"], D, D, c["action_dim"])
        self.beta_dist = Beta(c["noise_beta_alpha"], c["noise_beta_beta"])
        # One ViT layer is kept frozen, as in the trained checkpoint.
        for i in c["frozen_vision_layers"]:
            self.vision.backbone.encoder.layer[i].requires_grad_(False)

    # -- encoding -------------------------------------------------------------------------
    def encode(self, batch) -> torch.Tensor:
        """Visual context (B, max_vl_tokens, D): each sample's image tokens right-aligned,
        zeros in front (the zeros are attended to, exactly as in training)."""
        x = self.mm_projector(self.vision_projector(self.vision(batch["images"])))
        x = x + self.view_embedding(batch["view_ids"]).unsqueeze(1)
        n_tok = batch["n_images"] * self.cfg["tokens_per_image"]
        L = self.cfg["max_vl_tokens"]
        assert int(n_tok.max()) <= L, f"{int(n_tok.max())} image tokens > max_vl_tokens={L}"
        pos = torch.arange(L, device=x.device)
        mask = (pos[None] >= (L - n_tok)[:, None]).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        vl = torch.zeros(len(n_tok), L, x.shape[-1], dtype=x.dtype, device=x.device)
        return self.vl_self_attention(vl.masked_scatter(mask, x))

    def _velocity(self, context, actions, t_disc, t_dit, emb_id):
        sa = self.action_encoder(actions, t_disc, emb_id)
        T = sa.shape[1]
        sa = sa + self.position_embedding(torch.arange(T, device=sa.device)).unsqueeze(0)
        out = self.dit(sa, context, t_dit)
        return self.action_decoder(out, emb_id)[:, -T:]

    # -- training -------------------------------------------------------------------------
    def forward(self, batch) -> torch.Tensor:
        """Flow-matching loss over the native (unpadded) action dimensions."""
        context = self.encode(batch)
        actions, mask = batch["actions"], batch["actions_mask"].to(batch["actions"].dtype)
        noise = torch.randn_like(actions)
        s = self.cfg["noise_s"]
        t = (s - self.beta_dist.sample([actions.shape[0]]).to(actions.device, actions.dtype)) / s
        t = t[:, None, None]
        noisy = ((1 - t) * noise + t * actions) * mask
        t_disc = (t[:, 0, 0] * self.cfg["num_timestep_buckets"]).long()
        pred = self._velocity(context, noisy, t_disc, t_disc, batch["embodiment_id"])
        loss = F.mse_loss(pred, actions - noise, reduction="none") * mask
        return loss.sum() / mask.sum()

    # -- inference ------------------------------------------------------------------------
    @torch.inference_mode()
    def get_action(self, batch, generator: torch.Generator | None = None) -> torch.Tensor:
        """Integrate the flow from noise; returns normalised, padded actions (B, H, A)."""
        emb_id = batch["embodiment_id"]
        images = batch["images"]
        B, H, A = len(emb_id), self.cfg["action_horizon"], self.cfg["action_dim"]
        x = torch.randn((B, H, A), dtype=images.dtype, device=images.device, generator=generator)
        mask = batch["actions_mask"].to(x.dtype)
        x = x * mask
        context = self.encode(batch)
        n = self.cfg["num_inference_timesteps"]
        for i in range(n):
            t = int(i / float(n) * self.cfg["num_timestep_buckets"])
            v = self._velocity(context, x, (torch.ones(B) * t).to(x.device),
                               torch.tensor([t], device=x.device, dtype=torch.long), emb_id)
            x = (x + (1.0 / n) * v) * mask
        return x

    # -- I/O ------------------------------------------------------------------------------
    def save_pretrained(self, out_dir: str | Path, metadata: dict | None = None) -> None:
        from safetensors.torch import save_file

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
        (out / "idm_config.json").write_text(json.dumps(self.cfg, indent=1))
        if metadata is not None:
            (out / "metadata.json").write_text(json.dumps(metadata, indent=1))

    @classmethod
    def from_pretrained(cls, ckpt_dir: str | Path, device="cpu") -> "IDM":
        """Load a checkpoint written by ``save_pretrained`` or by the GR00T-Dreams IDM trainer."""
        from safetensors.torch import load_file

        d = Path(ckpt_dir)
        if (d / "idm_config.json").exists():
            cfg = json.loads((d / "idm_config.json").read_text())
        else:
            cfg = _legacy_config(json.loads((d / "config.json").read_text()))
        model = cls(cfg, pretrained_vision=False)
        sd = load_file(str(d / "model.safetensors"))
        if any(k.startswith("action_head.") for k in sd):
            sd = _legacy_state_dict(sd)
        model.load_state_dict(sd, strict=True)
        return model.to(device).eval()


def load_metadata(ckpt_dir: str | Path) -> dict:
    """Per-embodiment normalisation statistics saved with the checkpoint, keyed by name."""
    d = Path(ckpt_dir)
    for p in (d / "metadata.json", d / "experiment_cfg" / "metadata.json"):
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"no metadata.json in {d}")


def _legacy_config(c: dict) -> dict:
    h = c["action_head_cfg"]["config"]
    keep = ("num_layers", "num_attention_heads", "attention_head_dim", "dropout")
    return dict(
        vision_tower=h["siglip_model_cfg"]["pretrained_model_name_or_path"],
        vision_hidden_size=h["siglip_hidden_size"], hidden_size=h["hidden_size"],
        action_dim=h["action_dim"], action_horizon=h["action_horizon"],
        max_num_embodiments=h.get("max_num_embodiments", 32), max_num_views=h["max_num_views"],
        max_seq_len=h.get("max_seq_len", 1024),
        num_inference_timesteps=h["num_inference_timesteps"],
        num_timestep_buckets=h["num_timestep_buckets"],
        noise_beta_alpha=h["noise_beta_alpha"], noise_beta_beta=h["noise_beta_beta"],
        noise_s=h["noise_s"],
        dit={**{k: h["diffusion_model_cfg"][k] for k in keep},
             "output_dim": h["diffusion_model_cfg"]["output_dim"],
             "interleave_self_attention": h["diffusion_model_cfg"]["interleave_self_attention"]},
        vl_self_attention={k: h["vl_self_attention_cfg"][k] for k in keep},
    )


_LEGACY_RENAMES = (
    (r"^action_head\.siglip_model\.vision_model\.backbone\.", "vision.backbone."),
    (r"^action_head\.model\.", "dit."),
    (r"^action_head\.vl_self_attention_model\.", "vl_self_attention."),
    (r"^action_head\.", ""),
)


def _legacy_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.endswith(("siglip_model.logit_scale", "siglip_model.logit_bias")):
            continue
        for pat, rep in _LEGACY_RENAMES:
            k2 = re.sub(pat, rep, k)
            if k2 != k:
                k = k2
                break
        out[k] = v
    return out

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
# Adapted from GR00T-Dreams (Apache-2.0), gr00t/model/action_head/cross_attention_dit.py.
# Changes: plain nn.Module instead of diffusers ModelMixin/ConfigMixin, unused options removed.
# Parameter names are unchanged so GR00T-Dreams IDM checkpoints load as-is.
"""Diffusion transformer (DiT) of the flow-matching action head, and the self-attention
transformer that mixes the visual tokens before the DiT cross-attends to them."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, False)

    def forward(self, x, temb):
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    """Attention (self, or cross when encoder states are given) + feed-forward, pre-norm."""

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,
                 dropout: float = 0.0, activation_fn: str = "geglu", attention_bias: bool = False,
                 upcast_attention: bool = False, norm_elementwise_affine: bool = True,
                 norm_type: str = "layer_norm", norm_eps: float = 1e-5, final_dropout: bool = False):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim,
                               dropout=dropout, bias=attention_bias, cross_attention_dim=None,
                               upcast_attention=upcast_attention, out_bias=True)
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn,
                              final_dropout=final_dropout, inner_dim=None, bias=True)
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(self, hidden_states, encoder_hidden_states=None, temb=None):
        if self.norm_type == "ada_norm":
            h = self.norm1(hidden_states, temb)
        else:
            h = self.norm1(hidden_states)
        attn = self.attn1(h, encoder_hidden_states=encoder_hidden_states, attention_mask=None)
        if self.final_dropout:
            attn = self.final_dropout(attn)
        hidden_states = attn + hidden_states
        return self.ff(self.norm3(hidden_states)) + hidden_states


class DiT(nn.Module):
    """Action-token transformer conditioned on the flow time (AdaLN) and on the visual tokens.
    With ``interleave_self_attention`` the odd blocks are self-attention only."""

    def __init__(self, num_attention_heads: int = 16, attention_head_dim: int = 64,
                 output_dim: int = 1024, num_layers: int = 8, dropout: float = 0.2,
                 attention_bias: bool = True, activation_fn: str = "gelu-approximate",
                 upcast_attention: bool = False, norm_type: str = "ada_norm",
                 norm_elementwise_affine: bool = False, norm_eps: float = 1e-5,
                 final_dropout: bool = True, interleave_self_attention: bool = True):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.interleave_self_attention = interleave_self_attention
        self.timestep_encoder = TimestepEncoder(self.inner_dim)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim,
                                  dropout=dropout, activation_fn=activation_fn,
                                  attention_bias=attention_bias, upcast_attention=upcast_attention,
                                  norm_type=norm_type, norm_elementwise_affine=norm_elementwise_affine,
                                  norm_eps=norm_eps, final_dropout=final_dropout)
            for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        for idx, block in enumerate(self.transformer_blocks):
            cross = not (idx % 2 == 1 and self.interleave_self_attention)
            hidden_states = block(hidden_states, encoder_hidden_states if cross else None, temb)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)


class SelfAttentionTransformer(nn.Module):
    def __init__(self, num_attention_heads: int = 16, attention_head_dim: int = 64,
                 num_layers: int = 4, dropout: float = 0.2, attention_bias: bool = True,
                 activation_fn: str = "gelu-approximate", upcast_attention: bool = False,
                 final_dropout: bool = True):
        super().__init__()
        inner = num_attention_heads * attention_head_dim
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner, num_attention_heads, attention_head_dim, dropout=dropout,
                                  activation_fn=activation_fn, attention_bias=attention_bias,
                                  upcast_attention=upcast_attention, final_dropout=final_dropout)
            for _ in range(num_layers)])

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None):
        hidden_states = hidden_states.contiguous()
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states

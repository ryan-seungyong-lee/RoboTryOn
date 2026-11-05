"""The on-disk text-embedding cache of the image model's prompts (``robotryon.captions``).

Qwen3 prompt embeddings are ``(512, 12288)`` per prompt, too large to keep for every prompt in
memory, so they are computed once at trainer start-up and stored as one safetensors file per
prompt, named by a hash of the text.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MAX_SEQUENCE_LENGTH = 512


class TextEmbedCache:
    """One ``<sha1>.safetensors`` file per prompt holding ``prompt_embeds`` ``(L, D)``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, prompt: str) -> Path:
        return self.root / f"{hashlib.sha1(prompt.encode('utf-8')).hexdigest()}.safetensors"

    def missing(self, prompts) -> list[str]:
        return sorted({p for p in prompts if not self.path(p).exists()})

    def load(self, prompt: str) -> torch.Tensor:
        p = self.path(prompt)
        if not p.exists():
            raise FileNotFoundError(f"no cached embedding for prompt {prompt!r}")
        return load_file(str(p))["prompt_embeds"]

    @torch.no_grad()
    def fill(self, text_pipe, prompts, device, batch_size: int = 16) -> int:
        """Encode and store the given prompts with a ``Flux2KleinPipeline`` holding the text encoder."""
        self.root.mkdir(parents=True, exist_ok=True)
        prompts = list(prompts)
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            emb, _ = text_pipe.encode_prompt(prompt=chunk, device=device,
                                             max_sequence_length=MAX_SEQUENCE_LENGTH)
            for p, e in zip(chunk, emb):
                tmp = self.path(p).with_suffix(".tmp")
                save_file({"prompt_embeds": e.to(torch.bfloat16).contiguous().cpu()}, str(tmp))
                tmp.replace(self.path(p))
        return len(prompts)


def text_ids_for(prompt_embeds: torch.Tensor) -> torch.Tensor:
    """``(B, L, D)`` -> ``(B, L, 4)`` text position ids (same as ``Flux2KleinPipeline._prepare_text_ids``)."""
    b, n, _ = prompt_embeds.shape
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(n))
    return ids.unsqueeze(0).expand(b, -1, -1).to(prompt_embeds.device)

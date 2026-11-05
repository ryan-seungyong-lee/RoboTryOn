"""Training plumbing for the video LoRA: gradient checkpointing under DDP and LoRA weight I/O.

diffusers' own (non-reentrant) checkpointing does not reproduce the LTX-2 block on recompute
under DDP, so each transformer block is wrapped in a reentrant checkpoint here. Reentrant
checkpointing needs the block inputs to require grad, hence the hook on ``proj_in``, and pairs
with DDP ``static_graph``. Wrapping inserts ``.block.`` into parameter names; saved LoRA
weights strip it so they load into a plain transformer.
"""

from __future__ import annotations

import re

import torch
from torch import nn

LORA_TARGETS = ["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
                "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
                "ff.net.0.proj", "ff.net.2"]              # video modules only (official IC-LoRA)


class CheckpointedBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        if not self.training or not torch.is_grad_enabled():
            return self.block(*args, **kwargs)
        keys = list(kwargs)

        def run(*pos):
            return self.block(*pos[:len(args)], **dict(zip(keys, pos[len(args):])))
        return torch.utils.checkpoint.checkpoint(run, *args, *[kwargs[k] for k in keys],
                                                 use_reentrant=True)


def enable_checkpointing(transformer) -> None:
    blocks = transformer.transformer_blocks
    for i, b in enumerate(blocks):
        if not isinstance(b, CheckpointedBlock):
            blocks[i] = CheckpointedBlock(b)

    def hook(_m, _inp, out):
        if isinstance(out, torch.Tensor):
            out.requires_grad_(True)
    transformer.proj_in.register_forward_hook(hook)
    transformer.audio_proj_in.register_forward_hook(hook)


def lora_state_dict(model) -> dict:
    """PEFT LoRA weights with the checkpoint wrapper's ``.block.`` removed."""
    from peft.utils import get_peft_model_state_dict
    return {re.sub(r"(transformer_blocks\.\d+)\.block\.", r"\1.", k): v
            for k, v in get_peft_model_state_dict(model).items()}


def set_lora_state(model, want: dict) -> None:
    """Inverse of ``lora_state_dict``: load LoRA weights, re-inserting ``.block.`` if wrapped."""
    from peft import set_peft_model_state_dict
    cur = lora_state_dict(model)
    if set(want) != set(cur):
        raise ValueError(f"LoRA keys differ: {len(set(want) - set(cur))} only in file, "
                         f"{len(set(cur) - set(want))} only in model (rank or targets changed?)")
    wrapped = any(".block." in n for n, _ in model.named_parameters())
    set_peft_model_state_dict(model, {re.sub(r"(transformer_blocks\.\d+)\.", r"\1.block.", k)
                                      if wrapped else k: v for k, v in want.items()})


def load_lora_into(model, path) -> int:
    """Initialise the model's LoRA from a saved ``pytorch_lora_weights.safetensors`` (stage 2
    starts from stage 1). Verifies by value that every tensor was loaded."""
    from safetensors.torch import load_file
    want = {k.removeprefix("transformer."): v for k, v in load_file(str(path)).items()}
    set_lora_state(model, want)
    after = lora_state_dict(model)
    bad = [k for k in want if not torch.equal(after[k].float().cpu(), want[k].float().cpu())]
    if bad:
        raise RuntimeError(f"{len(bad)} LoRA tensors did not load, e.g. {bad[0]}")
    return len(want)

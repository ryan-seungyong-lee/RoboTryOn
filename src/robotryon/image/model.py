"""Build, save and load the keyframe-translation model f_theta (FLUX.2 [klein] 9B Base).

Two training modes share everything but the trainable set:

* ``fft``: the whole transformer is trained (FSDP, paper Table 4). Checkpoint:
  ``<ckpt>/transformer/`` written by ``save_pretrained`` (sharded safetensors, config records the
  widened ``in_channels=512``).
* ``lora``: LoRA adapters on the attention / fused QKV-MLP projections of every block plus the
  widened ``x_embedder`` trained in full. Checkpoint: ``<ckpt>/pytorch_lora_weights.safetensors``
  (diffusers LoRA format) and ``<ckpt>/x_embedder.safetensors``.

``load_pipeline`` recognises either checkpoint type.
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .fill_channels import FILL_IN, widen_x_embedder

LORA_FILE = "pytorch_lora_weights.safetensors"
X_EMBEDDER_FILE = "x_embedder.safetensors"
NUM_SINGLE_BLOCKS = 24


def lora_targets() -> list[str]:
    """Attention projections of the double-stream blocks, the fused QKV+MLP input projection of
    the single-stream blocks, and the single-stream attention output projections."""
    return (["to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj"]
            + [f"single_transformer_blocks.{i}.attn.to_out" for i in range(NUM_SINGLE_BLOCKS)])


def load_transformer(base: str, dtype: torch.dtype):
    from diffusers import Flux2Transformer2DModel
    return Flux2Transformer2DModel.from_pretrained(base, subfolder="transformer", torch_dtype=dtype)


def build_trainable(base: str, mode: str, lora_cfg=None):
    """Transformer ready for training (x_embedder widened, trainable set marked).

    fft: loaded in fp32 (FSDP keeps fp32 master shards and computes in bf16), all parameters
    trainable. lora: base weights frozen in bf16, adapters and x_embedder trainable in fp32.
    """
    if mode == "fft":
        tr = load_transformer(base, torch.float32)
        widen_x_embedder(tr, FILL_IN)
        tr.requires_grad_(True)
    elif mode == "lora":
        from peft import LoraConfig
        tr = load_transformer(base, torch.bfloat16)
        tr.requires_grad_(False)
        tr.add_adapter(LoraConfig(r=int(lora_cfg.rank), lora_alpha=int(lora_cfg.alpha),
                                  lora_dropout=float(lora_cfg.get("dropout", 0.0)),
                                  init_lora_weights="gaussian", target_modules=lora_targets()))
        widen_x_embedder(tr, FILL_IN).requires_grad_(True)
        for p in tr.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    else:
        raise ValueError(f"train.mode must be 'fft' or 'lora', got {mode!r}")
    return tr


def save_lora(transformer, out_dir: str | Path) -> None:
    """LoRA tensors in diffusers format plus the full widened x_embedder."""
    from diffusers import Flux2KleinPipeline
    from peft.utils import get_peft_model_state_dict
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lora = get_peft_model_state_dict(transformer)
    lora = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in lora.items()}
    Flux2KleinPipeline.save_lora_weights(out_dir, transformer_lora_layers=lora)
    save_file({"x_embedder.weight": transformer.x_embedder.weight.detach().float().cpu().contiguous()},
              str(out_dir / X_EMBEDDER_FILE))


def save_full(transformer, out_dir: str | Path, state_dict: dict | None = None) -> None:
    """Full transformer as a diffusers model directory ``<out_dir>/transformer`` (bf16 shards).

    ``state_dict`` is the gathered full state dict when training under FSDP (only the main
    process calls this); the unwrapped module supplies the config.
    """
    import json
    from huggingface_hub import split_torch_state_dict_into_shards
    d = Path(out_dir) / "transformer"
    d.mkdir(parents=True, exist_ok=True)
    sd = state_dict if state_dict is not None else transformer.state_dict()
    sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in sd.items()}
    split = split_torch_state_dict_into_shards(
        sd, filename_pattern="diffusion_pytorch_model{suffix}.safetensors", max_shard_size="5GB")
    for fname, keys in split.filename_to_tensors.items():
        save_file({k: sd[k] for k in keys}, str(d / fname), metadata={"format": "pt"})
    if split.is_sharded:
        index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
        (d / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    transformer.save_config(d)


def checkpoint_kind(ckpt: str | Path) -> str:
    ckpt = Path(ckpt)
    if (ckpt / LORA_FILE).exists():
        return "lora"
    if (ckpt / "transformer" / "config.json").exists():
        return "fft"
    raise FileNotFoundError(f"{ckpt} holds neither {LORA_FILE} nor transformer/config.json")


def load_pipeline(base: str, ckpt: str | Path, device: str = "cuda", text_encoder: bool = True):
    """``Flux2KleinPipeline`` in bf16 with the trained keyframe-translation weights.

    ``text_encoder=False`` skips the text encoder (prompts encoded beforehand).

    LoRA: widen ``x_embedder`` BEFORE loading the adapter (it was fitted to the 512-wide
    embedding), copy the trained embedding, then load the adapter unfused.
    """
    from diffusers import Flux2KleinPipeline
    ckpt = Path(ckpt)
    kind = checkpoint_kind(ckpt)
    kw = {} if text_encoder else {"text_encoder": None, "tokenizer": None}
    if kind == "fft":
        from diffusers import Flux2Transformer2DModel
        tr = Flux2Transformer2DModel.from_pretrained(ckpt / "transformer", torch_dtype=torch.bfloat16)
        pipe = Flux2KleinPipeline.from_pretrained(base, transformer=tr, torch_dtype=torch.bfloat16, **kw)
        pipe.to(device)
    else:
        pipe = Flux2KleinPipeline.from_pretrained(base, torch_dtype=torch.bfloat16, **kw).to(device)
        widen_x_embedder(pipe.transformer, FILL_IN)
        w = load_file(str(ckpt / X_EMBEDDER_FILE))["x_embedder.weight"]
        with torch.no_grad():
            pipe.transformer.x_embedder.weight.copy_(w.to(pipe.transformer.x_embedder.weight.dtype))
        pipe.load_lora_weights(str(ckpt), weight_name=LORA_FILE)
    if pipe.transformer.x_embedder.in_features != FILL_IN:
        raise ValueError(f"{ckpt}: x_embedder has {pipe.transformer.x_embedder.in_features} inputs")
    pipe.transformer.requires_grad_(False)
    return pipe


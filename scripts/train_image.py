#!/usr/bin/env python
"""Train the keyframe-translation model f_theta (FLUX.2 [klein] 9B Base), full fine-tune or LoRA.

    accelerate launch --num_processes 4 scripts/train_image.py \
        --config configs/image/klein9b_fft.yaml model.base=<FLUX.2-klein-base-9B dir> \
        data.roots='[/data/robot_a, /data/robot_b]' output_dir=runs/image_fft

``train.mode: fft`` shards the transformer with FSDP (full shard, bf16 compute, fp32 master
weights); ``train.mode: lora`` uses data parallelism. Global batch =
``train.batch_size`` (per GPU) x GPUs x ``train.grad_accum``.

Checkpoints go to ``<output_dir>/checkpoint-<step>/`` and ``<output_dir>/final/``:
fft -> ``transformer/`` (diffusers model dir); lora -> ``pytorch_lora_weights.safetensors`` +
``x_embedder.safetensors``. Both load with ``robotryon.image.model.load_pipeline``.

With ``train.resume`` (default) the full training state (weights, optimizer, scheduler, step) is
also kept in ``<output_dir>/resume_state/`` at every checkpoint, and a restarted run continues
from it.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import set_seed

from robotryon.common.config import parse_args, save_config
from robotryon.image.dataset import FillBatchSampler, KeyframeDataset, collate
from robotryon.image.model import build_trainable, save_full, save_lora
from robotryon.image.sampler import LatentCodec
from robotryon.image.text import TextEmbedCache
from robotryon.image.training import flow_matching_loss

BLOCKS = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]


def make_accelerator(cfg) -> Accelerator:
    plugin = None
    if cfg.train.mode == "fft":
        plugin = FullyShardedDataParallelPlugin(
            sharding_strategy="FULL_SHARD", auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=BLOCKS, state_dict_type="FULL_STATE_DICT",
            use_orig_params=True, limit_all_gathers=True)
    return Accelerator(mixed_precision="bf16", gradient_accumulation_steps=int(cfg.train.grad_accum),
                       fsdp_plugin=plugin)


def ensure_text_cache(cfg, accel: Accelerator, prompts: set[str]) -> TextEmbedCache:
    """Encode prompts missing from the cache, split across ranks."""
    cache = TextEmbedCache(cfg.text.cache_dir)
    todo = cache.missing(prompts)[accel.process_index::accel.num_processes]
    if todo:
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(cfg.model.base, transformer=None, vae=None,
                                                  torch_dtype=torch.bfloat16).to(accel.device)
        cache.fill(pipe, todo, accel.device)
        del pipe
        torch.cuda.empty_cache()
    accel.wait_for_everyone()
    return cache


def save(cfg, accel, transformer, out: Path) -> None:
    if cfg.train.mode == "fft":
        sd = accel.get_state_dict(transformer)          # every rank must take part in the gather
        if accel.is_main_process:
            save_full(accel.unwrap_model(transformer), out, sd)
    elif accel.is_main_process:
        save_lora(accel.unwrap_model(transformer), out)
    accel.wait_for_everyone()
    if accel.is_main_process:
        print(f"[train] saved {out}", flush=True)


def save_resume_state(accel, out_dir: Path, step: int) -> None:
    """Write the full training state to ``resume_state/``, replacing the previous one atomically."""
    tmp, final = out_dir / "resume_state.tmp", out_dir / "resume_state"
    if accel.is_main_process and tmp.exists():
        shutil.rmtree(tmp)
    accel.wait_for_everyone()
    accel.save_state(str(tmp))
    accel.wait_for_everyone()
    if accel.is_main_process:
        (tmp / "step.json").write_text(json.dumps({"step": step}))
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
    accel.wait_for_everyone()


def main():
    cfg = parse_args(__doc__)
    accel = make_accelerator(cfg)
    set_seed(int(cfg.train.seed) + accel.process_index)
    out_dir = Path(cfg.output_dir)
    if accel.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")
        g = cfg.train.batch_size * accel.num_processes * cfg.train.grad_accum
        print(f"[train] mode={cfg.train.mode} gpus={accel.num_processes} global batch={g}", flush=True)

    dataset = KeyframeDataset(cfg, None)
    dataset.text_cache = ensure_text_cache(cfg, accel, dataset.all_prompts())

    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from diffusers.optimization import get_scheduler
    vae = AutoencoderKLFlux2.from_pretrained(cfg.model.base, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.requires_grad_(False)
    codec = LatentCodec(vae.to(accel.device), accel.device)
    fm = FlowMatchEulerDiscreteScheduler.from_pretrained(cfg.model.base, subfolder="scheduler")

    transformer = build_trainable(cfg.model.base, cfg.train.mode, cfg.get("lora"))
    if cfg.train.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer = accel.prepare(transformer)

    # Param groups by name after wrapping (FSDP flattens modules but keeps original names).
    xe = [p for n, p in transformer.named_parameters() if p.requires_grad and "x_embedder" in n]
    rest = [p for n, p in transformer.named_parameters() if p.requires_grad and "x_embedder" not in n]
    o = cfg.optim
    xe_lr = float(o.x_embedder_lr) if o.get("x_embedder_lr") is not None else float(o.lr)
    optimizer = torch.optim.AdamW([{"params": rest, "lr": float(o.lr)}, {"params": xe, "lr": xe_lr}],
                                  betas=tuple(o.betas), weight_decay=float(o.weight_decay), eps=float(o.eps))
    optimizer = accel.prepare(optimizer)
    max_steps = int(cfg.train.max_steps)
    lr_sched = get_scheduler(o.schedule, optimizer=optimizer, num_warmup_steps=int(o.warmup_steps),
                             num_training_steps=max_steps)
    accel.register_for_checkpointing(lr_sched)

    step, state_dir = 0, out_dir / "resume_state"
    if cfg.train.get("resume", True) and (state_dir / "step.json").exists():
        accel.load_state(str(state_dir))
        step = int(json.loads((state_dir / "step.json").read_text())["step"])
        accel.print(f"[train] resumed from {state_dir} at step {step}")

    # the data order continues with a fresh seed after a resume
    sampler = FillBatchSampler(dataset, cfg.train.batch_size, cfg.data.stereo_p,
                               int(cfg.train.seed) + step, accel.process_index)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
                                         num_workers=int(cfg.train.num_workers), pin_memory=True,
                                         persistent_workers=int(cfg.train.num_workers) > 0)

    transformer.train()
    start, t0, it = step, time.time(), iter(loader)
    while step < max_steps:
        batch = next(it)
        with accel.accumulate(transformer):
            loss = flow_matching_loss(transformer, codec, fm.sigmas, fm.timesteps, batch, accel.device)
            accel.backward(loss)
            if accel.sync_gradients:
                accel.clip_grad_norm_(transformer.parameters(), float(o.max_grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if not accel.sync_gradients:
            continue
        lr_sched.step()
        step += 1
        if accel.is_main_process and (step % int(cfg.train.log_every) == 0 or step == 1):
            print(f"[train] step {step}/{max_steps} loss {loss.item():.4f} "
                  f"lr {lr_sched.get_last_lr()[0]:.2e} views {batch['target'].shape[1]} "
                  f"{(time.time() - t0) / (step - start):.2f}s/step", flush=True)
        if step % int(cfg.train.checkpoint_every) == 0 and step < max_steps:
            save(cfg, accel, transformer, out_dir / f"checkpoint-{step}")
            if cfg.train.get("resume", True):
                save_resume_state(accel, out_dir, step)
    save(cfg, accel, transformer, out_dir / "final")
    accel.end_training()


if __name__ == "__main__":
    main()

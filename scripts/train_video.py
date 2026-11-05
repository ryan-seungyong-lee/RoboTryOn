"""Train the video model g_phi: an in-context LoRA on LTX-2.5 (paper Sec. 3.3, App. B.2.4).

Two stages, each one run of this script:

    accelerate launch --num_processes 4 scripts/train_video.py \
        --config configs/video/ltx25_iclora_stage1.yaml data.roots=[path/to/robot_data]
    accelerate launch --num_processes 4 scripts/train_video.py \
        --config configs/video/ltx25_iclora_stage2.yaml data.roots=[path/to/robot_data] \
        lora.init=outputs/video/stage1/checkpoint-004000/pytorch_lora_weights.safetensors

Stage 1 trains 4k steps at 512x288 per view, stage 2 continues 1k steps at 1024x576 from the
stage-1 LoRA (fresh optimizer). The global batch is ``num_processes * grad_accum`` clips of
batch size 1 (4 x 4 = 16). Caption embeddings are computed once at start-up and cached
at ``text.embeds``.

Each step: sample a clip and its keyframes, VAE-encode target, keyframes and rendered skeleton
flow per view, noise the target (shifted logit-normal sigma), and regress the velocity
(``robotryon.video.conditioning``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from robotryon.common.config import parse_args, save_config
from robotryon.skeleton import SkeletonAugment
from robotryon.video.conditioning import (AUDIO_TOKENS, build_step, build_video_coords,
                                          flow_matching_loss, sample_keyframe_indices,
                                          shifted_logit_normal_sigma)
from robotryon.video.dataset import VideoClipDataset, load_episodes
from robotryon.video.pipeline import snapshot_dir
from robotryon.video.rope4d import check_bands_preserved, install_view_axis_rope
from robotryon.video.training import (LORA_TARGETS, enable_checkpointing, load_lora_into,
                                      lora_state_dict, set_lora_state)


def caption_embeds(cfg, acc, captions: set[str]) -> dict:
    """Connector outputs of every training caption, cached at ``text.embeds``.

    Missing captions are encoded once on the main process before the transformer is loaded, so
    the text encoder never shares the GPU with training.
    """
    path = Path(cfg.text.embeds)
    have = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else {}
    missing = sorted(captions - set(have))
    if missing and acc.is_main_process:
        from robotryon.video.text import encode_for_training
        have.update(encode_for_training(missing, cfg.model.base))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(have, path)
        torch.cuda.empty_cache()
    acc.wait_for_everyone()
    if missing:
        have = torch.load(path, map_location="cpu", weights_only=False)
    return have


def main() -> int:
    cfg = parse_args(__doc__)
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from diffusers import AutoencoderKLLTX2Video, LTX2VideoTransformer3DModel
    from diffusers.pipelines.ltx2 import LTX2Pipeline
    from peft import LoraConfig

    acc = Accelerator(mixed_precision="bf16",
                      kwargs_handlers=[DistributedDataParallelKwargs(static_graph=True)])
    set_seed(cfg.train.seed)
    dev, out_dir = acc.device, Path(cfg.train.output_dir)
    d, t = cfg.data, cfg.train
    skeleton_size = tuple(d.skeleton_size)
    ref_ds = d.out_size[0] // skeleton_size[0]
    if tuple(d.out_size) != (ref_ds * skeleton_size[0], ref_ds * skeleton_size[1]):
        raise ValueError("data.out_size must be an integer multiple of data.skeleton_size")

    # ── data ──
    episodes, weights = load_episodes(d.roots, d.frames_range[0], d.fps, d.caption_template)
    ds = VideoClipDataset(episodes, d.out_size, skeleton_size, d.frames_range, d.min_scale,
                          d.flip_p, d.mono_p, SkeletonAugment.from_config(cfg.skeleton.augment),
                          cfg.skeleton.draw, d.fps, t.seed)
    loader = DataLoader(ds, batch_size=1, num_workers=d.num_workers, pin_memory=True,
                        sampler=WeightedRandomSampler(weights, len(ds), replacement=True),
                        collate_fn=lambda b: b[0])
    text = caption_embeds(cfg, acc, {e.caption for e in episodes})
    acc.print(f"{len(episodes)} episodes | {d.out_size[0]}x{d.out_size[1]} per view | skeleton "
              f"{skeleton_size[0]}x{skeleton_size[1]} | frames {d.frames_range} | "
              f"{t.steps} steps x {acc.num_processes} GPUs x accum {t.grad_accum}")

    # ── model ──
    root = snapshot_dir(cfg.model.base)
    vae = AutoencoderKLLTX2Video.from_pretrained(root, subfolder="vae",
                                                 torch_dtype=torch.bfloat16).to(dev).eval()
    vae.requires_grad_(False)
    lat_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(dev, torch.bfloat16)
    lat_std = vae.latents_std.view(1, -1, 1, 1, 1).to(dev, torch.bfloat16)
    tr = LTX2VideoTransformer3DModel.from_pretrained(
        root, subfolder=cfg.model.transformer_subfolder, torch_dtype=torch.bfloat16)
    tr.requires_grad_(False)
    install_view_axis_rope(tr)
    chk = check_bands_preserved(tr.rope)
    if chk["n_changed"] != chk["n_expected"] or chk["max_abs_diff_off_view"] > 1e-3:
        raise RuntimeError(f"view-axis RoPE disturbed the t/h/w bands: {chk}")
    tr.add_adapter(LoraConfig(r=cfg.lora.rank, lora_alpha=cfg.lora.alpha,
                              init_lora_weights="gaussian", target_modules=LORA_TARGETS))
    tr.set_attention_backend(cfg.model.attention_backend)
    enable_checkpointing(tr)
    if cfg.lora.init:
        acc.print(f"LoRA initialised from {cfg.lora.init} ({load_lora_into(tr, cfg.lora.init)} "
                  f"tensors)")
    tr = tr.to(dev).train()
    params = [p for p in tr.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=t.lr, betas=tuple(t.betas), weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, t.lr_end_factor, total_iters=t.steps)
    tr, opt, loader, sched = acc.prepare(tr, opt, loader, sched)
    base = acc.unwrap_model(tr)
    acc.print(f"LoRA rank {cfg.lora.rank}: {sum(p.numel() for p in params) / 1e6:.1f}M params")

    def encode(pix: torch.Tensor):
        """``(V, 3, F, H, W)`` -> ``(1, V * tokens, C)`` packed view-major, and one view's grid."""
        packed = []
        with torch.no_grad():
            for v in range(pix.shape[0]):
                lat = vae.encode(pix[v:v + 1].to(dev, torch.bfloat16)).latent_dist.sample()
                lat = (lat - lat_mean) / lat_std
                packed.append(LTX2Pipeline._pack_latents(lat, 1, 1))
        return torch.cat(packed, dim=1), lat.shape[2:]

    def step_loss(batch) -> torch.Tensor:
        views, F, fps = batch["video"].shape[0], int(batch["num_frames"]), float(batch["fps"])
        tgt, (lf, lh, lw) = encode(batch["video"])
        ref, _ = encode(batch["condition"])
        sigma = shifted_logit_normal_sigma(tgt.shape[1], device=dev).to(torch.bfloat16)
        kf_idx = sample_keyframe_indices(F, cfg.keyframes.count_weights,
                                         cfg.keyframes.include_last_p)
        kf = {f: encode(batch["video"][:, :, f:f + 1])[0] for f in kf_idx}
        hidden, ts, velocity, pinned, appended = build_step(tgt, ref, kf, sigma, lh * lw, views)
        emb = text[batch["caption"]]
        mask = emb["mask"][None].to(dev)
        pred = tr(hidden_states=hidden,
                  audio_hidden_states=torch.zeros(1, AUDIO_TOKENS, base.config.audio_in_channels,
                                                  device=dev, dtype=torch.bfloat16),
                  encoder_hidden_states=emb["video"][None].to(dev, torch.bfloat16),
                  audio_encoder_hidden_states=emb["audio"][None].to(dev, torch.bfloat16),
                  timestep=ts, sigma=sigma * 1000.0,
                  audio_timestep=torch.zeros(1, AUDIO_TOKENS, device=dev, dtype=torch.bfloat16),
                  audio_sigma=sigma * 1000.0, encoder_attention_mask=mask,
                  audio_encoder_attention_mask=mask, num_frames=lf, height=lh, width=lw,
                  fps=fps, audio_num_frames=AUDIO_TOKENS,
                  video_coords=build_video_coords(base.rope, lf, lh, lw, ref_ds, dev, fps,
                                                  appended, views),
                  return_dict=False)[0]
        return flow_matching_loss(pred[:, :tgt.shape[1]], velocity, pinned)

    def save(step: int) -> None:
        ck = out_dir / f"checkpoint-{step:06d}"
        ck.mkdir(parents=True, exist_ok=True)
        LTX2Pipeline.save_lora_weights(ck, transformer_lora_layers=lora_state_dict(base))
        save_config(cfg, ck / "train_config.yaml")
        torch.save({"step": step, "epoch": epoch, "lora": lora_state_dict(base),
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict()},
                   out_dir / "resume_state.pt.tmp")
        (out_dir / "resume_state.pt.tmp").replace(out_dir / "resume_state.pt")
        acc.print(f"  saved {ck}")

    step, epoch, micro = 0, 0, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    if acc.is_main_process:
        save_config(cfg, out_dir / "config.yaml")
    state = out_dir / "resume_state.pt"
    if t.resume and state.exists():
        if cfg.lora.init:                      # the saved LoRA supersedes the initialisation
            acc.print(f"{state} exists; resuming from it instead of lora.init")
        ck = torch.load(state, map_location="cpu", weights_only=False)
        set_lora_state(base, ck["lora"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, epoch = ck["step"], ck["epoch"]
        acc.print(f"resumed at step {step}")

    checked, t0, running = False, time.time(), 0.0
    while step < t.steps:
        ds.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            if step >= t.steps:
                break
            loss = step_loss(batch) / t.grad_accum
            acc.backward(loss)
            if not checked:                    # reentrant checkpointing can drop gradients
                checked = True
                dead = [n for n, p in tr.named_parameters() if p.requires_grad and p.grad is None]
                if dead:
                    raise RuntimeError(f"{len(dead)} LoRA tensors got no gradient, e.g. {dead[0]}")
            micro += 1
            running += loss.item()
            if micro % t.grad_accum:
                continue
            acc.clip_grad_norm_(params, t.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % t.log_every == 0:
                acc.print(f"step {step:6d}/{t.steps}  loss {running / t.log_every:.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  F={batch['num_frames']}  "
                          f"V={batch['video'].shape[0]}  "
                          f"{(time.time() - t0) / t.log_every:.2f}s/step  "
                          f"peak {torch.cuda.max_memory_allocated() / 1e9:.0f}GB", flush=True)
                running, t0 = 0.0, time.time()
            if acc.is_main_process and (step % t.save_every == 0 or step == t.steps):
                save(step)
    acc.wait_for_everyone()
    acc.print(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

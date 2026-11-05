"""Synthesize the robot video of one human episode (paper Sec. 3.3, Eq. 3.4, App. B.2.4 / B.4).

    python scripts/infer_video.py --config configs/video/infer.yaml \
        input.episode=path/to/human/ep0 input.keyframes_dir=path/to/keyframes/ep0 \
        output.dir=path/to/video/ep0 two_stage.mode=same_res

Inputs:
    <episode>/meta.json              fps (must equal ``fps``), views, num_frames, caption
    <episode>/keypoints_<view>.npz   retargeted hand keypoints at video resolution
    <episode>/video_<view>.mp4       the human video (its frame size; its pixels too with
                                     ``mask_human: false``)
    <keyframes_dir>/keyframes.json   {"indices": [...], ...} written by scripts/infer_keyframes.py
                                     (falls back to <episode>/keyframes.json)
    <keyframes_dir>/keyframe_<view>_<index:06d>.png   translated robot keyframe per index

The skeleton flow S^h_{1:T} is rendered from the keypoints at ``skeleton_size`` on black
(``mask_human: true``, paper default). With ``mask_human: false`` it is drawn on the unmasked human
video instead; the keyframes always come from the masked image condition. The episode is
cut at its keyframes into clips of at most ``frames_range[1]`` frames that share their boundary
keyframe (``robotryon.video.chunks``); each clip is generated in two stages from its keyframes
and skeleton flow, with both stereo views in one sequence by default.

Outputs in ``output.dir``:
    video_<view>.mp4   the robot video, ``resolution`` per view, ``fps`` frames per second,
                       frames keyframes[0] .. keyframes[-1] of the episode
    chunks.json        the clip plan (start, length, keyframes of every clip)
    meta.json          episode metadata in the shared format (for the IDM)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from robotryon.common.config import parse_args, save_config
from robotryon.skeleton import draw_skeleton_video, paint
from robotryon.video.chunks import concat_chunks, drop_colliding_keyframes, plan_chunks
from robotryon.video.dataset import caption_text
from robotryon.video.io import read_image, write_video
from robotryon.video.pipeline import build_pipeline, generate
from robotryon.video.text import encode_for_inference


def to_unit(a: np.ndarray) -> torch.Tensor:
    """uint8 ``(..., H, W, 3)`` -> float in [-1, 1] with channels moved before the frame axes."""
    t = torch.from_numpy(np.ascontiguousarray(a)).float() / 127.5 - 1
    return t.permute(2, 0, 1) if a.ndim == 3 else t.permute(3, 0, 1, 2)


def skeleton_flow(ep: Path, view: str, total: int, size, draw_kw: dict) -> np.ndarray:
    """``(T, h, w, 3)`` uint8 skeleton flow of one view at ``size``."""
    import cv2
    cap = cv2.VideoCapture(str(ep / f"video_{view}.mp4"))
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    z = np.load(ep / f"keypoints_{view}.npz")
    kp = z["kp"][:total].astype(np.float32).copy()
    kp[..., 0] = (kp[..., 0] + 0.5) * size[0] / W - 0.5
    kp[..., 1] = (kp[..., 1] + 0.5) * size[1] / H - 0.5
    return draw_skeleton_video(kp, z["vis"][:total].astype(bool), size[0], size[1], **draw_kw)


def human_frames(ep: Path, view: str, total: int, size) -> np.ndarray:
    """``(T, h, w, 3)`` uint8 RGB frames of the human video of one view, resized to ``size``."""
    import cv2
    cap = cv2.VideoCapture(str(ep / f"video_{view}.mp4"))
    frames = []
    while len(frames) < total:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(f, tuple(size), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < total:
        raise SystemExit(f"{ep / f'video_{view}.mp4'}: {len(frames)} frames, expected {total}")
    return np.stack(frames)


def reference_flow(ep: Path, view: str, total: int, cfg) -> np.ndarray:
    """Reference video of one view: the skeleton flow, on black or on the human video."""
    skel = skeleton_flow(ep, view, total, cfg.skeleton_size, cfg.skeleton.draw)
    if cfg.get("mask_human", True):
        return skel
    human = human_frames(ep, view, total, cfg.skeleton_size)
    return np.stack([paint(h, s) for h, s in zip(human, skel)])


def plan(cfg):
    """``(episode dir, views, view groups, frame count, keyframes, clips)`` of one episode."""
    ep = Path(cfg.input.episode)
    meta = json.loads((ep / "meta.json").read_text())
    if abs(float(meta["fps"]) - cfg.fps) > 1e-6:
        raise SystemExit(f"episode is {meta['fps']} fps, the model generates {cfg.fps}: "
                         f"resample the episode first")
    views = list(cfg.input.views or meta["views"])
    groups = [views] if cfg.input.stereo and len(views) == 2 else [[v] for v in views]
    total = int(meta["num_frames"])
    kf_json = Path(cfg.input.keyframes_dir) / "keyframes.json"
    kf_json = kf_json if kf_json.exists() else ep / "keyframes.json"
    kf = sorted(set(int(i) for i in json.loads(kf_json.read_text())["indices"]))
    chunks = plan_chunks(kf, total, cfg.frames_range)
    if not chunks:
        raise SystemExit(f"no clip can be planned from keyframes {kf} in {total} frames")
    return ep, meta, views, groups, total, kf, chunks


def episode_caption(cfg) -> str:
    meta = json.loads((Path(cfg.input.episode) / "meta.json").read_text())
    return caption_text(cfg.text.template, meta, cfg.text.embodiment)


def load_pipeline(cfg, views: int, device: str = "cuda"):
    return build_pipeline(cfg.model.lora, base=cfg.model.base,
                          transformer_subfolder=cfg.model.transformer_subfolder,
                          views=views, attention_backend=cfg.model.attention_backend,
                          upsample=cfg.two_stage.mode == "upsample", device=device)


def synthesize(cfg, pipe, text) -> Path:
    """Generate and write the robot video of ``cfg.input.episode`` with a built pipeline and the
    encoded caption ``text`` (``robotryon.video.text.encode_for_inference``)."""
    ep, meta, views, groups, total, kf, chunks = plan(cfg)
    kf_dir, out = Path(cfg.input.keyframes_dir), Path(cfg.output.dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{ep.name}: {total} frames, keyframes {kf}, {len(chunks)} clips, views {groups}, "
          f"mode {cfg.two_stage.mode}, mask_human {cfg.get('mask_human', True)}")
    skel = {v: reference_flow(ep, v, total, cfg) for v in views}

    clips: dict[str, list[np.ndarray]] = {v: [] for v in views}
    for ci, ch in enumerate(chunks):
        keep = drop_colliding_keyframes(ch.kf_local, ch.frames)
        print(f"  clip {ci}: frames {ch.start}..{ch.start + ch.frames - 1} ({ch.frames}f), "
              f"keyframes at {[ch.kf_local[k] for k in keep]}", flush=True)
        for group in groups:
            sk = torch.stack([to_unit(skel[v][ch.start:ch.start + ch.frames]) for v in group])
            kfs = [(ch.kf_local[k], torch.stack([
                       to_unit(read_image(kf_dir / f"keyframe_{v}_{kf[ch.kf_slots[k]]:06d}.png"))
                       for v in group])) for k in keep]
            video = generate(pipe, skeleton=sk, keyframes=kfs, text=text,
                             num_frames=ch.frames, size=tuple(cfg.resolution), fps=cfg.fps,
                             seed=cfg.seed + ci, mode=cfg.two_stage.mode)
            for vi, v in enumerate(group):
                clips[v].append(video[vi].numpy())

    for v in views:
        full = concat_chunks(chunks, clips[v])
        write_video(full, out / f"video_{v}.mp4", cfg.fps)
    (out / "chunks.json").write_text(json.dumps(
        {"keyframes": kf, "chunks": [c.to_dict() for c in chunks]}, indent=1))
    (out / "meta.json").write_text(json.dumps(
        {"fps": cfg.fps, "views": views, "num_frames": int(full.shape[0]),
         "caption": meta.get("caption", ""), "embodiment": cfg.text.embodiment or "robot",
         "source_episode": ep.name, "first_frame": kf[0]}, indent=1))
    save_config(cfg, out / "infer_config.yaml")
    print(f"wrote {out}: {len(views)} x {full.shape[0]} frames at "
          f"{cfg.resolution[0]}x{cfg.resolution[1]}, {cfg.fps} fps")
    return out


def main() -> int:
    cfg = parse_args(__doc__)
    groups = plan(cfg)[3]
    caption = episode_caption(cfg)
    text = encode_for_inference([caption], cfg.model.base)[caption]
    synthesize(cfg, load_pipeline(cfg, len(groups[0])), text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

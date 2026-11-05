#!/usr/bin/env python
"""Translate the keyframes of preprocessed human episodes into robot keyframes.

    python scripts/infer_keyframes.py --config configs/image/infer.yaml \
        model.base=<FLUX.2-klein-base-9B dir> checkpoint=<trained checkpoint dir> \
        episodes='[human/ep0, human/ep1]' reference.left=robot_left.png reference.right=robot_right.png \
        output_dir=out/keyframes

Each episode directory uses the shared data layout plus ``keyframes.json``; its keypoints are
already retargeted to the target robot hand. ``reference`` maps every camera view to a
robot-only image of the target robot (the robot's pixels on a white background, see
``scripts/make_reference.py``), which selects the embodiment. Output per episode (``<output_dir>/<episode_id>/``):

    keyframe_<view>_<frame:06d>.png   translated robot keyframe, RGB, model resolution (1024x576)
    cond_<view>_<frame:06d>.png       conditioned human frame Phi given to the model
    mask_<view>_<frame:06d>.png       inference mask (255 = repainted region)
    keyframes.json                    indices, views, prompt and sampler settings
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from robotryon.common.config import parse_args
from robotryon.image.ar_keyframes import KeyframeTranslator
from robotryon.image.episodes import Episode
from robotryon.image.model import load_pipeline


def main():
    cfg = parse_args(__doc__)
    torch.set_grad_enabled(False)
    pipe = load_pipeline(cfg.model.base, cfg.checkpoint, device="cuda")
    tr = KeyframeTranslator(pipe, cfg)
    ref = cfg.reference
    episodes = cfg.episodes if isinstance(cfg.episodes, (list, tuple)) else [cfg.episodes]
    for ep in episodes:
        views = Episode.load(ep).views
        refs = {v: np.asarray(Image.open(ref[v] if isinstance(ref, dict) else ref).convert("RGB"))
                for v in views}
        out = tr.translate_episode(ep, refs, cfg.output_dir)
        print(f"[keyframes] {ep} -> {out}", flush=True)


if __name__ == "__main__":
    main()

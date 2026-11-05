#!/usr/bin/env python
"""Label a generated robot video with joint actions and states (paper Sec. 3.4).

    python scripts/label_actions.py --config configs/idm/label.yaml \
        label.checkpoint=/path/idm label.embodiment=openarm_rh56f1 \
        label.videos=[clip_left.mp4,clip_right.mp4] label.output_dir=outputs/clip

Writes to ``label.output_dir``:
    actions.npy      (T, D) float32  joint actions a_t in the embodiment's raw joint order
                                     (``action_groups`` concatenated), after chunk averaging
                                     and SG smoothing
    states.npy       (T, D) float32  q_t = a_{t-d} (d = label.state_delay), q_t = a_0 for t < d
    meta.json        embodiment, groups and their column ranges, settings

The wrist positions can then be corrected with ``scripts/correct_wrists.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from robotryon.common.config import parse_args
from robotryon.idm.data import embodiment_specs
from robotryon.idm.labeling import group_slices, predict_actions, smooth_actions, state_from_action
from robotryon.idm.model import IDM, load_metadata


def read_video(path: str, size) -> np.ndarray:
    """(T, H, W, 3) uint8 RGB, resized (INTER_AREA) to ``size`` = (W, H)."""
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if (f.shape[1], f.shape[0]) != tuple(size):
            f = cv2.resize(f, tuple(size), interpolation=cv2.INTER_AREA)
        out.append(f[..., ::-1])
    cap.release()
    if not out:
        raise RuntimeError(f"no frames in {path}")
    return np.ascontiguousarray(np.stack(out))



def main():
    cfg = parse_args(__doc__)
    lc = cfg.label
    meta = load_metadata(lc.checkpoint)[lc.embodiment]
    spec = meta.get("spec") or embodiment_specs(cfg.data)[lc.embodiment]
    stats = meta["statistics"]["action"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDM.from_pretrained(lc.checkpoint, device=device)

    views = [read_video(p, spec["video_size"]) for p in lc.videos]
    if len(views) == 1 and len(spec["views"]) > 1:
        views = views * len(spec["views"])
    assert len(views) == len(spec["views"]), f"{len(views)} videos for views {spec['views']}"
    T = min(len(v) for v in views)
    views = [v[:T] for v in views]

    gen = torch.Generator(device=device).manual_seed(int(lc.seed))
    raw = predict_actions(model, views, spec, stats, cfg.data.frame_gap, lc.batch_size, gen)
    groups = group_slices(spec, {g: len(stats[g]["q01"]) for g in spec["action_groups"]})
    actions = smooth_actions(raw, groups, set(spec["hand_groups"]), lc.smooth_window,
                             lc.hand_smooth_window, lc.smooth_order)

    out = Path(lc.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    actions = actions.astype(np.float32)
    np.save(out / "actions.npy", actions)
    np.save(out / "states.npy", state_from_action(actions, lc.state_delay))
    (out / "meta.json").write_text(json.dumps(dict(
        embodiment=lc.embodiment, frames=T, videos=list(lc.videos),
        groups={g: [s.start, s.stop] for g, s in groups.items()},
        state=f"q_t = a_(t-{lc.state_delay})", smooth=[lc.smooth_window, lc.hand_smooth_window],
        ), indent=1))
    print(f"{lc.embodiment}: {T} frames -> {out}")


if __name__ == "__main__":
    main()

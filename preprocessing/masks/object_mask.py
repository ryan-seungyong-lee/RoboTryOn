#!/usr/bin/env python
"""Mask of the manipulated object with SAM3, from a text prompt or a box.

The object region is removed from the arm mask so the object is preserved when the arm is
replaced (always at inference, with some probability in training; Appendix B.2.3). Several
anchor frames are tried because the object is often hidden by the hand on the first frame.
A box prompt (``--box FRAME X Y W H`` in pixels) tracks one object from that frame.

Writes ``object_mask_<view>.mp4`` (gray, >127 = object).
"""

from __future__ import annotations

from preprocessing.episode_io import parse, parser, video_info
from preprocessing.masks.sam3_video import segment_to_video


def main():
    p = parser(__doc__.split("\n")[0], "configs/preprocess/masks.yaml")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="object_mask_<view>.mp4")
    p.add_argument("--prompt", action="append", default=[], help="object description")
    p.add_argument("--box", nargs=5, type=float, action="append", default=[],
                   metavar=("FRAME", "X", "Y", "W", "H"))
    a, cfg = parse(p)
    if not a.prompt and not a.box:
        p.error("give --prompt and/or --box")
    _, W, H, _ = video_info(a.video)
    boxes = [(int(f), (x / W, y / H, w / W, h / H)) for f, x, y, w, h in a.box]
    segment_to_video(a.video, a.out, cfg.object, prompts=a.prompt, boxes=boxes)


if __name__ == "__main__":
    main()

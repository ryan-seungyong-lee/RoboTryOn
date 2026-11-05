#!/usr/bin/env python
"""Human arm and hand mask of a video with SAM3 (Appendix B.2.3).

At inference the region to replace is the human arm and hand, segmented by SAM3 alone from
text prompts and propagated through the clip. Sleeved arms need several prompts, so the
default config takes the union of all prompts. The mask is saved tight; it is enlarged at
inference by ``robotryon.masks``.

Writes ``mask_<view>.mp4`` (gray, >127 = masked).
"""

from __future__ import annotations

from preprocessing.episode_io import parse, parser
from preprocessing.masks.sam3_video import segment_to_video


def main():
    p = parser(__doc__.split("\n")[0], "configs/preprocess/masks.yaml")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="mask_<view>.mp4")
    a, cfg = parse(p)
    segment_to_video(a.video, a.out, cfg.human)


if __name__ == "__main__":
    main()

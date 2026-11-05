#!/usr/bin/env python
"""Robot arm and hand mask of a video with SAM3, the image half of the robot mask.

SAM3 alone misses parts of the robot and sometimes fires on people in the background; the
final robot mask combines it with the rendered silhouette (``urdf_mask.py``). Prompts are a
fallback chain of wordings for the robot; set them for your robot in the config.

Writes a gray mask mp4 (>127 = robot).
"""

from __future__ import annotations

from preprocessing.episode_io import parse, parser
from preprocessing.masks.sam3_video import segment_to_video


def main():
    p = parser(__doc__.split("\n")[0], "configs/preprocess/masks.yaml")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    a, cfg = parse(p)
    segment_to_video(a.video, a.out, cfg.robot)


if __name__ == "__main__":
    main()

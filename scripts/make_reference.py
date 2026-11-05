#!/usr/bin/env python
"""Make the robot-only reference image that selects the target robot (first keyframe).

    python scripts/make_reference.py --episode path/to/robot_episode --view left --frame 0 \
        --out robot_left.png

Takes one frame of a robot episode in the training-data layout (``video_<view>.mp4`` and
``mask_<view>.mp4``, e.g. from ``preprocessing/prepare_robot_episode.py``) and keeps only the
robot, on a white background, exactly as the image model's robot-only training references.
Pick a frame where the arms and hands are clearly visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robotryon.masks import robot_only  # noqa: E402


def read_frame(path: Path, index: int):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {index} of {path}")
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--episode", required=True, help="robot episode directory")
    p.add_argument("--view", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--erode", type=int, default=7, help="px, as in training (data.robot_only.erode)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ep = Path(a.episode)
    img = read_frame(ep / f"video_{a.view}.mp4", a.frame)
    mask = read_frame(ep / f"mask_{a.view}.mp4", a.frame)[..., 0] > 127
    cv2.imwrite(a.out, robot_only(img, mask, a.erode))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

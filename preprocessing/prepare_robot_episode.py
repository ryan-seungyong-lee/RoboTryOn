#!/usr/bin/env python
"""LeRobot v2.1 robot episode with joint states -> episode directory for image/video training.

For every camera of the embodiment YAML:

* ``video_<view>.mp4``: the episode's video re-encoded as H.264 (same fps and size).
* ``keypoints_<view>.npz``: forward kinematics of ``observation.state`` projected into the view
  (``skeleton/robot_skeleton.py``), and ``hand_state_<view>.npz``: each hand's closure / facing
  in that camera (the image model's prompt, ``robotryon.captions``).
* ``mask_<view>.mp4``: rendered robot silhouette, united with the SAM3 robot segmentation
  components that agree with it when ``--sam3`` is given (``masks/urdf_mask.py``).
* ``object_mask_<view>.mp4`` with ``--object_prompt`` (``masks/object_mask.py``).

Files written (T = min(video frames, state rows), H x W = source video size):
    meta.json             {"fps": float (dataset fps), "views": [str], "num_frames": T,
                           "caption": str (task text), "embodiment": str}
    video_<view>.mp4      RGB H.264, T frames
    keypoints_<view>.npz  kp float32 (T, 2, 17, 2) px at W x H; vis bool (T, 2, 17)
    hand_state_<view>.npz closure, dorsumness float32 (T, 2): hand state for the image prompt
    mask_<view>.mp4       gray lossless H.264, T frames, >127 = robot arm/hand
    object_mask_<view>.mp4  same encoding, >127 = object (optional)

Rendering needs ``MUJOCO_GL=egl`` (GPU) or ``osmesa`` (CPU); SAM3 needs a GPU.
"""

from __future__ import annotations

from pathlib import Path

from preprocessing.episode_io import (LeRobotEpisode, parse, parser, save_keypoints, save_meta,
                                      stage_config, transcode, video_info)
from preprocessing.masks.urdf_mask import write_robot_mask
from preprocessing.skeleton.robot_skeleton import robot_hand_state, robot_keypoints
from robotryon.captions import save_hand_state
from robotryon.embodiment import load_embodiment


def main():
    p = parser(__doc__.split("\n")[0], "configs/preprocess/prepare.yaml")
    p.add_argument("--embodiment", required=True)
    p.add_argument("--lerobot_root", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--out", required=True, help="episode directory")
    p.add_argument("--views", nargs="*", default=None, help="default: every camera in the YAML")
    p.add_argument("--sam3", action="store_true", help="add SAM3 segmentation to the mask")
    p.add_argument("--object_prompt", action="append", default=[])
    p.add_argument("--caption", default=None, help="default: the episode's task text")
    a, cfg = parse(p)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    emb = load_embodiment(a.embodiment)
    ep = LeRobotEpisode(a.lerobot_root, a.episode)
    states = ep.column("observation.state")
    views = a.views or emb.views
    mcfg = stage_config(cfg, a.config, "masks")

    T = min([len(states)] + [video_info(ep.video(emb.spec.cameras[v].video_key))[3]
                             for v in views])
    sam = None
    if a.sam3 or a.object_prompt:
        from preprocessing.masks.sam3_video import Sam3Video, segment_to_video
        sam = Sam3Video()
    for v in views:
        video = out / f"video_{v}.mp4"
        transcode(ep.video(emb.spec.cameras[v].video_key), video, max_frames=T)
        _, W, H, _ = video_info(video)
        kp, vis = robot_keypoints(emb, states[:T], v, (W, H))
        save_keypoints(out / f"keypoints_{v}.npz", kp, vis)
        save_hand_state(out / f"hand_state_{v}.npz", *robot_hand_state(emb, states[:T], v))
        sam_mask = None
        if a.sam3:
            sam_mask = out / f"sam3_robot_{v}.mp4"
            segment_to_video(str(video), str(sam_mask), mcfg.robot, sam=sam)
        write_robot_mask(emb, states[:T], v, out / f"mask_{v}.mp4", ep.fps, (W, H),
                         mcfg.robot_mask, sam_mask)
        if sam_mask is not None:
            sam_mask.unlink()
        if a.object_prompt:
            segment_to_video(str(video), str(out / f"object_mask_{v}.mp4"), mcfg.object,
                             prompts=a.object_prompt, sam=sam)
        print(f"{v}: T={T} {W}x{H} keypoints visible={vis.mean():.3f}")
    if sam is not None:
        sam.shutdown()
    caption = ep.task() if a.caption is None else a.caption
    save_meta(out, ep.fps, views, T, caption, emb.name)


if __name__ == "__main__":
    main()

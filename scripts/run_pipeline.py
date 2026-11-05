#!/usr/bin/env python
"""Run the full RoboTryOn pipeline on one human video (paper Fig. 1).

    python scripts/run_pipeline.py --config configs/pipeline.yaml [key=value ...]

Stages, each a separate process with its own interpreter (``python.<stage>``):

1. ``preprocess``  human video -> episode directory: hand reconstruction, keyframes, hand
                   skeleton retargeted to the target robot, arm and object masks
                   (``preprocessing/prepare_human_clip.py``).
2. ``keyframes``   image model: keyframes translated autoregressively (``scripts/infer_keyframes.py``).
3. ``video``       video model: robot video between the keyframes under the skeleton flow
                   (``scripts/infer_video.py``).
4. ``actions``     IDM: joint actions and states (``scripts/label_actions.py``), then the
                   wrists re-solved from the reconstructed human wrists
                   (``scripts/correct_wrists.py``).

Output layout (``output_dir``):
    episode/     preprocessed human episode (meta.json, video/keypoints/mask per view,
                 keyframes.json, hands.npz)
    keyframes/   robot keyframes  keyframe_<view>_<frame:06d>.png
    video/       robot video      video_<view>.mp4
    actions/     actions.npy, states.npy, meta.json
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))          # usable without `pip install -e .`

from robotryon.common.config import parse_args  # noqa: E402


def run(cmd: list[str], env_paths: bool = True) -> None:
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    env = None
    if env_paths:
        import os
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO / "src"), str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def as_list(paths) -> str:
    return "[" + ",".join(str(p) for p in paths) + "]"


def main() -> None:
    cfg = parse_args(__doc__)
    out = Path(cfg.output_dir).resolve()
    ep, kf_root, vid, act = out / "episode", out / "keyframes", out / "video", out / "actions"
    views = list(cfg.input.videos)
    if not views:
        sys.exit("input.videos is empty")
    py, conf, extra = cfg.python, cfg.configs, cfg.overrides

    if "preprocess" in cfg.stages:
        cmd = [py.preprocess, "-m", "preprocessing.prepare_human_clip",
               "--config", conf.preprocess, "--embodiment", cfg.embodiment,
               "--out", str(ep), "--caption", cfg.input.caption]
        for v in views:
            cmd += ["--video", v, str(cfg.input.videos[v])]
        if cfg.input.hands:
            cmd += ["--hands", str(cfg.input.hands)]
        else:
            cmd += ["--hawor_root", str(cfg.hawor_root), "--hawor_python", py.hawor]
        for prompt in cfg.input.object_prompts or []:
            cmd += ["--object_prompt", prompt]
        run(cmd + list(extra.preprocess))

    if "keyframes" in cfg.stages:
        refs = [f"reference.{v}={cfg.reference[v]}" for v in views]
        run([py.image, "scripts/infer_keyframes.py", "--config", conf.image,
             f"model.base={cfg.checkpoints.image_base}", f"checkpoint={cfg.checkpoints.image}",
             f"episodes={as_list([ep])}", f"output_dir={kf_root}", *refs, *extra.image])

    # infer_keyframes writes <output_dir>/<episode_id>/ with the episode directory's name.
    kf_dir = kf_root / ep.name
    if "video" in cfg.stages:
        run([py.video, "scripts/infer_video.py", "--config", conf.video,
             f"model.lora={cfg.checkpoints.video}", f"input.episode={ep}",
             f"input.keyframes_dir={kf_dir}", f"output.dir={vid}",
             f"mask_human={str(cfg.mask_human).lower()}", *extra.video])

    if "actions" in cfg.stages:
        meta = json.loads((vid / "meta.json").read_text())
        videos = [vid / f"video_{v}.mp4" for v in meta["views"]]
        run([py.idm, "scripts/label_actions.py", "--config", conf.idm,
             f"label.checkpoint={cfg.checkpoints.idm}", f"label.embodiment={cfg.idm_embodiment}",
             f"label.videos={as_list(videos)}", f"label.output_dir={act}", *extra.idm])
        # robot kinematics only (MuJoCo), so it runs with the preprocessing interpreter
        run([py.preprocess, "scripts/correct_wrists.py", "--config", conf.idm,
             f"label.output_dir={act}", f"wrist_correction.hand3d={ep / 'hands.npz'}",
             f"wrist_correction.embodiment={cfg.embodiment}", *extra.idm])

    print(f"\ndone -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Correct the wrist positions of IDM action labels with the reconstructed human wrists
(paper App. B.2.7).

    python scripts/correct_wrists.py --config configs/idm/label.yaml \
        label.output_dir=outputs/labels/clip wrist_correction.hand3d=episode/hands.npz \
        wrist_correction.embodiment=path/to/my_robot.yaml

Reads ``actions.npy`` from ``label.output_dir`` (written by ``scripts/label_actions.py``) and
rewrites it there with the arm joints re-solved so that the robot wrists follow the human wrists
(hand joints and wrist orientation are kept); ``states.npy`` is re-derived, the uncorrected
actions are kept as ``actions_idm.npy``, and ``meta.json`` gets the correction statistics.

Only needs numpy/scipy and MuJoCo (for the robot's forward kinematics), not the IDM itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robotryon.common.config import parse_args
from robotryon.embodiment import load_embodiment
from robotryon.idm.labeling import state_from_action
from robotryon.idm.wrist_correction import camera_to_base, correct_wrists


def main():
    cfg = parse_args(__doc__)
    wc, out = cfg.wrist_correction, Path(cfg.label.output_dir)
    if not wc.get("embodiment") or not wc.get("hand3d"):
        raise SystemExit("set wrist_correction.embodiment (robot YAML) and wrist_correction.hand3d")
    src = out / ("actions_idm.npy" if (out / "actions_idm.npy").exists() else "actions.npy")
    actions = np.load(src).astype(np.float64)
    T = len(actions)

    kin = load_embodiment(wc.embodiment)
    z = np.load(wc.hand3d)
    if len(z["wrist"]) < T:
        raise SystemExit(f"{wc.hand3d} has {len(z['wrist'])} frames, the labels have {T}")
    wrist_cam, conf = z["wrist"][:T].astype(np.float64), z["confidence"][:T]
    if hasattr(kin, "camera_pose"):
        # hands.npz holds the wrists in the first camera's frame
        T_base_cam = np.stack([kin.camera_pose(q, kin.views[0]) for q in actions])
    else:
        T_base_cam = np.asarray(wc.T_base_cam, np.float64)
    params = {k: v for k, v in wc.items() if k not in ("hand3d", "T_base_cam", "embodiment")}
    corrected, info = correct_wrists(actions, kin, camera_to_base(wrist_cam, T_base_cam), conf,
                                     params)

    np.save(out / "actions_idm.npy", actions.astype(np.float32))
    corrected = corrected.astype(np.float32)
    np.save(out / "actions.npy", corrected)
    np.save(out / "states.npy", state_from_action(corrected, int(cfg.label.state_delay)))
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["wrist_correction"] = info
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"wrist correction: {T} frames -> {out}")


if __name__ == "__main__":
    main()

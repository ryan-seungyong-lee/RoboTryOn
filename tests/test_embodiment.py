"""Embodiment YAML parsing (pure Python) and MujocoEmbodiment on a tiny synthetic robot."""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from robotryon.embodiment import load_spec, parse_spec

ROOT = Path(__file__).resolve().parents[1]
FINGERS = {"thumb": 4, "index": 3, "middle": 3, "ring": 3, "little": 3}


def _mjcf() -> str:
    """Two hands on slide/hinge joints, each finger a chain of bodies, one head camera site."""
    def hand(s, y):
        out = [f'<body name="{s}_wrist" pos="0.4 {y} 0"><joint name="{s}_x" type="slide" axis="1 0 0"/>'
               f'<geom type="sphere" size="0.02"/>']
        for i, (f, n) in enumerate(FINGERS.items()):
            off = -0.03 + 0.015 * i
            out.append(f'<body name="{s}_{f}_0" pos="0.05 {off} 0">'
                       f'<joint name="{s}_{f}_j" axis="0 1 0"/><geom type="sphere" size="0.005"/>')
            for k in range(1, n):
                out.append(f'<body name="{s}_{f}_{k}" pos="0.025 0 0"><geom type="sphere" size="0.005"/>')
            out.append("</body>" * n)
        out.append("</body>")
        return "".join(out)
    return (f'<mujoco><worldbody><site name="cam" pos="0 0 0.3" euler="0 0 0"/>'
            f'{hand("left", 0.15)}{hand("right", -0.15)}</worldbody></mujoco>')


def _cfg(model_path) -> dict:
    kps = ["body:{s}_wrist"]
    for f, n in FINGERS.items():
        kps += [f"body:{{s}}_{f}_{k}" for k in range(n)]
    return {
        "name": "toy", "model_path": str(model_path),
        "joints": ["left_x", "right_x", None, "left_index_j"],
        "arm_joints": {"left": [0, 1], "right": [1, 2]},
        "coupling": [{"joint": "{s}_middle_j", "source": "{s}_index_j", "ratio": 1.0}],
        "hands": {"sides": {"left": "left", "right": "right"}, "keypoints": kps,
                  "finger_scale": {"thumb": 1.0}},
        "cameras": {"main": {
            "video_key": "observation.images.main",
            "intrinsics": {"width": 640, "height": 480, "fx": 400.0, "fy": 400.0,
                           "cx": 320.0, "cy": 240.0},
            # camera looks along +x of the world: optical z = world x, x = -world y, y = -world z
            "mount": {"frame": "site:cam",
                      "rotation": [[0, -1, 0], [0, 0, -1], [1, 0, 0]]}}},
        "silhouette": {"bodies": ["{s}_"]},
        "retarget": {"joints": ["{s}_" + f + "_j" for f in FINGERS if f != "middle"],
                     "limits": {"{s}_thumb_j": [-1.0, 1.0]}},
    }


def test_gr1_example_parses():
    spec = load_spec(ROOT / "configs/embodiments/gr1.yaml")
    assert spec.hands == ["left", "right"] and spec.views == ["ego_view"]
    assert all(len(v) == 17 for v in spec.keypoints.values())
    assert spec.keypoints["right"][0] == "body:R_hand_base_link"
    assert spec.keypoint_offsets["left"][0] is not None and spec.keypoint_offsets["left"][1] is None
    assert spec.joint_scale["R_index_proximal_joint"] == -1.0 and len(spec.joints) == 44
    assert spec.arm_joints == {"left": (0, 7), "right": (22, 29)}
    assert spec.retarget.joints["left"][0] == "L_index_proximal_joint"

def test_spec_expansion_and_errors(tmp_path):
    spec = parse_spec(_cfg("x.xml"), base_dir=tmp_path)
    assert spec.model_path == str(tmp_path / "x.xml")
    assert ("left_middle_j", 1.0, "left_index_j") in spec.coupling
    assert spec.retarget.joints["right"] == ["right_thumb_j", "right_index_j", "right_ring_j",
                                             "right_little_j"]
    assert spec.retarget.limits["left_thumb_j"] == (-1.0, 1.0)
    bad = copy.deepcopy(_cfg("x.xml"))
    bad["hands"]["keypoints"] = bad["hands"]["keypoints"][:16]
    with pytest.raises(ValueError):
        parse_spec(bad)


@pytest.fixture
def emb(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    if mujoco.__version__ != "3.11.0":
        pytest.skip("mujoco 3.11.0 required")
    (tmp_path / "toy.xml").write_text(_mjcf())
    (tmp_path / "toy.yaml").write_text(yaml.safe_dump(_cfg("toy.xml")))
    from robotryon.embodiment import load_embodiment
    return load_embodiment(tmp_path / "toy.yaml")


def test_fk_project_and_coupling(emb):
    out = emb.fk([0.1, 0.0, 99.0, 0.5])
    assert set(out) == {"left", "right"} and out["left"]["keypoints"].shape == (17, 3)
    assert np.allclose(out["left"]["wrist_pos"], out["left"]["keypoints"][0])
    assert emb.arm_joint_slices == {"left": slice(0, 1), "right": slice(1, 2)}
    kp = emb.hand_keypoints([0.1, 0.0, 99.0, 0.5])
    assert kp.shape == (2, 17, 3)
    assert np.isclose(kp[0, 0, 0], 0.5) and np.isclose(kp[1, 0, 0], 0.4)
    q = emb.joint_dict([0.1, 0.0, 99.0, 0.5])
    assert q["left_middle_j"] == 0.5               # coupled joint follows its source
    T = emb.camera_pose(None, "main")
    assert np.allclose(T[:3, 2], [1, 0, 0])        # optical axis along world +x
    px, ok = emb.project(kp, "main", q=[0.1, 0.0, 0.0, 0.5])
    assert ok.all()
    # the left hand (world +y) appears on the left half of the image
    assert (px[0, :, 0] < 320).all() and (px[1, :, 0] > 320).all()
    kp2, vis = emb.keypoints_in_view([0.1, 0.0, 0.0, 0.5], "main", (640, 480))
    assert kp2.shape == (2, 17, 2) and vis.all() and np.allclose(kp2, px, atol=1e-4)


def test_retarget_recovers_robot_pose(emb):
    """A 'human' hand that is the robot hand itself is retargeted back to the same joints."""
    from robotryon.embodiment.retarget import to21
    q_true = {"left_thumb_j": 0.3, "left_index_j": 0.6, "left_ring_j": 0.9, "left_little_j": 0.2,
              "right_thumb_j": -0.2, "right_index_j": 0.4, "right_ring_j": 0.1, "right_little_j": 0.7}
    human = to21(emb.hand_keypoints(q_true))
    out, q = emb.retarget_hand(human)
    assert out.shape == (2, 17, 3) and np.isfinite(out).all()
    assert np.allclose(out[:, 0], human[:, 0])            # wrists coincide
    assert emb._rt_bounds["left"][0][0] == -1.0 and emb._rt_bounds["left"][1][0] == 1.0  # retarget.limits
    for n, v in q_true.items():
        assert abs(q[n] - v) < 1e-3, (n, q[n], v)
    # the coupled middle finger follows the index finger through FK
    assert np.allclose(out[0, 9], emb.hand_keypoints(q)[0, 9], atol=1e-6)
    # warm start + smoothing from the previous frame keeps the solution
    out2, q2 = emb.retarget_hand(human, prev=q)
    assert all(abs(q2[n] - v) < 1e-3 for n, v in q_true.items())


def test_robot_hand_state(emb):
    from preprocessing.skeleton.robot_skeleton import robot_hand_state
    q = np.zeros((3, 4))
    clo, dor = robot_hand_state(emb, q, "main")
    assert clo.shape == dor.shape == (3, 2)
    assert np.isfinite(clo).all() and np.isfinite(dor).all()

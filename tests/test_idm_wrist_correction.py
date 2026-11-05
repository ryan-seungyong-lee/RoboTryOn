"""Wrist correction on a toy robot: two planar 3-link arms (no URDF)."""

import numpy as np

from robotryon.idm.wrist_correction import camera_to_base, correct_wrists, ik_wrist

LINKS = np.array([0.40, 0.35, 0.10])
BASE = {"left": np.array([0.0, 0.25, 0.0]), "right": np.array([0.0, -0.25, 0.0])}


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class ToyArms:
    """q = [left arm 3, right arm 3, left hand 2, right hand 2]."""

    arm_joint_slices = {"left": slice(0, 3), "right": slice(3, 6)}

    def fk(self, q):
        out = {}
        for side, sl in self.arm_joint_slices.items():
            ang = np.cumsum(q[sl])
            pos = BASE[side] + np.array([np.sum(LINKS * np.cos(ang)), np.sum(LINKS * np.sin(ang)), 0.0])
            R = rz(ang[-1])
            kp = np.repeat(pos[None], 17, 0)
            kp[1:] += R @ np.array([0.08, 0.0, 0.0])          # a crude hand in front of the wrist
            out[side] = {"wrist_pos": pos, "wrist_rot": R, "keypoints": kp}
        return out


def _wrist(kin, q, side):
    return kin.fk(q)[side]["wrist_pos"]


def test_ik_reaches_target_keeping_orientation():
    kin = ToyArms()
    q_true = np.array([0.3, 0.6, -0.4, -0.2, -0.5, 0.3, 0.1, 0.2, 0.1, 0.2])
    q_idm = q_true + np.array([0.15, -0.25, 0.1, 0, 0, 0, 0, 0, 0, 0])     # same wrist yaw
    target = _wrist(kin, q_true, "left") * 1000
    R_ref = kin.fk(q_idm)["left"]["wrist_rot"]
    before = np.linalg.norm(_wrist(kin, q_idm, "left") * 1000 - target)
    # weak damping: converges onto the target
    q, _ = ik_wrist(kin.fk, q_idm, "left", slice(0, 3), target, R_ref, q_idm, damping=0.01)
    after = np.linalg.norm(_wrist(kin, q, "left") * 1000 - target)
    assert before > 30 and after < 0.05 * before
    # default damping: a compromise between the target and the IDM joints
    q_d, _ = ik_wrist(kin.fk, q_idm, "left", slice(0, 3), target, R_ref, q_idm)
    assert after < np.linalg.norm(_wrist(kin, q_d, "left") * 1000 - target) < before
    np.testing.assert_allclose(kin.fk(q)["left"]["wrist_rot"], R_ref, atol=2e-2)
    np.testing.assert_array_equal(q[3:], q_idm[3:])


def test_correct_wrists_track():
    kin = ToyArms()
    T = 60
    t = np.linspace(0, 1, T)[:, None]
    q_true = np.concatenate([[0.3, 0.6, -0.4] + 0.3 * t * [1, -1, 0.5],
                             [-0.2, -0.5, 0.3] + 0.2 * t * [-1, 1, 0],
                             0.5 * t.repeat(4, 1)], 1)
    bias = np.array([0.08, -0.12, 0.04, -0.06, 0.1, -0.04, 0, 0, 0, 0])
    actions = (q_true + bias).astype(np.float32)
    human = np.stack([[_wrist(kin, q, s) for s in ("left", "right")] for q in q_true])
    conf = np.ones((T, 2))
    conf[40:50, 1] = 0.0                                   # right hand lost for 10 frames
    human[20, 0] += [0.4, 0.0, 0.0]                        # a single misdetection on the left
    out, info = correct_wrists(actions, kin, human, conf, {"damping": 0.01})
    assert out.shape == actions.shape and out.dtype == actions.dtype
    np.testing.assert_array_equal(out[:, 6:], actions[:, 6:])     # hands untouched
    assert info["left"]["rejected"] >= 1
    err = lambda q, s: np.linalg.norm([_wrist(kin, q[i], s) - human[i, "lr".index(s[0])]
                                       for i in range(T)], axis=1)
    for side in ("left", "right"):
        keep = np.ones(T, bool)
        keep[[20]] = False
        if side == "right":
            keep[36:54] = False
        before, after = err(actions.astype(float), side)[keep], err(out.astype(float), side)[keep]
        assert np.median(after) < 0.1 * np.median(before), (side, np.median(before), np.median(after))
    # orientation of the IDM wrists is kept
    for i in range(T):
        np.testing.assert_allclose(kin.fk(out[i].astype(float))["left"]["wrist_rot"],
                                   kin.fk(actions[i].astype(float))["left"]["wrist_rot"], atol=3e-2)


def test_camera_to_base():
    T_bc = np.eye(4)
    T_bc[:3, :3] = rz(np.pi / 2)
    T_bc[:3, 3] = [1.0, 0.0, 0.5]
    p = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    np.testing.assert_allclose(camera_to_base(p, T_bc)[0], [[1.0, 1.0, 0.5], [0.0, 0.0, 0.5]], atol=1e-12)

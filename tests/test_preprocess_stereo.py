import numpy as np

from preprocessing.hand_recon.fuse_views import fuse, triangulate
from robotryon.common.config import Config
from robotryon.embodiment.camera import CameraModel

CAM = CameraModel(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                  dist=(-0.05, 0.03, 0.0, 0.0, -0.01))
BASE = 0.063
T_L_R = np.eye(4)
T_L_R[0, 3] = BASE        # right camera sits 6.3 cm to the right of the left one


def _project(X):
    return CAM.project(X)[0]


def test_triangulation_recovers_points():
    rng = np.random.default_rng(0)
    X = rng.uniform([-0.2, -0.2, 0.3], [0.2, 0.2, 0.8], (50, 3))
    XR = X - np.array([BASE, 0, 0])
    xa = CAM.normalized(_project(X))
    xb = CAM.normalized(_project(XR))
    Xh = triangulate(xa, xb, T_L_R)
    assert np.abs(Xh - X).max() < 1e-6


def test_fuse_corrects_depth_and_complements():
    T = 12
    rng = np.random.default_rng(1)
    hand = rng.normal(0, 0.03, (21, 3))
    hand[0] = 0
    wrist = np.array([0.05, 0.05, 0.5])
    true_L = np.broadcast_to(wrist + hand, (T, 2, 21, 3)).copy()
    true_L[:, 1] += np.array([-0.15, 0, 0.05])
    true_R = true_L - np.array([BASE, 0, 0])

    def view(true, scale):
        j3 = true * scale                   # HaWoR depth off by a scale; the 2D is right
        return dict(joints3d=j3.astype(np.float32), joints2d=_project(true).astype(np.float32),
                    valid=np.ones((T, 2), bool), conf=np.ones((T, 2), np.float32),
                    img_wh=np.array([640, 480]))

    pv = {"left": view(true_L, 1.3), "right": view(true_R, 0.8)}
    pv["right"]["valid"][5:, 1] = False     # right hand lost in the right view
    out = fuse(pv, {"left": CAM, "right": CAM}, {"left": np.eye(4), "right": T_L_R},
               Config(depth_smooth=1, max_reproj_px=8.0, min_depth=0.1, max_depth=2.0))
    j = out["joints3d"]
    # wrist of every hand moved onto the true depth (same ray)
    assert np.abs(j[0, :, 0, 0] - true_L[:, 0, 0]).max() < 1e-4
    assert np.abs(j[1, :5, 0, 0] - true_R[:5, 0, 0]).max() < 1e-4
    # the lost hand is filled from the left view, transformed into the right camera
    assert out["valid"][1, 5:, 1].all() and (out["source"][1, 5:, 1] == 0).all()
    assert np.abs(j[1, 5:, 1, 0] - true_R[5:, 1, 0]).max() < 1e-4


def test_complement_skips_a_hand_the_views_disagree_on():
    T = 6
    hand = np.zeros((21, 3))
    A = np.array([0.05, 0.05, 0.5]) + hand
    j0 = np.zeros((T, 2, 21, 3))
    j0[:, 0] = A                              # left view: this hand is the left one
    j1 = np.zeros((T, 2, 21, 3))
    j1[:, 1] = A - np.array([BASE, 0, 0])     # right view: the same hand, called the right one
    v0 = np.zeros((T, 2), bool)
    v0[:, 0] = True
    v1 = v0[:, ::-1].copy()

    def view(j, v):
        return dict(joints3d=j.astype(np.float32), joints2d=np.zeros((T, 2, 21, 2), np.float32),
                    valid=v, conf=v.astype(np.float32), img_wh=np.array([640, 480]))

    out = fuse({"left": view(j0, v0), "right": view(j1, v1)}, {"left": CAM, "right": CAM},
               {"left": np.eye(4), "right": T_L_R}, Config(triangulate_wrist=False))
    assert (out["valid"] == np.stack([v0, v1])).all()          # nothing copied either way

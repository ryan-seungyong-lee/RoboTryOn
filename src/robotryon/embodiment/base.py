"""The interface every robot is used through.

The generative models never see joint angles, only images, a mask of the arm and hand, and a
17-keypoint hand skeleton. What a robot must provide is therefore small:

* ``fk(q)``: wrist pose and the 17 hand keypoints of each hand in the robot base frame;
* ``camera_pose(q, view)`` and ``project(points, view)``: the head camera(s), to draw the
  skeleton of a robot frame and to place a human hand recorded from the same viewpoint;
* ``silhouette(q, view)``: the rendered robot, which together with SAM3 gives the robot mask;
* ``retarget_hand(human_kp3d)``: the robot hand posed like a reconstructed human hand.

Arrays over hands always use the order ``(left, right)``; a hand the robot does not have is NaN.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from robotryon.skeleton.topology import NUM_HANDS, NUM_KEYPOINTS

from .spec import HANDS, EmbodimentSpec

# A keypoint off screen is still drawn (OpenCV clips the bone) unless it lies further than
# this many frame sizes outside the image, where the projection is degenerate.
OFFSCREEN_TOLERANCE = 3.0


class Embodiment(ABC):
    spec: EmbodimentSpec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def views(self) -> list[str]:
        return self.spec.views

    @property
    def joint_names(self) -> list[str | None]:
        """Layout of the recorded joint vector ``q`` (``None`` = slot not in the model)."""
        return self.spec.joints

    # --- kinematics ----------------------------------------------------------------------
    @abstractmethod
    def fk(self, q) -> dict[str, dict[str, np.ndarray]]:
        """Forward kinematics in the base frame, metres.

        Args:
            q: joint vector in ``joint_names`` order, or a ``{joint: value}`` dict.
        Returns:
            ``{hand: {"wrist_pos": (3,), "wrist_rot": (3, 3), "keypoints": (17, 3)}}`` for each
            hand the robot has (``"left"``/``"right"``); keypoint 0 is the wrist and
            ``wrist_rot`` columns are the wrist frame's axes in the base frame.
        """

    @property
    def arm_joint_slices(self) -> dict[str, slice]:
        """Slice of the joint/action vector holding each arm's joints (YAML ``arm_joints``)."""
        return {h: slice(*r) for h, r in self.spec.arm_joints.items()}

    def hand_keypoints(self, q) -> np.ndarray:
        """``(2, 17, 3)`` keypoints of ``fk(q)`` in (left, right) order, NaN for a missing hand."""
        out, f = empty_hands(), self.fk(q)
        for hi, h in enumerate(HANDS):
            if h in f:
                out[hi] = f[h]["keypoints"]
        return out

    @abstractmethod
    def camera_pose(self, q, view: str) -> np.ndarray:
        """4x4 camera-to-base pose of ``view`` (optical frame) at joints ``q``.

        ``q=None`` uses the default pose (useful for fixed or stereo-relative quantities).
        """

    @abstractmethod
    def silhouette(self, q, view: str, image_size=None) -> np.ndarray:
        """Rendered robot arm+hand mask, bool ``(H, W)``."""

    @abstractmethod
    def retarget_hand(self, human_kp3d: np.ndarray, valid=None, prev: dict | None = None):
        """Robot hand posed like a human hand (vector retargeting, see ``retarget``).

        Args:
            human_kp3d: ``(2, 21|17, 3)`` human joints (MediaPipe order) in any frame.
            valid: ``(2,)`` bool, hands to retarget (default: finite ones).
            prev: the joints returned for the previous frame of the clip; they warm-start the
                optimization and its smoothing term (None for the first frame).
        Returns:
            ``(2, 17, 3)`` robot keypoints in the same frame (NaN for skipped hands) and a
            ``{joint: value}`` dict of the retargeted robot hand joints.
        """

    # --- cameras -------------------------------------------------------------------------
    def camera(self, view: str):
        return self.spec.cameras[view].intrinsics

    def stereo_extrinsic(self, view_a: str, view_b: str) -> np.ndarray:
        """4x4 transform from ``view_b``'s camera frame to ``view_a``'s."""
        Ta, Tb = self.camera_pose(None, view_a), self.camera_pose(None, view_b)
        return np.linalg.inv(Ta) @ Tb

    def project(self, points, view: str, q=None, frame: str = "base", image_size=None):
        """``(..., 3)`` points -> ``(..., 2)`` pixels and ``(...)`` depth-ok mask.

        ``frame="base"`` needs ``q`` (the camera may move with the head);
        ``frame="camera"`` takes points already in ``view``'s optical frame.
        """
        pts = np.asarray(points, np.float64)
        if frame == "base":
            T = np.linalg.inv(self.camera_pose(q, view))
            pts = pts @ T[:3, :3].T + T[:3, 3]
        elif frame != "camera":
            raise ValueError(f"frame must be 'base' or 'camera', got {frame!r}")
        return self.camera(view).project(pts, image_size)

    def keypoints_2d(self, points, view: str, image_size, q=None, frame: str = "base"):
        """Project ``(2, 17, 3)`` keypoints into the skeleton format.

        Returns ``kp`` float32 ``(2, 17, 2)`` (0 where invisible) and ``vis`` bool ``(2, 17)``:
        in front of the camera and within ``OFFSCREEN_TOLERANCE`` frames of the image.
        """
        px, ok = self.project(points, view, q=q, frame=frame, image_size=image_size)
        W, H = image_size
        m = OFFSCREEN_TOLERANCE
        u, v = px[..., 0], px[..., 1]
        with np.errstate(invalid="ignore"):
            vis = ok & (u > -m * W) & (u < (1 + m) * W) & (v > -m * H) & (v < (1 + m) * H)
        kp = np.where(vis[..., None], px, 0.0).astype(np.float32)
        return kp, vis

    def keypoints_in_view(self, q, view: str, image_size):
        """Projected skeleton ``(kp, vis)`` of the robot at joints ``q`` (see keypoints_2d)."""
        return self.keypoints_2d(self.hand_keypoints(q), view, image_size, q=q)


def empty_hands(d: int = 3) -> np.ndarray:
    return np.full((NUM_HANDS, NUM_KEYPOINTS, d), np.nan)


__all__ = ["Embodiment", "HANDS", "OFFSCREEN_TOLERANCE", "empty_hands"]

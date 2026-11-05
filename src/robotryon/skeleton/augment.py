"""Training-time skeleton augmentation.

Human keypoints at inference come from a reconstructed and retargeted hand, which is imprecise.
To make the generative models treat the skeleton as a coarse guide rather than an exact pose,
robot keypoints are corrupted during training:

* each keypoint is displaced by up to half the length of the finger segment it belongs to,
* the whole skeleton is dropped with probability ``p_drop_all``,
* each keypoint is dropped independently with probability ``p_drop_keypoint``.

For video, the displacement is sampled once per clip and per keypoint and then held fixed in
time, so the skeleton flow stays temporally coherent; keypoint dropout is also per clip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .topology import segment_length_per_keypoint


@dataclass
class SkeletonAugment:
    max_shift_frac: float = 0.5
    p_drop_all: float = 0.2
    p_drop_keypoint: float = 0.2
    enabled: bool = True

    @classmethod
    def from_config(cls, cfg) -> "SkeletonAugment":
        if cfg is None:
            return cls(enabled=False)
        return cls(**{k: cfg[k] for k in ("max_shift_frac", "p_drop_all", "p_drop_keypoint",
                                          "enabled") if k in cfg})

    def __call__(self, kp: np.ndarray, vis: np.ndarray, rng: np.random.Generator,
                 temporal: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Corrupt keypoints.

        Args:
            kp: ``(H, 17, 2)`` for one frame or ``(T, H, 17, 2)`` for a clip, in pixels.
            vis: matching visibility, ``(H, 17)`` or ``(T, H, 17)``.
            rng: random generator.
            temporal: treat the first axis as time. Inferred from ``kp.ndim == 4`` if omitted.

        Returns:
            ``(kp, vis)``: new arrays, the inputs are not modified.
        """
        kp = np.array(kp, dtype=np.float32, copy=True)
        vis = np.array(vis, dtype=bool, copy=True)
        if not self.enabled:
            return kp, vis
        if temporal is None:
            temporal = kp.ndim == 4
        shape = kp.shape[1:-1] if temporal else kp.shape[:-1]          # (H, 17)

        if rng.random() < self.p_drop_all:
            vis[...] = False
            return kp, vis

        # Segment length from the clip median, so a single bad frame cannot blow up the shift.
        seg = segment_length_per_keypoint(kp, vis)
        if temporal:
            seg = np.median(np.where(vis, seg, np.nan), axis=0)
            seg = np.nan_to_num(seg, nan=0.0)
        radius = self.max_shift_frac * seg * np.sqrt(rng.random(shape))   # uniform in a disc
        angle = rng.uniform(0.0, 2 * np.pi, shape)
        shift = np.stack([radius * np.cos(angle), radius * np.sin(angle)], -1).astype(np.float32)
        kp = kp + (shift[None] if temporal else shift)

        keep = rng.random(shape) >= self.p_drop_keypoint
        vis &= keep[None] if temporal else keep
        return kp, vis

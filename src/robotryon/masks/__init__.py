"""Shape-agnostic arm/hand masks for keyframe translation (paper Sec. 3.2, App. B.2.3).

The mask must hide the silhouette of whatever embodiment it covers, so it is never the raw
segmentation: every arm component becomes its convex hull, which is then dilated. During training
the dilation is random per arm, the manipulated object is cut out of the hole with some
probability, and the whole sample is jointly rescaled (field-of-view crop / pad). At inference the
dilation is fixed and the eroded object mask is always subtracted.
"""
from .hull import arm_hulls, fill_hulls, scale_hulls
from .augment import MaskAug, fov_augment, inference_mask
from .reference import robot_only

__all__ = ["arm_hulls", "fill_hulls", "scale_hulls", "MaskAug", "fov_augment", "inference_mask",
           "robot_only"]

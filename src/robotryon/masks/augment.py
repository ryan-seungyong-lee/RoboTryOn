"""Train-time mask randomisation, the fixed inference mask, and the joint field-of-view augmentation.

Training (``MaskAug.train_mask``):
  * one random dilation per arm hull, so two arms in one frame already carry different
    hole-to-arm ratios and the model cannot read the hand size off the hole;
  * episodes flagged as "bonus" (thin human arms) get a raised dilation floor, because a hull
    hugging a thin arm leaves no room for a thicker robot arm;
  * with probability ``p_object_cut``, where an object mask exists, the eroded object is cut out
    of the hole (a concave hole), which is what inference always does.

Inference (``inference_mask``): every hull dilated by a fixed kernel (the middle of the training
range plus the bonus), then the eroded object mask is subtracted so the object is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .hull import fill_hulls


def _ellipse(k: int) -> np.ndarray:
    k = int(k) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


@dataclass
class MaskAug:
    dilate_min: int = 3
    dilate_max: int = 71
    bonus: int = 21                 # added to the dilation floor for "bonus" (human) episodes
    p_object_cut: float = 0.5
    obj_erode_min: int = 3
    obj_erode_max: int = 21
    min_keep_frac: float = 0.4      # reject a cut that leaves less than this fraction of the hole

    @classmethod
    def from_config(cls, cfg) -> "MaskAug":
        cfg = cfg or {}
        return cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})

    def sample_dilate(self, rng: np.random.Generator, bonus: bool = False) -> int:
        lo = self.dilate_min + (self.bonus if bonus else 0)
        return int(rng.integers(lo, max(lo + 1, self.dilate_max) + 1)) | 1

    def sample_obj_erode(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.obj_erode_min, self.obj_erode_max + 1)) | 1

    def infer_dilate(self) -> int:
        """Fixed inference dilation: middle of the training range plus the bonus."""
        return (self.dilate_min + self.dilate_max) // 2 + self.bonus

    def train_mask(self, polys: list[np.ndarray], width: int, height: int,
                   rng: np.random.Generator, bonus: bool = False,
                   object_mask: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        """Randomised training hole. Returns ``(mask uint8 {0,1}, object erode kernel or 0)``."""
        mask = np.zeros((height, width), np.uint8)
        for one in fill_hulls(polys, width, height):
            mask |= cv2.dilate(one, _ellipse(self.sample_dilate(rng, bonus)))
        if object_mask is None:
            return mask, 0
        if float(rng.random()) >= self.p_object_cut:
            return mask, 0
        om = (np.asarray(object_mask) > 0).astype(np.uint8)
        if om.sum() == 0:
            return mask, 0
        e = self.sample_obj_erode(rng)
        om = cv2.erode(om, _ellipse(e))
        cut = (mask.astype(bool) & ~om.astype(bool)).astype(np.uint8)
        hull_px = int(mask.sum())
        if cut.sum() > 0 and (hull_px == 0 or cut.sum() / hull_px > self.min_keep_frac):
            return cut, e
        return mask, 0


def inference_mask(polys: list[np.ndarray], width: int, height: int, dilate: int,
                   object_mask: np.ndarray | None = None, obj_erode: int = 9) -> np.ndarray:
    """Fixed-dilation hull mask minus the eroded object -> ``(H, W)`` uint8 {0, 1}."""
    mask = np.zeros((height, width), np.uint8)
    for one in fill_hulls(polys, width, height):
        mask |= one
    if mask.any():
        mask = cv2.dilate(mask, _ellipse(dilate))
    if object_mask is not None:
        om = (np.asarray(object_mask) > 0).astype(np.uint8)
        if obj_erode > 0:
            om = cv2.erode(om, _ellipse(obj_erode))
        mask = (mask.astype(bool) & ~om.astype(bool)).astype(np.uint8)
    return mask


def fov_augment(imgs: list[Image.Image], rng: np.random.Generator, crop_min: float,
                scale_max: float = 1.0, flip_prob: float = 0.0) -> tuple[list[Image.Image], float]:
    """One shared field-of-view draw for every stream of a sample (all views included).

    ``scale ~ U[crop_min, max(1, scale_max)]``. Below 1 a ``scale``-sized window is cropped and
    resized back; above 1 the image is pasted at a random offset onto a black canvas ``scale``
    times larger and resized back, so the scene shrinks and black borders appear (the mask is
    black there too: "not a hole"). The model thus learns that the camera's field of view is not
    fixed.
    """
    if not imgs:
        return imgs, 1.0
    w, h = imgs[0].size
    if any(im.size != (w, h) for im in imgs):
        raise ValueError("fov_augment needs equally sized images")
    scale = float(rng.uniform(crop_min, max(1.0, float(scale_max))))
    if scale > 1.0 + 1e-6:
        cw, ch = int(round(w * scale)), int(round(h * scale))
        left = int(rng.integers(0, cw - w + 1))
        top = int(rng.integers(0, ch - h + 1))
        flip = float(rng.random()) < flip_prob
        out = []
        for im in imgs:
            canvas = Image.new(im.mode, (cw, ch), 0)
            canvas.paste(im, (left, top))
            im2 = canvas.resize((w, h), Image.LANCZOS)
            out.append(im2.transpose(Image.FLIP_LEFT_RIGHT) if flip else im2)
        return out, scale
    cw, ch = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    left = int(rng.integers(0, w - cw + 1)) if w > cw else 0
    top = int(rng.integers(0, h - ch + 1)) if h > ch else 0
    flip = float(rng.random()) < flip_prob
    out = []
    for im in imgs:
        im2 = im.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)
        out.append(im2.transpose(Image.FLIP_LEFT_RIGHT) if flip else im2)
    return out, scale

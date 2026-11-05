"""RoPE position ids for one or two camera views in one token sequence.

FLUX.2 gives every image token a 4D position ``(T, H, W, L)``. The pipeline sets ``L = 0`` for all
image tokens (only text tokens use it), so for images ``L`` is a free axis; stereo generation
uses it as the camera-view index. ``T`` keeps its learned meaning: targets at ``T = 0`` and the
reference at ``T = 10``. The token order is::

    [target_v0, target_v1, ..., reference_v0, reference_v1, ...]

With one view this is byte-identical to the stock ``_prepare_latent_ids`` + ``_prepare_image_ids``.
"""
from __future__ import annotations

import torch

TARGET_T = 0
REFERENCE_T = 10


def view_image_ids(sizes: list[tuple[int, int]], view_ids: list[int],
                   t_coords: list[int]) -> torch.Tensor:
    """``sizes[i] = (h, w)`` token grid of image i -> ``(1, sum h*w, 4)`` int64 position ids."""
    if not (len(sizes) == len(view_ids) == len(t_coords)):
        raise ValueError("one size, view id and t coordinate per image")
    out = [torch.cartesian_prod(torch.tensor([int(t)]), torch.arange(h), torch.arange(w),
                                torch.tensor([int(v)]))
           for (h, w), v, t in zip(sizes, view_ids, t_coords)]
    return torch.cat(out, dim=0).unsqueeze(0)


def fill_ids(h: int, w: int, n_views: int) -> torch.Tensor:
    """Ids for ``n_views`` targets followed by ``n_views`` references, all on an ``h x w`` grid."""
    return view_image_ids([(h, w)] * (2 * n_views), list(range(n_views)) * 2,
                          [TARGET_T] * n_views + [REFERENCE_T] * n_views)

"""A fourth RoPE axis for the camera view.

Both stereo views are generated in one sequence: each view is encoded by the VAE on its own
and the token runs are concatenated. To let attention relate the same (t, h, w) position in the
two cameras, every token also carries a view coordinate.

Passing four axes to the stock LTX-2 rotary embedding would re-cut its channel-to-frequency
map (``dim // (2 * num_axes)`` frequencies per axis) and perturb every pretrained head at once.
Instead the t/h/w frequencies stay exactly where they were, and the view rotation is written
into the last ``VIEW_CHANNELS_PER_HEAD`` channels of every head's slice (64 of 2048 channels),
using the lowest frequency of the table. With two views the coordinates sit a quarter turn
apart. Three-axis coordinates fall through to the original implementation unchanged, so mono
clips see the pretrained embedding exactly.
"""

from __future__ import annotations

import types

import torch

VIEW_CHANNELS_PER_HEAD = 2
BASE_VIEWS = 2


def view_channel_indices(dim: int, num_heads: int, per_head: int, device=None) -> torch.Tensor:
    """Channels taken by the view axis: the last ``per_head`` of each head's slice, head-major."""
    head_half = (dim // 2) // num_heads
    if per_head >= head_half:
        raise ValueError(f"per_head={per_head} would take a head's whole {head_half} channels")
    return torch.tensor([h * head_half + head_half - per_head + c
                         for h in range(num_heads) for c in range(per_head)],
                        dtype=torch.long, device=device)


def _forward_with_view_axis(self, coords: torch.Tensor, device=None):
    """Drop-in for ``LTX2AudioVideoRotaryPosEmbed.forward`` accepting a 4th (view) axis."""
    device = device or coords.device
    if coords.shape[1] != 4:
        return self._forward_3axis(coords, device)
    if self.rope_type != "split":
        raise NotImplementedError(f"view axis needs rope_type='split', got {self.rope_type!r}")
    if coords.ndim == 4:                                      # [start, end) -> midpoints
        start, end = coords.chunk(2, dim=-1)
        coords = ((start + end) / 2.0).squeeze(-1)
    spatial, view = coords[:, :3], coords[:, 3]

    # t/h/w exactly as the original: three axes, dim // 6 frequencies each
    max_positions = (self.base_num_frames, self.base_height, self.base_width)
    grid = torch.stack([spatial[:, i] / max_positions[i] for i in range(3)], dim=-1).to(device)
    freqs_dtype = torch.float64 if self.double_precision else torch.float32
    pow_indices = torch.pow(self.theta, torch.linspace(0.0, 1.0, steps=self.dim // 6,
                                                       dtype=freqs_dtype, device=device))
    freqs_base = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)
    angles = ((grid.unsqueeze(-1) * 2 - 1) * freqs_base).transpose(-1, -2).flatten(2)

    # pad the angle (cos 1 / sin 0 in the original) to dim // 2
    pad = self.dim // 2 - angles.shape[-1]
    if pad < 0:
        raise RuntimeError(f"angle width {angles.shape[-1]} exceeds dim//2")
    if pad > 0:
        angles = torch.cat([torch.zeros_like(angles[:, :, :pad]), angles], dim=-1)

    # the view rotation in the tail channels of every head
    idx = view_channel_indices(self.dim, self.num_attention_heads, VIEW_CHANNELS_PER_HEAD,
                               device=device)
    view_angle = ((view.to(device).unsqueeze(-1) / BASE_VIEWS * 2 - 1)
                  * freqs_base[:VIEW_CHANNELS_PER_HEAD])
    angles = angles.clone()
    angles[:, :, idx] = view_angle.repeat(1, 1, self.num_attention_heads).to(angles.dtype)

    b, n = angles.shape[:2]
    cos = angles.cos().reshape(b, n, self.num_attention_heads, -1).swapaxes(1, 2)
    sin = angles.sin().reshape(b, n, self.num_attention_heads, -1).swapaxes(1, 2)
    return cos, sin


def install_view_axis_rope(transformer_or_rope) -> None:
    """Patch the video rope (only) to accept view coordinates. Idempotent."""
    rope = getattr(transformer_or_rope, "rope", transformer_or_rope)
    if getattr(rope, "_view_axis_installed", False):
        return
    rope._forward_3axis = types.MethodType(type(rope).forward, rope)
    rope.forward = types.MethodType(_forward_with_view_axis, rope)
    rope._view_axis_installed = True


def add_view_coord(coords: torch.Tensor, view: int) -> torch.Tensor:
    """``(B, 3, N, 2)`` -> ``(B, 4, N, 2)`` with the view axis constant at ``[view, view+1)``."""
    b, _, n, _ = coords.shape
    v = torch.empty(b, 1, n, 2, dtype=coords.dtype, device=coords.device)
    v[..., 0] = float(view)
    v[..., 1] = float(view + 1)
    return torch.cat([coords, v], dim=1)


def set_view_coord(coords: torch.Tensor, view: int) -> torch.Tensor:
    """Set the view axis, appending it if the coordinates have only three axes."""
    if coords.shape[1] != 4:
        return add_view_coord(coords, view)
    out = coords.clone()
    out[:, 3, :, 0] = float(view)
    out[:, 3, :, 1] = float(view + 1)
    return out


@torch.no_grad()
def check_bands_preserved(rope, device=None, atol: float = 1e-3) -> dict:
    """Verify the patched rope changes only the view channels (run before training).

    Channels count as changed when they differ by more than ``atol``: the 3-axis and 4-axis
    paths are computed separately and can differ by float rounding on some CPUs."""
    device = device or torch.device("cpu")
    base = rope.prepare_video_coords(1, 4, 9, 16, device=device, fps=20.0)
    cos3, sin3 = rope._forward_3axis(base, device)
    cos4, sin4 = rope(add_view_coord(base, 0))
    idx = view_channel_indices(rope.dim, rope.num_attention_heads, VIEW_CHANNELS_PER_HEAD,
                               device=cos3.device)
    keep = torch.ones(rope.dim // 2, dtype=torch.bool, device=cos3.device)
    keep[idx] = False
    keep = keep.view(rope.num_attention_heads, -1)[None, :, None, :]
    d_cos, d_sin = (cos4 - cos3).abs(), (sin4 - sin3).abs()
    changed = ((d_cos > atol) | (d_sin > atol)).any(dim=2).squeeze(0)
    return {"n_changed": int(changed.sum()),
            "n_expected": rope.num_attention_heads * VIEW_CHANNELS_PER_HEAD,
            "heads_touched": int(changed.any(dim=-1).sum()),
            "num_heads": rope.num_attention_heads,
            "max_abs_diff_off_view": float(torch.max((d_cos * keep).max(),
                                                     (d_sin * keep).max()))}

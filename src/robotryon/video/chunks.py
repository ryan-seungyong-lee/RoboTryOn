"""Splitting a long demonstration into clips the video model can generate at once.

The video VAE takes clips of ``8k + 1`` frames, and the model is trained on clips of at most
145 frames (about 7 s at 20 fps). A longer demonstration is therefore cut at its keyframes:
each clip runs from one keyframe to a later one, holds every keyframe in between, and the next
clip starts at the keyframe the previous one ended on. Adjacent clips thus **share their
boundary keyframe**, so the arm pose cannot jump at the seam, and the concatenation drops the
duplicated boundary frame of every clip after the first.

A clip is ``8k + 1`` frames long, which usually reaches past its last keyframe; those trailing
frames are anchored to nothing and are discarded at concatenation. Two corner cases follow the
original generation code:

* two consecutive keyframes further apart than the cap: the clip is stretched past the cap
  rather than leaving a hole in the video (logged as off-distribution);
* the end of the video leaves less than the minimum clip length: a short tail clip of at least
  9 frames is generated instead of dropping the end of the demonstration, pulling the last
  keyframe onto the clip's final frame if the ``8k + 1`` constraint requires it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

VAE_TEMPORAL = 8


def allowed_frame_counts(lo: int, hi: int) -> list[int]:
    """Clip lengths in ``[lo, hi]`` the VAE accepts (``F = 8k + 1``)."""
    return [f for f in range(lo, hi + 1) if (f - 1) % VAE_TEMPORAL == 0]


def next_8k1(n: int) -> int:
    """Smallest ``8k + 1`` that is >= n."""
    return n if (n - 1) % VAE_TEMPORAL == 0 else n + (VAE_TEMPORAL - (n - 1) % VAE_TEMPORAL)


def prev_8k1(n: int) -> int:
    """Largest ``8k + 1`` that is <= n (0 if none)."""
    return n - (n - 1) % VAE_TEMPORAL if n >= 1 else 0


def latent_index(pixel: int, num_frames: int) -> int:
    """Latent frame holding a pixel frame: latent 0 is pixel 0, latent j>0 is [8j-7, 8j]."""
    lat_n = (num_frames - 1) // VAE_TEMPORAL + 1
    return 0 if pixel == 0 else min((pixel + 7) // VAE_TEMPORAL, lat_n - 1)


@dataclass
class Chunk:
    """One generated clip: ``frames`` frames starting at ``start`` of the episode.

    ``kf_slots`` index into the episode's keyframe list, ``kf_local`` are the same keyframes
    as frame indices inside the clip (first is always 0).
    """
    start: int
    frames: int
    kf_slots: list[int]
    kf_local: list[int]

    def to_dict(self) -> dict:
        return asdict(self)


def _tail_chunk(kf: list[int], i: int, start: int, avail: int, log) -> Chunk | None:
    """A short last clip covering what is left after keyframe ``i``."""
    F = prev_8k1(avail)
    if F < 9:
        return None
    ids = [t for t in range(i, len(kf)) if kf[t] <= start + F - 1] or [i]
    if ids[-1] + 1 < len(kf):
        ids.append(ids[-1] + 1)                 # include the last keyframe, pulled to F-1
    local = [min(kf[t] - start, F - 1) for t in ids]
    if len(set(local)) < 2:
        return None
    log(f"  tail clip: frames {start}..{start + F - 1} ({F}f, below the training minimum) "
        f"keyframes {ids} @ {local}")
    return Chunk(start, F, ids, local)


def plan_chunks(kf: list[int], total: int, frames_range=(49, 145), log=print) -> list[Chunk]:
    """Cut a ``total``-frame video with keyframes ``kf`` (sorted frame indices) into clips.

    Each clip starts at a keyframe and ends at a later one, holding as many keyframes as fit
    in ``frames_range[1]`` frames; its length is the smallest allowed ``8k + 1`` that covers
    them (at least ``frames_range[0]``).
    """
    choices = allowed_frame_counts(*frames_range)
    lo, hi = choices[0], choices[-1]
    out: list[Chunk] = []
    i = 0
    while i < len(kf) - 1:
        j = i + 1
        while j + 1 < len(kf) and (kf[j + 1] - kf[i] + 1) <= hi:
            j += 1
        span = kf[j] - kf[i] + 1
        start = kf[i]
        if span > hi:
            # Two consecutive keyframes are further apart than the cap: stretch the clip.
            F = next_8k1(span)
            if start + F > total:
                F = prev_8k1(total - start)
                if F < lo:
                    tc = _tail_chunk(kf, i, start, total - start, log)
                    if tc is not None:
                        out.append(tc)
                    break
            log(f"  keyframes {i}->{j} are {span} frames apart, above the {hi}-frame cap: "
                f"generating {F} frames")
            out.append(Chunk(start, F, [i, j], [0, min(kf[j] - start, F - 1)]))
            i = j
            continue
        avail = total - start
        F = next((f for f in choices if f >= max(span, lo) and f <= avail), None)
        if F is None:
            # The video ends before a long-enough clip: keep the start (frame 0 of a clip is
            # the substituted keyframe) and pull the last keyframe onto the clip's last frame.
            F = max((f for f in choices if f <= avail), default=None)
            if F is None or F < lo:
                tc = _tail_chunk(kf, i, start, avail, log)
                if tc is not None:
                    out.append(tc)
                break
            log(f"  last clip: {span} -> {F} frames, last keyframe pulled to {F - 1}")
        local = [min(k - start, F - 1) for k in kf[i:j + 1]]
        out.append(Chunk(start, F, list(range(i, j + 1)), local))
        i = j
    return out


def drop_colliding_keyframes(kf_local: list[int], num_frames: int) -> list[int]:
    """Positions (into ``kf_local``) to keep so no two keyframes share a latent frame.

    Short clips can put two keyframes into the same latent slab (17 frames, keyframes at
    [0, 11, 16] -> latents [0, 2, 2]), which the pipeline cannot represent. The first and last
    keyframes are kept; colliding interior ones are dropped.
    """
    if len(kf_local) <= 2:
        return list(range(len(kf_local)))
    last = latent_index(kf_local[-1], num_frames)
    keep, seen = [], set()
    for n, loc in enumerate(kf_local):
        li = latent_index(loc, num_frames)
        if n < len(kf_local) - 1 and (li in seen or li == last):
            continue
        seen.add(li)
        keep.append(n)
    return keep


def concat_chunks(chunks: list[Chunk], clips: list[np.ndarray]) -> np.ndarray:
    """Join generated clips ``(F, H, W, 3)`` into one video.

    Clip 0 contributes frames ``0..kf_local[-1]``; every later clip starts one frame in, since
    its frame 0 is the boundary keyframe the previous clip already ended on. Frames after a
    clip's last keyframe are dropped.
    """
    parts = []
    for n, (ch, clip) in enumerate(zip(chunks, clips)):
        lo = ch.kf_local[0] + (1 if n else 0)
        parts.append(clip[lo:ch.kf_local[-1] + 1])
    return np.concatenate(parts, axis=0)

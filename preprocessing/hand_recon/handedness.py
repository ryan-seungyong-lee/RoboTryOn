"""Decide which hand every hand-detector tracklet is, before HaWoR regresses its pose.

HaWoR's pose regressor only knows right hands; a left hand is regressed on a mirrored crop.
Chirality must therefore be settled before regression. HaWoR's own rule (the majority of the
detector's per-box left/right class) is often wrong, and a hand it mislabels is regressed
through the wrong mirror branch and collides with the other hand in the same slot.

Each tracklet (one tracker id) gets a score; positive means "right hand":

* **pair**: +1 for every frame in which it is the image-right one of two simultaneous
  detections, -1 when it is the image-left one. The camera looks forward from the person's
  head, so whenever both hands are in view the left hand is on the image left.
* **stereo** (optional): with a second, horizontally displaced view of the same moment,
  +1 / -1 for every frame in which that view sees two hands and this box centre is clearly
  nearer to its image-right / image-left one, after removing the median disparity of the two
  views. It decides tracklets that never share a frame with the other hand in their own view.
* **detector**: ``w_det`` times the sum of the detector's votes (+1 right, -1 left). The weight
  is small, so the class only decides tracklets without geometric evidence, e.g. a single hand
  seen by a single camera.

A tracklet is the right hand if its score is >= 0, else the left. Should two detections of one
frame still be the same hand, the image-left one becomes the left hand.

Detections are flat arrays, at most two per frame: ``frame`` (N,) int, ``track`` (N,) tracker
id, ``x`` (N,) box-centre column, ``cls`` (N,) detector class (1 = right).
"""

from __future__ import annotations

import numpy as np


def _two_hand_frames(frame: np.ndarray, x: np.ndarray) -> dict[int, np.ndarray]:
    """frame -> sorted box-centre columns, for frames with exactly two detections."""
    out = {}
    for t in np.unique(frame):
        i = np.flatnonzero(frame == t)
        if len(i) == 2:
            out[int(t)] = np.sort(x[i])
    return out


def tracklet_scores(frame, track, x, cls, twin=None, w_det: float = 0.25,
                    margin: float = 0.5) -> dict:
    """Chirality score of every tracklet (see module docstring).

    Args:
        twin: ``(frame, x)`` of the other stereo view's detections, or None.
        margin: a stereo vote counts only when the distances to the twin's two hands differ by
            at least this fraction of their separation.
    """
    frame, track, x, cls = (np.asarray(a) for a in (frame, track, x, cls))
    vote = w_det * (2.0 * (cls > 0.5) - 1.0)
    pairs = _two_hand_frames(frame, x)
    for t in pairs:
        i = np.flatnonzero(frame == t)
        vote[i[np.argsort(x[i])]] += (-1.0, 1.0)
    if twin is not None:
        theirs = _two_hand_frames(np.asarray(twin[0]), np.asarray(twin[1], float))
        common = pairs.keys() & theirs.keys()
        dx = float(np.median([pairs[t] - theirs[t] for t in common])) if common else 0.0
        for k, t in enumerate(frame):
            if int(t) not in theirs:
                continue
            xl, xr = theirs[int(t)] + dx
            d = abs(x[k] - xl) - abs(x[k] - xr)            # > 0: nearer the twin's right hand
            if abs(d) >= margin * (xr - xl):
                vote[k] += np.sign(d)
    return {tr: float(vote[track == tr].sum()) for tr in np.unique(track)}


def assign_hands(frame, track, x, cls, twin=None, **kw) -> np.ndarray:
    """Hand of every detection, 0 = left, 1 = right (see module docstring)."""
    frame, x = np.asarray(frame), np.asarray(x)
    score = tracklet_scores(frame, track, x, cls, twin, **kw)
    hand = np.array([int(score[tr] >= 0) for tr in np.asarray(track)], int)
    for t in _two_hand_frames(frame, x):
        i = np.flatnonzero(frame == t)
        if hand[i[0]] == hand[i[1]]:
            hand[i[np.argsort(x[i])]] = (0, 1)
    return hand


def hand_boxes(frame, track, box, hand, h: int, n_frames: int):
    """Boxes of hand ``h`` over ``n_frames`` frames.

    Detections of the hand are taken as they are; gaps inside one tracklet are bridged by
    linear interpolation of its boxes where the hand has no detection.

    Returns:
        boxes (n_frames, 5) xyxy + score, has (n_frames,) bool, real (n_frames,) bool
        (a detection, not a bridge).
    """
    frame, track, box, hand = (np.asarray(a) for a in (frame, track, box, hand))
    boxes = np.zeros((n_frames, box.shape[1]), np.float64)
    m = hand == h
    boxes[frame[m]] = box[m]
    real = np.zeros(n_frames, bool)
    real[frame[m]] = True
    has = real.copy()
    for tr in np.unique(track[m]):
        k = np.flatnonzero(m & (track == tr))
        k = k[np.argsort(frame[k])]
        for a, b in zip(k[:-1], k[1:]):
            gap = np.arange(frame[a] + 1, frame[b])
            gap = gap[~has[gap]]
            if len(gap):
                w = ((gap - frame[a]) / (frame[b] - frame[a]))[:, None]
                boxes[gap] = (1 - w) * box[a] + w * box[b]
                boxes[gap, 4] = 0.0
                has[gap] = True
    return boxes, has, real

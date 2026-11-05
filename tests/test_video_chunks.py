"""Chunk planning for long videos: shared boundary keyframes, <=145 frames, 8k+1 lengths."""
import numpy as np
import pytest

from robotryon.video.chunks import (concat_chunks, drop_colliding_keyframes, latent_index,
                                    plan_chunks)

CASES = [
    ([0, 60, 130, 200, 290, 412], 413),
    ([0, 40, 80, 120, 160, 199], 200),
    ([0, 144], 145),
    ([0, 30, 100, 170, 250, 300], 330),
    ([0, 50, 96], 97),
]


@pytest.mark.parametrize("kf,total", CASES)
def test_plan(kf, total):
    chunks = plan_chunks(kf, total, (49, 145), log=lambda *_: None)
    assert chunks
    for c in chunks:
        assert (c.frames - 1) % 8 == 0 and c.frames <= 145
        assert c.start + c.frames <= total
        assert c.kf_local[0] == 0 and c.kf_local == sorted(c.kf_local)
        # the last keyframe may be pulled onto the clip's last frame at the end of the video
        assert [min(kf[s] - c.start, c.frames - 1) for s in c.kf_slots] == c.kf_local
    for a, b in zip(chunks, chunks[1:]):           # shared boundary keyframe
        assert a.kf_slots[-1] == b.kf_slots[0]
        assert a.start + a.kf_local[-1] == b.start
    assert chunks[0].start == kf[0] and chunks[-1].kf_slots[-1] == len(kf) - 1


@pytest.mark.parametrize("kf,total", CASES)
def test_concat_drops_duplicate_boundary(kf, total):
    chunks = plan_chunks(kf, total, (49, 145), log=lambda *_: None)
    clips = [np.arange(c.start, c.start + c.frames)[:, None, None, None] for c in chunks]
    full = concat_chunks(chunks, clips)[:, 0, 0, 0]
    end = chunks[-1].start + chunks[-1].kf_local[-1]
    assert full.tolist() == list(range(kf[0], end + 1)) and kf[-1] - end < 8


def test_wide_gap_is_stretched():
    chunks = plan_chunks([0, 200], 260, (49, 145), log=lambda *_: None)
    assert len(chunks) == 1 and chunks[0].frames == 201 and chunks[0].kf_local == [0, 200]


def test_short_tail():
    chunks = plan_chunks([0, 100, 120], 125, (49, 145), log=lambda *_: None)
    assert chunks[-1].frames >= 9 and (chunks[-1].frames - 1) % 8 == 0
    assert chunks[-1].start + chunks[-1].frames <= 125


def test_colliding_keyframes():
    assert latent_index(11, 17) == latent_index(16, 17) == 2
    assert drop_colliding_keyframes([0, 11, 16], 17) == [0, 2]
    assert drop_colliding_keyframes([0, 40, 96], 97) == [0, 1, 2]

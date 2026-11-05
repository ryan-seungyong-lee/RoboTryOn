"""Training keyframes: 1-4 per clip, the first frame always one of them."""
import torch

from robotryon.video.conditioning import build_step, sample_keyframe_indices

W = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.3}


def test_keyframe_sampling():
    g = torch.Generator().manual_seed(0)
    counts, last = set(), 0
    for _ in range(2000):
        F = int(torch.randint(9, 146, (1,), generator=g))
        idx = sample_keyframe_indices(F, W, 0.5, generator=g)
        assert idx[0] == 0 and idx == sorted(set(idx))
        assert 1 <= len(idx) <= 4 and idx[-1] < F
        counts.add(len(idx))
        last += idx[-1] == F - 1 and len(idx) > 1
    assert counts == {1, 2, 3, 4}
    assert 600 < last < 1200            # roughly include_last_p of the multi-keyframe clips


def test_build_step_layout():
    V, tpf, lat_f, C = 2, 6, 3, 4
    tgt = torch.randn(1, V * tpf * lat_f, C)
    ref = torch.randn(1, V * tpf * lat_f, C)
    kf = {0: torch.randn(1, V * tpf, C), 20: torch.randn(1, V * tpf, C)}
    hidden, ts, vel, pinned, app = build_step(tgt, ref, kf, torch.tensor([0.5]), tpf, V)
    assert app == [20] and hidden.shape[1] == tgt.shape[1] + V * tpf + ref.shape[1]
    assert pinned.sum() == V * tpf                               # frame 0 of each view
    per_view = tpf * lat_f
    assert torch.equal(hidden[0, per_view:per_view + tpf], kf[0][0, tpf:])
    assert (ts[0, tgt.shape[1]:] == 0).all() and (ts[0, tpf:per_view] == 500).all()

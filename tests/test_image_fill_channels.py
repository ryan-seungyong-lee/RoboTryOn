import torch

from robotryon.image.fill_channels import (FILL_IN, fill_tokens, fold_mask_to_latent, pack,
                                           pad_condition_tokens, unpack)
from robotryon.image.stereo_ids import fill_ids


def test_fold_mask_shape_and_content():
    m = torch.zeros(2, 1, 64, 96)
    m[:, :, :8, :8] = 1
    f = fold_mask_to_latent(m)
    assert f.shape == (2, 256, 4, 6)
    assert f[:, :, 0, 0].sum() == 2 * 64 and f[:, :, 1:, :].sum() == 0


def test_fill_tokens_width_and_pack_roundtrip():
    lat = torch.randn(2, 128, 4, 6)
    m = torch.zeros(2, 1, 64, 96)
    t = fill_tokens(lat, lat, m)
    assert t.shape == (2, 24, FILL_IN)
    assert torch.equal(unpack(pack(lat), 4, 6), lat)
    assert pad_condition_tokens(pack(lat)).shape == (2, 24, FILL_IN)


def test_ids_match_stock_single_view():
    ids = fill_ids(4, 6, 1)[0]
    assert ids.shape == (48, 4)
    assert (ids[:24, 0] == 0).all() and (ids[24:, 0] == 10).all() and (ids[:, 3] == 0).all()
    st = fill_ids(4, 6, 2)[0]
    assert st.shape == (96, 4)
    assert st[:, 3].tolist() == [0] * 24 + [1] * 24 + [0] * 24 + [1] * 24


def test_widen_x_embedder_zero_init():
    import torch.nn as nn
    from robotryon.image.fill_channels import widen_x_embedder

    class Cfg(dict):
        __getattr__ = dict.get

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.x_embedder = nn.Linear(128, 16, bias=False)
            self.config = Cfg(in_channels=128, out_channels=None)

        def register_to_config(self, **kw):
            self.config.update(kw)

    m = Tiny()
    x = torch.randn(3, 128)
    y0 = m.x_embedder(x)
    widen_x_embedder(m)
    assert m.x_embedder.in_features == FILL_IN and m.config["in_channels"] == FILL_IN
    assert m.config["out_channels"] == 128
    y1 = m.x_embedder(torch.cat([x, torch.randn(3, FILL_IN - 128)], -1))
    assert torch.allclose(y0, y1)

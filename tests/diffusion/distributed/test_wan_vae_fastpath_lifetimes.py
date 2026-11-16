# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Spatial fast-path gating and decode buffer lifetime regressions."""

import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard as spatial
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import forwards as fp

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@torch.no_grad()
@pytest.mark.parametrize("axis", ["height", "width"])
def test_spatial_conv2d_input_buffer_requires_installed_fastpath(monkeypatch, axis):
    monkeypatch.setattr(spatial, "_rank_world", lambda group: (0, 1))
    source = torch.nn.Conv2d(2, 2, 3, padding=1)
    conv = spatial.WanDistConv2d(source, group=None, split_dim=axis)
    x = torch.randn(3, 2, 4, 5)
    expected = source(x)
    torch.testing.assert_close(conv(x), expected, rtol=0, atol=0)
    assert conv._input_buf is None

    setattr(conv, fp.CFG_ATTR, fp.FastPathConfig())
    torch.testing.assert_close(conv(x), expected, rtol=0, atol=0)
    buf = conv._input_buf
    assert buf is not None
    torch.testing.assert_close(conv(x), expected, rtol=0, atol=0)
    assert conv._input_buf is buf

    delattr(conv, fp.CFG_ATTR)
    conv._input_buf = None
    torch.testing.assert_close(conv(x), expected, rtol=0, atol=0)
    assert conv._input_buf is None


@torch.no_grad()
@pytest.mark.parametrize("failure", [None, "callback", "decoder"])
def test_spatial_decode_releases_conv2d_input_buffers(monkeypatch, failure):
    monkeypatch.setattr(spatial, "install_wan_spatial_shard_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(spatial, "_rank_world", lambda group: (0, 1))
    decoder = torch.nn.Module()
    decoder.conv = spatial.WanDistConv2d(torch.nn.Conv2d(2, 2, 3, padding=1), group=None)
    setattr(decoder.conv, fp.CFG_ATTR, fp.FastPathConfig())
    buffers = []

    def decode(x, **kwargs):
        out = decoder.conv(x.squeeze(2)).unsqueeze(2)
        buffers.append(weakref.ref(decoder.conv._input_buf))
        if failure == "decoder":
            raise RuntimeError("decoder failed")
        return out

    decoder.forward = decode
    vae = SimpleNamespace(
        decoder=decoder,
        post_quant_conv=lambda z: z,
        config=SimpleNamespace(patch_size=None),
        clear_cache=Mock(),
        _feat_map=[],
    )
    callback = Mock(side_effect=RuntimeError("callback failed") if failure == "callback" else None)
    z = torch.randn(1, 2, 3, 4, 5)
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            spatial.spatial_shard_decode(vae, z, group=None, on_chunk=callback)
    else:
        spatial.spatial_shard_decode(vae, z, group=None, on_chunk=callback)
    assert buffers
    assert decoder.conv._input_buf is None
    assert all(ref() is None for ref in buffers)

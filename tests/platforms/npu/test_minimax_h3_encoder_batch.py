# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Encoder tile batching must preserve order and avoid padded samples."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image import MiniMaxH3ImagePipeline
from vllm_omni.platforms.npu.models import minimax_h3_image as adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def _encode_moments(self, value):
        self.calls.append(value.shape[0])
        return value * 2

    def _decode_tile(self, value):
        return value

    def prepare_decoder_weights(self, dtype):
        pass


def setup_adapter(monkeypatch, batch):
    monkeypatch.setattr(adapter.NPUExactGraphRunner, "is_supported", lambda: True)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(set_compile_mode=lambda **kw: None, config=SimpleNamespace(allow_internal_format=True)),
        raising=False,
    )
    model = MiniMaxH3ImagePipeline.__new__(MiniMaxH3ImagePipeline)
    torch.nn.Module.__init__(model)
    model.vae = FakeVAE()
    config = OmniDiffusionConfig(
        additional_config={
            "npu_vae_tile_graph": True,
            "npu_vae_encoder_precision": "fp16",
            "npu_vae_encoder_batch": batch,
        },
        enforce_eager=False,
        enable_cuda_graph_decode=True,
    )
    adapter.apply_minimax_h3_image_graphs(model, config)
    return model.vae


def test_encoder_batch_preserves_order_and_partial_tail(monkeypatch):
    vae = setup_adapter(monkeypatch, 4)
    tiles = [torch.full((1, 3, 1, 2, 2), float(i)) for i in range(7)]
    output = vae._encode_moments.map(tiles)
    assert vae.calls == [4, 3]
    assert len(output) == len(tiles)
    assert all(torch.equal(a, b * 2) for a, b in zip(output, tiles))


def test_encoder_mixed_shapes_use_individual_calls(monkeypatch):
    vae = setup_adapter(monkeypatch, 4)
    tiles = [torch.ones(1, 3, 1, 2, 2), torch.ones(1, 3, 1, 3, 3)]
    output = vae._encode_moments.map(tiles)
    assert vae.calls == [1, 1]
    assert [t.shape for t in output] == [t.shape for t in tiles]


@pytest.mark.parametrize("batch", [True, 0, 3, 32, "8"])
def test_encoder_batch_rejects_invalid_values(monkeypatch, batch):
    with pytest.raises(ValueError, match="npu_vae_encoder_batch"):
        setup_adapter(monkeypatch, batch)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in routing and weight lifetime for H3 image decode graphs."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image import MiniMaxH3ImagePipeline
from vllm_omni.platforms.npu.models import minimax_h3_image as adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Linear(2, 2)

    def prepare_decoder_weights(self, dtype=torch.bfloat16):
        self.post_quant_conv.to(dtype)
        self.decoder.to(dtype)

    def _decode_tile(self, value):
        return value.value * 2


def _model():
    model = MiniMaxH3ImagePipeline.__new__(MiniMaxH3ImagePipeline)
    torch.nn.Module.__init__(model)
    model.vae = _VAE()
    return model


def _config(**additional):
    return OmniDiffusionConfig(additional_config=additional, enforce_eager=False, enable_cuda_graph_decode=True)


def test_default_preserves_native_decoder():
    model = _model()
    original = model.vae._decode_tile
    adapter.apply_minimax_h3_image_graphs(model, _config())
    assert model.vae._decode_tile == original


def test_graph_flag_requires_boolean():
    with pytest.raises(ValueError, match="must be a bool"):
        adapter.apply_minimax_h3_image_graphs(_model(), _config(npu_vae_tile_graph="true"))


def test_enforce_eager_disables_graphs():
    model = _model()
    config = _config(npu_vae_tile_graph=True)
    config.enforce_eager = True
    original = model.vae._decode_tile
    adapter.apply_minimax_h3_image_graphs(model, config)
    assert model.vae._decode_tile == original


def test_non_image_model_is_untouched():
    model = torch.nn.Linear(2, 2)
    adapter.apply_minimax_h3_image_graphs(model, _config(npu_vae_tile_graph=True))
    assert not hasattr(model, "vae")


def test_weight_relocation_invalidates_graph_runner(monkeypatch):
    created = []

    class Runner:
        @staticmethod
        def is_supported():
            return True

        def __init__(self, **kwargs):
            created.append(self)

        def run(self, operation, inputs, constants, compute):
            return compute(*inputs)

    monkeypatch.setattr(adapter, "NPUExactGraphRunner", Runner)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(set_compile_mode=lambda **kwargs: None, config=SimpleNamespace(allow_internal_format=True)),
        raising=False,
    )
    model = _model()
    adapter.apply_minimax_h3_image_graphs(model, _config(npu_vae_tile_graph=True))
    value = SimpleNamespace(device=SimpleNamespace(type="npu"), value=torch.tensor([3.0]))
    model.vae.prepare_decoder_weights()
    assert model.vae._decode_tile(value).item() == 6
    model.vae.prepare_decoder_weights()
    model.vae._decode_tile(value)
    assert len(created) == 1
    model.vae.prepare_decoder_weights(torch.float32)
    model.vae._decode_tile(value)
    assert len(created) == 2


@pytest.mark.parametrize("initial,warn_only", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("fail", [False, True])
def test_deterministic_execution_restores_caller_settings(initial, warn_only, fail):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    model = _model()

    def forward(request):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        if fail:
            raise RuntimeError("request failure")
        return request

    model.forward = forward
    try:
        torch.use_deterministic_algorithms(initial, warn_only=warn_only)
        adapter.apply_minimax_h3_image_graphs(model, _config(npu_deterministic=True))
        assert torch.are_deterministic_algorithms_enabled() == initial
        if fail:
            with pytest.raises(RuntimeError, match="request failure"):
                model(torch.tensor([1]))
        else:
            assert model(torch.tensor([1])).item() == 1
        assert torch.are_deterministic_algorithms_enabled() == initial
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


def test_failed_graph_setup_does_not_change_determinism(monkeypatch):
    initial = torch.are_deterministic_algorithms_enabled()
    monkeypatch.setattr(adapter.NPUExactGraphRunner, "is_supported", staticmethod(lambda: False))
    with pytest.raises(RuntimeError, match="NPUGraph support"):
        adapter.apply_minimax_h3_image_graphs(_model(), _config(npu_deterministic=True, npu_vae_tile_graph=True))
    assert torch.are_deterministic_algorithms_enabled() == initial


def test_graph_off_control_can_match_internal_format(monkeypatch):
    config = SimpleNamespace(allow_internal_format=True)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(config=config), raising=False)
    model = _model()
    original = model.vae._decode_tile
    adapter.apply_minimax_h3_image_graphs(model, _config(npu_disable_internal_format=True))
    assert not config.allow_internal_format
    assert model.vae._decode_tile == original


def test_internal_format_flag_requires_boolean():
    with pytest.raises(ValueError, match="npu_disable_internal_format must be a bool"):
        adapter.apply_minimax_h3_image_graphs(_model(), _config(npu_disable_internal_format="true"))

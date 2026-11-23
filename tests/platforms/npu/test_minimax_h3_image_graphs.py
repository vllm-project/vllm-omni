# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Default routing, opt-outs and weight lifetime for H3 image graphs."""

import sys
from contextlib import nullcontext
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
        self.encoder = torch.nn.Linear(2, 2)
        self.quant_conv = torch.nn.Linear(2, 2)
        self.encode_batches = []
        self.decode_batches = []

    def prepare_decoder_weights(self, dtype=torch.bfloat16):
        self.post_quant_conv.to(dtype)
        self.decoder.to(dtype)

    def _decode_tile(self, value):
        if isinstance(value, torch.Tensor):
            self.decode_batches.append(value.shape[0])
            return value * 2
        return value.value * 2

    def _encode_moments(self, value):
        self.encode_batches.append(value.shape[0])
        return value * 3


def _model():
    model = MiniMaxH3ImagePipeline.__new__(MiniMaxH3ImagePipeline)
    torch.nn.Module.__init__(model)
    model.vae = _VAE()
    return model


def _config(**additional):
    return OmniDiffusionConfig(additional_config=additional, enforce_eager=False, enable_cuda_graph_decode=True)


def test_explicit_graph_off_preserves_native_vae():
    model = _model()
    original = model.vae._decode_tile
    original_encode = model.vae._encode_moments
    adapter.apply_minimax_h3_image_graphs(model, _config(npu_vae_tile_graph=False))
    assert model.vae._decode_tile == original
    assert model.vae._encode_moments == original_encode


def test_graph_flag_requires_boolean():
    with pytest.raises(ValueError, match="must be a bool"):
        adapter.apply_minimax_h3_image_graphs(_model(), _config(npu_vae_tile_graph="true"))


@pytest.mark.parametrize("option,value", [("enforce_eager", True), ("enable_cuda_graph_decode", False)])
def test_eager_override_disables_default_graphs(option, value):
    model = _model()
    config = _config()
    setattr(config, option, value)
    original = model.vae._decode_tile
    original_encode = model.vae._encode_moments
    adapter.apply_minimax_h3_image_graphs(model, config)
    assert model.vae._decode_tile == original
    assert model.vae._encode_moments == original_encode


def test_non_image_model_is_untouched():
    model = torch.nn.Linear(2, 2)
    adapter.apply_minimax_h3_image_graphs(model, _config())
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
        adapter.apply_minimax_h3_image_graphs(model, _config(npu_vae_tile_graph=False))
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
    with pytest.raises(ValueError, match="NPUGraph support"):
        adapter.apply_minimax_h3_image_graphs(_model(), _config(npu_deterministic=True, npu_vae_tile_graph=True))
    assert torch.are_deterministic_algorithms_enabled() == initial


def test_graph_off_control_can_match_internal_format(monkeypatch):
    config = SimpleNamespace(allow_internal_format=True)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(config=config), raising=False)
    model = _model()
    original = model.vae._decode_tile
    adapter.apply_minimax_h3_image_graphs(model, _config(npu_vae_tile_graph=False, npu_disable_internal_format=True))
    assert config.allow_internal_format
    assert model.vae._decode_tile == original


def test_internal_format_flag_requires_boolean():
    with pytest.raises(ValueError, match="npu_disable_internal_format must be a bool"):
        adapter.apply_minimax_h3_image_graphs(_model(), _config(npu_disable_internal_format="true"))


def test_graph_off_control_can_match_jit_mode(mocker, monkeypatch):
    """Matching the worker JIT mode must leave the eager VAE decoder intact."""
    set_compile_mode = mocker.Mock()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_compile_mode=set_compile_mode), raising=False)
    model = _model()
    original = model.vae._decode_tile
    adapter.apply_minimax_h3_image_graphs(model, _config(npu_vae_tile_graph=False, npu_disable_jit_compile=True))
    set_compile_mode.assert_not_called()
    assert model.vae._decode_tile == original


def test_graph_off_jit_control_requires_boolean():
    with pytest.raises(ValueError, match="npu_disable_jit_compile must be a bool"):
        adapter.apply_minimax_h3_image_graphs(_model(), _config(npu_disable_jit_compile="true"))


class _NpuTensor(torch.Tensor):
    """CPU storage with NPU routing, for testing packing without a device."""

    @property
    def device(self):
        return SimpleNamespace(type="npu")


@pytest.fixture
def graph_runtime(monkeypatch, mocker):
    class Runner:
        @staticmethod
        def is_supported():
            return True

        def __init__(self, **kwargs):
            pass

        def run(self, operation, inputs, constants, compute):
            return compute(*inputs)

    monkeypatch.setattr(adapter, "NPUExactGraphRunner", Runner)
    npu = SimpleNamespace(set_compile_mode=mocker.Mock(), config=SimpleNamespace(allow_internal_format=True))
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    autocast = mocker.patch.object(torch, "autocast", side_effect=lambda *args, **kwargs: nullcontext())
    return npu, autocast


def test_explicit_graphs_batch_decoder_and_fp16_encoder(graph_runtime):
    npu, autocast = graph_runtime
    model = _model()
    adapter.apply_minimax_h3_image_graphs(
        model, _config(npu_vae_tile_batch=32, npu_vae_encoder_precision="fp16", npu_vae_encoder_batch=8)
    )
    tiles = [torch.full((1, 2, 2), float(i)).as_subclass(_NpuTensor) for i in range(35)]
    decoded = model.vae._decode_tile.map(tiles)
    encoded = model.vae._encode_moments.map(tiles[:11])
    assert model.vae.decode_batches == [32, 3]
    assert model.vae.encode_batches == [8, 3]
    assert len(decoded) == 35 and len(encoded) == 11
    assert all(torch.equal(result, tile * 2) for result, tile in zip(decoded, tiles))
    assert all(torch.equal(result, tile * 3) for result, tile in zip(encoded, tiles))
    assert autocast.call_count == 2
    autocast.assert_called_with(device_type="npu", dtype=torch.float16, cache_enabled=False)
    npu.set_compile_mode.assert_not_called()
    assert npu.config.allow_internal_format


def test_explicit_streams_replace_default_decoder_batch(graph_runtime, monkeypatch):
    counts = []

    class Streams:
        replicas = 1

        def __init__(self, fn, *, autocast_dtype):
            self.fn = fn

        def map(self, tiles):
            counts.append(self.replicas)
            return [self.fn(tile) for tile in tiles]

    monkeypatch.setattr(adapter, "_NPUTileGraphs", Streams)
    model = _model()
    original_encode = model.vae._encode_moments
    adapter.apply_minimax_h3_image_graphs(model, _config(npu_vae_tile_streams=4, npu_vae_encoder_precision="fp32"))
    tiles = [torch.ones(1, 2, 2).as_subclass(_NpuTensor) for _ in range(3)]
    results = model.vae._decode_tile.map(tiles)
    assert counts == [4]
    assert model.vae.decode_batches == [1, 1, 1]
    assert len(results) == 3 and all(torch.equal(result, tile * 2) for result, tile in zip(results, tiles))
    assert model.vae._encode_moments == original_encode


def test_explicit_single_tile_batches_disable_default_batching(graph_runtime):
    model = _model()
    adapter.apply_minimax_h3_image_graphs(
        model, _config(npu_vae_tile_batch=1, npu_vae_encoder_precision="fp16", npu_vae_encoder_batch=1)
    )
    assert not hasattr(model.vae._decode_tile, "map")
    assert not hasattr(model.vae._encode_moments, "map")


@pytest.mark.parametrize(
    "settings,match",
    [
        ({"npu_vae_tile_streams": 4, "npu_vae_tile_batch": 32}, "batching and multiple streams"),
        ({"npu_vae_encoder_precision": "fp32", "npu_vae_encoder_batch": 8}, "requires reduced precision"),
    ],
)
def test_explicit_incompatible_choices_are_rejected(graph_runtime, settings, match):
    with pytest.raises(ValueError, match=match):
        adapter.apply_minimax_h3_image_graphs(_model(), _config(**settings))


def test_determinism_can_be_disabled():
    model = _model()
    original = model.forward
    adapter.apply_minimax_h3_image_graphs(model, _config(npu_vae_tile_graph=False, npu_deterministic=False))
    assert model.forward == original


@pytest.fixture(autouse=True)
def compatible_graph_environment(monkeypatch):
    """Host debug settings must not affect the unit test's graph admission."""
    monkeypatch.delenv("TASK_QUEUE_ENABLE", raising=False)
    monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)


def test_defaults_preserve_native_encoder_and_single_tile_decoder(graph_runtime):
    model = _model()
    original_encoder = model.vae._encode_moments
    original_decoder = model.vae._decode_tile
    adapter.apply_minimax_h3_image_graphs(model, _config())
    assert model.vae._encode_moments == original_encoder
    assert model.vae._decode_tile != original_decoder
    assert not hasattr(model.vae._decode_tile, "map")


@pytest.mark.parametrize("name,value", [("TASK_QUEUE_ENABLE", "2"), ("ASCEND_LAUNCH_BLOCKING", "1")])
@pytest.mark.parametrize("explicit", [False, True])
def test_incompatible_environment_is_rejected_before_capture(graph_runtime, monkeypatch, name, value, explicit):
    monkeypatch.setenv(name, value)
    model = _model()
    original_forward = model.forward
    original_encoder = model.vae._encode_moments
    original_decoder = model.vae._decode_tile
    config = _config(**({"npu_vae_tile_graph": True} if explicit else {}))
    if explicit:
        with pytest.raises(ValueError, match=name):
            adapter.apply_minimax_h3_image_graphs(model, config)
        assert model.forward == original_forward
    else:
        adapter.apply_minimax_h3_image_graphs(model, config)
    assert model.vae._encode_moments == original_encoder
    assert model.vae._decode_tile == original_decoder
    graph_runtime[0].set_compile_mode.assert_not_called()


def test_unavailable_graph_api_preserves_implicit_eager_default(monkeypatch):
    monkeypatch.setattr(adapter.NPUExactGraphRunner, "is_supported", staticmethod(lambda: False))
    model = _model()
    original_decoder = model.vae._decode_tile
    adapter.apply_minimax_h3_image_graphs(model, _config())
    assert model.vae._decode_tile == original_decoder


@pytest.mark.parametrize(
    "settings",
    [
        {"npu_vae_encoder_precision": "invalid"},
        {"npu_vae_tile_streams": 4, "npu_vae_tile_batch": 32},
    ],
)
def test_invalid_configuration_does_not_mutate_model_or_runtime(graph_runtime, settings):
    npu, _ = graph_runtime
    model = _model()
    original_forward = model.forward
    original_encoder = model.vae._encode_moments
    original_decoder = model.vae._decode_tile
    with pytest.raises(ValueError):
        adapter.apply_minimax_h3_image_graphs(model, _config(**settings))
    assert model.forward == original_forward
    assert model.vae._encode_moments == original_encoder
    assert model.vae._decode_tile == original_decoder
    npu.set_compile_mode.assert_not_called()
    assert npu.config.allow_internal_format


@pytest.mark.parametrize("internal_format", [False, True])
@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("graphs", [False, True])
def test_forward_restores_npu_runtime_settings(graph_runtime, monkeypatch, internal_format, jit_compile, fail, graphs):
    """Exercise the actual forward boundary, including graph-off matching controls."""
    npu, _ = graph_runtime
    npu.config.allow_internal_format = internal_format
    jit = [jit_compile]
    npu.set_compile_mode.side_effect = lambda *, jit_compile: jit.__setitem__(0, jit_compile)
    npu.is_jit_compile_false = lambda: not jit[0]
    # TorchNPU exposes allow_internal_format as write-only; its getter is in _C.
    native = SimpleNamespace(_npu_getOption=lambda name: b"enable" if npu.config.allow_internal_format else b"disable")
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(_C=native))
    model = _model()

    def forward(request):
        assert not npu.config.allow_internal_format
        assert not jit[0]
        if fail:
            raise RuntimeError("request failure")
        return request

    model.forward = forward
    adapter.apply_minimax_h3_image_graphs(
        model,
        _config(
            npu_vae_tile_graph=graphs,
            npu_disable_internal_format=not graphs,
            npu_disable_jit_compile=not graphs,
            npu_deterministic=False,
        ),
    )
    assert npu.config.allow_internal_format == internal_format
    assert jit[0] == jit_compile
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match="request failure"):
                model(torch.tensor([1]))
        else:
            assert model(torch.tensor([1])).item() == 1
        assert npu.config.allow_internal_format == internal_format
        assert jit[0] == jit_compile
    if not jit_compile:
        npu.set_compile_mode.assert_not_called()


@pytest.mark.parametrize("eager", [False, True])
def test_eager_override_precedes_environment_rejection(graph_runtime, monkeypatch, eager):
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "2")
    model = _model()
    original_decoder = model.vae._decode_tile
    config = _config(npu_vae_tile_graph=True)
    config.enforce_eager = eager
    config.enable_cuda_graph_decode = eager
    adapter.apply_minimax_h3_image_graphs(model, config)
    assert model.vae._decode_tile == original_decoder


def test_runtime_entry_failure_restores_preceding_settings(graph_runtime, monkeypatch):
    """A setter can fail after changing state, before the model is called."""
    npu, _ = graph_runtime
    jit = [True]
    npu.is_jit_compile_false = lambda: not jit[0]
    native = SimpleNamespace(_npu_getOption=lambda name: b"")
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(_C=native))

    def set_compile_mode(*, jit_compile):
        jit[0] = jit_compile
        if not jit_compile:
            raise RuntimeError("JIT setter failed")

    npu.set_compile_mode.side_effect = set_compile_mode
    model = _model()
    initial = torch.are_deterministic_algorithms_enabled()
    initial_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    adapter.apply_minimax_h3_image_graphs(model, _config())
    with pytest.raises(RuntimeError, match="JIT setter failed"):
        model(torch.tensor([1]))
    assert jit[0]
    assert npu.config.allow_internal_format
    assert torch.are_deterministic_algorithms_enabled() == initial
    assert torch.is_deterministic_algorithms_warn_only_enabled() == initial_warn

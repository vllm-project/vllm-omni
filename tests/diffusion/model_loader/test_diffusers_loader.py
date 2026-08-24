# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for the DiffusersPipelineLoader.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from vllm.config.load import LoadConfig

from vllm_omni.diffusion.config import get_current_diffusion_config, get_current_diffusion_config_or_none
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.host_weight_plan import (
    HostWeightPlan,
    HostWeightPlanResult,
    TensorBinding,
)
from vllm_omni.diffusion.models.helios import HeliosPipeline
from vllm_omni.diffusion.registry import initialize_model

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

model_path = "hf-internal-testing/tiny-helios-modular-pipe"


@pytest.fixture(scope="module")
def prefetch_helios_model():
    """Downloads the tiny helios model prior to running a test."""
    snapshot_download(model_path)


@pytest.fixture(scope="function")
def mock_tp_group(mocker):
    """Mocks the tensor parallel group; this is needed to initialize the Helios model."""
    mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1)
    mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0)
    mock_group = mocker.MagicMock()
    mock_group.world_size = 1
    mock_group.rank_in_group = 0
    mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_group)


class _DummyPipelineModel(nn.Module):
    def __init__(self, *, source_prefix: str):
        super().__init__()
        self.transformer = nn.Linear(2, 2, bias=False)
        self.vae = nn.Linear(2, 2, bias=False)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path="dummy",
                subfolder="transformer",
                revision=None,
                prefix=source_prefix,
                fall_back_to_pt=True,
            )
        ]

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name not in params:
                continue
            params[name].data.copy_(tensor.to(dtype=params[name].dtype))
            loaded.add(name)
        return loaded


def _make_loader_with_weights(weight_names: list[str]) -> DiffusersPipelineLoader:
    od_config = SimpleNamespace(
        dtype=torch.float32,
        parallel_config=SimpleNamespace(use_hsdp=False),
        quantization_config=None,
    )
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)

    loader.counter_before_loading_weights = 0.0
    loader.counter_after_loading_weights = 0.0

    def _iter_weights(_model):
        for name in weight_names:
            yield name, torch.zeros((2, 2))

    loader.get_all_weights = _iter_weights  # type: ignore[assignment]
    return loader


def _make_dlo_online_quant_config() -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model="",
        dtype=torch.float32,
        quantization_config="fp8",
        parallel_config=DiffusionParallelConfig(data_parallel_size=2),
        enable_distributed_layerwise_offload=True,
        dlo_use_allgather=True,
    )


def test_strict_check_only_validates_source_prefix_parameters():
    model = _DummyPipelineModel(source_prefix="transformer.")
    loader = _make_loader_with_weights(["transformer.weight"])

    # Should not require VAE parameters because they are outside weights_sources.
    loader.load_weights(model)


def test_strict_check_raises_when_source_parameters_are_missing():
    model = _DummyPipelineModel(source_prefix="transformer.")
    loader = _make_loader_with_weights([])

    with pytest.raises(ValueError, match="transformer.weight"):
        loader.load_weights(model)


def test_empty_source_prefix_keeps_full_model_strict_check():
    model = _DummyPipelineModel(source_prefix="")
    loader = _make_loader_with_weights(["transformer.weight"])

    with pytest.raises(ValueError, match="vae.weight"):
        loader.load_weights(model)


def test_stream_online_quant_weights_offloads_layers_after_processing():
    from vllm.model_executor.model_loader.reload.layerwise import (
        get_layerwise_info,
    )

    events: list[str] = []

    class _OnlineQuantMethod:
        uses_meta_device = True

    class _TrackedLayer(nn.Linear):
        def __init__(self, name: str):
            super().__init__(2, 2, bias=False)
            self.name = name
            self.quant_method = _OnlineQuantMethod()
            get_layerwise_info(self).load_numel_total = self.weight.numel()

        def to(self, *args, **kwargs):
            events.append(self.name)
            return super().to(*args, **kwargs)

    class _StreamingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = _TrackedLayer("first")
            self.second = _TrackedLayer("second")

    model = _StreamingModel()
    weights = iter(
        [
            ("first.weight", torch.zeros((2, 2))),
            ("second.weight", torch.zeros((2, 2))),
        ]
    )
    streamed = DiffusersPipelineLoader._stream_online_quant_weights_to_cpu(model, weights)

    assert next(streamed)[0] == "first.weight"
    get_layerwise_info(model.first).reset()
    assert next(streamed)[0] == "second.weight"
    assert events == ["first"]

    get_layerwise_info(model.second).reset()
    with pytest.raises(StopIteration):
        next(streamed)
    assert events == ["first", "second"]


def test_process_weights_skips_completed_online_quant_layer(monkeypatch):
    from unittest.mock import Mock

    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.model_loader.reload import layerwise

    class _TrackedLayer(nn.Linear):
        def __init__(self):
            super().__init__(2, 2, bias=False)
            self.to_calls: list[object] = []

        def to(self, *args, **kwargs):
            self.to_calls.append(args[0] if args else kwargs.get("device"))
            return super().to(*args, **kwargs)

    model = nn.Module()
    model.layer = _TrackedLayer()
    quant_method = Mock(spec=QuantizeMethodBase)
    quant_method.uses_meta_device = True
    model.layer.quant_method = quant_method
    model.layer._already_called_process_weights_after_loading = True
    finalize = Mock()
    monkeypatch.setattr(layerwise, "finalize_layerwise_processing", finalize)

    loader = _make_loader_with_weights([])
    loader._process_weights_after_loading(model, torch.device("cuda"))

    finalize.assert_called_once_with(model, model_config=None)
    quant_method.process_weights_after_loading.assert_not_called()
    assert model.layer.to_calls == []


class _ConfigAwareModel(nn.Module):
    def __init__(self, *, od_config):
        super().__init__()
        self.captured_config = get_current_diffusion_config()
        self.seen_config_during_init = get_current_diffusion_config_or_none()
        self.od_config = od_config


def test_initialize_model_sets_current_diffusion_config_during_model_construction(monkeypatch):
    import vllm_omni.diffusion.registry as registry_mod

    od_config = SimpleNamespace(
        model_class_name="DummyPipeline",
        parallel_config=SimpleNamespace(vae_patch_parallel_size=1, sequence_parallel_size=1),
        vae_use_slicing=False,
        vae_use_tiling=False,
    )

    monkeypatch.setattr(
        registry_mod.DiffusionModelRegistry,
        "_try_load_model_cls",
        staticmethod(lambda _name: _ConfigAwareModel),
    )
    monkeypatch.setattr(registry_mod, "_apply_sequence_parallel_if_enabled", lambda *_args, **_kwargs: None)

    model = initialize_model(od_config)

    assert model.captured_config is od_config
    assert model.seen_config_during_init is od_config
    assert get_current_diffusion_config_or_none() is None


def test_load_model_custom_pipeline_sets_current_diffusion_config(monkeypatch):
    import vllm_omni.diffusion.model_loader.diffusers_loader as loader_mod

    class _DeviceContext:
        def __init__(self, device_type: str):
            self.type = device_type

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    od_config = SimpleNamespace(
        dtype=torch.float32,
        parallel_config=SimpleNamespace(use_hsdp=False),
        quantization_config=None,
    )

    loader = DiffusersPipelineLoader(LoadConfig(), od_config)
    loader.load_weights = lambda model: None  # type: ignore[assignment]
    loader._process_weights_after_loading = lambda model, target_device: None  # type: ignore[assignment]

    monkeypatch.setattr(loader_mod, "resolve_obj_by_qualname", lambda _name: _ConfigAwareModel)
    monkeypatch.setattr(loader_mod.torch, "device", lambda _name: _DeviceContext("cpu"))

    model = loader.load_model(
        load_device="cpu",
        load_format="custom_pipeline",
        custom_pipeline_name="tests.dummy.ConfigAwarePipeline",
    )

    assert model.captured_config is od_config
    assert model.seen_config_during_init is od_config
    assert get_current_diffusion_config_or_none() is None


def test_dlo_transfers_loader_plan_and_skips_ordinary_weight_loading(monkeypatch):
    import vllm_omni.diffusion.model_loader.diffusers_loader as loader_mod

    od_config = SimpleNamespace(
        dtype=torch.float32,
        parallel_config=SimpleNamespace(use_hsdp=False, tensor_parallel_size=1),
        quantization_config=None,
        enable_distributed_layerwise_offload=True,
        dlo_use_allgather=False,
        model="unused",
    )
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)
    model = nn.Module()
    model.transformer = nn.Linear(2, 2, bias=False)
    plan = HostWeightPlan(
        backing_kind="checkpoint_mmap",
        bindings={},
    )
    loaded_ordinary_weights = False

    def load_weights(_model):
        nonlocal loaded_ordinary_weights
        loaded_ordinary_weights = True

    loader._init_from_load_format = lambda *_args, **_kwargs: model  # type: ignore[method-assign]
    loader.load_weights = load_weights  # type: ignore[method-assign]
    loader._apply_skip_softmax_calibration = lambda _model: None  # type: ignore[method-assign]
    monkeypatch.setattr(
        loader_mod,
        "build_checkpoint_mmap_plan",
        lambda *_args, **_kwargs: HostWeightPlanResult(plan),
    )

    assert loader.load_model(load_device="cpu") is model
    assert not loaded_ordinary_weights
    assert loader.take_host_weight_plan() is plan
    assert loader.take_host_weight_plan() is None


def test_dlo_plan_loads_component_sources_outside_planned_dit(monkeypatch):
    import vllm_omni.diffusion.model_loader.diffusers_loader as loader_mod

    od_config = SimpleNamespace(
        dtype=torch.float32,
        parallel_config=SimpleNamespace(use_hsdp=False, tensor_parallel_size=1),
        quantization_config=None,
        enable_distributed_layerwise_offload=True,
        dlo_use_allgather=False,
        model="unused",
    )
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)

    class MixedSourceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Linear(2, 2, bias=False)
            self.text_encoder = nn.Linear(2, 2, bias=False)
            self.weights_sources = (
                DiffusersPipelineLoader.ComponentSource("unused", None, None, prefix="transformer."),
                DiffusersPipelineLoader.ComponentSource("unused", None, None, prefix="text_encoder."),
            )
            self.loaded_weight_names: list[str] = []

        def load_weights(self, weights):
            self.loaded_weight_names = [name for name, _ in weights]
            return set(self.loaded_weight_names)

    model = MixedSourceModel()
    plan = HostWeightPlan(
        backing_kind="checkpoint_mmap",
        bindings={
            "transformer.weight": TensorBinding(
                checkpoint_key="weight",
                file_path="unused",
            )
        },
        planned_source_prefixes=frozenset({"transformer."}),
    )
    requested_prefixes: list[str] = []

    def get_weights(source, model=None):
        del model
        requested_prefixes.append(source.prefix)
        yield source.prefix + "weight", torch.ones(2, 2)

    loader._init_from_load_format = lambda *_args, **_kwargs: model  # type: ignore[method-assign]
    loader._get_weights_iterator = get_weights  # type: ignore[method-assign]
    loader._apply_skip_softmax_calibration = lambda _model: None  # type: ignore[method-assign]
    monkeypatch.setattr(
        loader_mod,
        "build_checkpoint_mmap_plan",
        lambda *_args, **_kwargs: HostWeightPlanResult(plan),
    )

    assert loader.load_model(load_device="cpu") is model
    assert requested_prefixes == ["text_encoder."]
    assert model.loaded_weight_names == ["text_encoder.weight"]
    assert loader.take_host_weight_plan() is plan


def test_dlo_plan_fallback_runs_ordinary_loader(monkeypatch):
    import vllm_omni.diffusion.model_loader.diffusers_loader as loader_mod

    od_config = SimpleNamespace(
        dtype=torch.float32,
        parallel_config=SimpleNamespace(use_hsdp=False, tensor_parallel_size=1),
        quantization_config=None,
        enable_distributed_layerwise_offload=True,
        dlo_use_allgather=False,
        model="unused",
    )
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)
    model = nn.Module()
    model.transformer = nn.Linear(2, 2, bias=False)
    calls: list[str] = []

    loader._init_from_load_format = lambda *_args, **_kwargs: model  # type: ignore[method-assign]
    loader.load_weights = lambda _model: calls.append("load")  # type: ignore[method-assign]
    loader._process_weights_after_loading = lambda *_args: calls.append("process")  # type: ignore[method-assign]
    loader._apply_skip_softmax_calibration = lambda _model: None  # type: ignore[method-assign]
    monkeypatch.setattr(
        loader_mod,
        "build_checkpoint_mmap_plan",
        lambda *_args, **_kwargs: HostWeightPlanResult(None, "not direct-compatible"),
    )

    assert loader.load_model(load_device="cpu") is model
    assert calls == ["load", "process"]
    assert loader.take_host_weight_plan() is None


def test_dlo_allgather_online_fp8_uses_ordinary_loader(monkeypatch):
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
    )

    import vllm_omni.diffusion.model_loader.diffusers_loader as loader_mod

    od_config = _make_dlo_online_quant_config()
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)
    model = nn.Module()
    model.transformer = nn.Linear(2, 2, bias=False)
    model.transformer.quant_method = object.__new__(Fp8PerTensorOnlineLinearMethod)
    model.transformer.quant_method.uses_meta_device = True
    calls: list[object] = []
    allowlist_models: list[nn.Module] = []

    original_allowlist_check = loader._unsupported_dlo_allgather_online_quant_methods

    def check_allowlist(candidate: nn.Module) -> tuple[str, ...]:
        allowlist_models.append(candidate)
        return original_allowlist_check(candidate)

    loader._init_from_load_format = lambda *_args, **_kwargs: model  # type: ignore[method-assign]
    monkeypatch.setattr(loader, "_unsupported_dlo_allgather_online_quant_methods", check_allowlist)
    loader._request_offload_after_quant = lambda _model: 1  # type: ignore[method-assign]
    loader.load_weights = (  # type: ignore[method-assign]
        lambda _model, *, stream_online_quant_to_cpu=False: calls.append(("load", stream_online_quant_to_cpu))
    )
    loader._process_weights_after_loading = lambda *_args: calls.append("process")  # type: ignore[method-assign]
    loader._apply_skip_softmax_calibration = lambda _model: None  # type: ignore[method-assign]
    monkeypatch.setattr(
        loader_mod,
        "build_checkpoint_mmap_plan",
        lambda *_args, **_kwargs: HostWeightPlanResult(
            None,
            "online quantization requires the ordinary loader",
        ),
    )

    assert loader.load_model(load_device="cpu", device=torch.device("cpu")) is model
    assert allowlist_models == [model]
    assert calls == [("load", True), "process"]
    assert loader.take_host_weight_plan() is None


def test_dlo_allgather_rejects_unvalidated_online_quant_method(monkeypatch):
    import vllm_omni.diffusion.model_loader.diffusers_loader as loader_mod

    class UnsupportedOnlineMethod:
        uses_meta_device = True

    od_config = _make_dlo_online_quant_config()
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)
    model = nn.Module()
    model.transformer = nn.Linear(2, 2, bias=False)
    model.transformer.quant_method = UnsupportedOnlineMethod()

    loader._init_from_load_format = lambda *_args, **_kwargs: model  # type: ignore[method-assign]
    loader._apply_skip_softmax_calibration = lambda _model: None  # type: ignore[method-assign]
    monkeypatch.setattr(
        loader_mod,
        "build_checkpoint_mmap_plan",
        lambda *_args, **_kwargs: HostWeightPlanResult(
            None,
            "online quantization requires the ordinary loader",
        ),
    )

    with pytest.raises(ValueError, match="per-tensor FP8 linears"):
        loader.load_model(load_device="cpu")


def test_hsdp_processes_quantized_weights_before_sharding(mocker):
    import vllm_omni.diffusion.model_loader.diffusers_loader as loader_mod
    from vllm_omni.diffusion.offloader.module_collector import PipelineModules

    od_config = SimpleNamespace(
        dtype=torch.float32,
        parallel_config=SimpleNamespace(
            use_hsdp=True,
            hsdp_replicate_size=1,
            hsdp_shard_size=2,
        ),
        quantization_config=None,
    )
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)
    loader.quant_config = object()

    model = nn.Module()
    model.transformer = nn.Linear(2, 2, bias=False)
    events: list[str] = []

    loader._init_from_load_format = mocker.Mock(return_value=model)  # type: ignore[method-assign]
    loader.load_weights = mocker.Mock(side_effect=lambda _model: events.append("load"))  # type: ignore[method-assign]
    loader._process_weights_after_loading = mocker.Mock(  # type: ignore[method-assign]
        side_effect=lambda _model, _device: events.append("process")
    )
    mocker.patch.object(
        loader_mod.ModuleDiscovery,
        "discover",
        return_value=PipelineModules(
            dits=[model.transformer],
            dit_names=["transformer"],
            vaes=[],
            encoders=[],
            encoder_names=[],
            resident_modules=[],
            resident_names=[],
        ),
    )
    mocker.patch(
        "vllm_omni.diffusion.quantization.hsdp_fp8.prepare_fp8_layers_for_fsdp",
        side_effect=lambda _model: events.append("prepare"),
    )
    mocker.patch.object(
        loader_mod,
        "apply_hsdp_to_model",
        side_effect=lambda *_args, **_kwargs: events.append("shard"),
    )

    loader._load_model_with_hsdp(torch.device("cpu"))

    assert events == ["load", "process", "prepare", "shard"]


def test_get_all_weights(prefetch_helios_model, mock_tp_group):
    """Ensure that get all weights on a tiny model resolves to nonempty weights."""
    od_config = OmniDiffusionConfig(
        model_class_name="HeliosPipeline",
        model=model_path,
    )
    loader = DiffusersPipelineLoader(
        load_config=LoadConfig(),
        od_config=od_config,
    )
    pipeline = HeliosPipeline(od_config=od_config)

    weights = list(loader.get_all_weights(pipeline))
    assert len(weights) > 0


def test_load_model(prefetch_helios_model, mock_tp_group):
    """Ensure that load model creates an instance of the expected pipeline class."""
    od_config = OmniDiffusionConfig(
        model_class_name="HeliosPipeline",
        model=model_path,
    )
    loader = DiffusersPipelineLoader(
        load_config=LoadConfig(),
        od_config=od_config,
    )
    model = loader.load_model(load_device="cpu")
    assert isinstance(model, HeliosPipeline)

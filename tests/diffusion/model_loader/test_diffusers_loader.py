# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Tests for the DiffusersPipelineLoader.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from vllm.config.load import LoadConfig

from vllm_omni.diffusion.config import get_current_diffusion_config, get_current_diffusion_config_or_none
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.host_weight_plan import (
    HostWeightPlan,
    HostWeightPlanResult,
    TensorBinding,
)
from vllm_omni.diffusion.model_loader.host_weights import source_identity as source_identity_module
from vllm_omni.diffusion.models.helios import HeliosPipeline
from vllm_omni.diffusion.models.host_weight_contract import FinalLayoutModelContract
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


class _HWRTransformer(nn.Module):
    host_weight_restore_contract = FinalLayoutModelContract(
        implementation_id="test-hwr-transformer",
        version="1",
    )

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.float32).to(torch.bfloat16).reshape(2, 2))

    def validate_restored_host_weights(self):
        assert self.weight.dtype is torch.bfloat16


class _HWRPipeline(nn.Module):
    def __init__(self, source_root: Path):
        super().__init__()
        self.transformer = _HWRTransformer()
        self.load_count = 0
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=str(source_root),
                subfolder=None,
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]

    def load_weights(self, weights):
        loaded: set[str] = set()
        params = dict(self.named_parameters())
        for name, tensor in weights:
            if name in params:
                params[name].data.copy_(tensor.to(dtype=params[name].dtype))
                loaded.add(name)
        self.load_count += 1
        return loaded


def _hwr_config(model: str | Path, root: Path, *, mode: str = "preferred") -> SimpleNamespace:
    return SimpleNamespace(
        model=str(model),
        dtype=torch.bfloat16,
        host_weight_runtime_mode=mode,
        host_weight_runtime_root=str(root),
        enable_distributed_layerwise_offload=True,
        dlo_use_allgather=False,
        lora_path=None,
        quantization_config=None,
        diffusion_attention_config=None,
        parallel_config=SimpleNamespace(
            use_hsdp=False,
            data_parallel_size=1,
            sequence_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            cfg_parallel_size=1,
            enable_expert_parallel=False,
            ulysses_degree=1,
            ring_degree=1,
            allgather_degree=1,
            ulysses_mode="strict",
        ),
    )


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


def test_hwr_cold_publication_and_warm_restore_skip_ordinary_dit_loading(
    tmp_path: Path,
    monkeypatch,
):
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    save_file(
        {"weight": torch.arange(4, dtype=torch.float32).to(torch.bfloat16).reshape(2, 2)},
        str(canonical_root / "model.safetensors"),
    )
    store_root = tmp_path / "hwr-store"
    hash_calls = 0
    original_sha256 = source_identity_module._sha256_file

    def counted_sha256(path: Path, state: object) -> str:
        nonlocal hash_calls
        hash_calls += 1
        return original_sha256(path, state)  # type: ignore[arg-type]

    monkeypatch.setattr(source_identity_module, "_sha256_file", counted_sha256)

    def make_loader() -> tuple[DiffusersPipelineLoader, _HWRPipeline]:
        loader = DiffusersPipelineLoader(LoadConfig(), _hwr_config(canonical_root, store_root))
        pipeline = _HWRPipeline(canonical_root)
        monkeypatch.setattr(loader, "_init_from_load_format", lambda *args, **kwargs: pipeline)
        return loader, pipeline

    cold_loader, cold_model = make_loader()
    cold = cold_loader.load_model(load_device="cpu", device=torch.device("cpu"))
    assert cold is cold_model
    assert cold_model.load_count == 1
    assert cold_loader._hwr_state is not None

    warm_loader, warm_model = make_loader()
    monkeypatch.setattr(
        warm_loader,
        "_process_weights_after_loading",
        lambda *args, **kwargs: pytest.fail("warm HWR restore re-entered byte-changing finalization"),
    )
    warm = warm_loader.load_model(load_device="cpu", device=torch.device("cpu"))

    assert warm is warm_model
    assert warm_model.load_count == 0
    assert torch.equal(warm_model.transformer.weight, cold_model.transformer.weight)
    from vllm_omni.diffusion.offloader.startup import take_offload_startup_state

    startup_state = take_offload_startup_state(warm)
    assert startup_state is not None
    warm_plan = startup_state.host_weight_plan
    assert warm_plan is not None
    assert warm_plan.lease_carrier is not None
    warm_plan.lease_carrier.close()
    assert hash_calls == 1
    assert len(tuple((store_root / "source-digests-v1" / "entries").glob("*.json"))) == 1


def test_hwr_commit_failure_discards_model_and_reloads_without_hwr_or_mmap(tmp_path: Path, monkeypatch):
    from vllm_omni.diffusion.model_loader import diffusers_loader as loader_module

    loader = DiffusersPipelineLoader(LoadConfig(), _hwr_config(tmp_path, tmp_path / "store"))
    models: list[_DummyPipelineModel] = []

    def init_model(*args, **kwargs):
        del args, kwargs
        model = _DummyPipelineModel(source_prefix="transformer.")
        models.append(model)
        return model

    def commit_error(*args, **kwargs):
        del args, kwargs
        raise loader_module._HWRCommitError("restore commit failed")

    monkeypatch.setattr(loader, "_init_from_load_format", init_model)
    monkeypatch.setattr(loader, "_get_weight_sources", lambda _model: ())
    monkeypatch.setattr(loader, "_resolve_hwr", commit_error)
    monkeypatch.setattr(loader, "load_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "_process_weights_after_loading", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "_apply_skip_softmax_calibration", lambda *args, **kwargs: None)

    recovered = loader.load_model(load_device="cpu", device=torch.device("cpu"))

    assert len(models) == 2
    assert recovered is models[1]
    assert loader.take_host_weight_plan() is None


def test_required_hwr_miss_fails_before_ordinary_loading_or_publication(
    tmp_path: Path,
    monkeypatch,
):
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    save_file(
        {"weight": torch.arange(4, dtype=torch.float32).to(torch.bfloat16).reshape(2, 2)},
        str(canonical_root / "model.safetensors"),
    )
    loader = DiffusersPipelineLoader(
        LoadConfig(),
        _hwr_config(canonical_root, tmp_path / "empty-store", mode="required"),
    )
    pipeline = _HWRPipeline(canonical_root)
    monkeypatch.setattr(loader, "_init_from_load_format", lambda *args, **kwargs: pipeline)

    with pytest.raises(RuntimeError, match="Host Weight Runtime resolution failed"):
        loader.load_model(load_device="cpu", device=torch.device("cpu"))

    assert pipeline.load_count == 0
    assert loader.take_host_weight_plan() is None


def _make_dlo_online_quant_config() -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model="",
        dtype=torch.float32,
        quantization_config="fp8",
        parallel_config=DiffusionParallelConfig(data_parallel_size=2),
        enable_distributed_layerwise_offload=True,
        dlo_use_allgather=True,
    )


@pytest.mark.parametrize(
    ("dist_offload", "use_allgather", "mode"),
    [
        (False, False, "preferred"),
        (True, True, "preferred"),
        (True, False, "disabled"),
    ],
)
def test_hwr_disabled_for_noneligible_dlo_paths_without_store_interaction(
    monkeypatch,
    tmp_path,
    dist_offload,
    use_allgather,
    mode,
):
    """Disabled and AllGather paths must never construct or probe HWR."""
    from vllm_omni.host_weight_runtime import HostWeightRuntime

    root = tmp_path / "must-not-be-touched"
    loader = DiffusersPipelineLoader(LoadConfig(), _hwr_config("dummy-model", root, mode=mode))
    model = _DummyPipelineModel(source_prefix="transformer.")
    modules = SimpleNamespace(dit_names=("transformer",), dits=(model.transformer,))

    def unexpected_store_construction(*args, **kwargs):
        raise AssertionError(f"HWR store interaction was not eligible: {args}, {kwargs}")

    monkeypatch.setattr(HostWeightRuntime, "from_config", unexpected_store_construction)
    assert (
        loader._resolve_hwr(
            model,
            modules,
            dist_offload=dist_offload,
            use_allgather=use_allgather,
            load_format="default",
            sources=tuple(model.weights_sources),
        )
        is None
    )
    assert not root.exists()


def test_required_hwr_rejects_a_model_without_a_restore_contract(tmp_path):
    loader = DiffusersPipelineLoader(
        LoadConfig(),
        _hwr_config("dummy-model", tmp_path / "store", mode="required"),
    )
    model = _DummyPipelineModel(source_prefix="transformer.")
    modules = SimpleNamespace(dit_names=("transformer",), dits=(model.transformer,))

    with pytest.raises(ValueError, match="restore contract"):
        loader._resolve_hwr(
            model,
            modules,
            dist_offload=True,
            use_allgather=False,
            load_format="default",
            sources=tuple(model.weights_sources),
        )


@pytest.mark.parametrize("offline", [False, True])
def test_prepare_weights_honors_component_index_and_explicit_override(tmp_path, mocker, offline):
    import vllm_omni.diffusion.model_loader.diffusers_loader as loader_mod

    snapshot = tmp_path / "snapshot"
    transformer = snapshot / "transformer"
    transformer.mkdir(parents=True)
    indexed_files = [f"diffusion_pytorch_model-{index:05d}-of-00002.safetensors" for index in (1, 2)]
    stale_file = "diffusion_pytorch_model-00001-of-00008.safetensors"
    index_path = transformer / "diffusion_pytorch_model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {f"weight.{index}": filename for index, filename in enumerate(indexed_files)}})
    )
    for filename in indexed_files + [stale_file]:
        (transformer / filename).touch()

    mocker.patch.object(loader_mod.huggingface_hub.constants, "HF_HUB_OFFLINE", offline)

    def download_index(*, filename, **_kwargs):
        if filename == "transformer/diffusion_pytorch_model.safetensors.index.json":
            return str(index_path)
        raise loader_mod.huggingface_hub.errors.EntryNotFoundError(filename)

    hub_api = mocker.Mock()
    hub_api.hf_hub_download.side_effect = download_index
    mocker.patch.object(loader_mod, "hf_api", return_value=hub_api)
    indexed_download = mocker.patch.object(loader_mod, "download_weights_from_hf_specific", return_value=str(snapshot))
    generic_download = mocker.patch.object(loader_mod, "download_weights_from_hf", return_value=str(snapshot))

    loader = _make_loader_with_weights([])
    cache_dir = str(tmp_path / "cache")
    loader.load_config.download_dir = cache_dir
    folder, files, use_safetensors = loader._prepare_weights(
        "org/model",
        subfolder="transformer",
        revision="revision",
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )

    assert folder == str(transformer)
    assert [str(transformer / filename) for filename in indexed_files] == files
    assert use_safetensors
    assert hub_api.hf_hub_download.call_count == 2
    hub_api.hf_hub_download.assert_any_call(
        repo_id="org/model",
        filename="transformer/diffusion_pytorch_model.safetensors.index.json",
        cache_dir=cache_dir,
        revision="revision",
        local_files_only=offline,
    )
    indexed_download.assert_called_once_with(
        model_name_or_path="org/model",
        cache_dir=cache_dir,
        allow_patterns=[f"transformer/{filename}" for filename in indexed_files],
        revision="revision",
        ignore_patterns=loader.load_config.ignore_patterns,
        require_all=True,
    )

    _, override_files, _ = loader._prepare_weights(
        "org/model",
        subfolder="transformer",
        revision="revision",
        fall_back_to_pt=True,
        allow_patterns_overrides=[stale_file],
    )
    assert override_files == [str(transformer / stale_file)]
    generic_download.assert_called_once_with(
        "org/model",
        cache_dir,
        [stale_file],
        "revision",
        subfolder="transformer",
        ignore_patterns=loader.load_config.ignore_patterns,
    )


def test_prepare_local_weights_honors_component_index(tmp_path):
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    indexed_file = "diffusion_pytorch_model-00001-of-00001.safetensors"
    stale_file = "diffusion_pytorch_model-00001-of-00008.safetensors"
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": indexed_file}})
    )
    for filename in (indexed_file, stale_file):
        (transformer / filename).touch()

    folder, files, use_safetensors = _make_loader_with_weights([])._prepare_weights(
        tmp_path,
        subfolder="transformer",
        revision=None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )

    assert folder == str(transformer)
    assert files == [str(transformer / indexed_file)]
    assert use_safetensors


def test_prepare_weights_rejects_polluted_offline_cache_without_index(tmp_path, mocker):
    import vllm_omni.diffusion.model_loader.diffusers_loader as loader_mod

    snapshot = tmp_path / "snapshot"
    transformer = snapshot / "transformer"
    transformer.mkdir(parents=True)
    for shard_count in (4, 8):
        for shard in range(1, shard_count + 1):
            (transformer / f"diffusion_pytorch_model-{shard:05d}-of-{shard_count:05d}.safetensors").touch()

    mocker.patch.object(loader_mod.huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    hub_api = mocker.Mock()
    hub_api.hf_hub_download.side_effect = loader_mod.huggingface_hub.errors.EntryNotFoundError("index is not cached")
    mocker.patch.object(loader_mod, "hf_api", return_value=hub_api)
    mocker.patch.object(loader_mod, "download_weights_from_hf", return_value=str(snapshot))

    loader = _make_loader_with_weights([])
    with pytest.raises(ValueError, match="conflicting shard totals"):
        loader._prepare_weights(
            "org/model",
            subfolder="transformer",
            revision="revision",
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )

    assert hub_api.hf_hub_download.call_count == 2
    assert all(call.kwargs["local_files_only"] for call in hub_api.hf_hub_download.call_args_list)


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

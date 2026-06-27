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
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
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

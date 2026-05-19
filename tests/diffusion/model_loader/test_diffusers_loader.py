# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.config import get_current_diffusion_config, get_current_diffusion_config_or_none
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.gguf_adapters import get_gguf_adapter
from vllm_omni.diffusion.registry import initialize_model

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


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
    loader = object.__new__(DiffusersPipelineLoader)
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


def test_qwen_model_class_selects_qwen_gguf_adapter():
    od_config = type(
        "Config",
        (),
        {
            "model_class_name": "QwenImagePipeline",
            "tf_model_config": {"model_type": "qwen_image"},
        },
    )()
    source = DiffusersPipelineLoader.ComponentSource(
        model_or_path="dummy",
        subfolder="transformer",
        revision=None,
        prefix="transformer.",
    )

    adapter = get_gguf_adapter("dummy.gguf", object(), source, od_config)

    assert adapter.__class__.__name__ == "QwenImageGGUFAdapter"


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

    loader = object.__new__(DiffusersPipelineLoader)
    loader.load_weights = lambda model: None  # type: ignore[assignment]
    loader._process_weights_after_loading = lambda model, target_device: None  # type: ignore[assignment]
    loader._is_gguf_quantization = lambda _od_config: False  # type: ignore[assignment]

    monkeypatch.setattr(loader_mod, "resolve_obj_by_qualname", lambda _name: _ConfigAwareModel)
    monkeypatch.setattr(loader_mod.torch, "device", lambda _name: _DeviceContext("cpu"))

    model = loader.load_model(
        od_config,
        load_device="cpu",
        load_format="custom_pipeline",
        custom_pipeline_name="tests.dummy.ConfigAwarePipeline",
    )

    assert model.captured_config is od_config
    assert model.seen_config_during_init is od_config
    assert get_current_diffusion_config_or_none() is None

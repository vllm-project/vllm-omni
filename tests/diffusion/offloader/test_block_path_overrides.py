# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""User ``block_paths`` overrides and staging around layerwise hooks."""

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

from tests.diffusion.offloader.helpers import _DummyBlock, _PlainEncoder, _SingleBlockModel, patch_offload_runtime
from vllm_omni.diffusion.offloader import component_utils as component_utils_module
from vllm_omni.diffusion.offloader import layerwise_backend as layerwise_backend_module
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.component_utils import LayerwiseAwareStager, non_block_modules
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
from vllm_omni.diffusion.offloader.plan_resolver import resolve_offload_plan

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

CPU = torch.device("cpu")


@pytest.fixture
def patched_offload_runtime(monkeypatch):
    patch_offload_runtime(monkeypatch, layerwise_backend_module.current_omni_platform)


class _RecordingStager:
    """Stand-in for PinnedModuleStager that records which modules it owns."""

    instances: ClassVar[list["_RecordingStager"]] = []

    def __init__(self, module, device, *, pin_memory, cache_retention=None):
        self.modules = (module,) if isinstance(module, nn.Module) else tuple(module)
        self.loads = 0
        self.offloads = 0
        _RecordingStager.instances.append(self)

    def load(self):
        self.loads += 1

    def offload(self):
        self.offloads += 1


@pytest.fixture
def recording_stager(monkeypatch):
    _RecordingStager.instances = []
    monkeypatch.setattr(component_utils_module, "PinnedModuleStager", _RecordingStager)
    return _RecordingStager


class _QwenLikeModel(nn.Module):
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 2)
        self.layers = nn.ModuleList([_DummyBlock() for _ in range(num_layers)])
        self.norm = nn.LayerNorm(2)


class _InnerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = _QwenLikeModel()


class _StagedWrapper(nn.Module):
    """MAGI-2 shape: an on-demand wrapper around the real encoder."""

    _omni_supports_layerwise_staging: ClassVar[bool] = True

    def __init__(self):
        super().__init__()
        self.module = _InnerEncoder()
        self._stager = LayerwiseAwareStager(self, self.module, CPU, pin_memory=False)

    def load_to_device(self):
        self._stager.load()

    def offload_to_cpu(self):
        self._stager.offload()


class _LegacyStagedEncoder(nn.Module):
    """On-demand encoder that always stages its whole module."""

    def __init__(self):
        super().__init__()
        self.module = _InnerEncoder()

    def load_to_device(self):
        return None

    def offload_to_cpu(self):
        return None


def _pipeline(encoder: nn.Module, *, on_demand: bool):
    class Pipeline(nn.Module):
        _dit_modules: ClassVar[list[str]] = ["transformer"]
        _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
        _vae_modules: ClassVar[list[str]] = []
        _resident_modules: ClassVar[list[str]] = []
        _offload_plan = OffloadPlan(
            on_demand_component_paths=frozenset({"text_encoder"}) if on_demand else frozenset(),
        )

        def __init__(self):
            super().__init__()
            self.transformer = _SingleBlockModel(num_blocks=3)
            self.text_encoder = encoder

    return Pipeline()


def _config(components, block_paths=None) -> OffloadConfig:
    return OffloadConfig(
        strategy=OffloadStrategy.LAYER_WISE,
        pin_cpu_memory=False,
        components=frozenset(components),
        block_paths=block_paths or {},
    )


def _od_config(public):
    return SimpleNamespace(
        diffusion_offload_config=public,
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
        dlo_use_allgather=True,
        dlo_chunk_size_mb=64,
        dlo_resident_layers=0,
        pin_cpu_memory=True,
        parallel_config=None,
        model="/fake/model",
    )


class TestSchema:
    def test_list_and_mapping_forms_reach_the_offload_config(self):
        config = OffloadConfig.from_od_config(
            _od_config(
                {
                    "mode": "layer",
                    "components": ["dit", "text_encoder"],
                    "layer_options": {
                        "dit": {"block_paths": ["blocks"]},
                        "text_encoder": {"block_paths": {"text_encoder": ["module.text_model.layers"]}},
                    },
                }
            )
        )

        assert config.block_paths_for("dit", "transformer_2") == ("blocks",)
        assert config.block_paths_for("text_encoder", "text_encoder") == ("module.text_model.layers",)
        assert config.block_paths_for("text_encoder", "text_encoder_2") is None

    @pytest.mark.parametrize(
        ("component", "block_paths", "error", "match"),
        [
            ("text_encoder", "text_model.layers", TypeError, "non-empty list"),
            ("text_encoder", [], TypeError, "non-empty list"),
            ("text_encoder", {}, ValueError, "at least one component path"),
            ("text_encoder", ["a", "a"], ValueError, "must not repeat"),
            ("text_encoder", [" a"], ValueError, "surrounding spaces"),
            ("dit", ["inner.blocks"], ValueError, "direct DiT attributes"),
        ],
    )
    def test_invalid_block_paths_are_rejected(self, component, block_paths, error, match):
        with pytest.raises(error, match=match):
            OffloadConfig.from_od_config(
                _od_config(
                    {
                        "mode": "layer",
                        "components": [component],
                        "layer_options": {component: {"block_paths": block_paths}},
                    }
                )
            )


class TestResolver:
    def test_text_encoder_override_streams_a_model_without_a_plan(self):
        pipeline = _pipeline(_PlainEncoder(), on_demand=False)

        resolved = resolve_offload_plan(pipeline, _config({"text_encoder"}, {"text_encoder": ("encoder.block",)}))

        (encoder,) = resolved.encoders
        assert encoder.stacks[0].blocks == tuple(pipeline.text_encoder.encoder.block)

    def test_dit_override_replaces_the_class_declaration(self):
        pipeline = _pipeline(_PlainEncoder(), on_demand=False)
        pipeline.transformer.alt_blocks = nn.ModuleList([_DummyBlock() for _ in range(2)])

        resolved = resolve_offload_plan(pipeline, _config({"dit"}, {"dit": ("alt_blocks",)}))

        assert resolved.dits[0].stacks[0].blocks == tuple(pipeline.transformer.alt_blocks)

    def test_wrong_path_lists_the_candidates(self):
        pipeline = _pipeline(_PlainEncoder(), on_demand=False)

        with pytest.raises(ValueError, match=r"Candidates on text_encoder: encoder\.block"):
            resolve_offload_plan(pipeline, _config({"text_encoder"}, {"text_encoder": ("encoder.blocks",)}))

    def test_unknown_component_path_is_rejected(self):
        pipeline = _pipeline(_PlainEncoder(), on_demand=False)

        with pytest.raises(ValueError, match="unknown text_encoder path"):
            resolve_offload_plan(
                pipeline, _config({"text_encoder"}, {"text_encoder": {"text_encoder_2": ("encoder.block",)}})
            )

    def test_on_demand_encoder_must_support_hook_aware_staging(self):
        pipeline = _pipeline(_LegacyStagedEncoder(), on_demand=True)

        with pytest.raises(ValueError, match="does not support streaming its blocks"):
            resolve_offload_plan(pipeline, _config({"text_encoder"}, {"text_encoder": ("module.text_model.layers",)}))


class TestNonBlockModules:
    def test_returns_the_largest_block_free_submodules(self):
        encoder = _InnerEncoder()
        text_model = encoder.text_model

        found = non_block_modules(encoder, [text_model.layers])

        assert found == [text_model.embed_tokens, text_model.norm]

    def test_rejects_a_tensor_owned_by_a_block_ancestor(self):
        encoder = _InnerEncoder()
        encoder.text_model.scale = nn.Parameter(torch.ones(1))

        with pytest.raises(ValueError, match="text_model owns tensors next to streamed blocks"):
            non_block_modules(encoder, [encoder.text_model.layers])


class TestLayerwiseAwareStager:
    def test_without_hooks_the_whole_module_is_staged(self, recording_stager):
        wrapper = _StagedWrapper()

        wrapper.load_to_device()
        wrapper.offload_to_cpu()

        (stager,) = recording_stager.instances
        assert stager.modules == (wrapper.module,)
        assert (stager.loads, stager.offloads) == (1, 1)

    def test_layerwise_selection_without_the_encoder_stages_it_whole(self, recording_stager, patched_offload_runtime):
        pipeline = _pipeline(_StagedWrapper(), on_demand=True)
        backend = LayerWiseOffloadBackend(_config({"dit"}), CPU)

        backend.enable(pipeline)
        pipeline.text_encoder.load_to_device()

        assert not getattr(pipeline.text_encoder, "_omni_layerwise_enabled", False)
        assert recording_stager.instances[-1].modules == (pipeline.text_encoder.module,)
        backend.disable()

    def test_hooked_blocks_are_left_to_the_hooks(self, recording_stager, patched_offload_runtime):
        pipeline = _pipeline(_StagedWrapper(), on_demand=True)
        backend = LayerWiseOffloadBackend(
            _config({"dit", "text_encoder"}, {"text_encoder": ("module.text_model.layers",)}), CPU
        )

        backend.enable(pipeline)
        pipeline.text_encoder.load_to_device()

        text_model = pipeline.text_encoder.module.text_model
        assert pipeline.text_encoder._omni_layerwise_enabled
        assert all(block._hook_registry.get_hook("layerwise_offload") is not None for block in text_model.layers)
        (stager,) = recording_stager.instances
        assert stager.modules == (text_model.embed_tokens, text_model.norm)

        pipeline.text_encoder.offload_to_cpu()
        assert stager.offloads == 1
        backend.disable()

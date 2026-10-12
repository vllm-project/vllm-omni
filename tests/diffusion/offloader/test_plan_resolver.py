# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the shared offload plan resolver."""

from typing import ClassVar

import pytest
import torch
from torch import nn

from tests.diffusion.offloader.helpers import (
    _DummyBlock,
    _PlainEncoder,
    _SingleBlockModel,
    _StagedEncoder,
    _StagedVAE,
    patch_offload_runtime,
)
from vllm_omni.diffusion.offloader import layerwise_backend as layerwise_backend_module
from vllm_omni.diffusion.offloader import plan_resolver as plan_resolver_module
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
from vllm_omni.diffusion.offloader.plan_resolver import resolve_offload_plan

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

CPU = torch.device("cpu")


@pytest.fixture
def patched_offload_runtime(monkeypatch):
    patch_offload_runtime(monkeypatch, layerwise_backend_module.current_omni_platform)


def _config(
    components: set[str] | None = None,
    strategy: OffloadStrategy = OffloadStrategy.LAYER_WISE,
    **kwargs,
) -> OffloadConfig:
    return OffloadConfig(
        strategy=strategy,
        pin_cpu_memory=False,
        components=None if components is None else frozenset(components),
        **kwargs,
    )


def _stack_of(component):
    assert len(component.stacks) == 1
    return component.stacks[0]


class _WanStylePipeline(nn.Module):
    """Streamable DiT plus a plan-declared, replicated text encoder."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []
    _offload_plan = OffloadPlan(
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("encoder.block",)},
        encoder_dlo_weight_replication=frozenset({"text_encoder"}),
    )

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel(num_blocks=4)
        self.text_encoder = _PlainEncoder()
        self.vae = _StagedVAE()


class _StagedComponentPipeline(nn.Module):
    """MiniMax-H3 shape: two encoder stacks plus pipeline-staged components."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []
    _offload_plan = OffloadPlan(
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("vision.blocks", "text_model.layers")},
        on_demand_component_paths=frozenset({"text_encoder", "vae"}),
        resident_dit_paths=frozenset({"transformer"}),
    )

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel(num_blocks=4)
        self.text_encoder = _StagedEncoder()
        self.vae = _StagedVAE()


class _LegacyScanPipeline(nn.Module):
    """No SupportsComponentDiscovery declaration and no OffloadPlan."""

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel(num_blocks=3)
        self.text_encoder = _PlainEncoder()


class TestProducerPrecedence:
    def test_plan_block_attrs_win_over_class_attributes(self):
        class TwoStackDiT(nn.Module):
            _layerwise_offload_blocks_attrs = ["blocks"]

            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([_DummyBlock() for _ in range(2)])
                self.alt_blocks = nn.ModuleList([_DummyBlock() for _ in range(3)])

        class Pipeline(nn.Module):
            _offload_plan = OffloadPlan(block_attrs={"transformer": ("alt_blocks",)})

            def __init__(self):
                super().__init__()
                self.transformer = TwoStackDiT()

        pipeline = Pipeline()
        resolved = resolve_offload_plan(pipeline, _config({"dit"}))

        stack = _stack_of(resolved.dits[0])
        assert stack.attrs == ("alt_blocks",)
        assert list(stack.blocks) == list(pipeline.transformer.alt_blocks)

    def test_class_attributes_are_used_without_a_plan(self):
        pipeline = _LegacyScanPipeline()
        resolved = resolve_offload_plan(pipeline, _config({"dit"}))

        assert list(_stack_of(resolved.dits[0]).blocks) == list(pipeline.transformer.blocks)

    def test_declared_component_type_wins_over_leaf_name(self):
        class Pipeline(nn.Module):
            _dit_modules: ClassVar[list[str]] = ["transformer"]
            _encoder_modules: ClassVar[list[str]] = ["mllm"]
            _vae_modules: ClassVar[list[str]] = []
            _resident_modules: ClassVar[list[str]] = []
            _offload_plan = OffloadPlan(
                encoder_component_types={"mllm": "text_encoder"},
                encoder_block_attrs={"mllm": ("encoder.block",)},
            )

            def __init__(self):
                super().__init__()
                self.transformer = _SingleBlockModel(num_blocks=2)
                self.mllm = _PlainEncoder()

        resolved = resolve_offload_plan(Pipeline(), _config({"text_encoder"}))

        encoder = resolved.encoders[0]
        assert encoder.path == "mllm"
        assert encoder.selected
        assert len(_stack_of(encoder).blocks) == 2

    def test_component_declaration_wins_over_attribute_scan(self):
        class Pipeline(nn.Module):
            _dit_modules: ClassVar[list[str]] = ["custom_dit"]
            _encoder_modules: ClassVar[list[str]] = []
            _vae_modules: ClassVar[list[str]] = []
            _resident_modules: ClassVar[list[str]] = []

            def __init__(self):
                super().__init__()
                self.custom_dit = _SingleBlockModel(num_blocks=2)
                # Would be picked up by the deprecated attribute scan.
                self.transformer = _SingleBlockModel(num_blocks=2)

        resolved = resolve_offload_plan(Pipeline(), _config({"dit"}))

        assert [component.path for component in resolved.dits] == ["custom_dit"]

    def test_attribute_scan_warns_once_per_pipeline_class(self, monkeypatch):
        warnings: list[str] = []

        class _Recorder:
            def warning(self, message, *args):
                warnings.append(message % args if args else message)

            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        monkeypatch.setattr(plan_resolver_module, "logger", _Recorder())

        class UndeclaredPipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = _SingleBlockModel(num_blocks=2)

        resolve_offload_plan(UndeclaredPipeline(), _config({"dit"}))
        resolve_offload_plan(UndeclaredPipeline(), _config({"dit"}))

        scan_warnings = [message for message in warnings if "SupportsComponentDiscovery" in message]
        assert len(scan_warnings) == 1
        assert "deprecated" in scan_warnings[0]


class TestResolvedTopology:
    def test_encoder_plan_resolves_one_stack_per_declared_path(self):
        pipeline = _WanStylePipeline()
        resolved = resolve_offload_plan(pipeline, _config({"dit", "text_encoder"}))

        encoder = resolved.encoders[0]
        assert encoder.selected and not encoder.on_demand
        assert list(_stack_of(encoder).blocks) == list(pipeline.text_encoder.encoder.block)
        assert [component.path for component in resolved.vaes] == ["vae"]

    def test_staged_plan_resolves_every_declared_stack(self):
        pipeline = _StagedComponentPipeline()
        resolved = resolve_offload_plan(pipeline, _config({"text_encoder"}))

        encoder = resolved.encoders[0]
        assert encoder.on_demand
        assert [len(stack.blocks) for stack in encoder.stacks] == [2, 2]
        # An explicit selection never stages a VAE through the legacy plan.
        assert not resolved.vaes[0].on_demand
        # DiT is unselected, so it keeps no streaming topology.
        assert resolved.dits[0].selected is False
        assert resolved.dits[0].stacks == ()

    def test_omitted_selector_keeps_the_legacy_staged_vae(self):
        resolved = resolve_offload_plan(_StagedComponentPipeline(), _config())

        assert resolved.vaes[0].on_demand
        assert resolved.encoders[0].on_demand
        assert resolved.dits[0].selected

    def test_on_demand_only_plan_resolves_without_stacks(self):
        class Pipeline(nn.Module):
            _dit_modules: ClassVar[list[str]] = ["transformer"]
            _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
            _vae_modules: ClassVar[list[str]] = ["vae"]
            _resident_modules: ClassVar[list[str]] = []
            _offload_plan = OffloadPlan(on_demand_component_paths=frozenset({"text_encoder", "vae"}))

            def __init__(self):
                super().__init__()
                self.transformer = _SingleBlockModel(num_blocks=2)
                self.text_encoder = _StagedEncoder()
                self.vae = _StagedVAE()

        resolved = resolve_offload_plan(Pipeline(), _config({"text_encoder"}))

        assert resolved.encoders[0].on_demand
        assert resolved.encoders[0].stacks == ()

    def test_module_strategy_resolves_selection_without_stacks(self):
        resolved = resolve_offload_plan(
            _WanStylePipeline(),
            _config({"dit", "text_encoder"}, strategy=OffloadStrategy.MODEL_LEVEL),
        )

        assert [component.path for component in resolved.components] == ["transformer", "text_encoder", "vae"]
        assert all(component.stacks == () for component in resolved.components)
        assert resolved.encoders[0].selected

    def test_module_strategy_leaves_staged_residency_to_the_pipeline(self):
        resolved = resolve_offload_plan(
            _StagedComponentPipeline(),
            _config({"dit", "text_encoder"}, strategy=OffloadStrategy.MODEL_LEVEL),
        )

        assert not any(component.on_demand for component in resolved.components)


class TestResidentLayers:
    def _dlo_config(self, resident_layers: int, components: set[str] | None = None) -> OffloadConfig:
        return _config(
            components,
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            dlo_transfers={"dit": "rank-local", "text_encoder": "rank-local"},
            dlo_resident_layers=resident_layers,
        )

    def test_resident_head_splits_the_declared_dit(self):
        pipeline = _StagedComponentPipeline()
        resolved = resolve_offload_plan(pipeline, self._dlo_config(2, {"dit"}))

        stack = _stack_of(resolved.dits[0])
        assert stack.resident_head == 2
        assert list(stack.resident) == list(pipeline.transformer.blocks)[:2]
        assert list(stack.streaming) == list(pipeline.transformer.blocks)[2:]

    def test_resident_count_beyond_the_stack_keeps_every_block(self):
        resolved = resolve_offload_plan(_StagedComponentPipeline(), self._dlo_config(9, {"dit"}))

        stack = _stack_of(resolved.dits[0])
        assert stack.resident_head == 4
        assert stack.streaming == ()

    def test_resident_head_leaving_one_streaming_block_is_rejected(self):
        with pytest.raises(ValueError, match="leaves only one streaming block"):
            resolve_offload_plan(_StagedComponentPipeline(), self._dlo_config(3, {"dit"}))

    def test_legacy_single_streaming_block_stays_resident(self):
        resolved = resolve_offload_plan(_StagedComponentPipeline(), self._dlo_config(3))

        stack = _stack_of(resolved.dits[0])
        assert stack.resident_head == 4
        assert stack.streaming == ()

    def test_resident_layers_without_a_declared_path_is_rejected(self):
        with pytest.raises(ValueError, match="no matching resident_dit_paths"):
            resolve_offload_plan(_WanStylePipeline(), self._dlo_config(2, {"dit"}))


class TestValidation:
    def test_selected_dit_requires_a_discovered_module(self):
        pipeline = nn.Module()
        pipeline.text_encoder = _PlainEncoder()

        with pytest.raises(ValueError, match="No DiT/transformer modules found"):
            resolve_offload_plan(pipeline, _config({"dit"}))

    def test_model_level_selection_requires_both_swap_sides(self):
        pipeline = nn.Module()
        pipeline.transformer = _SingleBlockModel(num_blocks=2)

        with pytest.raises(ValueError, match="requires an encoder execution stage"):
            resolve_offload_plan(pipeline, _config({"dit"}, strategy=OffloadStrategy.MODEL_LEVEL))

    def test_selected_text_encoder_requires_a_matching_module(self):
        pipeline = nn.Module()
        pipeline.transformer = _SingleBlockModel(num_blocks=2)

        with pytest.raises(ValueError, match="No text encoder modules found"):
            resolve_offload_plan(pipeline, _config({"text_encoder"}))

    def test_selected_encoder_requires_a_streamable_or_staged_plan(self):
        pipeline = nn.Module()
        pipeline.transformer = _SingleBlockModel(num_blocks=2)
        pipeline.text_encoder = _PlainEncoder()

        with pytest.raises(ValueError, match="Selected text encoder 'text_encoder' requires"):
            resolve_offload_plan(pipeline, _config({"text_encoder"}))

    def test_staged_component_requires_the_explicit_lifecycle(self):
        class Pipeline(nn.Module):
            _offload_plan = OffloadPlan(on_demand_component_paths=frozenset({"text_encoder"}))

            def __init__(self):
                super().__init__()
                self.transformer = _SingleBlockModel(num_blocks=2)
                self.text_encoder = _PlainEncoder()

        with pytest.raises(ValueError, match="must implement load_to_device"):
            resolve_offload_plan(Pipeline(), _config({"text_encoder"}))

    def test_single_block_dit_is_rejected_for_layer_offload(self):
        pipeline = nn.Module()
        pipeline.transformer = _SingleBlockModel(num_blocks=1)

        with pytest.raises(ValueError, match="requires at least two streamable"):
            resolve_offload_plan(pipeline, _config({"dit"}))

    def test_encoder_allgather_requires_declared_replication(self):
        config = _config(
            {"text_encoder"},
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            dp_size=2,
            dlo_transfers={"dit": "rank-local", "text_encoder": "allgather"},
        )

        with pytest.raises(ValueError, match="not declared replicated"):
            resolve_offload_plan(_StagedComponentPipeline(), config)

    def test_two_dits_cannot_own_the_same_blocks(self):
        class Pipeline(nn.Module):
            _dit_modules: ClassVar[list[str]] = ["transformer", "transformer_2"]
            _encoder_modules: ClassVar[list[str]] = []
            _vae_modules: ClassVar[list[str]] = []
            _resident_modules: ClassVar[list[str]] = []

            def __init__(self):
                super().__init__()
                self.transformer = _SingleBlockModel(num_blocks=2)
                self.transformer_2 = _SingleBlockModel(num_blocks=2)
                self.transformer_2.blocks = self.transformer.blocks

        with pytest.raises(ValueError, match="claimed by both"):
            resolve_offload_plan(Pipeline(), _config({"dit"}))

    def test_legacy_selection_keeps_unstreamable_components(self):
        pipeline = nn.Module()
        pipeline.transformer = _SingleBlockModel(num_blocks=1)
        pipeline.text_encoder = _PlainEncoder()

        resolved = resolve_offload_plan(pipeline, _config())

        assert resolved.dits[0].stacks == ()
        assert resolved.encoders[0].stacks == ()


class TestPurity:
    def _snapshot(self, pipeline: nn.Module):
        return {
            "devices": [(name, tensor.device) for name, tensor in pipeline.named_parameters()],
            "attrs": {name: sorted(vars(module)) for name, module in pipeline.named_modules()},
        }

    def test_resolution_does_not_touch_the_model(self):
        pipeline = _StagedComponentPipeline()
        before = self._snapshot(pipeline)

        resolve_offload_plan(pipeline, _config({"dit", "text_encoder"}))

        assert self._snapshot(pipeline) == before
        assert pipeline.text_encoder.offload_calls == 0
        assert pipeline.text_encoder.to_calls == 0
        assert pipeline.vae.to_calls == 0

    def test_resolution_is_repeatable(self):
        pipeline = _WanStylePipeline()
        config = _config({"dit", "text_encoder"})

        first = resolve_offload_plan(pipeline, config)
        second = resolve_offload_plan(pipeline, config)

        assert [component.path for component in first.components] == [component.path for component in second.components]
        for left, right in zip(first.components, second.components, strict=True):
            assert left.selected == right.selected
            assert left.on_demand == right.on_demand
            assert [list(stack.blocks) for stack in left.stacks] == [list(stack.blocks) for stack in right.stacks]

    def test_backend_enable_rejects_before_touching_earlier_components(self, patched_offload_runtime):
        class Pipeline(nn.Module):
            _dit_modules: ClassVar[list[str]] = ["transformer"]
            _encoder_modules: ClassVar[list[str]] = ["text_encoder", "text_encoder_2"]
            _vae_modules: ClassVar[list[str]] = ["vae"]
            _resident_modules: ClassVar[list[str]] = []
            _offload_plan = OffloadPlan(
                encoder_component_types={
                    "text_encoder": "text_encoder",
                    "text_encoder_2": "text_encoder",
                },
                encoder_block_attrs={"text_encoder": ("encoder.block",)},
            )

            def __init__(self):
                super().__init__()
                self.transformer = _SingleBlockModel(num_blocks=2)
                self.text_encoder = _PlainEncoder()
                self.text_encoder_2 = _PlainEncoder()
                self.vae = _StagedVAE()

        pipeline = Pipeline()
        backend = LayerWiseOffloadBackend(_config({"text_encoder"}), CPU)

        with pytest.raises(ValueError, match="Selected text encoder 'text_encoder_2' requires"):
            backend.enable(pipeline)

        assert not backend.enabled
        assert not getattr(pipeline.text_encoder, "_omni_layerwise_enabled", False)
        for block in pipeline.text_encoder.encoder.block:
            assert getattr(block, "_hook_registry", None) is None
        assert pipeline.vae.to_calls == 0

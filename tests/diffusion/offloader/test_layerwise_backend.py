# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for LayerwiseOffloadHook and LayerWiseOffloadBackend utilities."""

from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate

import vllm_omni.diffusion.offloader.layerwise_backend as layerwise_backend_module
import vllm_omni.diffusion.offloader.tensor_utils as tensor_utils_module
from tests.diffusion.offloader.helpers import (
    DummyStream,
    _DummyBlock,
    _PlainEncoder,
    _SingleBlockModel,
    _StagedEncoder,
    _StagedVAE,
    patch_offload_runtime,
)
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.layerwise_backend import (
    LayerWiseOffloadBackend,
    LayerwiseOffloadHook,
)
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture
def patched_offload_runtime(monkeypatch):
    patch_offload_runtime(monkeypatch, layerwise_backend_module.current_omni_platform)


class TinyBlock(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        mesh = DeviceMesh("cpu", [0])
        dtensor = DTensor.from_local(values, mesh, [Replicate()])
        self.weight = nn.Parameter(dtensor)


def _make_values(start: float) -> torch.Tensor:
    return torch.arange(start, start + 4, dtype=torch.float32)


def test_clear_tensor_storage_rolls_back_partial_commit(monkeypatch):
    first = nn.Parameter(torch.tensor([1.0, 2.0]))
    second = nn.Parameter(torch.tensor([3.0, 4.0]))
    expected_first = first.detach().clone()
    expected_second = second.detach().clone()
    second_data_ptr = second.data_ptr()
    original_set_storage = tensor_utils_module.set_tensor_storage
    calls = 0

    def fail_second_commit(target, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected storage clear failure")
        original_set_storage(target, value)

    monkeypatch.setattr(tensor_utils_module, "set_tensor_storage", fail_second_commit)

    with pytest.raises(RuntimeError, match="injected storage clear failure"):
        tensor_utils_module.clear_tensor_storage([first, second])

    assert calls == 3  # first commit, failed second commit, first rollback
    torch.testing.assert_close(first, expected_first)
    torch.testing.assert_close(second, expected_second)
    assert second.data_ptr() == second_data_ptr


class TestLayerwiseOffloadHook:
    def test_buffer_only_block_reports_offloaded_state(self, patched_offload_runtime):
        current_block = nn.Module()
        current_block.register_buffer("state", torch.ones(2))
        next_block = nn.Module()
        next_block.register_buffer("state", torch.ones(2))
        hook = LayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            stream=DummyStream(),
            pin_memory=False,
        )
        hook.initialize_hook(current_block)

        assert hook.is_materialized
        hook.offload_layer()
        assert not hook.is_materialized

    def test_ring_probes_are_captured_before_any_block_is_cleared(self, patched_offload_runtime):
        blocks = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(3)])
        expected_middle = blocks[1].weight.detach().clone()

        hooks = layerwise_backend_module._install_layerwise_hook_group(
            blocks,
            torch.device("cpu"),
            DummyStream(),
            pin_memory=False,
        )

        assert all(not hook.is_materialized for hook in hooks)
        hooks[2].pre_forward(blocks[1])
        torch.testing.assert_close(blocks[1].weight, expected_middle)

    def test_dtensor_wrapper_is_preserved_across_prefetch_and_offload(self, dist_group, patched_offload_runtime):
        current_block = TinyBlock(_make_values(1.0))
        next_block = TinyBlock(_make_values(10.0))

        hook = LayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            stream=DummyStream(),
            pin_memory=False,
        )

        hook.initialize_hook(current_block)

        assert isinstance(next_block.weight, DTensor)
        assert next_block.weight.to_local().is_meta
        assert next_block.weight.to_local().shape == torch.Size([4])
        assert hook.dtype_metadata[next_block.weight.dtype][0]["shape"] == torch.Size([4])

        hook.prefetch_layer(non_blocking=False)
        assert isinstance(next_block.weight, DTensor)
        assert torch.equal(next_block.weight.to_local(), _make_values(10.0))
        assert next_block.weight.to_local().shape == torch.Size([4])

        hook.offload_layer()
        assert isinstance(current_block.weight, DTensor)
        assert current_block.weight.to_local().is_meta
        assert current_block.weight.to_local().shape == torch.Size([4])
        assert not hook.is_materialized

    def test_prefetch_preserves_transposed_weight_stride(self, patched_offload_runtime):
        """Online-FP8 Cutlass weights must retain their transposed layout."""

        class StridedBlock(nn.Module):
            def __init__(self, start: float):
                super().__init__()
                base = torch.arange(start, start + 12).reshape(3, 4)
                self.weight = nn.Parameter(base.t(), requires_grad=False)

        current_block = StridedBlock(1.0)
        next_block = StridedBlock(20.0)
        expected = next_block.weight.detach().clone()
        expected_stride = next_block.weight.stride()
        assert expected_stride == (1, 4)

        hook = LayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            stream=DummyStream(),
            pin_memory=False,
        )
        hook.initialize_hook(current_block)
        hook.prefetch_layer(non_blocking=False)

        assert next_block.weight.stride() == expected_stride
        assert torch.equal(next_block.weight, expected)


class _MultiBlockModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]

    def __init__(self, num_transformer: int = 2, num_single: int = 2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_DummyBlock() for _ in range(num_transformer)])
        self.single_transformer_blocks = nn.ModuleList([_DummyBlock() for _ in range(num_single)])


class _EmptyBlocksModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([])


class _InvalidAttrModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["nonexistent_blocks", "blocks"]

    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class _DeprecatedSingleAttrModel(nn.Module):
    _layerwise_offload_blocks_attr = "blocks"

    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class _NoAttrsModel(nn.Module):
    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class TestGetBlocksFromDit:
    def test_get_blocks_from_dit_single_block_attr(self):
        model = _SingleBlockModel(num_blocks=3)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == ["blocks"]
        assert len(blocks) == 3
        assert all(isinstance(b, _DummyBlock) for b in blocks)

    def test_get_blocks_from_dit_multi_block_attrs(self):
        model = _MultiBlockModel(num_transformer=2, num_single=3)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert set(attr_names) == {"transformer_blocks", "single_transformer_blocks"}
        assert len(blocks) == 5
        assert all(isinstance(b, _DummyBlock) for b in blocks)

    def test_get_blocks_from_dit_empty_blocks(self):
        model = _EmptyBlocksModel()
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == []
        assert blocks == []

    def test_get_blocks_from_dit_invalid_attr_name(self):
        model = _InvalidAttrModel(num_blocks=2)
        with pytest.raises(
            AttributeError,
            match="Attribute 'nonexistent_blocks' declared in _layerwise_offload_blocks_attrs does not exist",
        ):
            LayerWiseOffloadBackend.get_blocks_from_dit(model)

    def test_get_blocks_from_dit_no_attrs_defined(self):
        model = _NoAttrsModel(num_blocks=3)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == []
        assert blocks == []

    def test_get_blocks_from_dit_deprecated_single_attr(self):
        model = _DeprecatedSingleAttrModel(num_blocks=2)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == ["blocks"]
        assert len(blocks) == 2


class TestGetBlocksAttrNames:
    def test_get_blocks_attr_names_new_format(self):
        model = _MultiBlockModel()
        attrs = LayerWiseOffloadBackend.get_blocks_attr_names(model)
        assert attrs == ["transformer_blocks", "single_transformer_blocks"]

    def test_get_blocks_attr_names_no_attrs(self):
        model = _NoAttrsModel()
        attrs = LayerWiseOffloadBackend.get_blocks_attr_names(model)
        assert attrs == []

    def test_set_blocks_attr_names(self):
        model = _NoAttrsModel()
        LayerWiseOffloadBackend.set_blocks_attr_names(model, ["new_blocks"])
        assert hasattr(model.__class__, "_layerwise_offload_blocks_attrs")
        assert model.__class__._layerwise_offload_blocks_attrs == ["new_blocks"]


class _ComponentPipeline(nn.Module):
    _offload_plan = OffloadPlan(
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("vision.blocks", "text_model.layers")},
        on_demand_component_paths=frozenset({"text_encoder", "vae"}),
    )

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel()
        self.text_encoder = _StagedEncoder()
        self.vae = _StagedVAE()


class _LegacyComponentPipeline(nn.Module):
    """Pipeline with DiT block metadata but no auxiliary OffloadPlan."""

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel()
        self.text_encoder = _StagedEncoder()
        self.vae = _StagedVAE()


class _LegacyEncoderOnlyPipeline(nn.Module):
    _offload_plan = _ComponentPipeline._offload_plan

    def __init__(self):
        super().__init__()
        self.text_encoder = _StagedEncoder()
        self.vae = _StagedVAE()


class _PlannedDitPipeline(nn.Module):
    _offload_plan = OffloadPlan(block_attrs={"transformer": ("blocks",)})

    def __init__(self):
        super().__init__()
        self.transformer = _NoAttrsModel()


class _GenericEncoderPipeline(nn.Module):
    _offload_plan = OffloadPlan(
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("encoder.block",)},
    )

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel()
        self.text_encoder = _PlainEncoder()


def _layer_backend(components: set[str] | None = None) -> LayerWiseOffloadBackend:
    options = {}
    if components is not None:
        options = {"components": frozenset(components)}
    return LayerWiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.LAYER_WISE,
            pin_cpu_memory=False,
            **options,
        ),
        torch.device("cpu"),
    )


class TestLayerwiseComponentSelection:
    def test_plan_block_attrs_drive_dit_discovery(self, patched_offload_runtime):
        pipeline = _PlannedDitPipeline()
        backend = _layer_backend({"dit"})

        backend.enable(pipeline)

        assert hasattr(pipeline.transformer.blocks[0], "_hook_registry")
        backend.disable()

    def test_legacy_missing_dit_preserves_noop(self, patched_offload_runtime):
        pipeline = _LegacyEncoderOnlyPipeline()
        backend = _layer_backend()

        backend.enable(pipeline)

        assert not backend.enabled
        assert pipeline.text_encoder.offload_calls == 0
        assert pipeline.vae.offload_calls == 0

    def test_encoder_only_streams_planned_blocks(self, patched_offload_runtime, monkeypatch):
        pipeline = _ComponentPipeline()
        backend = _layer_backend({"text_encoder"})
        move_non_block_state = Mock()
        monkeypatch.setattr(layerwise_backend_module, "move_non_block_state_to_device", move_non_block_state)

        backend.enable(pipeline)

        assert pipeline.text_encoder._omni_layerwise_enabled
        assert len(pipeline.text_encoder._omni_layerwise_hooks) == 4
        assert pipeline.text_encoder.offload_calls == 1
        assert pipeline.vae.offload_calls == 0
        assert pipeline.vae.to_calls == 1
        assert not hasattr(pipeline.transformer.blocks[0], "_hook_registry")
        assert backend.enabled
        move_non_block_state.assert_not_called()

        backend.disable()
        assert not pipeline.text_encoder._omni_layerwise_enabled

    def test_single_gpu_dit_only_keeps_encoder_and_vae_resident(self, patched_offload_runtime):
        pipeline = _ComponentPipeline()
        expected_weights = [block.weight.detach().clone() for block in pipeline.transformer.blocks]
        backend = _layer_backend({"dit"})

        backend.enable(pipeline)

        assert not hasattr(pipeline.text_encoder, "_omni_layerwise_enabled")
        assert pipeline.text_encoder.to_calls == 1
        assert pipeline.text_encoder.offload_calls == 0
        assert pipeline.vae.to_calls == 1
        assert pipeline.vae.offload_calls == 0
        assert hasattr(pipeline.transformer.blocks[0], "_hook_registry")

        backend.disable()

        for block, expected in zip(pipeline.transformer.blocks, expected_weights, strict=True):
            torch.testing.assert_close(block.weight, expected)

        backend.enable(pipeline)
        backend.disable()
        for block, expected in zip(pipeline.transformer.blocks, expected_weights, strict=True):
            torch.testing.assert_close(block.weight, expected)

    def test_partial_dit_enable_failure_restores_weights_and_hooks(self, patched_offload_runtime, monkeypatch):
        pipeline = _ComponentPipeline()
        expected_weights = [block.weight.detach().clone() for block in pipeline.transformer.blocks]
        backend = _layer_backend({"dit"})
        original_apply = layerwise_backend_module.apply_block_hook
        calls = 0

        def fail_second_hook(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected hook failure")
            return original_apply(*args, **kwargs)

        monkeypatch.setattr(layerwise_backend_module, "apply_block_hook", fail_second_hook)

        with pytest.raises(RuntimeError, match="injected hook failure"):
            backend.enable(pipeline)

        assert not backend.enabled
        for block, expected in zip(pipeline.transformer.blocks, expected_weights, strict=True):
            registry = getattr(block, "_hook_registry", None)
            assert registry is None or registry.get_hook("layerwise_offload") is None
            torch.testing.assert_close(block.weight, expected)

    def test_later_component_failure_removes_installed_encoder_hooks(
        self,
        patched_offload_runtime,
        monkeypatch,
    ):
        pipeline = _ComponentPipeline()
        blocks = [*pipeline.text_encoder.vision.blocks, *pipeline.text_encoder.text_model.layers]
        expected = [block.weight.detach().clone() for block in blocks]
        backend = _layer_backend({"text_encoder"})

        def fail_vae_placement(*_args, **_kwargs):
            raise RuntimeError("injected VAE placement failure")

        monkeypatch.setattr(pipeline.vae, "to", fail_vae_placement)

        with pytest.raises(RuntimeError, match="injected VAE placement failure"):
            backend.enable(pipeline)

        assert not backend.enabled
        assert not pipeline.text_encoder._omni_layerwise_enabled
        assert not backend._encoder_modules
        for block, original in zip(blocks, expected, strict=True):
            assert block._hook_registry.get_hook("layerwise_offload") is None
            torch.testing.assert_close(block.weight, original)

    def test_encoder_only_requires_streamable_offload_plan(self, patched_offload_runtime):
        pipeline = nn.Module()
        pipeline.transformer = _SingleBlockModel()
        pipeline.text_encoder = _StagedEncoder()
        backend = _layer_backend({"text_encoder"})

        with pytest.raises(ValueError, match="Selected text encoder 'text_encoder' requires"):
            backend.enable(pipeline)

    def test_explicit_text_encoder_selection_requires_a_matching_module(self, patched_offload_runtime):
        pipeline = nn.Module()
        pipeline.transformer = _SingleBlockModel(num_blocks=2)
        backend = _layer_backend({"dit", "text_encoder"})

        with pytest.raises(ValueError, match="No text encoder modules found"):
            backend.enable(pipeline)

        assert not hasattr(pipeline.transformer.blocks[0], "_hook_registry")

    def test_every_selected_encoder_requires_its_own_plan(self, patched_offload_runtime):
        class Pipeline(nn.Module):
            _offload_plan = OffloadPlan(
                encoder_component_types={
                    "text_encoder": "text_encoder",
                    "text_encoder_2": "text_encoder",
                },
                encoder_block_attrs={"text_encoder": ("encoder.block",)},
            )

            def __init__(self):
                super().__init__()
                self.transformer = _SingleBlockModel()
                self.text_encoder = _PlainEncoder()
                self.text_encoder_2 = _PlainEncoder()

        pipeline = Pipeline()
        backend = _layer_backend({"text_encoder"})

        with pytest.raises(ValueError, match="Selected text encoder 'text_encoder_2' requires"):
            backend.enable(pipeline)

        assert not pipeline.text_encoder._omni_layerwise_enabled

    def test_default_selection_preserves_unplanned_auxiliaries(self, patched_offload_runtime):
        pipeline = _LegacyComponentPipeline()
        backend = _layer_backend()

        backend.enable(pipeline)

        assert hasattr(pipeline.transformer.blocks[0], "_hook_registry")
        assert not hasattr(pipeline.text_encoder, "_omni_layerwise_enabled")
        assert pipeline.text_encoder.to_calls == 1
        assert pipeline.text_encoder.offload_calls == 0
        assert pipeline.vae.to_calls == 1
        assert pipeline.vae.offload_calls == 0

        backend.disable()

    def test_disable_failure_keeps_host_masters_for_retry(self, patched_offload_runtime, monkeypatch):
        pipeline = nn.Module()
        pipeline.transformer = _SingleBlockModel(num_blocks=3)
        expected = {name: tensor.detach().clone() for name, tensor in pipeline.state_dict().items()}
        backend = _layer_backend()
        backend.enable(pipeline)
        failing_hook = backend._dit_hooks[0]
        restore = failing_hook.restore_next_block
        attempts = 0

        def fail_once():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected restore failure")
            restore()

        monkeypatch.setattr(failing_hook, "restore_next_block", fail_once)

        with pytest.raises(RuntimeError, match="Failed to fully disable"):
            backend.disable()

        assert backend.enabled
        assert backend._dit_hooks

        backend.disable()

        assert not backend.enabled
        for name, tensor in pipeline.state_dict().items():
            torch.testing.assert_close(tensor, expected[name])

    def test_encoder_disable_failure_keeps_host_masters_for_retry(
        self,
        patched_offload_runtime,
        monkeypatch,
    ):
        pipeline = _ComponentPipeline()
        encoder = pipeline.text_encoder
        expected = {name: tensor.detach().clone() for name, tensor in encoder.state_dict().items()}
        backend = _layer_backend({"text_encoder"})
        backend.enable(pipeline)
        failing_hook = encoder._omni_layerwise_hooks[0]
        restore = failing_hook.restore_next_block
        attempts = 0

        def fail_once():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected encoder restore failure")
            restore()

        monkeypatch.setattr(failing_hook, "restore_next_block", fail_once)

        with pytest.raises(RuntimeError, match="Failed to fully disable"):
            backend.disable()

        assert backend.enabled
        assert backend._encoder_modules == [encoder]
        assert encoder._omni_layerwise_enabled
        assert encoder._omni_layerwise_hooks

        backend.disable()

        assert not backend.enabled
        assert not encoder._omni_layerwise_enabled
        for name, tensor in encoder.state_dict().items():
            torch.testing.assert_close(tensor, expected[name])

    def test_legacy_selector_preserves_planned_encoder_and_vae_lifecycle(self, patched_offload_runtime):
        pipeline = _ComponentPipeline()
        backend = _layer_backend()

        backend.enable(pipeline)

        assert pipeline.text_encoder._omni_layerwise_enabled
        assert pipeline.text_encoder.offload_calls == 1
        assert pipeline.vae.offload_calls == 1

        backend.disable()

    def test_standard_encoder_needs_only_declared_block_paths(self, patched_offload_runtime):
        pipeline = _GenericEncoderPipeline()
        backend = _layer_backend({"text_encoder"})

        backend.enable(pipeline)

        blocks = pipeline.text_encoder.encoder.block
        assert pipeline.text_encoder._omni_layerwise_enabled
        assert all(block._hook_registry.get_hook("layerwise_offload") is not None for block in blocks)
        assert pipeline.text_encoder.final_norm.weight.numel() == 4

        backend.disable()

        assert not pipeline.text_encoder._omni_layerwise_enabled
        assert all(block._hook_registry.get_hook("layerwise_offload") is None for block in blocks)
        assert all(block.weight.numel() == 100 for block in blocks)


def _offload_od_config(**overrides):
    values = {
        "diffusion_offload_config": None,
        "enable_cpu_offload": False,
        "enable_layerwise_offload": False,
        "enable_distributed_layerwise_offload": False,
        "dlo_use_allgather": True,
        "dlo_resident_layers": 0,
        "pin_cpu_memory": True,
        "parallel_config": None,
        "model": "/fake/model",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolve_offload_config(public=None, **overrides) -> OffloadConfig:
    return OffloadConfig.from_od_config(_offload_od_config(diffusion_offload_config=public, **overrides))


class TestLayerwiseComponentConfig:
    def test_layer_mode_selects_components_without_backend_jargon(self):
        config = _resolve_offload_config(
            {
                "mode": "layer",
                "components": ["dit", "text_encoder"],
            }
        )

        assert config.strategy is OffloadStrategy.LAYER_WISE
        assert config.components == frozenset({"dit", "text_encoder"})
        assert not config.uses_allgather("dit")
        assert not config.uses_allgather("text_encoder")
        plan = OffloadPlan(encoder_component_types={"mllm": "text_encoder"})
        assert config.offloads_encoder("mllm", plan)

    @pytest.mark.parametrize("component", ["image_encoder", "vae", "scheduler", "text-encoder"])
    def test_unknown_or_noncanonical_component_is_rejected(self, component):
        with pytest.raises(ValueError, match="Unknown diffusion offload component"):
            OffloadConfig.from_od_config(
                _offload_od_config(diffusion_offload_config={"mode": "layer", "components": [component]})
            )

    def test_components_is_selection_only_not_an_options_mapping(self):
        with pytest.raises(TypeError, match="components must be a non-empty list"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={
                        "mode": "layer",
                        "components": {"dit": {}},
                    }
                )
            )

    def test_layer_options_requires_component_name_keys(self):
        with pytest.raises(TypeError, match="layer_options keys must be strings"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={
                        "mode": "layer",
                        "components": ["dit"],
                        "layer_options": {0: {}},
                    }
                )
            )

    @pytest.mark.parametrize(
        ("config", "message"),
        [
            ({0: "layer"}, "diffusion_offload_config keys must be strings"),
            (
                {
                    "mode": "layer",
                    "components": ["dit"],
                    "layer_options": {"dit": {0: "allgather"}},
                },
                "layer_options\\['dit'\\] keys must be strings",
            ),
        ],
    )
    def test_all_public_mapping_levels_require_string_keys(self, config, message):
        with pytest.raises(TypeError, match=message):
            OffloadConfig.from_od_config(_offload_od_config(diffusion_offload_config=config))

    @pytest.mark.parametrize(
        "config,match",
        [
            ({"components": ["dit"]}, "requires 'mode'"),
            ({"mode": "layer"}, "requires 'components'"),
            ({"mode": "layer", "components": []}, "must not be empty"),
            ({"mode": "layer", "components": [" dit "]}, "Unknown diffusion offload component"),
            ({"mode": "layerwise", "components": ["dit"]}, "Unknown diffusion offload mode"),
            (
                {
                    "mode": "layer",
                    "components": ["dit"],
                    "layer_options": {"dit": {"weight_transfer": "rank_local"}},
                },
                "Unknown offload transfer",
            ),
            (
                {
                    "mode": "layer",
                    "components": ["dit"],
                    "layer_options": {"dit": {"prefetch": 2}},
                },
                "Unknown diffusion offload setting",
            ),
            (
                {
                    "mode": "layer",
                    "components": ["text_encoder"],
                    "layer_options": {"text_encoder": {"resident_layers": 1}},
                },
                "supports only the 'dit'",
            ),
            (
                {
                    "mode": "layer",
                    "components": ["dit"],
                    "layer_options": {"text_encoder": {}},
                },
                "requires selecting the same component",
            ),
        ],
    )
    def test_schema_rejects_ambiguous_or_unsupported_values(self, config, match):
        with pytest.raises(ValueError, match=match):
            OffloadConfig.from_od_config(_offload_od_config(diffusion_offload_config=config))

    def test_module_mode_accepts_component_selection(self):
        config = _resolve_offload_config(
            {
                "mode": "module",
                "components": ["dit", "text_encoder"],
            }
        )

        assert config.strategy is OffloadStrategy.MODEL_LEVEL
        assert config.components == frozenset({"dit", "text_encoder"})

    @pytest.mark.parametrize("setting", [{"weight_transfer": "rank-local"}, {"resident_layers": 1}])
    def test_module_mode_rejects_layer_settings(self, setting):
        with pytest.raises(ValueError, match="layer_options requires mode='layer'"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={
                        "mode": "module",
                        "components": ["dit"],
                        "layer_options": {"dit": setting},
                    }
                )
            )

    def test_each_component_can_choose_its_transfer(self):
        config = _resolve_offload_config(
            {
                "mode": "layer",
                "components": ["dit", "text_encoder"],
                "layer_options": {
                    "dit": {"weight_transfer": "rank-local"},
                    "text_encoder": {"weight_transfer": "allgather"},
                },
            }
        )

        assert config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
        assert not config.uses_allgather("dit")
        assert config.uses_allgather("text_encoder")
        assert config.dlo_use_allgather is False

    def test_dit_residency_selects_capable_backend(self):
        config = _resolve_offload_config(
            {
                "mode": "layer",
                "components": ["dit"],
                "layer_options": {"dit": {"resident_layers": 20}},
            }
        )

        assert config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
        assert config.dlo_resident_layers == 20
        assert not config.uses_allgather("dit")

    def test_dit_residency_rejects_allgather(self):
        with pytest.raises(ValueError, match="resident_layers requires dit.weight_transfer='rank-local'"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={
                        "mode": "layer",
                        "components": ["dit"],
                        "layer_options": {"dit": {"weight_transfer": "allgather", "resident_layers": 1}},
                    }
                )
            )

    def test_host_weight_runtime_remains_separate_from_compact_config(self):
        with pytest.raises(ValueError, match="cannot be combined with Host Weight Runtime"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={"mode": "layer", "components": ["dit"]},
                    host_weight_runtime_mode="preferred",
                )
            )

    def test_pin_memory_override_is_scoped_to_offload_config(self):
        config = _resolve_offload_config(
            {
                "mode": "layer",
                "components": ["dit"],
                "pin_memory": False,
            },
            pin_cpu_memory=True,
        )

        assert config.pin_cpu_memory is False

    def test_legacy_omitted_selector_preserves_dit_only_behavior(self):
        config = _resolve_offload_config(enable_layerwise_offload=True)

        assert config.strategy is OffloadStrategy.LAYER_WISE
        assert config.components is None

    def test_offload_resolution_is_cached_for_runtime_accessors(self, monkeypatch):
        import vllm_omni.diffusion.offloader.config as config_module

        od_config = _offload_od_config(
            diffusion_offload_config={
                "mode": "layer",
                "components": ["dit", "text_encoder"],
            }
        )
        parse_calls = 0
        original_parse = config_module.parse_diffusion_offload_config

        def track_parse(value):
            nonlocal parse_calls
            parse_calls += 1
            return original_parse(value)

        monkeypatch.setattr(config_module, "parse_diffusion_offload_config", track_parse)

        assert config_module.resolve_offload_strategy(od_config) is OffloadStrategy.LAYER_WISE
        assert config_module.selected_offload_components(od_config) == {"dit", "text_encoder"}
        assert not config_module.component_uses_allgather(od_config, "dit")
        assert not config_module.any_selected_component_uses_allgather(od_config)
        assert parse_calls == 1

    def test_cached_policy_is_deeply_immutable_and_pickle_safe(self):
        import vllm_omni.diffusion.offloader.config as config_module

        od_config = _offload_od_config(
            diffusion_offload_config={
                "mode": "layer",
                "components": ["dit"],
                "layer_options": {"dit": {"weight_transfer": "rank-local"}},
            }
        )
        resolved = config_module.resolve_offload(od_config)

        with pytest.raises(TypeError):
            resolved.transfers["dit"] = config_module.DLOTransfer.ALLGATHER  # type: ignore[index]
        assert resolved.public is not None
        with pytest.raises(TypeError):
            resolved.public.layer_options["dit"] = config_module.LayerOffloadOptions()  # type: ignore[index]

        restored = ForkingPickler.loads(ForkingPickler.dumps(resolved))
        assert not restored.uses_allgather("dit")

    def test_dp_text_encoder_allgather_rejects_rank_local_prompt_cache(self):
        config = _offload_od_config(
            diffusion_offload_config={
                "mode": "layer",
                "components": ["text_encoder"],
                "layer_options": {"text_encoder": {"weight_transfer": "allgather"}},
            },
            enable_prompt_embed_cache=True,
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                sequence_parallel_size=1,
                use_hsdp=False,
            ),
        )

        with pytest.raises(ValueError, match="rank-local cache hits would skip different encoder collectives"):
            OffloadConfig.from_od_config(config)

    def test_dp_allgather_rejects_rank_local_dit_cache_decisions(self):
        config = _offload_od_config(
            diffusion_offload_config={
                "mode": "layer",
                "components": ["dit"],
                "layer_options": {"dit": {"weight_transfer": "allgather"}},
            },
            cache_backend="tea_cache",
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                sequence_parallel_size=1,
                use_hsdp=False,
            ),
        )

        with pytest.raises(ValueError, match="rank-local cache decisions can skip different weight collectives"):
            OffloadConfig.from_od_config(config)

    def test_internal_transfer_map_requires_every_component(self):
        with pytest.raises(ValueError, match="missing: text_encoder"):
            OffloadConfig(
                strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
                dlo_transfers={"dit": "rank-local"},
            )

    def test_legacy_distributed_settings_still_work(self):
        config = _resolve_offload_config(
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=False,
            dlo_resident_layers=3,
        )

        assert config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
        assert config.dlo_resident_layers == 3
        assert not config.uses_allgather("dit")
        assert not config.uses_allgather("text_encoder")

    def test_legacy_allgather_remains_dit_only(self):
        config = _resolve_offload_config(
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=True,
        )

        assert config.uses_allgather("dit")
        assert not config.uses_allgather("text_encoder")

    @pytest.mark.parametrize(
        ("legacy_flags", "expected_strategy"),
        [
            (
                {"enable_cpu_offload": True, "enable_layerwise_offload": True},
                OffloadStrategy.LAYER_WISE,
            ),
            (
                {
                    "enable_cpu_offload": True,
                    "enable_layerwise_offload": True,
                    "enable_distributed_layerwise_offload": True,
                },
                OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            ),
        ],
    )
    def test_legacy_aliases_preserve_existing_priority(self, legacy_flags, expected_strategy):
        config = _resolve_offload_config(**legacy_flags)

        assert config.strategy is expected_strategy

    def test_compact_config_cannot_mix_with_legacy_alias(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={"mode": "layer", "components": ["dit"]},
                    enable_cpu_offload=True,
                )
            )

    @pytest.mark.parametrize(
        "legacy_options",
        [
            {"dlo_use_allgather": False},
            {"dlo_resident_layers": 12},
        ],
    )
    def test_compact_config_cannot_mix_with_legacy_dlo_options(self, legacy_options):
        with pytest.raises(ValueError, match="cannot be combined with legacy DLO option"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={"mode": "layer", "components": ["dit"]},
                    **legacy_options,
                )
            )

    def test_legacy_allgather_resident_layers_fail_during_config_resolution(self):
        with pytest.raises(ValueError, match="requires the DiT DLO transfer to be rank-local"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    enable_distributed_layerwise_offload=True,
                    dlo_use_allgather=True,
                    dlo_resident_layers=12,
                )
            )

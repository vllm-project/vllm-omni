# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Sequence Parallelism (SP) framework.

These tests verify the SP plan mechanism and hooks work correctly without
requiring a distributed environment. They test:
1. _sp_plan validation (sp_plan.py)
2. Hook utilities and submodule resolution (sequence_parallel.py)
3. Model _sp_plan definitions
4. Tensor sharding simulation

Note: Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism" (CP)
in diffusers. We use "Sequence Parallelism" to align with vLLM-Omni terminology.
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
    SequenceParallelPartialInput,
    get_sp_plan_from_model,
    validate_sp_plan,
)


def is_distributed_initialized() -> bool:
    """Check if distributed environment is initialized."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

        get_sp_group()
        return True
    except (AssertionError, ImportError):
        return False


# Decorator to skip tests that require distributed environment
requires_distributed = pytest.mark.skipif(
    not is_distributed_initialized(),
    reason="Requires initialized distributed environment (SP group)",
)

# Module-level markers: these tests are diffusion + parallel related
pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.parallel,
]

# =============================================================================
# Tests for sp_plan.py
# =============================================================================


@pytest.mark.cpu
class TestSequenceParallelPlanValidation:
    """Test _sp_plan validation logic."""

    def test_valid_simple_plan(self):
        """Test a simple valid _sp_plan."""
        plan = {
            "rope": {
                0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
                1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
            },
            "blocks.0": {
                "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
            },
            "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
        }
        # Should not raise
        validate_sp_plan(plan)

    def test_valid_partial_input_plan(self):
        """Test a valid _sp_plan with SequenceParallelPartialInput."""
        plan = {
            "pos_embed": {
                0: SequenceParallelPartialInput(
                    split_dim=0,
                    text_len_source="txt_ids",
                    expected_dims=2,
                    split_output=True,
                ),
            },
            "blocks.0": {
                "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
            },
        }
        # Should not raise
        validate_sp_plan(plan)

    def test_invalid_plan_type(self):
        """Test that non-dict plan raises error."""
        with pytest.raises(ValueError, match="must be a dict"):
            validate_sp_plan("not a dict")

    def test_invalid_module_key_type(self):
        """Test that non-string module keys raise error."""
        plan = {123: {"hidden_states": SequenceParallelInput(split_dim=1)}}
        with pytest.raises(ValueError, match="keys must be strings"):
            validate_sp_plan(plan)

    def test_invalid_output_index_without_split_output(self):
        """Test that integer keys require split_output=True."""
        plan = {
            "rope": {
                0: SequenceParallelInput(split_dim=1, split_output=False),  # Invalid
            }
        }
        with pytest.raises(ValueError, match="split_output=True"):
            validate_sp_plan(plan)


@pytest.mark.cpu
class TestGetSpPlanFromModel:
    """Test get_sp_plan_from_model utility."""

    def test_model_with_sp_plan(self):
        """Test getting _sp_plan from a model that has one."""

        class ModelWithPlan(nn.Module):
            _sp_plan = {
                "layer": {
                    "x": SequenceParallelInput(split_dim=1),
                }
            }

        model = ModelWithPlan()
        plan = get_sp_plan_from_model(model)
        assert plan is not None
        assert "layer" in plan

    def test_model_without_sp_plan(self):
        """Test getting _sp_plan from a model without one."""

        class ModelWithoutPlan(nn.Module):
            pass

        model = ModelWithoutPlan()
        plan = get_sp_plan_from_model(model)
        assert plan is None


@pytest.mark.cpu
class TestSequenceParallelInputTypes:
    """Test SequenceParallelInput and related types."""

    def test_sequence_parallel_input_repr(self):
        """Test SequenceParallelInput repr."""
        spi = SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True)
        assert "split_dim=1" in repr(spi)
        assert "expected_dims=3" in repr(spi)
        assert "split_output=True" in repr(spi)

    def test_sequence_parallel_output_repr(self):
        """Test SequenceParallelOutput repr."""
        spo = SequenceParallelOutput(gather_dim=1, expected_dims=3)
        assert "gather_dim=1" in repr(spo)
        assert "expected_dims=3" in repr(spo)

    def test_sequence_parallel_partial_input_repr(self):
        """Test SequenceParallelPartialInput repr."""
        sppi = SequenceParallelPartialInput(
            split_dim=0,
            text_len_source="txt_ids",
            expected_dims=2,
            split_output=True,
        )
        assert "split_dim=0" in repr(sppi)
        assert "txt_ids" in repr(sppi)
        assert "expected_dims=2" in repr(sppi)
        assert "split_output=True" in repr(sppi)

    def test_sequence_parallel_partial_input_with_int_source(self):
        """Test SequenceParallelPartialInput with integer text_len_source."""
        sppi = SequenceParallelPartialInput(
            split_dim=0,
            text_len_source=512,  # Fixed length
            expected_dims=2,
        )
        assert sppi.text_len_source == 512


# =============================================================================
# Tests for sequence_parallel.py
# =============================================================================


@pytest.mark.cpu
class TestModuleForwardMetadata:
    """Test ModuleForwardMetadata parameter resolution."""

    def test_get_parameter_from_kwargs(self):
        """Test getting parameter from kwargs."""
        from vllm_omni.diffusion.hooks.sequence_parallel import ModuleForwardMetadata

        class DummyModule(nn.Module):
            def forward(self, hidden_states, encoder_hidden_states):
                pass

        metadata = ModuleForwardMetadata()
        metadata._cls = DummyModule

        kwargs = {"hidden_states": torch.randn(2, 4, 8)}
        val, is_kwarg, index = metadata._get_parameter_from_args_kwargs("hidden_states", (), kwargs)
        assert is_kwarg is True
        assert index is None
        assert val.shape == (2, 4, 8)

    def test_get_parameter_from_args(self):
        """Test getting parameter from positional args."""
        from vllm_omni.diffusion.hooks.sequence_parallel import ModuleForwardMetadata

        class DummyModule(nn.Module):
            def forward(self, hidden_states, encoder_hidden_states):
                pass

        metadata = ModuleForwardMetadata()
        metadata._cls = DummyModule

        tensor = torch.randn(2, 4, 8)
        args = (tensor,)
        val, is_kwarg, index = metadata._get_parameter_from_args_kwargs("hidden_states", args, {})
        assert is_kwarg is False
        assert index == 0
        assert torch.equal(val, tensor)

    def test_parameter_caching(self):
        """Test that parameter indices are cached."""
        from vllm_omni.diffusion.hooks.sequence_parallel import ModuleForwardMetadata

        class DummyModule(nn.Module):
            def forward(self, a, b, c):
                pass

        metadata = ModuleForwardMetadata()
        metadata._cls = DummyModule

        # First call - should populate cache
        args = (torch.randn(1), torch.randn(1), torch.randn(1))
        metadata._get_parameter_from_args_kwargs("b", args, {})

        # Check cache was populated
        assert metadata.cached_parameter_indices is not None
        assert metadata.cached_parameter_indices["a"] == 0
        assert metadata.cached_parameter_indices["b"] == 1
        assert metadata.cached_parameter_indices["c"] == 2


@pytest.mark.cpu
class TestGetSubmoduleByName:
    """Test _get_submodule_by_name function."""

    def test_root_module(self):
        """Test getting root module with empty string."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _get_submodule_by_name

        model = nn.Linear(10, 10)
        submodule = _get_submodule_by_name(model, "")
        assert submodule is model

    def test_simple_submodule(self):
        """Test getting a simple submodule."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _get_submodule_by_name

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)

        model = Model()
        submodule = _get_submodule_by_name(model, "layer")
        assert submodule is model.layer

    def test_nested_submodule(self):
        """Test getting a nested submodule."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _get_submodule_by_name

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(10, 10), nn.ReLU())

        model = Model()
        submodule = _get_submodule_by_name(model, "encoder.0")
        assert isinstance(submodule, nn.Linear)

    def test_module_list_by_index(self):
        """Test getting element from ModuleList by index."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _get_submodule_by_name

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

        model = Model()
        submodule = _get_submodule_by_name(model, "blocks.0")
        assert submodule is model.blocks[0]

        submodule = _get_submodule_by_name(model, "blocks.2")
        assert submodule is model.blocks[2]

    def test_wildcard_modulelist(self):
        """Test wildcard matching for ModuleList."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _get_submodule_by_name

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

        model = Model()
        submodules = _get_submodule_by_name(model, "blocks.*")
        assert isinstance(submodules, list)
        assert len(submodules) == 3
        for i, sm in enumerate(submodules):
            assert sm is model.blocks[i]

    def test_module_dict(self):
        """Test getting submodule from ModuleDict."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _get_submodule_by_name

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.outputs = nn.ModuleDict({"main": nn.Linear(10, 10), "aux": nn.Linear(10, 5)})

        model = Model()
        submodule = _get_submodule_by_name(model, "outputs.main")
        assert submodule is model.outputs["main"]

        submodule = _get_submodule_by_name(model, "outputs.aux")
        assert submodule is model.outputs["aux"]

    def test_invalid_submodule_raises(self):
        """Test that invalid submodule path raises error."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _get_submodule_by_name

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)

        model = Model()
        with pytest.raises(ValueError, match="not a submodule"):
            _get_submodule_by_name(model, "nonexistent")

    def test_multiple_wildcards_raises(self):
        """Test that multiple wildcards raise error."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _get_submodule_by_name

        model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="only be used once"):
            _get_submodule_by_name(model, "a.*.b.*")


@pytest.mark.cpu
class TestHookRegistration:
    """Test hook registration logic (without distributed backend)."""

    def test_plan_validation_before_apply(self):
        """Test that invalid plans are rejected before hook registration."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_in = nn.Linear(10, 10)
                self.proj_out = nn.Linear(10, 10)

            def forward(self, x):
                return self.proj_out(self.proj_in(x))

        # Invalid plan (non-string key)
        invalid_plan = {
            123: {"x": SequenceParallelInput(split_dim=1)},
        }

        with pytest.raises(ValueError):
            validate_sp_plan(invalid_plan)

    def test_valid_plan_structure_for_model(self):
        """Test that a valid plan can be defined for a model."""

        class SimpleModel(nn.Module):
            _sp_plan = {
                "proj_in": {"x": SequenceParallelInput(split_dim=1, expected_dims=3)},
                "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
            }

            def __init__(self):
                super().__init__()
                self.proj_in = nn.Linear(10, 10)
                self.proj_out = nn.Linear(10, 10)

            def forward(self, x):
                return self.proj_out(self.proj_in(x))

        model = SimpleModel()
        plan = get_sp_plan_from_model(model)

        assert plan is not None
        assert "proj_in" in plan
        assert "proj_out" in plan

        # Verify submodules exist
        from vllm_omni.diffusion.hooks.sequence_parallel import _get_submodule_by_name

        assert _get_submodule_by_name(model, "proj_in") is model.proj_in
        assert _get_submodule_by_name(model, "proj_out") is model.proj_out


# =============================================================================
# Tests for model _sp_plan definitions
# =============================================================================


@pytest.mark.L4
class TestModelSpPlans:
    """Test that model _sp_plan definitions are valid.

    These tests import actual model classes to verify _sp_plan structure.
    May require GPU for model imports.
    """

    def test_zimage_transformer_sp_plan(self):
        """Test ZImageTransformer2DModel _sp_plan structure.

        The plan specifies:
        - unified_prepare: Shard all 4 outputs (unified, cos, sin, attn_mask)
        - all_final_layer.2-1: Gather outputs after final layer

        Note: _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism)
        """
        try:
            from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel

            plan = getattr(ZImageTransformer2DModel, "_sp_plan", None)
            assert plan is not None, "ZImageTransformer2DModel should define _sp_plan"
            assert isinstance(plan, dict)

            assert "unified_prepare" in plan
            unified_prepare_plan = plan["unified_prepare"]
            # Check all 4 outputs are sharded with split_output=True
            assert 0 in unified_prepare_plan  # unified
            assert 1 in unified_prepare_plan  # unified_cos
            assert 2 in unified_prepare_plan  # unified_sin
            assert 3 in unified_prepare_plan  # unified_attn_mask

            # Check output gathering
            assert "all_final_layer.2-1" in plan

            validate_sp_plan(plan)
        except ImportError:
            pytest.skip("ZImageTransformer2DModel not available")

    def test_qwen_image_transformer_sp_plan(self):
        """Test QwenImageTransformer2DModel _sp_plan structure.

        Qwen-Image follows the diffusers pattern similar to Z-Image:
        - image_rope_prepare: Shards hidden_states and vid_freqs together
        - proj_out: Gathers output

        Key insight: hidden_states and vid_freqs MUST be sharded together
        to maintain dimension alignment for RoPE computation.

        Note: _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism)
        """
        try:
            from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
                QwenImageTransformer2DModel,
            )

            plan = getattr(QwenImageTransformer2DModel, "_sp_plan", None)
            assert plan is not None, "QwenImageTransformer2DModel should define _sp_plan"
            assert isinstance(plan, dict)

            # Check image_rope_prepare sharding
            assert "image_rope_prepare" in plan
            rope_plan = plan["image_rope_prepare"]
            # hidden_states (index 0)
            assert 0 in rope_plan
            assert rope_plan[0].split_dim == 1
            assert rope_plan[0].split_output is True
            # vid_freqs (index 1)
            assert 1 in rope_plan
            assert rope_plan[1].split_dim == 0
            assert rope_plan[1].split_output is True
            # txt_freqs (index 2) should NOT be in plan (kept replicated)
            assert 2 not in rope_plan

            # Check output gathering at proj_out
            assert "proj_out" in plan
            proj_out_plan = plan["proj_out"]
            assert proj_out_plan.gather_dim == 1

            validate_sp_plan(plan)
        except ImportError:
            pytest.skip("QwenImageTransformer2DModel not available")


# =============================================================================
# Tests for tensor sharding simulation (no distributed required)
# =============================================================================


@pytest.mark.cpu
class TestMockSharding:
    """Test tensor sharding logic (mocked, no distributed)."""

    def test_shard_tensor_simulation(self):
        """Simulate tensor sharding without distributed backend."""
        # Create a test tensor
        batch_size, seq_len, hidden_dim = 2, 16, 64
        tensor = torch.randn(batch_size, seq_len, hidden_dim)

        # Simulate sharding for world_size=4
        world_size = 4
        rank = 1

        # Manual chunking (what sp_shard does internally)
        chunks = tensor.chunk(world_size, dim=1)
        sharded = chunks[rank]

        assert sharded.shape == (batch_size, seq_len // world_size, hidden_dim)
        assert sharded.shape == (2, 4, 64)

    def test_partial_shard_simulation(self):
        """Simulate partial sharding (text kept, image sharded)."""
        # Create a test tensor with [text, image] concatenated
        batch_size = 2
        text_len = 8
        image_len = 16
        hidden_dim = 64

        text_part = torch.randn(batch_size, text_len, hidden_dim)
        image_part = torch.randn(batch_size, image_len, hidden_dim)
        tensor = torch.cat([text_part, image_part], dim=1)

        assert tensor.shape == (batch_size, text_len + image_len, hidden_dim)

        # Simulate partial sharding for world_size=4, rank=1
        world_size = 4
        rank = 1
        dim = 1

        # Extract parts
        text_kept = tensor.narrow(dim, 0, text_len)
        image_full = tensor.narrow(dim, text_len, image_len)

        # Shard only image part
        image_chunks = image_full.chunk(world_size, dim=dim)
        image_sharded = image_chunks[rank]

        # Concatenate back
        result = torch.cat([text_kept, image_sharded], dim=dim)

        expected_len = text_len + image_len // world_size
        assert result.shape == (batch_size, expected_len, hidden_dim)
        assert result.shape == (2, 8 + 4, 64)  # text_len + image_len/4

    def test_gather_tensor_simulation(self):
        """Simulate tensor gathering without distributed backend."""
        # Create sharded tensors (as if from different ranks)
        batch_size, shard_seq_len, hidden_dim = 2, 4, 64
        world_size = 4

        shards = [torch.randn(batch_size, shard_seq_len, hidden_dim) for _ in range(world_size)]

        # Simulate gathering (concatenate along dim 1)
        gathered = torch.cat(shards, dim=1)

        assert gathered.shape == (batch_size, shard_seq_len * world_size, hidden_dim)
        assert gathered.shape == (2, 16, 64)

    def test_padding_simulation(self):
        """Simulate padding for non-divisible sequence lengths."""
        # Create tensor with non-divisible sequence length
        batch_size, seq_len, hidden_dim = 2, 17, 64  # 17 not divisible by 4
        tensor = torch.randn(batch_size, seq_len, hidden_dim)

        world_size = 4
        dim = 1

        # Calculate padding needed
        remainder = seq_len % world_size
        if remainder != 0:
            pad_size = world_size - remainder
        else:
            pad_size = 0

        assert pad_size == 3  # 17 + 3 = 20, divisible by 4

        # Pad tensor
        if pad_size > 0:
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad_size
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            padded = torch.cat([tensor, padding], dim=dim)
        else:
            padded = tensor

        assert padded.shape == (batch_size, seq_len + pad_size, hidden_dim)
        assert padded.shape == (2, 20, 64)

        # Now can shard evenly
        chunks = padded.chunk(world_size, dim=dim)
        assert all(c.shape == (2, 5, 64) for c in chunks)


# =============================================================================
# Additional tests for sequence_parallel.py coverage
# =============================================================================


@pytest.mark.cpu
class TestUnwrapModule:
    """Test _unwrap_module function."""

    def test_unwrap_simple_module(self):
        """Test that a simple module returns itself."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _unwrap_module

        module = nn.Linear(10, 10)
        result = _unwrap_module(module)
        assert result is module

    def test_unwrap_sequential_single(self):
        """Test unwrapping a Sequential with single child."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _unwrap_module

        inner = nn.Linear(10, 10)
        wrapper = nn.Sequential(inner)
        result = _unwrap_module(wrapper)
        # Should unwrap to the inner module
        assert result is inner

    def test_unwrap_nested_wrapper(self):
        """Test unwrapping nested single-child wrappers."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _unwrap_module

        inner = nn.Linear(10, 10)
        wrapper1 = nn.Sequential(inner)
        wrapper2 = nn.Sequential(wrapper1)
        result = _unwrap_module(wrapper2)
        # Should fully unwrap to the innermost module
        assert result is inner


@pytest.mark.cpu
class TestSequenceParallelSplitHookInit:
    """Test SequenceParallelSplitHook initialization and setup."""

    def test_hook_init(self):
        """Test SequenceParallelSplitHook initialization."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
        from vllm_omni.diffusion.hooks.sequence_parallel import SequenceParallelSplitHook

        metadata = {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        }
        config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)

        hook = SequenceParallelSplitHook(metadata, config)
        assert hook.metadata == metadata
        assert hook.config == config
        assert hook.module_forward_metadata is None  # Not initialized until initialize_hook

    def test_hook_initialize(self):
        """Test SequenceParallelSplitHook.initialize_hook."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
        from vllm_omni.diffusion.hooks.sequence_parallel import SequenceParallelSplitHook

        class DummyModule(nn.Module):
            def forward(self, hidden_states, encoder_hidden_states):
                return hidden_states

        metadata = {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        }
        config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)

        hook = SequenceParallelSplitHook(metadata, config)
        module = DummyModule()

        # Initialize hook
        result = hook.initialize_hook(module)
        assert result is module
        assert hook.module_forward_metadata is not None
        assert hook.module_forward_metadata._cls is DummyModule


@pytest.mark.cpu
class TestSequenceParallelGatherHookInit:
    """Test SequenceParallelGatherHook initialization."""

    def test_hook_init_single_output(self):
        """Test SequenceParallelGatherHook with single output."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
        from vllm_omni.diffusion.hooks.sequence_parallel import SequenceParallelGatherHook

        metadata = SequenceParallelOutput(gather_dim=1, expected_dims=3)
        config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)

        hook = SequenceParallelGatherHook(metadata, config)
        # Single output should be wrapped in a list
        assert isinstance(hook.metadata, list)
        assert len(hook.metadata) == 1
        assert hook.metadata[0].gather_dim == 1

    def test_hook_init_multiple_outputs(self):
        """Test SequenceParallelGatherHook with multiple outputs."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
        from vllm_omni.diffusion.hooks.sequence_parallel import SequenceParallelGatherHook

        metadata = [
            SequenceParallelOutput(gather_dim=1, expected_dims=3),
            SequenceParallelOutput(gather_dim=2, expected_dims=4),
        ]
        config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)

        hook = SequenceParallelGatherHook(metadata, config)
        assert len(hook.metadata) == 2
        assert hook.metadata[0].gather_dim == 1
        assert hook.metadata[1].gather_dim == 2


@pytest.mark.cpu
class TestResolveTextLen:
    """Test _resolve_text_len in SequenceParallelSplitHook."""

    def test_resolve_int_source(self):
        """Test resolving text length from integer source."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
        from vllm_omni.diffusion.hooks.sequence_parallel import SequenceParallelSplitHook

        class DummyModule(nn.Module):
            def forward(self, x, txt_ids):
                return x

        partial_input = SequenceParallelPartialInput(
            split_dim=1,
            text_len_source=256,  # Fixed integer
            expected_dims=3,
        )
        config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)

        hook = SequenceParallelSplitHook({"x": partial_input}, config)
        hook.initialize_hook(DummyModule())

        # Resolve with integer source
        text_len = hook._resolve_text_len(partial_input, (), {})
        assert text_len == 256

    def test_resolve_string_source_from_tensor(self):
        """Test resolving text length from tensor parameter."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
        from vllm_omni.diffusion.hooks.sequence_parallel import SequenceParallelSplitHook

        class DummyModule(nn.Module):
            def forward(self, x, txt_ids):
                return x

        partial_input = SequenceParallelPartialInput(
            split_dim=1,
            text_len_source="txt_ids",  # Get from parameter
            expected_dims=3,
        )
        config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)

        hook = SequenceParallelSplitHook({"x": partial_input}, config)
        hook.initialize_hook(DummyModule())

        # Provide txt_ids tensor
        txt_ids = torch.randn(128, 64)  # shape[0] = 128
        kwargs = {"txt_ids": txt_ids}

        text_len = hook._resolve_text_len(partial_input, (), kwargs)
        assert text_len == 128

    def test_resolve_text_len_caching(self):
        """Test that text length is cached."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
        from vllm_omni.diffusion.hooks.sequence_parallel import SequenceParallelSplitHook

        class DummyModule(nn.Module):
            def forward(self, x, txt_ids):
                return x

        partial_input = SequenceParallelPartialInput(
            split_dim=1,
            text_len_source="txt_ids",
            expected_dims=3,
        )
        config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)

        hook = SequenceParallelSplitHook({"x": partial_input}, config)
        hook.initialize_hook(DummyModule())

        txt_ids = torch.randn(64, 32)
        kwargs = {"txt_ids": txt_ids}

        # First call - should populate cache
        hook._resolve_text_len(partial_input, (), kwargs)
        assert "txt_ids" in hook._text_len_cache
        assert hook._text_len_cache["txt_ids"] == 64

        # Second call - should use cache
        text_len = hook._resolve_text_len(partial_input, (), kwargs)
        assert text_len == 64


@pytest.mark.cpu
class TestHookNameTemplates:
    """Test hook name template generation."""

    def test_input_hook_name(self):
        """Test input hook name format."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _SP_INPUT_HOOK_TEMPLATE

        name = _SP_INPUT_HOOK_TEMPLATE.format("blocks.0")
        assert name == "sp_input---blocks.0"

    def test_output_hook_name(self):
        """Test output hook name format."""
        from vllm_omni.diffusion.hooks.sequence_parallel import _SP_OUTPUT_HOOK_TEMPLATE

        name = _SP_OUTPUT_HOOK_TEMPLATE.format("proj_out")
        assert name == "sp_output---proj_out"


@pytest.mark.cpu
class TestApplyRemoveSequenceParallel:
    """Test apply_sequence_parallel and remove_sequence_parallel functions."""

    def test_apply_sp_registers_hooks(self):
        """Test that apply_sequence_parallel registers hooks on modules."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
        from vllm_omni.diffusion.hooks.sequence_parallel import (
            _SP_INPUT_HOOK_TEMPLATE,
            _SP_OUTPUT_HOOK_TEMPLATE,
            apply_sequence_parallel,
        )

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_in = nn.Linear(10, 10)
                self.proj_out = nn.Linear(10, 10)

            def forward(self, hidden_states):
                x = self.proj_in(hidden_states)
                return self.proj_out(x)

        model = SimpleModel()
        config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)
        plan = {
            "proj_in": {"hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3)},
            "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
        }

        # Apply SP
        apply_sequence_parallel(model, config, plan)

        # Check hooks are registered

        assert hasattr(model.proj_in, "_hook_registry")
        assert hasattr(model.proj_out, "_hook_registry")

        proj_in_registry = model.proj_in._hook_registry
        proj_out_registry = model.proj_out._hook_registry

        assert _SP_INPUT_HOOK_TEMPLATE.format("proj_in") in proj_in_registry._hooks
        assert _SP_OUTPUT_HOOK_TEMPLATE.format("proj_out") in proj_out_registry._hooks

    def test_remove_sp_removes_hooks(self):
        """Test that remove_sequence_parallel removes hooks from modules."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
        from vllm_omni.diffusion.hooks.sequence_parallel import (
            _SP_INPUT_HOOK_TEMPLATE,
            _SP_OUTPUT_HOOK_TEMPLATE,
            apply_sequence_parallel,
            remove_sequence_parallel,
        )

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_in = nn.Linear(10, 10)
                self.proj_out = nn.Linear(10, 10)

            def forward(self, hidden_states):
                x = self.proj_in(hidden_states)
                return self.proj_out(x)

        model = SimpleModel()
        config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)
        plan = {
            "proj_in": {"hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3)},
            "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
        }

        # Apply then remove SP
        apply_sequence_parallel(model, config, plan)
        remove_sequence_parallel(model, plan)

        # Check hooks are removed
        proj_in_registry = model.proj_in._hook_registry
        proj_out_registry = model.proj_out._hook_registry

        assert _SP_INPUT_HOOK_TEMPLATE.format("proj_in") not in proj_in_registry._hooks
        assert _SP_OUTPUT_HOOK_TEMPLATE.format("proj_out") not in proj_out_registry._hooks

    def test_apply_sp_with_wildcard(self):
        """Test apply_sequence_parallel with wildcard module names."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
        from vllm_omni.diffusion.hooks.sequence_parallel import (
            _SP_INPUT_HOOK_TEMPLATE,
            apply_sequence_parallel,
        )

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block() for _ in range(3)])

            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return x

        model = Model()
        config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)
        plan = {
            "blocks.*": {"x": SequenceParallelInput(split_dim=1, expected_dims=3)},
        }

        # Apply SP
        apply_sequence_parallel(model, config, plan)

        # Check all blocks have hooks registered
        for i, block in enumerate(model.blocks):
            assert hasattr(block, "_hook_registry")
            registry = block._hook_registry
            assert _SP_INPUT_HOOK_TEMPLATE.format("blocks.*") in registry._hooks


@pytest.mark.cpu
class TestDimensionValidation:
    """Test expected_dims validation in hooks."""

    def test_skip_shard_on_wrong_dims(self):
        """Test that sharding is skipped when tensor dims don't match expected."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
        from vllm_omni.diffusion.hooks.sequence_parallel import SequenceParallelSplitHook

        class DummyModule(nn.Module):
            def forward(self, x):
                return x

        # Expect 3D tensor
        metadata = {
            "x": SequenceParallelInput(split_dim=1, expected_dims=3),
        }
        config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)

        hook = SequenceParallelSplitHook(metadata, config)
        hook.initialize_hook(DummyModule())

        # Provide 4D tensor (wrong dims)
        tensor_4d = torch.randn(2, 4, 8, 16)

        # _prepare_sp_input should return tensor unchanged when dims don't match
        result = hook._prepare_sp_input(tensor_4d, metadata["x"], (), {})
        # Since expected_dims=3 but tensor has 4 dims, should return original
        assert result.shape == tensor_4d.shape


@pytest.mark.cpu
class TestSequenceParallelConfig:
    """Test SequenceParallelConfig dataclass."""

    def test_config_defaults_invalid(self):
        """Test that SequenceParallelConfig with default values raises error.

        At least one of ulysses_degree or ring_degree must be > 1 to enable SP.
        """
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig

        with pytest.raises(ValueError, match="must be > 1"):
            SequenceParallelConfig()  # Both defaults are 1, which is invalid

    def test_config_ulysses_only(self):
        """Test SequenceParallelConfig with Ulysses only."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig

        config = SequenceParallelConfig(ulysses_degree=4, ring_degree=1)
        assert config.sequence_parallel_size == 4

    def test_config_ring_only(self):
        """Test SequenceParallelConfig with Ring only."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig

        config = SequenceParallelConfig(ulysses_degree=1, ring_degree=4)
        assert config.sequence_parallel_size == 4

    def test_config_hybrid(self):
        """Test SequenceParallelConfig with hybrid (Ulysses + Ring)."""
        from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig

        config = SequenceParallelConfig(ulysses_degree=2, ring_degree=4)
        assert config.sequence_parallel_size == 8

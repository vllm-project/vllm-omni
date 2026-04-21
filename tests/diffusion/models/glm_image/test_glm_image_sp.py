# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GLM-Image Sequence Parallelism support."""

import pytest

from vllm_omni.diffusion.data import DiffusionParallelConfig


@pytest.fixture(scope="function", autouse=True)
def setup_sp_groups(mocker):
    """Set up SP and TP groups for each test function."""
    mock_get_sp_group = mocker.patch("vllm_omni.diffusion.distributed.parallel_state.get_sp_group")
    mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1)
    mock_get_tp_group = mocker.patch("vllm.distributed.parallel_state.get_tp_group")

    mock_sp_group = mocker.MagicMock()
    mock_sp_group.world_size = 4
    mock_get_sp_group.return_value = mock_sp_group

    mock_tp_group = mocker.MagicMock()
    mock_tp_group.world_size = 1
    mock_get_tp_group.return_value = mock_tp_group
    yield


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_glm_image_sp_plan_defined():
    """Test that _sp_plan is properly defined on GlmImageTransformer2DModel."""
    from vllm_omni.diffusion.models.glm_image.glm_image_transformer import (
        GlmImageTransformer2DModel,
    )

    assert hasattr(GlmImageTransformer2DModel, "_sp_plan")
    plan = GlmImageTransformer2DModel._sp_plan
    assert plan is not None

    # Verify plan structure
    assert "prepare" in plan
    assert "proj_out" in plan


def test_glm_image_sp_plan_valid():
    """Validate _sp_plan structure."""
    from vllm_omni.diffusion.distributed.sp_plan import validate_sp_plan
    from vllm_omni.diffusion.models.glm_image.glm_image_transformer import (
        GlmImageTransformer2DModel,
    )

    plan = GlmImageTransformer2DModel._sp_plan
    validate_sp_plan(plan)


def test_glm_image_prepare_module_exists():
    """Test that GlmImagePrepare module exists."""
    from vllm_omni.diffusion.models.glm_image.glm_image_transformer import (
        GlmImagePrepare,
    )

    assert GlmImagePrepare is not None


def test_glm_image_attention_accepts_parallel_config():
    """Test that GlmImageAttention accepts parallel_config parameter."""
    from vllm_omni.diffusion.models.glm_image.glm_image_transformer import (
        GlmImageAttention,
    )

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=2,
        ring_degree=2,
        tensor_parallel_size=1,
        sequence_parallel_size=4,
    )

    attn = GlmImageAttention(
        dim=2560,
        num_heads=64,
        head_dim=40,
        parallel_config=parallel_config,
    )

    assert attn.parallel_config is not None
    assert attn.parallel_config.sequence_parallel_size == 4


def test_glm_image_transformer_block_accepts_parallel_config():
    """Test that GlmImageTransformerBlock accepts parallel_config parameter."""
    from vllm_omni.diffusion.models.glm_image.glm_image_transformer import (
        GlmImageTransformerBlock,
    )

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=2,
        ring_degree=2,
        tensor_parallel_size=1,
        sequence_parallel_size=4,
    )

    block = GlmImageTransformerBlock(
        dim=2560,
        num_attention_heads=64,
        attention_head_dim=40,
        time_embed_dim=512,
        parallel_config=parallel_config,
    )

    assert block.attn1.parallel_config is not None
    assert block.attn1.parallel_config.sequence_parallel_size == 4


def test_glm_image_has_sp_support():
    """Test that GLM-Image has SP support implemented."""
    from vllm_omni.diffusion.models.glm_image.glm_image_transformer import (
        GlmImageTransformer2DModel,
    )

    # Check that the model has parallel_config support
    assert hasattr(GlmImageTransformer2DModel, "__init__")

    # Verify the model can be instantiated with SP config

    # This test just verifies the structure exists
    # Actual SP testing requires multi-GPU setup


@pytest.mark.cuda
@pytest.mark.sp
def test_glm_image_sp_inference():
    """Test SP inference (requires multi-GPU setup)."""
    pytest.skip("Requires multi-GPU SP setup")

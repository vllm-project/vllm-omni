"""Tests for Flux2 sequence parallel support."""

import pytest
from pytest_mock import MockerFixture

from vllm_omni.diffusion.data import DiffusionParallelConfig


@pytest.fixture(scope="function", autouse=True)
def setup_sp_groups(mocker: MockerFixture):
    """Set up mock TP/SP groups for Flux2 SP structure tests."""
    mock_get_sp_group = mocker.patch("vllm_omni.diffusion.distributed.parallel_state.get_sp_group")
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        return_value=0,
    )
    mock_get_tp_group = mocker.patch("vllm.distributed.parallel_state.get_tp_group")

    mock_sp_group = mocker.MagicMock()
    mock_sp_group.world_size = 4
    mock_get_sp_group.return_value = mock_sp_group

    mock_tp_group = mocker.MagicMock()
    mock_tp_group.world_size = 1
    mock_tp_group.rank_in_group = 0
    mock_tp_group.rank = 0
    mock_get_tp_group.return_value = mock_tp_group
    yield


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_parallel_config() -> DiffusionParallelConfig:
    return DiffusionParallelConfig(
        tensor_parallel_size=1,
        ulysses_degree=2,
        ring_degree=2,
        sequence_parallel_size=4,
    )


def test_flux2_sp_plan_defined():
    from vllm_omni.diffusion.models.flux2.flux2_transformer import (
        Flux2Transformer2DModel,
    )

    assert hasattr(Flux2Transformer2DModel, "_sp_plan")
    plan = Flux2Transformer2DModel._sp_plan
    assert plan is not None
    assert "" in plan
    assert "rope_prepare" in plan
    assert "proj_out" in plan


def test_flux2_sp_plan_valid():
    from vllm_omni.diffusion.distributed.sp_plan import validate_sp_plan
    from vllm_omni.diffusion.models.flux2.flux2_transformer import (
        Flux2Transformer2DModel,
    )

    validate_sp_plan(Flux2Transformer2DModel._sp_plan)


def test_flux2_rope_prepare_exists():
    from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2RopePrepare

    assert Flux2RopePrepare is not None


def test_flux2_attention_accepts_parallel_config():
    from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2Attention

    parallel_config = _make_parallel_config()
    attn = Flux2Attention(
        parallel_config=parallel_config,
        query_dim=256,
        heads=8,
        dim_head=32,
        added_kv_proj_dim=256,
    )

    assert attn.parallel_config.sequence_parallel_size == 4


def test_flux2_parallel_self_attention_accepts_parallel_config():
    from vllm_omni.diffusion.models.flux2.flux2_transformer import (
        Flux2ParallelSelfAttention,
    )

    parallel_config = _make_parallel_config()
    attn = Flux2ParallelSelfAttention(
        parallel_config=parallel_config,
        query_dim=256,
        heads=8,
        dim_head=32,
        out_dim=256,
    )

    assert attn.parallel_config.sequence_parallel_size == 4


def test_flux2_blocks_accept_parallel_config():
    from vllm_omni.diffusion.models.flux2.flux2_transformer import (
        Flux2SingleTransformerBlock,
        Flux2TransformerBlock,
    )

    parallel_config = _make_parallel_config()
    block = Flux2TransformerBlock(
        parallel_config=parallel_config,
        dim=256,
        num_attention_heads=8,
        attention_head_dim=32,
    )
    single_block = Flux2SingleTransformerBlock(
        parallel_config=parallel_config,
        dim=256,
        num_attention_heads=8,
        attention_head_dim=32,
    )

    assert block.attn.parallel_config.sequence_parallel_size == 4
    assert single_block.attn.parallel_config.sequence_parallel_size == 4

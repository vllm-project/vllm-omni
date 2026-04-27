# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that FLUX transformer blocks correctly propagate `quant_config` and
`prefix` through all sub-layers.

The tests instantiate blocks with a known prefix and verify that all quantization-
aware sub-layers (AdaLayerNorm, FeedForward, Attention projections) receive the
prefix rooted at the block prefix.  This is critical for quantized weight loading
to match checkpoint keys to the correct model parameters.
"""

import os

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Standard dimensions for a minimal FLUX block
_DIM = 64
_HEADS = 2
_HEAD_DIM = _DIM // _HEADS


@pytest.fixture(autouse=True)
def _init_distributed():
    """Initialize the minimal distributed environment required by
    vLLM parallel linear layers (tensor-parallel group must exist)."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29502")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


def _param_names(module) -> set[str]:
    """Return the set of all parameter names in a module."""
    return {name for name, _ in module.named_parameters()}


def test_flux_transformer_block_passes_prefix():
    """FluxTransformerBlock propagates prefix to norm1, norm1_context, attn, ff, ff_context."""
    from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformerBlock

    prefix = "transformer_blocks.0"
    block = FluxTransformerBlock(
        dim=_DIM,
        num_attention_heads=_HEADS,
        attention_head_dim=_HEAD_DIM,
        quant_config=None,
        prefix=prefix,
    )

    params = _param_names(block)

    # norm1 and norm1_context (AdaLayerNormZero) should have linear weights
    assert any(name.startswith("norm1.linear.") for name in params), (
        f"norm1.linear.* not found in params: {sorted(params)}"
    )
    assert any(name.startswith("norm1_context.linear.") for name in params), (
        f"norm1_context.linear.* not found in params: {sorted(params)}"
    )

    # attn should have QKV projections
    assert any(name.startswith("attn.to_qkv.") for name in params), (
        f"attn.to_qkv.* not found in params: {sorted(params)}"
    )

    # ff and ff_context should have net layers
    assert any(name.startswith("ff.net.") for name in params), f"ff.net.* not found in params: {sorted(params)}"
    assert any(name.startswith("ff_context.net.") for name in params), (
        f"ff_context.net.* not found in params: {sorted(params)}"
    )


def test_flux_single_transformer_block_passes_prefix():
    """FluxSingleTransformerBlock propagates prefix to norm, proj_mlp, attn."""
    from vllm_omni.diffusion.models.flux.flux_transformer import FluxSingleTransformerBlock

    prefix = "single_transformer_blocks.0"
    block = FluxSingleTransformerBlock(
        dim=_DIM,
        num_attention_heads=_HEADS,
        attention_head_dim=_HEAD_DIM,
        quant_config=None,
        prefix=prefix,
    )

    params = _param_names(block)

    # norm (AdaLayerNormZeroSingle) should have linear weights
    assert any(name.startswith("norm.linear.") for name in params), (
        f"norm.linear.* not found in params: {sorted(params)}"
    )

    # proj_mlp (ReplicatedLinear) should have weight
    assert any(name.startswith("proj_mlp.") for name in params), f"proj_mlp.* not found in params: {sorted(params)}"

    # attn should have QKV projection
    assert any(name.startswith("attn.to_qkv.") for name in params), (
        f"attn.to_qkv.* not found in params: {sorted(params)}"
    )


def test_flux_feedforward_passes_prefix():
    """FeedForward propagates prefix to net.0 (GELU proj) and net.2 (output proj)."""
    from vllm_omni.diffusion.models.flux.flux_transformer import FeedForward

    prefix = "transformer_blocks.0.ff"
    ff = FeedForward(
        dim=_DIM,
        dim_out=_DIM,
        quant_config=None,
        prefix=prefix,
    )

    params = _param_names(ff)

    # net.0 is ColumnParallelApproxGELU which wraps a ColumnParallelLinear
    assert any("net.0" in name for name in params), f"net.0 not found in params: {sorted(params)}"

    # net.2 is RowParallelLinear
    assert any("net.2" in name for name in params), f"net.2 not found in params: {sorted(params)}"

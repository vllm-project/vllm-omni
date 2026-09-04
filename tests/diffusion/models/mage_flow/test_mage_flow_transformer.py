# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Padded-batch attention and the sequence-parallel padding guard."""

import os

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _init_distributed():
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29531")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def _force_default_gemm(monkeypatch):
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


def _tiny_transformer(**overrides):
    from vllm_omni.diffusion.models.mage_flow.mage_flow_transformer import (
        MageFlowTransformer2DModel,
    )

    kwargs = {
        "in_channels": 4,
        "out_channels": 4,
        "context_in_dim": 6,
        "hidden_size": 8,
        "num_heads": 1,
        "depth": 1,
        "axes_dim": [2, 2, 4],
    }
    kwargs.update(overrides)
    return MageFlowTransformer2DModel(**kwargs)


def _initialize_parameters(module: nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.02, 0.02)


def test_single_request_explicit_padding_is_handed_to_the_backend_as_a_mask():
    """Padding travels as a joint attention mask, not as per-request slicing.

    The backend unpads, runs varlen attention and repads with zeros, the same
    contract Qwen-Image and Flux use. The kernel therefore sees the full padded
    width, the mask carries text tokens in front, and the padded tail must still
    come back zeroed.
    """
    from vllm_omni.diffusion.models.mage_flow import mage_flow_layers

    class _LocalAttentionRecorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_lengths = []
            self.masks = []

        def forward(self, query, _key, value, attn_metadata=None):
            self.query_lengths.append(query.shape[1])
            self.masks.append(None if attn_metadata is None else attn_metadata.attn_mask)
            return value

    attention = mage_flow_layers.MageJointAttention(
        query_dim=8,
        heads=1,
        dim_head=8,
        added_kv_proj_dim=8,
    )
    _initialize_parameters(attention)
    recorder = _LocalAttentionRecorder()
    attention.attention = recorder
    image_mask = torch.tensor([[True, True, False]])
    text_mask = torch.tensor([[True, False]])

    image_output, text_output = attention(
        torch.randn(1, 3, 8),
        torch.randn(1, 2, 8),
        torch.ones(1, 3, 4, dtype=torch.complex64),
        image_attention_mask=image_mask,
        encoder_attention_mask=text_mask,
    )

    # One call over the padded width of 5, with the two text tokens in front.
    assert recorder.query_lengths == [5]
    assert torch.equal(
        recorder.masks[0],
        torch.tensor([[True, False, True, True, False]]),
    )
    assert not image_output[:, 2:].count_nonzero()
    assert not text_output[:, 1:].count_nonzero()


def test_sp_rejects_request_level_image_padding():
    """Sharding splits a padded sequence blindly, so it must refuse one.

    Real tokens and filler would land on different ranks with no mask to tell
    them apart. The pipeline refuses ``max_num_seqs > 1`` under SP at startup
    and routes guided requests through sequential CFG rather than the packed
    path, so this runtime guard is defensive: it pins the invariant that keeps
    a future caller from reintroducing a padded batch under sharding.
    """
    from vllm_omni.diffusion.forward_context import (
        ForwardContext,
        override_forward_context,
    )

    model = _tiny_transformer()
    context = ForwardContext(
        sp_plan_hooks_applied=True,
        _sp_shard_depth=1,
    )

    with (
        override_forward_context(context),
        pytest.raises(ValueError, match="does not support padded token"),
    ):
        model(
            hidden_states=torch.randn(1, 3, 4),
            encoder_hidden_states=torch.randn(1, 2, 6),
            timestep=torch.zeros(1),
            image_grid_hw=(1, 2),
            image_attention_mask=torch.tensor([[True, True, False]]),
        )

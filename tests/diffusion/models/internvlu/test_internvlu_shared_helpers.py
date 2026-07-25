# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.norm import LayerNorm
from vllm_omni.diffusion.models.internvlu.internvlu_transformer import (
    InternVLUTransformerBlock,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    ColumnParallelApproxGELU,
    FeedForward,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_DIM = 64


@pytest.fixture(autouse=True)
def _init_distributed():
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29503")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


def _make_block() -> InternVLUTransformerBlock:
    return InternVLUTransformerBlock(
        dim=_DIM,
        num_heads=2,
        head_dim=32,
        quant_config=None,
        prefix="transformer_blocks.0",
    )


def test_block_reuses_parameter_free_shared_fp32_layer_norm():
    block = _make_block()
    inputs = torch.tensor(
        [[[1000.0 + index / 16 for index in range(_DIM)]]],
        dtype=torch.bfloat16,
    )
    expected = F.layer_norm(
        inputs.float(),
        (_DIM,),
        weight=None,
        bias=None,
        eps=1e-6,
    ).to(inputs.dtype)

    for norm in (
        block.img_norm1,
        block.img_norm2,
        block.txt_norm1,
        block.txt_norm2,
    ):
        assert isinstance(norm, LayerNorm)
        assert norm.elementwise_affine is False
        assert norm.state_dict() == {}
        torch.testing.assert_close(norm(inputs), expected, rtol=0, atol=0)


def test_block_reuses_qwen_feed_forward_without_checkpoint_or_numeric_drift():
    block = _make_block()

    for stream, feed_forward in (
        ("img_mlp", block.img_mlp),
        ("txt_mlp", block.txt_mlp),
    ):
        assert isinstance(feed_forward, FeedForward)
        assert isinstance(feed_forward.net[0], ColumnParallelApproxGELU)
        assert set(feed_forward.state_dict()) == {
            "net.0.proj.weight",
            "net.0.proj.bias",
            "net.2.weight",
            "net.2.bias",
        }, stream
        assert feed_forward.net[0].proj.bias is not None
        assert feed_forward.net[2].bias is not None

    feed_forward = block.img_mlp
    with torch.no_grad():
        for parameter in feed_forward.parameters():
            parameter.copy_(
                torch.linspace(
                    -0.05,
                    0.05,
                    parameter.numel(),
                    dtype=parameter.dtype,
                ).reshape_as(parameter)
            )
    hidden_states = torch.randn(2, 3, _DIM)
    projected = F.linear(
        hidden_states,
        feed_forward.net[0].proj.weight,
        feed_forward.net[0].proj.bias,
    )
    activated = F.gelu(projected, approximate="tanh")
    expected = F.linear(
        activated,
        feed_forward.net[2].weight,
        feed_forward.net[2].bias,
    )

    torch.testing.assert_close(feed_forward(hidden_states), expected)

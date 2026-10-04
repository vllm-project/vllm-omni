# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed import ProcessGroup
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import (
    Transformer2DModel,
    TransformerBlock,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def set_tp_group(world: int, rank: int) -> None:
    group = get_tp_group()
    group.device_group = MagicMock(spec=ProcessGroup)
    group.world_size, group.rank_in_group = world, rank


@pytest.fixture(autouse=True)
def setup_mock_tp_env() -> Iterator[None]:
    """Set up CPU SDPA and a mock TP group, restoring both after each test."""
    config = OmniDiffusionConfig(diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}})
    with (
        set_current_diffusion_config(config),
        patch("vllm.distributed.parallel_state._TP", SimpleNamespace(world_size=1, rank_in_group=0)),
    ):
        yield


@pytest.mark.parametrize("world", [2, 4, 7, 8])
@torch.inference_mode()
def test_checkpoint_shards_reconstruct_projections(world: int) -> None:
    """Exercise actual loader callbacks, local attention and uneven FFN slices."""
    torch.manual_seed(42)
    original = TransformerBlock(84, 21, 7, 16, None, 1e-5).eval()
    x = torch.randn(2, 9, 84)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[0, -3:] = False
    angle = torch.randn(9, 2).repeat_interleave(2, -1)
    kwargs = dict(
        encoder_hidden_states=x,
        attention_mask=mask,
        image_rotary_emb=(angle.cos(), angle.sin()),
    )
    expected_attn, expected_ffn = original.attn(x, **kwargs), original.feed_forward(x)
    attn_parts, ffn_parts = [], []
    for rank in range(world):
        set_tp_group(world, rank)
        local = TransformerBlock(84, 21, 7, 16, None, 1e-5).eval()
        loaded = local.load_weights(original.state_dict().items())
        assert loaded == set(local.state_dict())
        with patch("vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce", side_effect=lambda x: x):
            attn_parts.append(local.attn(x, **kwargs))
            ffn_parts.append(local.feed_forward(x))
    torch.testing.assert_close(sum(attn_parts), expected_attn, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(sum(ffn_parts), expected_ffn, atol=2e-6, rtol=2e-5)


def test_tp8_padded_attention_heads_are_zero() -> None:
    dense = TransformerBlock(84, 21, 7, 16, None, 1e-5)
    set_tp_group(8, 7)
    local = TransformerBlock(84, 21, 7, 16, None, 1e-5)
    local.load_weights(dense.state_dict().items())
    for projection in (local.attn.qkv_proj, local.attn.to_out[0]):
        assert torch.count_nonzero(projection.weight) == 0
    assert torch.count_nonzero(local.feed_forward.linear_1.weight) > 0


def test_all_refiners_load_tp_weights() -> None:
    config = dict(
        hidden_size=126,
        num_layers=2,
        num_refiner_layers=1,
        num_attention_heads=21,
        num_kv_heads=7,
        multiple_of=16,
        axes_dim_rope=(2, 2, 2),
        axes_lens=(8, 8, 8),
        text_feat_dim=12,
        in_channels=4,
    )
    dense = Transformer2DModel.from_config(config)
    set_tp_group(4, 0)
    local = Transformer2DModel.from_config(config)
    assert AutoWeightsLoader(local).load_weights(dense.state_dict().items()) == set(local.state_dict())
    for family in (local.noise_refiner, local.ref_image_refiner, local.context_refiner, local.layers):
        for block in family:
            assert isinstance(block.attn.qkv_proj, QKVParallelLinear)
            assert isinstance(block.feed_forward.linear_2, RowParallelLinear)

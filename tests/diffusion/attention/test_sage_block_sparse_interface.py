# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Structural checks stay usable without loading FlashInfer or a CUDA runtime."""

import pytest
import torch

from vllm_omni.diffusion.attention.ops.flashinfer_block_sparse import validate_sparse_inputs
from vllm_omni.diffusion.attention.ops.sage_block_sparse_attention import (
    sage_block_sparse_attention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _inputs(sq=129, sk=191, batch=2, heads=3):
    q = torch.empty(batch, sq, heads, 128, dtype=torch.bfloat16)
    k = torch.empty(batch, sk, heads, 128, dtype=torch.bfloat16)
    indices = torch.zeros(batch, heads, (sq + 63) // 64, 2, dtype=torch.int32)
    counts = torch.ones(indices.shape[:3], dtype=torch.int32)
    sizes = torch.full(((sk + 63) // 64,), 64, dtype=torch.int32)
    return q, k, torch.empty_like(k), indices, counts, sizes


@pytest.mark.parametrize("metadata_rank", [0, 1, 2, 3])
def test_rectangular_ragged_attention_accepts_public_block_size_layouts(metadata_rank):
    args = list(_inputs())
    sizes = args[-1]
    args[-1] = {
        0: None,
        1: sizes,
        2: sizes.repeat(2, 1),
        3: sizes.repeat(2, 3, 1),
    }[metadata_rank]
    validate_sparse_inputs(*args)


@pytest.mark.parametrize("operand", [0, 1, 2])
def test_rejects_incorrect_operand_dtype(operand):
    args = list(_inputs())
    args[operand] = args[operand].float()
    with pytest.raises(TypeError, match="BF16"):
        validate_sparse_inputs(*args)


@pytest.mark.parametrize("metadata", [3, 4, 5])
def test_rejects_non_int32_metadata(metadata):
    args = list(_inputs())
    args[metadata] = args[metadata].long()
    with pytest.raises(TypeError, match="INT32"):
        validate_sparse_inputs(*args)


def test_rejects_metadata_for_wrong_query_length():
    args = list(_inputs())
    args[3] = args[3][:, :, :-1].contiguous()
    with pytest.raises(ValueError, match="Sparse indices"):
        validate_sparse_inputs(*args)


def test_rejects_wrong_key_block_sizes():
    args = list(_inputs(sq=65, sk=191))
    args[-1] = torch.full((2,), 64, dtype=torch.int32)
    with pytest.raises(ValueError, match="Block sizes"):
        validate_sparse_inputs(*args)


def test_rejects_cross_device_metadata_before_launch():
    args = list(_inputs())
    args[4] = args[4].to("meta")
    with pytest.raises(ValueError, match="share one device"):
        validate_sparse_inputs(*args)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_scale_fails_before_cuda_import(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        sage_block_sparse_attention(*_inputs(), scale)


def test_prepared_query_cannot_reuse_scales_from_another_length():
    q_int8 = torch.empty(2, 3, 129, 128, dtype=torch.int8)
    wrong_scales = torch.empty(2, 3, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="prepared_q must match"):
        sage_block_sparse_attention(*_inputs(), 0.125, prepared_q=(q_int8, wrong_scales))

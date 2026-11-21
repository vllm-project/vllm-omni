# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.ops.video_tiles import get_tile_metadata, pack_video_tiles

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="requires CUDA and Triton")


def _reference(x, seq_shape, tile_size):
    partition, sizes, non_pad, _ = get_tile_metadata(seq_shape, tile_size, math.prod(tile_size), x.device)
    output = x.new_zeros(x.shape[0], sizes.numel() * math.prod(tile_size), x.shape[2], x.shape[3])
    output[:, non_pad] = x[:, partition]
    return output


@pytest.mark.cpu
@pytest.mark.parametrize("seq_shape", [(1, 1, 1), (4, 8, 8), (3, 10, 10)])
def test_pack_video_tiles_cpu_and_gradients(seq_shape):
    x = torch.randn(2, math.prod(seq_shape), 3, 8, requires_grad=True)
    output = pack_video_tiles(x, seq_shape, (4, 8, 8))
    torch.testing.assert_close(output, _reference(x, seq_shape, (4, 8, 8)), rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("seq_len", [7, 9])
def test_pack_video_tiles_rejects_sequence_length_mismatch(seq_len):
    x = torch.randn(1, seq_len, 1, 8)
    with pytest.raises(ValueError, match="sequence length"):
        pack_video_tiles(x, (2, 2, 2), (4, 8, 8))


@pytest.mark.cpu
@pytest.mark.parametrize(
    "seq_shape,tile_size",
    [
        ((0, 1, 1), (4, 8, 8)),
        ((1, 0, 1), (4, 8, 8)),
        ((1, 1, 0), (4, 8, 8)),
        ((-1, 1, 1), (4, 8, 8)),
        ((1, 1, 1), (0, 8, 8)),
        ((1, 1, 1), (4, 0, 8)),
        ((1, 1, 1), (4, 8, 0)),
        ((1, 1, 1), (4, -1, 8)),
    ],
)
def test_pack_video_tiles_rejects_nonpositive_dimensions(seq_shape, tile_size):
    x = torch.randn(1, 1, 1, 8)
    with pytest.raises(ValueError, match="positive"):
        pack_video_tiles(x, seq_shape, tile_size)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
@pytest.mark.parametrize("seq_shape", [(1, 1, 1), (4, 8, 8), (3, 10, 10), (5, 9, 17), (31, 22, 40)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["bshd", "bhsd", "sliced"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_pack_video_tiles_cuda(seq_shape, dtype, layout, head_dim):
    batch, seq_len, heads = 2, math.prod(seq_shape), 3
    if layout == "bhsd":
        x = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=dtype).transpose(1, 2)
    elif layout == "sliced":
        x = torch.randn(batch, seq_len + 1, heads, head_dim * 2, device="cuda", dtype=dtype)[:, :seq_len, :, ::2]
    else:
        x = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=dtype)
    output = pack_video_tiles(x, seq_shape, (4, 8, 8))
    assert output.is_contiguous()
    torch.testing.assert_close(output, _reference(x, seq_shape, (4, 8, 8)), rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pack_video_tiles_compile(dtype):
    seq_shape = (3, 10, 10)
    x = torch.randn(1, math.prod(seq_shape), 2, 128, device="cuda", dtype=dtype)
    compiled = torch.compile(pack_video_tiles, fullgraph=True)
    for source in (x, x + 1):
        torch.testing.assert_close(
            compiled(source, seq_shape, (4, 8, 8)), _reference(source, seq_shape, (4, 8, 8)), rtol=0, atol=0
        )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Triton/cuDNN sparse attention against an independent FP32 token oracle."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd
from vllm_omni.diffusion.attention.ops.cudnn_block_sparse import require_cudnn_sparse

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def runtime(provider):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires SM80+")
    if provider == "triton":
        pytest.importorskip("fastvideo_kernel.block_sparse_attn")
    else:
        if torch.cuda.get_device_capability()[0] not in (9, 10, 11, 12):
            pytest.skip("requires a cuDNN BSA supported GPU")
        pytest.importorskip("cudnn", minversion="1.31.0")
        require_cudnn_sparse(torch.device("cuda", torch.accelerator.current_device_index()))
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize(
    "provider,tolerance",
    [
        pytest.param("triton", 0.02, marks=hardware_marks(res={"cuda": ["L4", "H100", "RTX5090"]}), id="triton"),
        pytest.param("cudnn", 0.01, marks=hardware_marks(res={"cuda": ["H100", "B200", "RTX5090"]}), id="cudnn"),
    ],
)
@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize(
    "batch,heads,q_blocks,empty,scale",
    [
        (1, 7, 5, False, 1.0),
        (1, 7, 5, True, 1.0),
        (2, 3, 3, False, 1.0),
        (2, 3, 3, False, 4.0),
        (2, 3, 3, True, 1.0),
    ],
)
@torch.inference_mode()
def test_sparse_ragged_tiles_and_graph_replay(provider, tolerance, tile, batch, heads, q_blocks, empty, scale):
    torch.manual_seed(1919)
    q = torch.randn(batch, q_blocks * tile, heads, 128, device="cuda", dtype=torch.bfloat16) * scale
    k = torch.randn(batch, 5 * tile, heads, 128, device="cuda", dtype=torch.bfloat16) * scale
    v = torch.randn_like(k)
    sizes = torch.tensor([tile, 17, 0, tile - 1, 1], device="cuda", dtype=torch.int32)
    if empty:
        sizes.zero_()
    mask = torch.rand(batch, heads, q_blocks, 5, device="cuda") < 0.5
    mask[..., 0] = True
    mask[..., -1, :] = False
    valid = (torch.arange(tile, device="cuda")[None] < sizes[:, None]).flatten()
    v[:, ~valid] = 100  # Padded keys must not contribute to attention.

    def reference():
        valid = (torch.arange(tile, device="cuda")[None] < sizes[:, None]).flatten()
        token_mask = mask.repeat_interleave(tile, -2).repeat_interleave(tile, -1) & valid[None, None, None, :]
        scores = q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2) / 128**0.5
        probs = scores.masked_fill(~token_mask, -torch.inf).softmax(-1).nan_to_num()
        return (probs @ v.float().transpose(1, 2)).transpose(1, 2)

    def run():
        return block_sparse_attn_bshd(q, k, v, mask, sizes, 128**-0.5, provider=provider, block_size=tile)

    actual = run()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference(), atol=tolerance, rtol=tolerance)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    mask.zero_()
    graph.replay()
    assert torch.count_nonzero(captured).item() == 0
    mask[..., 0] = True
    sizes.copy_(torch.tensor([17, tile, 1, 0, tile - 1], device="cuda", dtype=torch.int32))
    for tensor in (q, k, v):
        tensor.copy_(torch.randn_like(tensor))
    graph.replay()
    torch.testing.assert_close(captured.float(), reference(), atol=tolerance, rtol=tolerance)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU sparse attention against masked FP32 attention and graph replay."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd, resolve_block_sparse_provider

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


@pytest.mark.parametrize("lengths", [(128, 128, 128), (5, 77, 2), (1, 1, 1), (0, 0, 0)])
@torch.inference_mode()
def test_sparse_partial_blocks_and_graph_replay(lengths: tuple[int, int, int]) -> None:
    torch.manual_seed(43)
    q, k, v = [torch.randn(1, 384, 2, 128, device="npu", dtype=torch.bfloat16) for _ in range(3)]
    sizes = torch.tensor(lengths, device=q.device, dtype=torch.int32)
    blocks = torch.tensor(
        [[[[1, 0, 1], [1, 1, 0], [0, 0, 0]], [[1, 1, 0], [0, 1, 1], [0, 0, 0]]]],
        device=q.device,
        dtype=torch.bool,
    )
    provider = resolve_block_sparse_provider("auto", "bf16", q.device, q.dtype, 128, block_size=128)
    assert provider == "npu"

    def reference() -> torch.Tensor:
        valid = torch.arange(384, device=q.device) % 128 < sizes.repeat_interleave(128)
        mask = blocks.repeat_interleave(128, -2).repeat_interleave(128, -1) & valid[None, None, None, :]
        return F.scaled_dot_product_attention(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(), attn_mask=mask
        ).transpose(1, 2)

    def run() -> torch.Tensor:
        return block_sparse_attn_bshd(q, k, v, blocks, sizes, 128**-0.5, provider=provider, block_size=128)

    actual = run()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference(), atol=0.01, rtol=0.01)
    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    blocks.zero_()
    graph.replay()
    torch.npu.synchronize()
    assert torch.count_nonzero(captured) == 0
    blocks[..., 0] = True
    sizes.copy_(torch.tensor([17, 128, 1], device=q.device, dtype=torch.int32))
    for tensor in (q, k, v):
        tensor.copy_(torch.randn_like(tensor))
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured.float(), reference(), atol=0.01, rtol=0.01)

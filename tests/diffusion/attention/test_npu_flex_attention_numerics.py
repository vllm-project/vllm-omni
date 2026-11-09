# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device numerics for MindIE-SD NPU FlexAttention with block-causal metadata."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.utils import npu_flex_attention
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires an Ascend NPU")
@torch.inference_mode()
def test_catlass_npu_flex_matches_dense_across_requests():
    mindiesd = pytest.importorskip("mindiesd")
    if not hasattr(mindiesd, "make_block_causal_mask"):
        pytest.skip("requires a MindIE-SD build with CATLASS DSL FlexAttention")
    torch.manual_seed(42)
    scale = 128**-0.5
    # Exercise changing lengths, bidirectional blocks and partially filled mask blocks
    # in one process with the real CATLASS DSL execution path.
    for index, length in enumerate((257, 299, 385, 409, 512, 677, 781, 896, 913, 1024)):
        spans = [(16 + index, length)] if index % 2 else [(8, 152), (170 + index, length)]
        q, k, v = [torch.randn(1, length, 2, 128, device="npu", dtype=torch.bfloat16) * 0.1 for _ in range(3)]
        metadata = AttentionMetadata(full_attn_spans=[spans])
        actual = npu_flex_attention.try_npu_flex_attention(q, k, v, metadata, scale)
        assert actual is not None
        block_ids = torch.full((length,), -1, device=q.device)
        for block_id, (start, end) in enumerate(spans):
            block_ids[start:end] = block_id
        positions = torch.arange(length, device=q.device)
        keep = (positions[:, None] >= positions[None, :]) | (
            (block_ids[:, None] >= 0) & (block_ids[:, None] == block_ids[None, :])
        )
        scores = (q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-2, -1)) * scale
        expected = (scores.masked_fill(~keep, float("-inf")).softmax(-1) @ v.transpose(1, 2).float()).transpose(1, 2)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-2)

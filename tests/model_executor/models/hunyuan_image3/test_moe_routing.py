# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import _hunyuan_image3_topk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_moe_routing_preserves_expert_ids_and_rounded_weights(dtype):
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(17, 64, dtype=torch.float32, generator=generator)
    logits[0].zero_()
    logits[1, :9] = torch.linspace(10.0, 10.001, 9)
    hidden_states = torch.empty(17, 32, dtype=dtype)

    weights, indices = _hunyuan_image3_topk(hidden_states, logits, 8, False)
    gates = torch.softmax(logits, dim=-1, dtype=torch.float32)
    values, ids = torch.topk(gates, 8, dim=-1)
    normalized = values / values.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    expected = normalized.to(dtype).float()
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    torch.testing.assert_close(indices, ids.int(), rtol=0, atol=0)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for MiniMax Music 3 attention dtype dispatch."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.model_executor.models.minimax_music3 import dit

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def attention(mocker):
    backend = nn.Module()
    backend.attn_backend = SimpleNamespace(get_name=lambda: "FLASH_ATTN")
    mocker.patch.object(dit, "_build_native_attention", return_value=backend)
    return dit.Attention(dim=24, head_dim=8)


def test_fp32_attention_matches_sdpa(attention, mocker):
    native_backend_forward = mocker.patch.object(
        attention.backend, "forward", side_effect=AssertionError("FP32 must bypass the native backend currently")
    )
    generator = torch.Generator().manual_seed(42)
    q, k, v = (torch.randn(2, 7, 3, 8, generator=generator) for _ in range(3))
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=False,
        scale=8**-0.5,
    ).transpose(1, 2)

    actual = attention._attend(q, k, v)

    native_backend_forward.assert_not_called()
    assert actual.shape == q.shape
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_bf16_attention_dispatches_to_backend(attention, mocker):
    generator = torch.Generator().manual_seed(42)
    q, k, v = (torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.bfloat16) for _ in range(3))
    expected = torch.full_like(q, 0.25)
    native_backend_forward = mocker.patch.object(attention.backend, "forward", return_value=expected)
    sdpa = mocker.patch.object(
        dit.F, "scaled_dot_product_attention", side_effect=AssertionError("BF16 can use the available backend")
    )

    actual = attention._attend(q, k, v)

    native_backend_forward.assert_called_once()
    args, kwargs = native_backend_forward.call_args
    assert len(args) == 3
    assert args[0] is q
    assert args[1] is k
    assert args[2] is v
    assert kwargs == {}
    assert actual is expected
    sdpa.assert_not_called()

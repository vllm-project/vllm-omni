# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl


pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def test_sdpa_impl_force_math_env_uses_math_only_kernel(monkeypatch) -> None:
    calls = []

    @contextmanager
    def fake_sdp_kernel(**kwargs):
        calls.append(kwargs)
        yield

    def fake_attention(query, key, value, **kwargs):
        del key, value, kwargs
        return query

    monkeypatch.setenv("DIFFUSION_SDPA_FORCE_MATH", "1")
    monkeypatch.setattr(torch.backends.cuda, "sdp_kernel", fake_sdp_kernel)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_attention)

    impl = SDPAImpl(num_heads=2, head_size=4, softmax_scale=0.5, num_kv_heads=2)
    query = torch.zeros(1, 2, 3, 4)
    key = torch.zeros(1, 2, 3, 4)
    value = torch.zeros(1, 2, 3, 4)

    output = impl.forward_cuda(query, key, value)

    assert output.shape == query.shape
    assert calls == [
        {
            "enable_flash": False,
            "enable_math": True,
            "enable_mem_efficient": False,
            "enable_cudnn": False,
        }
    ]


def test_sdpa_impl_force_fp32_env_upcasts_kernel_inputs_and_restores_output_dtype(monkeypatch) -> None:
    seen = {}

    def fake_attention(query, key, value, **kwargs):
        del kwargs
        seen["query_dtype"] = query.dtype
        seen["key_dtype"] = key.dtype
        seen["value_dtype"] = value.dtype
        return query

    monkeypatch.setenv("DIFFUSION_SDPA_FORCE_FP32", "1")
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_attention)

    impl = SDPAImpl(num_heads=2, head_size=4, softmax_scale=0.5, num_kv_heads=2)
    query = torch.zeros(1, 2, 3, 4, dtype=torch.bfloat16)
    key = torch.zeros(1, 2, 3, 4, dtype=torch.bfloat16)
    value = torch.zeros(1, 2, 3, 4, dtype=torch.bfloat16)

    output = impl.forward_cuda(query, key, value)

    assert seen == {
        "query_dtype": torch.float32,
        "key_dtype": torch.float32,
        "value_dtype": torch.float32,
    }
    assert output.dtype == torch.bfloat16

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util

import pytest
import torch

from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]

requires_sage3_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((12, 0), (12, 1))
    or importlib.util.find_spec("sageattn3") is None,
    reason="requires SM120/SM121 CUDA GPU and a matching SageAttention3 build",
)


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@requires_sage3_blackwell
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_real_sage3_fullgraph_candidate(head_size, dtype, causal):
    from vllm_omni.diffusion.attention.backends import sage_attn3 as sage

    torch.manual_seed(42)
    scale = head_size**-0.5
    impl = sage.SageAttention3Impl(4, head_size, scale, causal=causal)
    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    compiled = torch.compile(impl.forward_cuda, fullgraph=True, dynamic=True)
    for batch, q_len, kv_len in ((1, 127, 127), (2, 193, 193 if causal else 257)):
        # Exercise noncontiguous inputs and sequence padding.
        q = torch.randn(batch, 4, q_len, head_size, device="cuda", dtype=dtype).transpose(1, 2)
        k, v = (torch.randn(batch, kv_len, 4, head_size, device="cuda", dtype=dtype) for _ in range(2))
        result = impl.resolve_execution_path(context, q, k, v, None)
        if torch.cuda.get_device_capability() == (12, 0):
            assert result.requested_support(context).status is SupportStatus.SUPPORTED
        else:
            assert result.support.status is SupportStatus.UNMIGRATED
        before = [t.clone() for t in (q, k, v)]
        eager = impl.forward_cuda(q, k, v)
        for _ in range(2):
            out = compiled(q, k, v)
            assert out.shape == q.shape and out.dtype == dtype and out.device == q.device
            assert out.is_contiguous() and torch.isfinite(out).all()
            torch.testing.assert_close(out, eager, atol=1e-3, rtol=1e-3)
        for actual, original in zip((q, k, v), before):
            torch.testing.assert_close(actual, original, atol=0, rtol=0)
    checks = torch.library.opcheck(
        sage._sageattn3_blackwell_op.default,
        (q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), causal),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert all(value == "SUCCESS" for value in checks.values())

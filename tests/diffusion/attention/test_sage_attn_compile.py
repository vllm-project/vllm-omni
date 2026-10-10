# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends import sage_attn as sage
from vllm_omni.diffusion.attention.capabilities import CompilationMode, ExecutionContext, SupportStatus
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

requires_sage_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0) or sage.sageattn is None,
    reason="requires SM90 CUDA GPU and SageAttention",
)


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@requires_sage_sm90
@pytest.mark.parametrize("head_size", [32, 64, 96, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_real_sage_fullgraph_contract(head_size, dtype, causal):
    torch.manual_seed(42)
    scale = 0.17
    impl = sage.SageAttentionImpl(4, head_size, scale, causal=causal)
    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    compiled = torch.compile(impl.forward_cuda, fullgraph=True, dynamic=True)
    for batch, q_len, kv_len in ((1, 127, 127), (2, 193, 193 if causal else 257)):
        # Contiguous head elements with noncontiguous NHD strides are accepted
        # by Sage; head sizes 32 and 96 also exercise its internal padding.
        q = torch.randn(batch, 4, q_len, head_size, device="cuda", dtype=dtype).transpose(1, 2)
        k, v = (torch.randn(batch, kv_len, 4, head_size, device="cuda", dtype=dtype) for _ in range(2))
        result = impl.resolve_execution_path(context, q, k, v, None)
        assert result.kernel_variant == "sage_sm90"
        assert result.requested_support(context).status is SupportStatus.SUPPORTED
        assert result.compilation_mode is CompilationMode.CUSTOM_OP
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
        sage._sage_attention_op.default,
        (q, k, v, causal, scale),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert all(value == "SUCCESS" for value in checks.values())


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@requires_sage_sm90
def test_real_sage_production_attention_entry(monkeypatch):
    from vllm_omni.diffusion.attention import layer as attention_layer

    monkeypatch.setattr(
        attention_layer, "get_attn_backend_for_role", lambda **kwargs: (sage.SageAttentionBackend, None)
    )
    monkeypatch.setattr(attention_layer, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    layer = attention_layer.Attention(num_heads=4, head_size=64, softmax_scale=0.17, causal=False)
    compiled = torch.compile(layer, fullgraph=True, dynamic=True)
    for length in (127, 193, 127):
        q, k, v = (torch.randn(1, length, 4, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        path = layer.resolve_execution_path(ExecutionContext(platform="cuda", require_fullgraph=True), q, k, v, None)
        assert path.support.status is SupportStatus.SUPPORTED
        torch.testing.assert_close(compiled(q, k, v), layer(q, k, v), atol=1e-3, rtol=1e-3)

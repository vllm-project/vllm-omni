# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
from vllm_omni.diffusion.attention.capabilities import CompilationMode, ExecutionContext, SupportStatus
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("q_heads,kv_heads,head_dim", [(8, 8, 64), (8, 2, 64), (8, 2, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("masked", [False, True])
def test_sdpa_dynamic_fullgraph_matches_eager(q_heads, kv_heads, head_dim, dtype, masked):
    impl = SDPAImpl(num_heads=q_heads, num_kv_heads=kv_heads, head_size=head_dim, softmax_scale=head_dim**-0.5)
    compiled = torch.compile(impl.forward_cuda, fullgraph=True, dynamic=True)
    with torch.inference_mode():
        for seed, (q_len, kv_len) in enumerate([(3, 7), (5, 9)]):
            torch.manual_seed(seed)
            query = torch.randn(2, q_len, q_heads, head_dim, device="cuda", dtype=dtype)
            key = torch.randn(2, kv_len, kv_heads, head_dim, device="cuda", dtype=dtype)
            value = torch.randn_like(key)
            metadata = None
            if masked:
                mask = torch.ones(2, kv_len, device="cuda", dtype=torch.bool)
                mask[:, -1] = False
                metadata = AttentionMetadata(attn_mask=mask, extra={"attention_mask_mode": "padding"})
            expected = impl.forward_cuda(query, key, value, metadata)
            actual = compiled(query, key, value, metadata)
            if q_heads != kv_heads:
                assert actual.is_contiguous()
                assert actual.flatten(2).untyped_storage().data_ptr() == actual.untyped_storage().data_ptr()
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("q_heads,kv_heads,head_dim", [(8, 8, 64), (8, 2, 64), (8, 2, 512)])
@pytest.mark.parametrize("masked", [False, True])
def test_sdpa_production_attention_fullgraph(monkeypatch, q_heads, kv_heads, head_dim, masked):
    from vllm_omni.diffusion.attention import layer as attention_layer

    monkeypatch.setattr(attention_layer, "get_attn_backend_for_role", lambda **_kwargs: (SDPABackend, None))
    monkeypatch.setattr(attention_layer, "build_parallel_attention_strategy", lambda **_kwargs: NoParallelAttention())
    layer = attention_layer.Attention(
        num_heads=q_heads, num_kv_heads=kv_heads, head_size=head_dim, softmax_scale=head_dim**-0.5, causal=False
    )
    compiled = torch.compile(layer, fullgraph=True, dynamic=True)
    with torch.inference_mode():
        for q_len, kv_len in [(3, 7), (5, 9)]:
            query = torch.randn(2, q_len, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
            key = torch.randn(2, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
            value = torch.randn_like(key)
            mask = torch.ones(2, kv_len, device="cuda", dtype=torch.bool) if masked else None
            if mask is not None:
                mask[:, -1] = False
            metadata = AttentionMetadata(attn_mask=mask, extra={"attention_mask_mode": "padding"})
            context = ExecutionContext(platform="cuda", require_fullgraph=True)
            result = layer.resolve_execution_path(context, query, key, value, metadata)
            expected_mode = CompilationMode.TRACEABLE if q_heads == kv_heads else CompilationMode.CUSTOM_OP
            assert result.compilation_mode is expected_mode
            assert result.requested_support(context).status is SupportStatus.SUPPORTED
            torch.testing.assert_close(compiled(query, key, value, metadata), layer(query, key, value, metadata))


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("head_dim", [64, 512])
@pytest.mark.parametrize("unequal_value_dim", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_sdpa_gqa_custom_op_schema_and_fake(head_dim, unequal_value_dim, masked):
    query = torch.randn(2, 3, 8, head_dim, device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
    key = torch.randn(2, 7, 2, head_dim, device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
    value_dim = head_dim // 2 if unequal_value_dim else head_dim
    value = torch.randn(2, 7, 2, value_dim, device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
    mask = torch.ones(2, 1, 1, 7, device="cuda", dtype=torch.bool) if masked else None
    output = torch.ops.vllm_omni.sdpa_gqa_attention(query, key, value, mask, False, head_dim**-0.5)
    assert output.shape == (2, 3, 8, value_dim)
    assert output.is_contiguous()
    torch.library.opcheck(
        torch.ops.vllm_omni.sdpa_gqa_attention.default,
        (query, key, value, mask, False, head_dim**-0.5),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_sdpa_compiled_fallback_preserves_mask_gradients():
    impl = SDPAImpl(num_heads=8, num_kv_heads=2, head_size=64, softmax_scale=0.125)
    query = torch.randn(2, 3, 8, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 7, 2, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.randn(2, 1, 3, 7, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    metadata = AttentionMetadata(attn_mask=mask)
    # Avoid a cuDNN mask-backward alignment limitation unrelated to this boundary.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        expected = impl.forward_cuda(query, key, value, metadata)
        expected_grad = torch.autograd.grad(expected.sum(), mask)[0]
        # Grad-enabled masks are not a fullgraph contract; preserve eager fallback.
        compiled = torch.compile(impl.forward_cuda, backend="eager", fullgraph=False)
        actual = compiled(query, key, value, metadata)
        actual_grad = torch.autograd.grad(actual.sum(), mask)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("kv_heads,head_dim", [(8, 64), (2, 64), (2, 512)])
@pytest.mark.parametrize(
    "source_dtype,target_dtype", [(torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16)]
)
@pytest.mark.parametrize("masked", [False, True])
def test_sdpa_fullgraph_autocast_output_dtype(kv_heads, head_dim, source_dtype, target_dtype, masked):
    impl = SDPAImpl(num_heads=8, num_kv_heads=kv_heads, head_size=head_dim, softmax_scale=head_dim**-0.5)
    query = torch.randn(2, 3, 8, head_dim, device="cuda", dtype=source_dtype)
    key = torch.randn(2, 7, kv_heads, head_dim, device="cuda", dtype=source_dtype)
    value = torch.randn_like(key)
    mask = torch.zeros(2, 1, 3, 7, device="cuda", dtype=source_dtype) if masked else None
    if mask is not None:
        mask[..., -1] = -2.5
    metadata = AttentionMetadata(attn_mask=mask)
    compiled = torch.compile(impl.forward_cuda, fullgraph=True, dynamic=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=target_dtype):
        expected = impl.forward_cuda(query, key, value, metadata)
        actual = compiled(query, key, value, metadata)
        assert actual.dtype is expected.dtype
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

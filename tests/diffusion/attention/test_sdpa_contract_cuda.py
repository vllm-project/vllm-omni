# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

import vllm_omni.diffusion.attention.backends.sdpa as sdpa_backend
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.attention.backends.utils.attn_runtime_selector import can_sdpa_use_fused_gqa
from vllm_omni.diffusion.attention.capabilities import (
    CompilationMode,
    ExecutionContext,
    ParallelStrategy,
    SupportStatus,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _tensor_signature(tensor):
    return tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize(
    "query_heads,kv_heads,fused_gqa,path",
    [
        (4, 4, False, "sdpa_equal_heads"),
        (4, 2, True, "sdpa_native_gqa"),
        (4, 2, False, "sdpa_expanded_kv"),
    ],
)
def test_sdpa_contract_reports_selected_gqa_path(monkeypatch, query_heads, kv_heads, fused_gqa, path):
    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: fused_gqa)
    query = torch.empty((1, 3, query_heads, 64), device="cuda", dtype=torch.bfloat16)
    key = torch.empty((1, 3, kv_heads, 64), device="cuda", dtype=torch.bfloat16)
    impl = SDPAImpl(num_heads=query_heads, num_kv_heads=kv_heads, head_size=64, softmax_scale=0.5)

    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, key, key, None)

    assert result.path == path
    assert result.support.status is SupportStatus.SUPPORTED


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_sdpa_contract_requires_published_padding_semantics(monkeypatch):
    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: False)
    query = torch.empty((1, 3, 4, 64), device="cuda", dtype=torch.bfloat16)
    key = torch.empty((1, 7, 2, 64), device="cuda", dtype=torch.bfloat16)
    mask = torch.ones((1, 7), device="cuda", dtype=torch.bool)
    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=64, softmax_scale=0.5)

    unpublished = impl.resolve_execution_path(
        ExecutionContext(platform="cuda"), query, key, key, AttentionMetadata(attn_mask=mask)
    )
    published = impl.resolve_execution_path(
        ExecutionContext(platform="cuda"),
        query,
        key,
        key,
        AttentionMetadata(attn_mask=mask, extra={"attention_mask_mode": "padding"}),
    )

    assert unpublished.support.status is SupportStatus.UNMIGRATED
    assert published.path == "sdpa_expanded_kv"
    assert published.support.status is SupportStatus.SUPPORTED


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_sdpa_contract_keeps_parallel_and_causal_paths_unmigrated():
    query = torch.empty((1, 3, 4, 64), device="cuda", dtype=torch.bfloat16)
    impl = SDPAImpl(num_heads=4, num_kv_heads=4, head_size=64, softmax_scale=0.5)
    causal_impl = SDPAImpl(num_heads=4, num_kv_heads=4, head_size=64, softmax_scale=0.5, causal=True)

    parallel = impl.resolve_execution_path(
        ExecutionContext(platform="cuda", parallel_strategy=ParallelStrategy.ULYSSES), query, query, query, None
    )
    causal = causal_impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, query, query, None)

    assert parallel.support.status is SupportStatus.UNMIGRATED
    assert causal.support.status is SupportStatus.UNMIGRATED


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_sdpa_contract_supports_fullgraph_for_inference(monkeypatch):
    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: True)
    query = torch.empty((1, 3, 4, 64), device="cuda", dtype=torch.bfloat16)
    key = torch.empty((1, 3, 2, 64), device="cuda", dtype=torch.bfloat16)
    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=64, softmax_scale=0.5)
    context = ExecutionContext(platform="cuda", require_fullgraph=True)

    result = impl.resolve_execution_path(context, query, key, key, None)

    assert result.compilation_mode is CompilationMode.CUSTOM_OP
    assert result.requested_support(context).status is SupportStatus.SUPPORTED


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_sdpa_contract_does_not_promise_gqa_autograd_fullgraph(kv_heads):
    query = torch.empty((1, 3, 4, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.empty((1, 3, kv_heads, 64), device="cuda", dtype=torch.bfloat16)
    impl = SDPAImpl(num_heads=4, num_kv_heads=kv_heads, head_size=64, softmax_scale=0.5)
    context = ExecutionContext(platform="cuda", require_fullgraph=True)

    result = impl.resolve_execution_path(context, query, key, key, None)

    assert result.compilation_mode is CompilationMode.EAGER_ONLY
    assert result.requested_support(context).status is SupportStatus.UNSUPPORTED


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_sdpa_autocast_contract_remains_unmigrated():
    query = torch.empty((1, 3, 4, 64), device="cuda", dtype=torch.bfloat16)
    key = torch.empty((1, 7, 2, 64), device="cuda", dtype=torch.bfloat16)
    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=64, softmax_scale=0.5)
    with torch.autocast("cuda", dtype=torch.float16):
        result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, key, key, None)
    assert result.support.status is SupportStatus.UNMIGRATED


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_sdpa_contract_rejects_nondivisible_gqa():
    query = torch.empty((1, 3, 3, 64), device="cuda", dtype=torch.bfloat16)
    key = torch.empty((1, 3, 2, 64), device="cuda", dtype=torch.bfloat16)
    impl = SDPAImpl(num_heads=3, num_kv_heads=2, head_size=64, softmax_scale=0.5)

    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, key, key, None)

    assert result.support.status is SupportStatus.UNSUPPORTED
    assert "multiple of KV heads" in result.support.reason


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_sdpa_resolver_and_forward_probe_identical_inputs(monkeypatch):
    probe_inputs = []
    forward_inputs = []

    def probe(query, key, value, mask, causal):
        probe_inputs.append(
            (
                _tensor_signature(query),
                _tensor_signature(key),
                _tensor_signature(value),
                _tensor_signature(mask),
                mask.clone(),
                causal,
            )
        )
        return False

    def fake_sdpa(query, key, value, **kwargs):
        forward_inputs.append((query.shape, key.shape, value.shape, kwargs["enable_gqa"]))
        return query

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", probe)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    query = torch.empty((1, 3, 4, 64), device="cuda", dtype=torch.bfloat16)
    key = torch.empty((1, 7, 2, 64), device="cuda", dtype=torch.bfloat16)
    mask = torch.ones((1, 7), device="cuda", dtype=torch.bool)
    metadata = AttentionMetadata(attn_mask=mask, extra={"attention_mask_mode": "padding"})
    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=64, softmax_scale=0.5)

    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, key, key, metadata)
    impl.forward_cuda(query, key, key, metadata)

    assert result.path == "sdpa_expanded_kv"
    assert probe_inputs[0][:4] == probe_inputs[1][:4]
    assert torch.equal(probe_inputs[0][4], probe_inputs[1][4])
    assert probe_inputs[0][5] == probe_inputs[1][5]
    assert forward_inputs == [((1, 4, 3, 64), (1, 4, 7, 64), (1, 4, 7, 64), False)]
    assert metadata.attn_mask is mask
    assert metadata.extra == {"attention_mask_mode": "padding"}


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("query_heads,kv_heads,head_dim", [(8, 8, 64), (8, 2, 64), (8, 2, 512)])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("q_len,kv_len", [(4, 4), (3, 7)])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sdpa_gqa_contract_matches_cuda_reference(query_heads, kv_heads, head_dim, batch, q_len, kv_len, masked, dtype):
    torch.manual_seed(0)
    query = torch.randn((batch, q_len, query_heads, head_dim), device="cuda", dtype=dtype)
    key = torch.randn((batch, kv_len, kv_heads, head_dim), device="cuda", dtype=dtype)
    value = torch.randn((batch, kv_len, kv_heads, head_dim), device="cuda", dtype=dtype)
    mask = torch.ones((batch, kv_len), device="cuda", dtype=torch.bool) if masked else None
    if mask is not None and kv_len > 1:
        mask[:, -1] = False
    metadata = AttentionMetadata(attn_mask=mask, extra={"attention_mask_mode": "padding"}) if mask is not None else None
    scale = head_dim**-0.5
    impl = SDPAImpl(num_heads=query_heads, num_kv_heads=kv_heads, head_size=head_dim, softmax_scale=scale)

    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, key, value, metadata)
    output = impl.forward_cuda(query, key, value, metadata)

    if query_heads == kv_heads:
        expected_runtime_path = "sdpa_equal_heads"
    else:
        normalized_mask = mask[:, None, None, :] if mask is not None else None
        native_gqa = can_sdpa_use_fused_gqa(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
            normalized_mask,
            False,
        )
        expected_runtime_path = "sdpa_native_gqa" if native_gqa else "sdpa_expanded_kv"
    assert result.path == expected_runtime_path
    assert result.support.status is SupportStatus.SUPPORTED
    reference = torch.nn.functional.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3),
        key.repeat_interleave(query_heads // kv_heads, dim=2).permute(0, 2, 1, 3),
        value.repeat_interleave(query_heads // kv_heads, dim=2).permute(0, 2, 1, 3),
        attn_mask=mask[:, None, None, :] if mask is not None else None,
        dropout_p=0.0,
        scale=scale,
    ).permute(0, 2, 1, 3)
    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_sdpa_contract_does_not_claim_support_for_cross_device_mask():
    query = torch.randn((1, 3, 8, 64), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((1, 7, 2, 64), device="cuda", dtype=torch.bfloat16)
    metadata = AttentionMetadata(
        attn_mask=torch.ones((1, 7), dtype=torch.bool),
        extra={"attention_mask_mode": "padding"},
    )
    impl = SDPAImpl(num_heads=8, num_kv_heads=2, head_size=64, softmax_scale=0.125)

    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, key, key, metadata)

    assert result.support.status is SupportStatus.UNMIGRATED

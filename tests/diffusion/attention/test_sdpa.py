# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

import vllm_omni.diffusion.attention.backends.sdpa as sdpa_backend
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("value_dim", [4, 8])
@pytest.mark.parametrize("masked", [False, True])
def test_sdpa_gqa_custom_op_returns_contiguous_bshd(value_dim, masked):
    query = torch.randn(2, 3, 4, 8).permute(0, 2, 1, 3)
    key = torch.randn(2, 5, 2, 8).permute(0, 2, 1, 3)
    value = torch.randn(2, 5, 2, value_dim).permute(0, 2, 1, 3)
    mask = torch.ones(2, 1, 1, 5, dtype=torch.bool) if masked else None
    if mask is not None:
        mask[..., -1] = False

    output = torch.ops.vllm_omni.sdpa_gqa_attention(query, key, value, mask, False, 8**-0.5)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key.repeat_interleave(2, dim=1), value.repeat_interleave(2, dim=1), attn_mask=mask, scale=8**-0.5
    ).permute(0, 2, 1, 3)

    assert output.shape == (2, 3, 4, value_dim)
    assert output.is_contiguous()
    assert output.flatten(2).untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
    torch.testing.assert_close(output, expected)


def test_sdpa_npu_packed_padding_matches_dense_mask():
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, 9, 2, 8, generator=generator)
    key = torch.randn(1, 11, 2, 8, generator=generator)
    value = torch.randn(1, 11, 2, 8, generator=generator)
    # Inflated padding so that wrongly reading the suffix changes the result.
    value[:, 7:] += 100
    metadata = AttentionMetadata(
        packed_padding=PackedPaddingMetadata(
            q_length=7,
            kv_length=7,
            cu_seqlens_q=torch.tensor([0, 7], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 7], dtype=torch.int32),
        )
    )
    impl = SDPAImpl(num_heads=2, head_size=8, softmax_scale=0.5)
    actual = impl.forward_npu(query, key, value, metadata)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
        attn_mask=(torch.arange(11) < 7)[None, None, None],
        scale=0.5,
    ).permute(0, 2, 1, 3)
    torch.testing.assert_close(actual, expected)


@pytest.mark.cpu
def test_sdpa_packed_mask_free_gate_is_npu_only(monkeypatch):
    """H3 skips the padding mask only behind this gate: on for NPU, off elsewhere.

    A revert of SDPABackend.supports_packed_mask_free to the base False would
    silently restore the Q x K padding mask on NPU while the forward_npu test
    above stays green, so pin the gate itself.
    """
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    assert SDPABackend.supports_packed_mask_free() is True

    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    assert SDPABackend.supports_packed_mask_free() is False


def test_sdpa_expands_kv_when_native_gqa_kernel_is_unavailable(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: False)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=8, softmax_scale=0.5)
    output = impl.forward_cuda(
        torch.randn(1, 3, 4, 8),
        torch.randn(1, 3, 2, 8),
        torch.randn(1, 3, 2, 8),
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == key_shape == value_shape == (1, 4, 3, 8)
    assert kwargs["enable_gqa"] is False
    assert output.shape == (1, 3, 4, 8)


def test_sdpa_keeps_compressed_kv_when_native_gqa_kernel_is_available(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: True)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=8, softmax_scale=0.5)
    output = impl.forward_cuda(
        torch.randn(1, 3, 4, 8),
        torch.randn(1, 3, 2, 8),
        torch.randn(1, 3, 2, 8),
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == (1, 4, 3, 8)
    assert key_shape == value_shape == (1, 2, 3, 8)
    assert kwargs["enable_gqa"] is True
    assert output.shape == (1, 3, 4, 8)


def test_sdpa_preconstruction_contract_is_runtime_dependent():
    result = SDPABackend.resolve_capabilities(ExecutionContext(platform="cuda"))

    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.path == "runtime_gqa_dependent"


def test_sdpa_contract_does_not_claim_cuda_support_for_cpu_tensors():
    impl = SDPAImpl(num_heads=4, num_kv_heads=4, head_size=64, softmax_scale=0.5)
    query = torch.empty((1, 3, 4, 64), dtype=torch.bfloat16)

    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, query, query, None)

    assert result.support.status is SupportStatus.UNMIGRATED

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
from vllm_omni.diffusion.attention.backends.utils import fa
from vllm_omni.diffusion.attention.selector import _load_backend_cls
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _CudaDeviceTensor(torch.Tensor):
    """Exercise CUDA/ROCm eligibility with CPU storage and mocked blocks."""

    @property
    def device(self):
        return torch.device("cuda")


@pytest.fixture
def decode_model(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(transformer, "is_forward_context_available", lambda: False)

    def make(num_blocks=3, backend=FlashAttentionBackend, requested="FLASH_ATTN", dtype=torch.bfloat16):
        model = transformer.QwenImage21Transformer2DModel.__new__(transformer.QwenImage21Transformer2DModel)
        nn.Module.__init__(model)
        model.causal_condition = True
        model.causal_block = True
        model.parallel_config = None
        model.prefix_kv_cache_dtype = None
        model._decode_graph_manager = None
        target = torch.zeros(2, 4, 8, dtype=dtype).as_subclass(_CudaDeviceTensor)
        model.sequence_prepare = Mock(return_value=(target[:, :0], target, torch.empty(0, 4), torch.empty(4, 4)))
        model.sequence_prepare.expand_image_pad_mask = transformer.QwenImage21SequencePrepare.expand_image_pad_mask
        model.time_text_embed = Mock(return_value=torch.zeros(3, 8, dtype=dtype))
        model.modulation = Mock(return_value=torch.zeros(3, 32, dtype=dtype))
        model.norm_out = Mock(side_effect=lambda values, *args: values)
        model.proj_out = nn.Identity()
        model.transformer_blocks = [
            Mock(
                attn=SimpleNamespace(attn=SimpleNamespace(attn_backend=backend, backend_pref=requested)),
                side_effect=lambda hidden_states, **kwargs: hidden_states.as_subclass(torch.Tensor),
            )
            for _ in range(num_blocks)
        ]
        inputs = dict(
            hidden_states=torch.zeros(2, 4, 8, dtype=dtype),
            encoder_hidden_states=None,
            timestep=torch.ones(2),
            img_shapes=[[(1, 2, 2)], [(1, 2, 2)]],
            img_mask=torch.tensor([[False, False, False, True]] * 2),
            encoder_hidden_states_mask=torch.tensor([[True, False, True, True], [False, True, False, True]]),
            kv_cache=[{"cond": {"key": torch.empty(2, 3, 1, 8)}} for _ in range(num_blocks)],
        )
        return model, inputs

    return make


@pytest.mark.parametrize("num_blocks", [1, 3, 32])
@pytest.mark.parametrize("hip_version", [None, "test"], ids=["cuda-build", "rocm-build"])
@pytest.mark.parametrize(
    ("dtype", "prefix_dtype", "grad_enabled"),
    [
        (torch.bfloat16, None, False),
        (torch.float16, None, False),
        (torch.float32, None, False),
        (torch.bfloat16, None, True),
        (torch.bfloat16, "fp8_e4m3", False),
        (torch.bfloat16, "fp8_e4m3_v", False),
    ],
    ids=["bf16", "fp16", "fp32-metadata", "grad-enabled", "fp8-kv", "fp8-v"],
)
def test_decode_prepares_metadata_once_per_forward_and_refreshes_changed_mask(
    monkeypatch, decode_model, num_blocks, hip_version, dtype, prefix_dtype, grad_enabled
):
    monkeypatch.setattr(torch.version, "hip", hip_version)
    model, inputs = decode_model(num_blocks=num_blocks, dtype=dtype)
    model.prefix_kv_cache_dtype = prefix_dtype
    prepare = Mock(wraps=fa._get_unpad_data)
    monkeypatch.setattr(fa, "_get_unpad_data", prepare)

    with torch.set_grad_enabled(grad_enabled):
        model(**inputs)
    first_metadata = model.transformer_blocks[0].call_args.kwargs["attn_metadata"]
    first = first_metadata.extra["qwen21_decode_varlen"]
    assert prepare.call_count == 1
    for block in model.transformer_blocks:
        assert block.call_args.kwargs["attn_metadata"] is first_metadata
    assert first_metadata.extra["attention_mask_mode"] == "padding"
    torch.testing.assert_close(first.indices_k, torch.tensor([0, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13]))
    torch.testing.assert_close(first.cu_seqlens_k, torch.tensor([0, 6, 11], dtype=torch.int32))
    torch.testing.assert_close(first.cu_seqlens_q, torch.tensor([0, 4, 8], dtype=torch.int32))
    assert (first.max_seqlen_k, first.max_seqlen_q) == (6, 4)

    # Reuse and mutate the same mask object, with both changed positions and lengths.
    inputs["encoder_hidden_states_mask"][:, :3] = torch.tensor([[False, True, False], [True, True, False]])
    with torch.set_grad_enabled(grad_enabled):
        model(**inputs)
    second_metadata = model.transformer_blocks[0].call_args.kwargs["attn_metadata"]
    second = second_metadata.extra["qwen21_decode_varlen"]
    assert prepare.call_count == 2
    assert second is not first
    for block in model.transformer_blocks:
        assert block.call_args.kwargs["attn_metadata"] is second_metadata
    torch.testing.assert_close(second.indices_k, torch.tensor([1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]))
    torch.testing.assert_close(second.cu_seqlens_k, torch.tensor([0, 5, 11], dtype=torch.int32))
    torch.testing.assert_close(first.cu_seqlens_k, torch.tensor([0, 6, 11], dtype=torch.int32))


@pytest.mark.parametrize("requested", ["TORCH_SDPA", "FLASH_ATTN"])
@torch.no_grad()
def test_resolved_sdpa_bypasses_decode_preparation(monkeypatch, decode_model, requested):
    # vLLM queries the architecture while importing its ROCm platform module.
    monkeypatch.setattr("torch.cuda.get_device_properties", lambda device: SimpleNamespace(gcnArchName="gfx942"))
    from vllm_omni.platforms.rocm.platform import RocmOmniPlatform

    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (9, 4))
    monkeypatch.setattr("vllm._aiter_ops.is_aiter_found_and_supported", lambda: False)
    backend_path = RocmOmniPlatform.get_diffusion_attn_backend_cls(requested, head_size=8)
    backend = _load_backend_cls(backend_path)
    assert backend is SDPABackend
    model, inputs = decode_model(backend=backend, requested=requested)
    prepare = Mock(side_effect=AssertionError("SDPA must not prepare FlashAttention metadata"))
    monkeypatch.setattr(fa, "_get_unpad_data", prepare)
    model(**inputs)
    prepare.assert_not_called()
    for block in model.transformer_blocks:
        metadata = block.call_args.kwargs["attn_metadata"]
        assert metadata.attn_mask is not None
        assert "qwen21_decode_varlen" not in metadata.extra
        assert "attention_mask_mode" not in metadata.extra

    expected_mask = torch.tensor(
        [[True, False, True, True, True, True, True], [False, True, False, True, True, True, True]]
    )
    query = torch.zeros(2, 4, 1, 1)
    key = torch.zeros(2, 7, 1, 1)
    value = torch.arange(14, dtype=torch.float32).reshape(2, 7, 1, 1)
    impl = SDPAImpl(num_heads=1, head_size=1, softmax_scale=1.0)
    actual = impl.forward_hip(query, key, value, metadata)
    expected = nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=expected_mask[:, None, None, :],
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("exclusion", ["cpu", "sp", "compile", "unpadded", "no_backend"])
@pytest.mark.parametrize("hip_version", [None, "test"], ids=["cuda-build", "rocm-build"])
@torch.no_grad()
def test_decode_preparation_exclusions(monkeypatch, decode_model, exclusion, hip_version):
    monkeypatch.setattr(torch.version, "hip", hip_version)
    model, inputs = decode_model(backend=None if exclusion == "no_backend" else FlashAttentionBackend)
    if exclusion == "cpu":
        values = model.sequence_prepare.return_value
        model.sequence_prepare.return_value = tuple(value.as_subclass(torch.Tensor) for value in values)
    elif exclusion == "sp":
        model.parallel_config = DiffusionParallelConfig(ulysses_degree=2)
    elif exclusion == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif exclusion == "unpadded":
        inputs["encoder_hidden_states_mask"].fill_(True)
    prepare = Mock(side_effect=AssertionError("ineligible forward prepared decode metadata"))
    monkeypatch.setattr(fa, "_get_unpad_data", prepare)
    model(**inputs)
    prepare.assert_not_called()


@torch.no_grad()
def test_prefill_does_not_prepare_decode_metadata(monkeypatch, decode_model):
    model, inputs = decode_model()
    _, target, _, freqs = model.sequence_prepare.return_value
    model.sequence_prepare.return_value = (target[:, :3], target, freqs[:3], freqs)
    model.txt_in = nn.Identity()
    inputs["encoder_hidden_states"] = torch.zeros(2, 3, 8, dtype=torch.bfloat16)
    inputs["kv_cache"] = [{} for _ in model.transformer_blocks]
    prepare = Mock(side_effect=AssertionError("prefill must not prepare decode metadata"))
    monkeypatch.setattr(fa, "_get_unpad_data", prepare)
    model(**inputs)
    prepare.assert_not_called()
    for block in model.transformer_blocks:
        metadata = block.call_args.kwargs["attn_metadata"]
        assert metadata.attn_mask.ndim == 4
        assert "qwen21_decode_varlen" not in metadata.extra


def _reference_varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    outputs = []
    for index in range(cu_seqlens_q.numel() - 1):
        query = q[cu_seqlens_q[index] : cu_seqlens_q[index + 1]].transpose(0, 1)
        key = k[cu_seqlens_k[index] : cu_seqlens_k[index + 1]].transpose(0, 1)
        value = v[cu_seqlens_k[index] : cu_seqlens_k[index + 1]].transpose(0, 1)
        outputs.append(nn.functional.scaled_dot_product_attention(query, key, value).transpose(0, 1))
    return torch.cat(outputs)


@pytest.mark.parametrize("case", ["eligible", "query_length", "batch_size"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    ("hip_version", "fa_version"),
    [("test", None), (None, 2), (None, 3), (None, 4)],
    ids=["rocm-provider", "cuda-fa2", "cuda-fa3", "cuda-fa4"],
)
@torch.no_grad()
def test_flash_decode_reuses_metadata_and_matches_existing_unpadding(monkeypatch, case, dtype, hip_version, fa_version):
    monkeypatch.setattr(torch.version, "hip", hip_version)
    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    kernel = Mock(side_effect=_reference_varlen)
    monkeypatch.setattr(fa, "flash_attn_varlen_func", kernel)
    # Exercise version forwarding without requiring a CUDA or AITER kernel.
    monkeypatch.setattr(fa, "_is_vllm_flash_attn_varlen", lambda func: fa_version is not None and func is kernel)
    resolver = Mock(return_value=fa_version)
    monkeypatch.setattr(fa, "resolve_vllm_flash_attn_version", resolver)
    impl = FlashAttentionImpl(num_heads=2, head_size=8, softmax_scale=8**-0.5)
    generator = torch.Generator().manual_seed(42)
    # The CPU reference accepts FP32; GPU dtype support remains provider-owned.
    query = torch.randn(2, 4, 2, 8, generator=generator).to(dtype)
    key = torch.randn(2, 7, 2, 8, generator=generator).to(dtype)
    value = torch.randn(2, 7, 2, 8, generator=generator).to(dtype)
    mask = torch.tensor([[True, False, True, True, True, True, True], [False, True, False, True, True, True, True]])
    metadata = AttentionMetadata(attn_mask=mask)
    expected = impl.forward_cuda(query, key, value, metadata)
    reference_call = kernel.call_args
    prepared = transformer._prepare_qwen21_decode_varlen(mask, query.shape[1])
    if case == "query_length":
        prepared = prepared._replace(max_seqlen_q=query.shape[1] + 1)
    elif case == "batch_size":
        prepared = prepared._replace(cu_seqlens_q=prepared.cu_seqlens_q[:2])
    metadata.extra.update(qwen21_decode_varlen=prepared, attention_mask_mode="padding")
    unpad = Mock(wraps=fa._unpad_input)
    monkeypatch.setattr(fa, "_unpad_input", unpad)
    for _ in range(3):
        actual = impl.forward_cuda(query, key, value, metadata)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_tensor, expected_tensor in zip(kernel.call_args.args, reference_call.args):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        for name in ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"):
            torch.testing.assert_close(kernel.call_args.kwargs[name], reference_call.kwargs[name], rtol=0, atol=0)
        if fa_version is None:
            assert "fa_version" not in kernel.call_args.kwargs
            resolver.assert_not_called()
        else:
            assert kernel.call_args.kwargs["fa_version"] == reference_call.kwargs["fa_version"] == fa_version
        if case == "eligible":
            assert kernel.call_args.kwargs["cu_seqlens_q"] is prepared.cu_seqlens_q
            assert kernel.call_args.kwargs["cu_seqlens_k"] is prepared.cu_seqlens_k
    assert unpad.call_count == (0 if case == "eligible" else 3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
@torch.enable_grad()
def test_prepared_decode_metadata_preserves_qkv_gradients(monkeypatch, dtype):
    # Test the real gathers with a differentiable CPU provider, not GPU backward support.
    monkeypatch.setattr(fa, "flash_attn_varlen_func", _reference_varlen)
    monkeypatch.setattr(fa, "_is_vllm_flash_attn_varlen", lambda func: False)
    impl = FlashAttentionImpl(num_heads=2, head_size=8, softmax_scale=8**-0.5)
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(2, 4, 2, 8, generator=generator).to(dtype).requires_grad_()
    key = torch.randn(2, 7, 2, 8, generator=generator).to(dtype).requires_grad_()
    value = torch.randn(2, 7, 2, 8, generator=generator).to(dtype).requires_grad_()
    mask = torch.tensor([[True, False, True, True, True, True, True], [False, True, False, True, True, True, True]])
    grad_output = torch.randn(query.shape, generator=generator).to(dtype)

    expected = impl._forward_varlen_masked(query, key, value, mask)
    expected_grads = torch.autograd.grad(expected, (query, key, value), grad_outputs=grad_output)
    prepared = transformer._prepare_qwen21_decode_varlen(mask, query.shape[1])
    unpad = Mock(wraps=fa._unpad_input)
    monkeypatch.setattr(fa, "_unpad_input", unpad)
    for _ in range(2):
        actual = impl._forward_varlen_masked(query, key, value, mask, prepared)
        actual_grads = torch.autograd.grad(actual, (query, key, value), grad_outputs=grad_output)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
            assert torch.count_nonzero(actual_grad) > 0
        assert torch.count_nonzero(actual_grads[1][~mask]) == 0
        assert torch.count_nonzero(actual_grads[2][~mask]) == 0
    unpad.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("prefix_dtype", ["fp8_e4m3", "fp8_e4m3_v"])
@torch.no_grad()
def test_quantized_prefix_decode_reuses_metadata_without_changing_kv(monkeypatch, dtype, prefix_dtype):
    # Keep Qwen's normalization, RoPE, dequantization, and concatenation; replace
    # projections and the GPU attention provider so the cache path runs on CPU.
    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    kernel = Mock(side_effect=_reference_varlen)
    monkeypatch.setattr(fa, "flash_attn_varlen_func", kernel)
    monkeypatch.setattr(fa, "_is_vllm_flash_attn_varlen", lambda func: False)
    impl = FlashAttentionImpl(num_heads=2, head_size=8, softmax_scale=8**-0.5)
    generator = torch.Generator().manual_seed(42)
    projection = torch.randn(2, 4, 48, generator=generator).to(dtype)
    attention = object.__new__(transformer.QwenImage21Attention)
    nn.Module.__init__(attention)
    attention.num_heads = attention.num_kv_heads = 2
    attention.head_dim = 8
    attention.prefix_kv_cache_dtype = prefix_dtype
    attention.norm_q = transformer.RMSNorm(8, eps=1e-6).to(dtype=dtype)
    attention.norm_k = transformer.RMSNorm(8, eps=1e-6).to(dtype=dtype)
    attention.to_qkv = lambda _: (projection, None)
    attention.to_out = nn.Identity()
    attention.attn = impl.forward_cuda

    prefix_key = torch.randn(2, 3, 2, 8, generator=generator).to(dtype)
    prefix_value = torch.randn(2, 3, 2, 8, generator=generator).to(dtype)
    branch_cache = {"key": prefix_key}
    if prefix_dtype == "fp8_e4m3":
        branch_cache["key"], branch_cache["key_scale"] = transformer._quantize_prefix_kv_fp8(prefix_key)
    branch_cache["value"], branch_cache["value_scale"] = transformer._quantize_prefix_kv_fp8(prefix_value)
    original_cache = {name: tensor.clone() for name, tensor in branch_cache.items()}
    cache = {"cond": branch_cache}
    hidden_states = torch.zeros(2, 4, 16, dtype=dtype)
    freqs = torch.ones(4, 4, dtype=torch.complex64)
    mask = torch.tensor([[True, False, True, True, True, True, True], [False, True, False, True, True, True, True]])
    metadata = AttentionMetadata(attn_mask=mask)
    expected = attention(hidden_states, freqs, attn_metadata=metadata, kv_cache=cache)
    reference_call = kernel.call_args

    prepared = transformer._prepare_qwen21_decode_varlen(mask, hidden_states.shape[1])
    metadata.extra.update(qwen21_decode_varlen=prepared, attention_mask_mode="padding")
    unpad = Mock(wraps=fa._unpad_input)
    monkeypatch.setattr(fa, "_unpad_input", unpad)
    dequantize = Mock(wraps=transformer._dequantize_prefix_kv_fp8)
    monkeypatch.setattr(transformer, "_dequantize_prefix_kv_fp8", dequantize)
    actual = attention(hidden_states, freqs, attn_metadata=metadata, kv_cache=cache)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_tensor, expected_tensor in zip(kernel.call_args.args, reference_call.args):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert kernel.call_args.kwargs["cu_seqlens_k"] is prepared.cu_seqlens_k
    assert kernel.call_args.kwargs["cu_seqlens_q"] is prepared.cu_seqlens_q
    assert dequantize.call_count == (2 if prefix_dtype == "fp8_e4m3" else 1)
    unpad.assert_not_called()
    for name, original in original_cache.items():
        torch.testing.assert_close(branch_cache[name].view(torch.uint8), original.view(torch.uint8), rtol=0, atol=0)


@torch.no_grad()
def test_mismatched_mask_shape_bypasses_prepared_metadata(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    kernel = Mock(side_effect=AssertionError("invalid mask must not reach attention"))
    monkeypatch.setattr(fa, "flash_attn_varlen_func", kernel)
    impl = FlashAttentionImpl(num_heads=2, head_size=8, softmax_scale=8**-0.5)
    query = torch.zeros(2, 4, 2, 8, dtype=torch.bfloat16)
    key = torch.zeros(2, 7, 2, 8, dtype=torch.bfloat16)
    valid_mask = torch.ones(2, 7, dtype=torch.bool)
    prepared = transformer._prepare_qwen21_decode_varlen(valid_mask, query.shape[1])
    invalid_mask = torch.ones(2, 8, dtype=torch.bool)
    unpad = Mock(wraps=fa._unpad_input)
    monkeypatch.setattr(fa, "_unpad_input", unpad)

    # The original path cannot unpad this invalid mask either. Verify the guard
    # preserves that path instead of silently accepting unrelated prepared data.
    for metadata in (None, prepared):
        unpad.reset_mock()
        with pytest.raises(IndexError):
            impl._forward_varlen_masked(query, key, key, invalid_mask, metadata)
        unpad.assert_called_once_with(key, invalid_mask)
    kernel.assert_not_called()

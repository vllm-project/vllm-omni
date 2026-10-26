# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the CUDA FA3 FP8 diffusion attention path.

The FA3 FP8 kernel itself requires SM90; these tests mock the kernel and the
device-capability probe so dispatch, validation, and quantization math are
covered in L1 (CPU) CI.
"""

import pytest
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization import input_quant_fp8
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.utils import fa as fa_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_impl(causal: bool = False) -> FlashAttentionImpl:
    return FlashAttentionImpl(num_heads=2, head_size=64, softmax_scale=64**-0.5, causal=causal)


def _randn_bshd(
    batch: int, seq: int, heads: int = 2, head_dim: int = 64, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    return torch.randn(batch, seq, heads, head_dim, dtype=dtype)


def _reference_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Independent normalized Sylvester matrix reference for the CUDA op."""
    dim = x.shape[-1]
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < dim:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return (x.float() @ (matrix / dim**0.5)).to(x.dtype)


def _use_cuda_e4m3fn_quant_constants(monkeypatch) -> None:
    """Pin QuantFP8.forward_native to NVIDIA E4M3FN ranges.

    The reference reads vLLM's platform FP8 constants. On ROCm those are
    E4M3FNUZ, whose max is 224. This CUDA numerical fixture and its fake
    FA3 kernel follow E4M3FN instead.
    """

    dtype = torch.float8_e4m3fn
    finfo = torch.finfo(dtype)
    monkeypatch.setattr(input_quant_fp8, "_FP8_DTYPE", dtype)
    monkeypatch.setattr(input_quant_fp8, "_FP8_MIN", finfo.min)
    monkeypatch.setattr(input_quant_fp8, "_FP8_MAX", finfo.max)
    monkeypatch.setattr(input_quant_fp8, "_FP8_MIN_SCALING_FACTOR", 1.0 / (finfo.max * 512.0))


@pytest.fixture
def cpu_fp8_quant(monkeypatch, mocker):
    # The production path uses vLLM's CUDA custom op. CPU tests substitute
    # its upstream reference implementation, returning the scale actually
    # used for quantization rather than calculating a separate test scale.
    _use_cuda_e4m3fn_quant_constants(monkeypatch)
    quantizer = QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR, compile_native=False)
    quantize = mocker.Mock(wraps=quantizer.forward_native)
    monkeypatch.setattr(ops, "scaled_fp8_quant", quantize)

    def hadacore(x, inplace=False):
        assert not inplace
        assert x.is_contiguous()
        return _reference_hadamard(x)

    monkeypatch.setattr(ops, "hadacore_transform", hadacore)
    return quantize


def test_quant_constants_follow_cuda_e4m3fn_not_rocm_fnuz(monkeypatch):
    monkeypatch.setattr(input_quant_fp8, "_FP8_DTYPE", torch.float8_e4m3fnuz)
    monkeypatch.setattr(input_quant_fp8, "_FP8_MIN", -224.0)
    monkeypatch.setattr(input_quant_fp8, "_FP8_MAX", 224.0)
    monkeypatch.setattr(input_quant_fp8, "_FP8_MIN_SCALING_FACTOR", 1.0 / (224.0 * 512.0))
    _use_cuda_e4m3fn_quant_constants(monkeypatch)

    quantizer = QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR, compile_native=False)
    values = torch.tensor([[1.0, -8.0, 3.5]])
    quantized, scale = quantizer.forward_native(values)

    assert quantized.dtype == torch.float8_e4m3fn
    expected_scale = values.abs().amax() / torch.finfo(torch.float8_e4m3fn).max
    torch.testing.assert_close(scale.reshape(()), expected_scale)


@pytest.fixture(params=["dense", "vllm_varlen"])
def fa3_kernel(monkeypatch, request):
    fake = _FakeFa3Fp8Kernel()
    monkeypatch.setattr(FlashAttentionImpl, "_fa3_fp8_device_unsupported_reason", staticmethod(lambda device: None))
    monkeypatch.setattr(fa_utils, "FLASH_ATTN_SUPPORTS_FP8", True)
    if request.param == "dense":
        monkeypatch.setattr(fa_utils, "flash_attn_func", fake)
    else:

        def fake_varlen(*, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, fa_version, **kwargs):
            assert fa_version == 3  # The vLLM dispatcher otherwise defaults to FA2.
            batch = kwargs["q_descale"].shape[0]
            for cu_seqlens, length in ((cu_seqlens_q, max_seqlen_q), (cu_seqlens_k, max_seqlen_k)):
                assert cu_seqlens.dtype == torch.int32
                torch.testing.assert_close(cu_seqlens, torch.arange(batch + 1, dtype=torch.int32) * length)
            assert q.shape[0] == batch * max_seqlen_q
            assert k.shape[0] == v.shape[0] == batch * max_seqlen_k
            out = fake(
                q=q.unflatten(0, (batch, max_seqlen_q)),
                k=k.unflatten(0, (batch, max_seqlen_k)),
                v=v.unflatten(0, (batch, max_seqlen_k)),
                **kwargs,
            )
            return out.flatten(0, 1)

        monkeypatch.setattr(fa_utils, "flash_attn_func", None)
        monkeypatch.setattr(fa_utils, "flash_attn_varlen_func", fake_varlen)
        monkeypatch.setattr(fa_utils, "resolve_vllm_flash_attn_version", lambda requested: 3)
    return fake


class TestForwardCudaQuantDispatch:
    def test_fp8_extra_routes_to_quant_path(self, monkeypatch):
        impl = _make_impl()
        calls = []

        def _quant_forward(self, q, k, v, metadata):
            calls.append(metadata)
            return q

        monkeypatch.setattr(
            FlashAttentionImpl,
            "forward_fa_quant_cuda",
            _quant_forward,
        )
        metadata = AttentionMetadata(extra={"kv_cache_dtype": "fp8"})
        impl.forward_cuda(_randn_bshd(1, 8), _randn_bshd(1, 8), _randn_bshd(1, 8), metadata)
        assert calls == [metadata]

    def test_non_fp8_precision_is_rejected_on_cuda(self):
        impl = _make_impl()
        with pytest.raises(ValueError, match="supports only 'fp8'"):
            impl.forward_cuda(
                _randn_bshd(1, 8),
                _randn_bshd(1, 8),
                _randn_bshd(1, 8),
                AttentionMetadata(extra={"kv_cache_dtype": "mxfp8"}),
            )

    @pytest.mark.parametrize("dtype_value", [None, "float", "auto"])
    def test_float_extra_does_not_route_to_quant_path(self, monkeypatch, dtype_value):
        impl = _make_impl()

        def _fail(self, q, k, v, m):
            raise AssertionError("quant path must not be entered")

        monkeypatch.setattr(FlashAttentionImpl, "forward_fa_quant_cuda", _fail)
        monkeypatch.setattr(FlashAttentionImpl, "_forward_varlen_dense", lambda self, q, k, v: q)
        # flash_attn_func may or may not exist on CPU; force the varlen-dense fallback.
        monkeypatch.setattr(fa_utils, "HAS_FLASH_ATTN", True)
        monkeypatch.setattr(fa_utils, "flash_attn_func", None)
        extra = {} if dtype_value is None else {"kv_cache_dtype": dtype_value}
        metadata = AttentionMetadata(extra=extra)
        impl.forward_cuda(_randn_bshd(1, 8), _randn_bshd(1, 8), _randn_bshd(1, 8), metadata)


class TestQuantValidation:
    def test_rejects_attention_mask(self):
        impl = _make_impl()
        metadata = AttentionMetadata(
            attn_mask=torch.ones(1, 8, dtype=torch.bool),
            extra={"kv_cache_dtype": "fp8"},
        )
        with pytest.raises(ValueError, match="caller masks require the float attention path"):
            impl.forward_fa_quant_cuda(_randn_bshd(1, 8), _randn_bshd(1, 8), _randn_bshd(1, 8), metadata)

    def test_rejects_piecewise_spans(self):
        impl = _make_impl()
        metadata = AttentionMetadata(
            full_attn_spans=[(0, 8)],
            extra={"kv_cache_dtype": "fp8"},
        )
        with pytest.raises(ValueError, match="piecewise attention requires the float attention path"):
            impl.forward_fa_quant_cuda(_randn_bshd(1, 8), _randn_bshd(1, 8), _randn_bshd(1, 8), metadata)

    def test_rejects_packed_varlen_metadata(self):
        impl = _make_impl()
        metadata = AttentionMetadata(
            extra={"kv_cache_dtype": "fp8", "cu_seqlens_q": torch.tensor([0, 8], dtype=torch.int32)},
        )
        with pytest.raises(ValueError, match="packed/varlen metadata requires the float attention path"):
            impl.forward_fa_quant_cuda(_randn_bshd(1, 8), _randn_bshd(1, 8), _randn_bshd(1, 8), metadata)

    @pytest.mark.parametrize("head_dim", [0, 8, 24, 44, 272])
    def test_rejects_unsupported_head_dim(self, head_dim):
        impl = _make_impl()
        with pytest.raises(ValueError, match="head_dim divisible by 16"):
            impl.forward_fa_quant_cuda(
                torch.randn(1, 8, 2, head_dim, dtype=torch.bfloat16),
                torch.randn(1, 8, 2, head_dim, dtype=torch.bfloat16),
                torch.randn(1, 8, 2, head_dim, dtype=torch.bfloat16),
            )

    def test_rejects_mismatched_key_value_shapes(self):
        impl = _make_impl()
        with pytest.raises(ValueError, match="K/V sequence lengths and head counts must match"):
            impl.forward_fa_quant_cuda(
                _randn_bshd(1, 8),
                _randn_bshd(1, 9),
                _randn_bshd(1, 8),
            )

    def test_device_gate_rejects_cpu(self):
        reason = FlashAttentionImpl._fa3_fp8_device_unsupported_reason(torch.device("cpu"))
        assert reason is not None and "requires a CUDA device" in reason

    def test_device_gate_rejects_non_sm90(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
        reason = FlashAttentionImpl._fa3_fp8_device_unsupported_reason(torch.device("cuda"))
        assert reason is not None and "require SM90" in reason

    def test_device_gate_accepts_sm90(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
        assert FlashAttentionImpl._fa3_fp8_device_unsupported_reason(torch.device("cuda")) is None

    @pytest.mark.parametrize("kv_seq_len,expected", [(1, 1), (256, 1), (2560, 1), (2561, 2), (19320, 8), (75600, 32)])
    def test_fp8_split_count_scales_with_kv_length(self, kv_seq_len, expected):
        assert FlashAttentionImpl._fa3_fp8_num_splits(kv_seq_len) == expected

    @pytest.mark.parametrize("q_heads,kv_heads", [(3, 2), (2, 4), (4, 0), (0, 2)])
    def test_rejects_invalid_gqa_head_counts(self, fa3_kernel, q_heads, kv_heads):
        with pytest.raises(ValueError, match="Q heads divisible by KV heads"):
            _make_impl().forward_fa_quant_cuda(
                _randn_bshd(1, 8, q_heads), _randn_bshd(1, 8, kv_heads), _randn_bshd(1, 8, kv_heads)
            )

    @pytest.mark.parametrize("supports_fp8,has_dense", [(False, True), (False, False), (True, False)])
    def test_rejects_unsupported_provider_before_quantization(
        self, monkeypatch, cpu_fp8_quant, supports_fp8, has_dense
    ):
        monkeypatch.setattr(FlashAttentionImpl, "_fa3_fp8_device_unsupported_reason", staticmethod(lambda device: None))
        monkeypatch.setattr(fa_utils, "FLASH_ATTN_SUPPORTS_FP8", supports_fp8)
        monkeypatch.setattr(fa_utils, "flash_attn_func", _FakeFa3Fp8Kernel() if has_dense else None)
        monkeypatch.setattr(fa_utils, "flash_attn_varlen_func", None)
        with pytest.raises(ValueError, match="selected diffusion FlashAttention implementation has no FA3 FP8"):
            _make_impl().forward_fa_quant_cuda(_randn_bshd(1, 8), _randn_bshd(1, 8), _randn_bshd(1, 8))
        cpu_fp8_quant.assert_not_called()

    def test_rejects_vllm_without_fa3_before_quantization(self, monkeypatch, cpu_fp8_quant):
        monkeypatch.setattr(FlashAttentionImpl, "_fa3_fp8_device_unsupported_reason", staticmethod(lambda device: None))
        monkeypatch.setattr(fa_utils, "FLASH_ATTN_SUPPORTS_FP8", True)
        monkeypatch.setattr(fa_utils, "flash_attn_func", None)
        monkeypatch.setattr(fa_utils, "flash_attn_varlen_func", lambda **kwargs: pytest.fail("must not run FA2"))

        def unavailable(requested):
            assert requested == 3
            raise RuntimeError("FlashAttention 3 is unavailable on this device")

        monkeypatch.setattr(fa_utils, "resolve_vllm_flash_attn_version", unavailable)
        with pytest.raises(ValueError, match="FlashAttention 3 is unavailable"):
            _make_impl().forward_fa_quant_cuda(_randn_bshd(1, 8), _randn_bshd(1, 8), _randn_bshd(1, 8))
        cpu_fp8_quant.assert_not_called()


class _FakeFa3Fp8Kernel:
    """Dequantize-then-reference-attention stand-in for the FA3 FP8 kernel."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        q,
        k,
        v,
        softmax_scale,
        causal,
        deterministic,
        q_descale,
        k_descale,
        v_descale,
        num_splits,
    ):
        self.calls.append(
            dict(
                qkv_shapes=tuple(tuple(t.shape) for t in (q, k, v)),
                deterministic=deterministic,
                q_descale_shape=tuple(q_descale.shape),
                k_descale_shape=tuple(k_descale.shape),
                v_descale_shape=tuple(v_descale.shape),
                num_splits=num_splits,
                descales=tuple(scale.clone() for scale in (q_descale, k_descale, v_descale)),
            )
        )
        batch, q_len, q_heads, _ = q.shape
        k_len, kv_heads = k.shape[1:3]
        assert q_heads % kv_heads == 0
        for descale in (q_descale, k_descale, v_descale):
            assert descale.shape == (batch, kv_heads)
            assert descale.dtype == torch.float32
            assert torch.equal(descale, descale[0, 0].expand_as(descale))
        for tensor in (q, k, v):
            assert tensor.dtype == torch.float8_e4m3fn
        # FA3 indexes Q descales by KV head, including when GQA is not packed.
        repeat = q_heads // kv_heads
        q_scales = q_descale.repeat_interleave(repeat, dim=1)
        qf = q.float() * q_scales[:, None, :, None]
        kf = k.float() * k_descale[:, None, :, None]
        vf = v.float() * v_descale[:, None, :, None]
        if kf.shape[2] != qf.shape[2]:  # GQA/MQA: replicate KV heads like the real kernel
            kf = kf.repeat_interleave(repeat, dim=2)
            vf = vf.repeat_interleave(repeat, dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * softmax_scale
        if causal:
            q_pos = torch.arange(q_len).unsqueeze(1)
            k_pos = torch.arange(k_len).unsqueeze(0)
            scores = scores.masked_fill(k_pos > q_pos + (k_len - q_len), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", probs, vf)
        return out.to(torch.bfloat16)


def _reference_attention(q, k, v, scale, causal=False):
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    if causal:
        q_pos = torch.arange(q.shape[1]).unsqueeze(1)
        k_pos = torch.arange(k.shape[1]).unsqueeze(0)
        scores = scores.masked_fill(k_pos > q_pos + (k.shape[1] - q.shape[1]), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v.float())


@pytest.mark.usefixtures("cpu_fp8_quant")
class TestForwardFaQuantCudaNumerics:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("deterministic", [False, True])
    def test_matches_float_reference_within_fp8_tolerance(self, fa3_kernel, causal, deterministic, dtype):
        torch.manual_seed(0)
        batch, seq, heads, head_dim = 2, 64, 4, 64
        scale = head_dim**-0.5
        impl = FlashAttentionImpl(num_heads=heads, head_size=head_dim, softmax_scale=scale, causal=causal)
        impl.fa_deterministic = deterministic

        query = _randn_bshd(batch, seq, heads, head_dim, dtype=dtype)
        key = _randn_bshd(batch, seq, heads, head_dim, dtype=dtype)
        value = _randn_bshd(batch, seq, heads, head_dim, dtype=dtype)
        out = impl.forward_fa_quant_cuda(query, key, value, AttentionMetadata(extra={"kv_cache_dtype": "fp8"}))

        assert out.shape == (batch, seq, heads, head_dim)
        assert out.dtype == query.dtype
        assert fa3_kernel.calls[0]["qkv_shapes"] == (tuple(query.shape), tuple(key.shape), tuple(value.shape))
        assert fa3_kernel.calls[0]["deterministic"] == deterministic
        assert fa3_kernel.calls[0]["num_splits"] == 1
        assert fa3_kernel.calls[0]["q_descale_shape"] == (batch, heads)
        assert fa3_kernel.calls[0]["k_descale_shape"] == (batch, heads)
        assert fa3_kernel.calls[0]["v_descale_shape"] == (batch, heads)

        expected = _reference_attention(query, key, value, scale, causal=causal)
        rel_err = (out.float() - expected).abs().mean() / expected.abs().mean()
        assert rel_err < 0.05

    @pytest.mark.parametrize("q_heads,kv_heads", [(4, 2), (8, 2), (4, 1)])
    @pytest.mark.parametrize("causal", [False, True])
    def test_gqa_tensor_scales_match_reference(self, fa3_kernel, cpu_fp8_quant, q_heads, kv_heads, causal):
        torch.manual_seed(1)
        batch, q_len, k_len, head_dim = 2, 32, 48, 64
        impl = FlashAttentionImpl(
            num_heads=q_heads,
            num_kv_heads=kv_heads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=causal,
        )

        def exact_fp8_values(shape):
            magnitudes = 2.0 ** torch.randint(0, 3, shape)
            signs = 2 * torch.randint(0, 2, shape) - 1
            return (magnitudes * signs).to(torch.bfloat16)

        # Noncontiguous BSHD inputs, with different magnitudes across Q heads.
        query = exact_fp8_values((batch, q_heads, q_len, head_dim))
        query *= (2.0 ** (torch.arange(q_heads) % 4 - 2))[None, :, None, None]
        query = query.transpose(1, 2)
        key = exact_fp8_values((batch, k_len, kv_heads, head_dim)) / 8
        value = exact_fp8_values((batch, k_len, kv_heads, head_dim))
        impl_out = impl.forward_fa_quant_cuda(query, key, value, AttentionMetadata(extra={"kv_cache_dtype": "fp8"}))
        assert impl_out.shape == query.shape
        for name in ("q_descale_shape", "k_descale_shape", "v_descale_shape"):
            assert fa3_kernel.calls[0][name] == (batch, kv_heads)
        assert cpu_fp8_quant.call_count == 3
        rotated_query = _reference_hadamard(query)
        rotated_key = _reference_hadamard(key)
        # Q/K rotation stays inside each head; V keeps its original basis.
        for call, tensor in zip(cpu_fp8_quant.call_args_list, (rotated_query, rotated_key, value)):
            assert not call.kwargs  # vLLM computes dynamic per-tensor scales.
            assert call.args[0].is_contiguous()
            torch.testing.assert_close(call.args[0], tensor.reshape(batch * tensor.shape[1], -1))
        for descale, tensor in zip(fa3_kernel.calls[0]["descales"], (rotated_query, rotated_key, value)):
            expected_scale = tensor.float().abs().amax() / torch.finfo(torch.float8_e4m3fn).max
            torch.testing.assert_close(descale, expected_scale.expand(batch, kv_heads))
        repeat = q_heads // kv_heads
        # Isolate layout/descale/GQA correctness from FP8 rounding by using
        # dequantized inputs for the independent attention reference.
        dequantized = []
        quantizer = QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR, compile_native=False)
        for tensor in (rotated_query, rotated_key, value):
            quantized, scale = quantizer.forward_native(tensor.reshape(batch * tensor.shape[1], -1))
            dequantized.append((quantized.float() * scale).reshape(tensor.shape))
        expected = _reference_attention(
            dequantized[0],
            dequantized[1].repeat_interleave(repeat, dim=2),
            dequantized[2].repeat_interleave(repeat, dim=2),
            head_dim**-0.5,
            causal=causal,
        )
        torch.testing.assert_close(impl_out, expected.to(torch.bfloat16))

    @pytest.mark.parametrize("head_dim", [16, 48, 64, 96, 128, 192, 256])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_rotation_preserves_gqa_scores_and_inputs(self, fa3_kernel, cpu_fp8_quant, head_dim, dtype):
        torch.manual_seed(12)
        batch, seq, q_heads, kv_heads = 2, 5, 4, 2
        # Views of a shared projection must not be modified by the rotation.
        projected = torch.randn(batch, seq, q_heads + 2 * kv_heads, head_dim, dtype=dtype)
        original = projected.clone()
        query, key, value = projected.split((q_heads, kv_heads, kv_heads), dim=2)
        impl = FlashAttentionImpl(q_heads, head_dim, head_dim**-0.5, num_kv_heads=kv_heads)
        impl.forward_fa_quant_cuda(query, key, value)
        rotated_q, rotated_k, unrotated_v = [call.args[0] for call in cpu_fp8_quant.call_args_list]
        qf = rotated_q.reshape(query.shape).float()
        kf = rotated_k.reshape(key.shape).float().repeat_interleave(q_heads // kv_heads, dim=2)
        actual_scores = torch.einsum("bqhd,bkhd->bhqk", qf, kf)
        expected_scores = torch.einsum(
            "bqhd,bkhd->bhqk", query.float(), key.float().repeat_interleave(q_heads // kv_heads, dim=2)
        )
        relative_error = (actual_scores - expected_scores).norm() / expected_scores.norm()
        assert relative_error < 0.005
        torch.testing.assert_close(unrotated_v, value.reshape(batch * seq, -1), rtol=0, atol=0)
        torch.testing.assert_close(projected, original, rtol=0, atol=0)

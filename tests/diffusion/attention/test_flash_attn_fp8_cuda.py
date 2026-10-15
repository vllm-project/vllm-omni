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
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_impl(causal: bool = False) -> FlashAttentionImpl:
    return FlashAttentionImpl(num_heads=2, head_size=64, softmax_scale=64**-0.5, causal=causal)


def _randn_bshd(batch: int, seq: int, heads: int = 2, head_dim: int = 64) -> torch.Tensor:
    return torch.randn(batch, seq, heads, head_dim, dtype=torch.bfloat16)


@pytest.fixture
def cpu_fp8_quant(monkeypatch, mocker):
    # The production path uses vLLM's CUDA custom op. CPU tests substitute
    # its upstream reference implementation, returning the scale actually
    # used for quantization rather than calculating a separate test scale.
    quantizer = QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR, compile_native=False)
    quantize = mocker.Mock(wraps=quantizer.forward_native)
    monkeypatch.setattr(ops, "scaled_fp8_quant", quantize)
    return quantize


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
        import vllm_omni.diffusion.attention.backends.utils.fa as fa_utils

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

    def test_rejects_unsupported_head_dim(self):
        impl = FlashAttentionImpl(num_heads=2, head_size=44, softmax_scale=44**-0.5)
        with pytest.raises(ValueError, match="head_dim divisible by 8"):
            impl.forward_fa_quant_cuda(
                torch.randn(1, 8, 2, 44, dtype=torch.bfloat16),
                torch.randn(1, 8, 2, 44, dtype=torch.bfloat16),
                torch.randn(1, 8, 2, 44, dtype=torch.bfloat16),
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

    @pytest.mark.parametrize("q_heads,kv_heads", [(3, 2), (2, 4), (4, 0), (0, 2)])
    def test_rejects_invalid_gqa_head_counts(self, monkeypatch, q_heads, kv_heads):
        monkeypatch.setattr(FlashAttentionImpl, "_fa3_fp8_device_unsupported_reason", staticmethod(lambda device: None))
        with pytest.raises(ValueError, match="Q heads divisible by KV heads"):
            _make_impl().forward_fa_quant_cuda(
                _randn_bshd(1, 8, q_heads), _randn_bshd(1, 8, kv_heads), _randn_bshd(1, 8, kv_heads)
            )


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
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        q_descale,
        k_descale,
        v_descale,
        fa_version,
        **kwargs,
    ):
        self.calls.append(
            dict(
                fa_version=fa_version,
                q_descale_shape=tuple(q_descale.shape),
                k_descale_shape=tuple(k_descale.shape),
                v_descale_shape=tuple(v_descale.shape),
                descales=tuple(scale.clone() for scale in (q_descale, k_descale, v_descale)),
            )
        )
        assert fa_version == 3
        batch = cu_seqlens_q.shape[0] - 1
        kv_heads = k.shape[1]
        assert q.shape[1] % kv_heads == 0
        for descale in (q_descale, k_descale, v_descale):
            assert descale.shape == (batch, kv_heads)
            assert descale.dtype == torch.float32
            assert torch.equal(descale, descale[0, 0].expand_as(descale))
        for tensor in (q, k, v):
            assert tensor.dtype == torch.float8_e4m3fn
        # FA3 indexes Q descales by KV head, including when GQA is not packed.
        repeat = q.shape[1] // kv_heads
        q_scales = q_descale.repeat_interleave(repeat, dim=1)
        qf = q.float().view(batch, max_seqlen_q, *q.shape[1:]) * q_scales[:, None, :, None]
        kf = k.float().view(batch, max_seqlen_k, *k.shape[1:]) * k_descale[:, None, :, None]
        vf = v.float().view(batch, max_seqlen_k, *v.shape[1:]) * v_descale[:, None, :, None]
        if kf.shape[2] != qf.shape[2]:  # GQA/MQA: replicate KV heads like the real kernel
            kf = kf.repeat_interleave(repeat, dim=2)
            vf = vf.repeat_interleave(repeat, dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * softmax_scale
        if causal:
            q_pos = torch.arange(max_seqlen_q).unsqueeze(1)
            k_pos = torch.arange(max_seqlen_k).unsqueeze(0)
            scores = scores.masked_fill(k_pos > q_pos + (max_seqlen_k - max_seqlen_q), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", probs, vf)
        return out.reshape(q.shape[0], *q.shape[1:]).to(torch.bfloat16)


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
    @pytest.mark.parametrize("causal", [False, True])
    def test_matches_float_reference_within_fp8_tolerance(self, monkeypatch, causal):
        torch.manual_seed(0)
        batch, seq, heads, head_dim = 2, 64, 4, 64
        scale = head_dim**-0.5
        impl = FlashAttentionImpl(num_heads=heads, head_size=head_dim, softmax_scale=scale, causal=causal)

        fake = _FakeFa3Fp8Kernel()
        monkeypatch.setattr(
            FlashAttentionImpl,
            "_fa3_fp8_device_unsupported_reason",
            staticmethod(lambda device: None),
        )
        import vllm.v1.attention.backends.fa_utils as fa3_interface

        monkeypatch.setattr(fa3_interface, "flash_attn_varlen_func", fake)

        query = _randn_bshd(batch, seq, heads, head_dim)
        key = _randn_bshd(batch, seq, heads, head_dim)
        value = _randn_bshd(batch, seq, heads, head_dim)
        out = impl.forward_fa_quant_cuda(query, key, value, AttentionMetadata(extra={"kv_cache_dtype": "fp8"}))

        assert out.shape == (batch, seq, heads, head_dim)
        assert out.dtype == torch.bfloat16  # FA3 FP8 kernels return BF16.
        assert fake.calls[0]["fa_version"] == 3
        assert fake.calls[0]["q_descale_shape"] == (batch, heads)
        assert fake.calls[0]["k_descale_shape"] == (batch, heads)
        assert fake.calls[0]["v_descale_shape"] == (batch, heads)

        expected = _reference_attention(query, key, value, scale, causal=causal)
        rel_err = (out.float() - expected).abs().mean() / expected.abs().mean()
        assert rel_err < 0.05

    @pytest.mark.parametrize("q_heads,kv_heads", [(4, 2), (8, 2), (4, 1)])
    @pytest.mark.parametrize("causal", [False, True])
    def test_gqa_tensor_scales_match_reference(self, monkeypatch, cpu_fp8_quant, q_heads, kv_heads, causal):
        torch.manual_seed(1)
        batch, q_len, k_len, head_dim = 2, 32, 48, 64
        impl = FlashAttentionImpl(
            num_heads=q_heads,
            num_kv_heads=kv_heads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=causal,
        )

        fake = _FakeFa3Fp8Kernel()
        monkeypatch.setattr(
            FlashAttentionImpl,
            "_fa3_fp8_device_unsupported_reason",
            staticmethod(lambda device: None),
        )
        import vllm.v1.attention.backends.fa_utils as fa3_interface

        monkeypatch.setattr(fa3_interface, "flash_attn_varlen_func", fake)

        # Signed powers of two map exactly to E4M3 under amax / 448 scaling.
        # This isolates scale broadcasting from lossy rounding/softmax amplification.
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
            assert fake.calls[0][name] == (batch, kv_heads)
        assert cpu_fp8_quant.call_count == 3
        # Flattening preserves each tensor's data without head permutations.
        for call, tensor in zip(cpu_fp8_quant.call_args_list, (query, key, value)):
            assert not call.kwargs  # vLLM computes dynamic per-tensor scales.
            assert call.args[0].is_contiguous()
            torch.testing.assert_close(call.args[0], tensor.reshape(batch * tensor.shape[1], -1))
        for descale, tensor in zip(fake.calls[0]["descales"], (query, key, value)):
            expected_scale = tensor.float().abs().amax() / torch.finfo(torch.float8_e4m3fn).max
            torch.testing.assert_close(descale, expected_scale.expand(batch, kv_heads))
        repeat = q_heads // kv_heads
        expected = _reference_attention(
            query,
            key.repeat_interleave(repeat, dim=2),
            value.repeat_interleave(repeat, dim=2),
            head_dim**-0.5,
            causal=causal,
        )
        torch.testing.assert_close(impl_out, expected.to(torch.bfloat16))

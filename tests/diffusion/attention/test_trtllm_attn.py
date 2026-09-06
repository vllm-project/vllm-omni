# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends import trtllm_attn as tg
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
)
from vllm_omni.diffusion.attention.backends.trtllm_attn import TrtllmAttentionImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]


def _has_trtllm_attn() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 10:
        return False
    try:
        import flashinfer  # noqa: F401
    except Exception:
        return False
    return tg.HAS_FLASHINFER


requires_trtllm_attn = pytest.mark.skipif(
    not _has_trtllm_attn(), reason="requires Blackwell SM100+ GPU with flashinfer"
)


def _impl(**bk):
    return TrtllmAttentionImpl(8, 128, 1.0 / math.sqrt(128), causal=False, num_kv_heads=8, backend_kwargs=bk)


def _packed_padding(cu_seqlens_q, cu_seqlens_k, q_length, kv_length):
    return PackedPaddingMetadata(
        q_length=q_length,
        kv_length=kv_length,
        cu_seqlens_q=cu_seqlens_q[:2],
        cu_seqlens_k=cu_seqlens_k[:2],
    )


def test_skip_config_pure_resolution():
    from vllm_omni.diffusion.attention.backends.trtllm_attn import SkipSoftmaxConfig

    assert not SkipSoftmaxConfig().enabled
    assert SkipSoftmaxConfig().resolve_factor(4096, timestep=0.3) is None

    cfg = SkipSoftmaxConfig.from_backend_kwargs({"skip_softmax_threshold": 0.01, "disabled_until_timestep": 0.6})
    assert cfg.enabled and cfg.gated
    assert cfg.resolve_factor(4096, timestep=0.9) is None
    assert cfg.resolve_factor(4096, timestep=0.3) == pytest.approx(0.01 * 4096)
    assert cfg.resolve_factor(4096, timestep=None) == pytest.approx(0.01 * 4096)


def test_quant_config_from_backend_kwargs():
    from vllm_omni.diffusion.attention.backends.trtllm_attn import QuantConfig

    assert not QuantConfig.from_backend_kwargs(None).enabled
    assert not QuantConfig.from_backend_kwargs({}).enabled
    assert QuantConfig.from_backend_kwargs({"quant": {"dtype_qk": "fp8_e4m3"}}) == QuantConfig("fp8_e4m3", 1, 16)
    qc = QuantConfig.from_backend_kwargs({"quant": {"dtype_qk": "int8", "q_block_size": 4, "k_block_size": 16}})
    assert qc.enabled and (qc.dtype_qk, qc.q_block_size, qc.k_block_size) == ("int8", 4, 16)


def test_quant_rejects_non_sage_dtype():
    with pytest.raises(RuntimeError, match="supports dtype_qk"):
        _impl(quant={"dtype_qk": "bfloat16"})


def test_quant_quantize_requires_flashinfer_routine(monkeypatch):
    import vllm_omni.diffusion.attention.backends.trtllm_attn as mod

    monkeypatch.setattr(mod, "_sage_kernel_available", lambda: True)
    monkeypatch.setattr(mod, "_sage_quantize_fn", lambda: None)
    with pytest.raises(RuntimeError, match="trtllm_sage_attention_quantize"):
        _impl(quant={"dtype_qk": "int8"})


def test_quant_quantize_calls_routine_and_shapes_sfs():
    from vllm_omni.diffusion.attention.backends.trtllm_attn import QuantConfig

    captured: dict[str, object] = {}

    def fake_quantize(q, k, v, q_block_size, k_block_size, qk_quant_dtype):
        captured.update(q_block_size=q_block_size, k_block_size=k_block_size, qk_quant_dtype=qk_quant_dtype)
        return "qq", "kq", "vq", "qsfs", "ksfs", "vsfs"

    q_q, k_q, v_q, sfs, blk = QuantConfig(dtype_qk="int8").quantize(object(), object(), object(), fake_quantize)
    assert (q_q, k_q, v_q, sfs, blk) == ("qq", "kq", "vq", ("qsfs", "ksfs", None, "vsfs"), (1, 16, 0, 1))
    assert captured == {"q_block_size": 1, "k_block_size": 16, "qk_quant_dtype": torch.int8}


def test_skip_factor_none_without_curve():
    assert _impl(target_sparsity=0.5)._resolve_skip_factor(4096) is None


def _calibrated(a, b, **bk):
    impl = _impl(**bk)
    impl.set_layer_calibration(a, b)
    return impl


def test_skip_factor_from_curve():
    assert _calibrated(100.0, 0.0, target_sparsity=0.5)._resolve_skip_factor(4096) == pytest.approx(100.0)
    f2 = _calibrated(2.0, 1.0, target_sparsity=0.5)._resolve_skip_factor(4096)
    assert f2 == pytest.approx(2.0 * math.exp(0.5))


def test_skip_factor_none_until_stamped():
    assert _impl(target_sparsity=0.5)._resolve_skip_factor(4096) is None


def test_skip_factor_from_threshold_no_calibration():
    impl = _impl(skip_softmax_threshold=0.01)
    assert impl._resolve_skip_factor(4096) == pytest.approx(0.01 * 4096)
    assert impl._resolve_skip_factor(8192) == pytest.approx(0.01 * 8192)

    impl2 = _calibrated(100.0, 0.0, skip_softmax_threshold=0.02, target_sparsity=0.5)
    assert impl2._resolve_skip_factor(4096) == pytest.approx(0.02 * 4096)


def test_skip_timestep_gate(monkeypatch):
    from vllm_omni.diffusion import forward_context as fc

    impl = _calibrated(100.0, 0.0, target_sparsity=0.5, disabled_until_timestep=0.6)
    ctx = fc.ForwardContext(denoise_timestep=0.9)
    monkeypatch.setattr(fc, "_forward_context", ctx)
    assert impl._resolve_skip_factor(4096) is None
    ctx.denoise_timestep = 0.3
    assert impl._resolve_skip_factor(4096) == pytest.approx(100.0)


def test_denoise_progress_mixin(monkeypatch):
    import types

    from vllm_omni.diffusion import forward_context as fc

    class _Pipe(fc.DenoiseProgressMixin):
        scheduler = types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=1000))

    ctx = fc.ForwardContext()
    monkeypatch.setattr(fc, "_forward_context", ctx)
    _Pipe().record_denoise_step(3, 250)
    assert ctx.denoise_step_idx == 3
    assert ctx.denoise_timestep == pytest.approx(0.25)

    _Pipe().record_denoise_step(4)
    assert ctx.denoise_step_idx == 4 and ctx.denoise_timestep == pytest.approx(0.25)

    _Pipe().record_denoise_step(5, normalized_timestep=0.7)
    assert ctx.denoise_step_idx == 5 and ctx.denoise_timestep == pytest.approx(0.7)


@pytest.mark.parametrize(
    "bk, field",
    [
        ({"target_sparsity": 1.5}, "target_sparsity"),
        ({"target_sparsity": -0.1}, "target_sparsity"),
        ({"disabled_until_timestep": 1.2}, "disabled_until_timestep"),
        ({"skip_softmax_threshold": -1.0}, "skip_softmax_threshold"),
        ({"target_sparsity": float("nan")}, "target_sparsity"),
        ({"skip_softmax_threshold": float("inf")}, "skip_softmax_threshold"),
    ],
)
def test_skip_control_validation_rejects_out_of_range(bk, field):
    from vllm_omni.diffusion.attention.backends.trtllm_attn import SkipSoftmaxConfig

    with pytest.raises(ValueError, match=field):
        SkipSoftmaxConfig.from_backend_kwargs(bk)


def test_skip_control_validation_accepts_valid():
    from vllm_omni.diffusion.attention.backends.trtllm_attn import SkipSoftmaxConfig

    cfg = SkipSoftmaxConfig.from_backend_kwargs(
        {"target_sparsity": 0.65, "disabled_until_timestep": 0.86, "skip_softmax_threshold": 0.01}
    )
    assert cfg.target_sparsity == pytest.approx(0.65)
    assert cfg.disabled_until_timestep == pytest.approx(0.86)
    assert cfg.threshold == pytest.approx(0.01)


def test_ring_sp_with_skip_softmax_rejected():
    import types

    from vllm_omni.diffusion.attention.layer import Attention

    fake = types.SimpleNamespace(attention=_impl(target_sparsity=0.5), ring_runner=None)
    with pytest.raises(NotImplementedError, match="ring sequence parallelism"):
        Attention._run_ring_attention(fake, None, None, None, None)


def test_mask_free_allowlist_gates_trtllm_default():
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata as meta

    assert meta("WanPipeline").attention_mask_free is True
    assert meta("WanVACEPipeline").attention_mask_free is True
    assert meta("MiniMaxH3Pipeline").attention_mask_free is True
    assert meta("FluxPipeline").attention_mask_free is False
    assert meta(None).attention_mask_free is False


def test_masked_layer_raises():
    b, s, h, d = 1, 8, 8, 128
    q, k, v = (torch.zeros(b, s, h, d) for _ in range(3))
    mask = torch.ones(b, s, dtype=torch.bool)
    mask[:, s // 2 :] = False

    class _Meta:
        attn_mask = mask

    with pytest.raises(ValueError, match="does not support attn_mask"):
        _impl().forward_cuda(q, k, v, _Meta())


def test_generic_packed_metadata_preserves_ragged_batch(monkeypatch):
    b, h, d = 1, 8, 128
    q = torch.zeros(b, 5, h, d)
    k, v = (torch.zeros(b, 7, h, d) for _ in range(2))
    metadata = AttentionMetadata(
        extra={
            "cu_seqlens_q": torch.tensor([0, 2, 5], dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, 3, 7], dtype=torch.int32),
            "max_seqlen_q": 3,
            "max_seqlen_k": 4,
        }
    )
    fake_attention = Mock(return_value=torch.zeros_like(q.reshape(5, h, d)))

    monkeypatch.setattr(tg, "HAS_FLASHINFER", True)
    monkeypatch.setattr(tg, "trtllm_ragged_attention_deepseek", fake_attention)
    monkeypatch.setattr(TrtllmAttentionImpl, "_get_workspace", classmethod(lambda cls, device: torch.empty(0)))

    _impl().forward_cuda(q, k, v, metadata)
    captured = fake_attention.call_args.kwargs

    assert captured["query"].shape[0] == 5
    assert captured["key"].shape[0] == 7
    assert captured["value"].shape[0] == 7
    assert captured["batch_size"] == 2
    assert captured["seq_lens"].tolist() == [3, 4]
    assert tg.TrtllmAttentionBackend.supports_multi_doc_packed_varlen()


def test_packed_padding_metadata_trims_without_mask(monkeypatch):
    q_len, kv_len = 8, 9
    valid_q_tokens, valid_kv_tokens = 5, 6
    b, h, d = 1, 8, 128
    q = torch.zeros(b, q_len, h, d)
    k, v = (torch.zeros(b, kv_len, h, d) for _ in range(2))
    cu_seqlens_q = torch.tensor([0, valid_q_tokens, q_len], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, valid_kv_tokens, kv_len], dtype=torch.int32)
    metadata = AttentionMetadata(
        packed_padding=_packed_padding(
            cu_seqlens_q,
            cu_seqlens_k,
            valid_q_tokens,
            valid_kv_tokens,
        ),
        extra={
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": valid_q_tokens,
            "max_seqlen_k": valid_kv_tokens,
            "valid_kv_length": valid_kv_tokens,
        },
    )
    fake_attention = Mock(return_value=torch.ones(valid_q_tokens, h, d))

    monkeypatch.setattr(tg, "HAS_FLASHINFER", True)
    monkeypatch.setattr(tg, "trtllm_ragged_attention_deepseek", fake_attention)
    monkeypatch.setattr(TrtllmAttentionImpl, "_get_workspace", classmethod(lambda cls, device: torch.empty(0)))

    out = _impl().forward_cuda(q, k, v, metadata)
    captured = fake_attention.call_args.kwargs

    assert tg.TrtllmAttentionBackend.supports_packed_mask_free()
    assert out.shape == q.shape
    assert captured["query"].shape[0] == valid_q_tokens
    assert captured["key"].shape[0] == valid_kv_tokens
    assert captured["value"].shape[0] == valid_kv_tokens
    assert captured["batch_size"] == 1
    assert captured["seq_lens"].tolist() == [valid_kv_tokens]
    assert captured["cum_seq_lens_q"].tolist() == [0, valid_q_tokens]
    assert captured["cum_seq_lens_kv"].tolist() == [0, valid_kv_tokens]
    assert captured["max_q_len"] == valid_q_tokens
    assert captured["max_kv_len"] == valid_kv_tokens
    assert torch.count_nonzero(out[:, :valid_q_tokens] != 1) == 0
    assert torch.count_nonzero(out[:, valid_q_tokens:]) == 0


def test_packed_metadata_must_be_complete():
    b, s, h, d = 1, 8, 8, 128
    q, k, v = (torch.zeros(b, s, h, d) for _ in range(3))
    metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, s], dtype=torch.int32)})

    with pytest.raises(ValueError, match="Incomplete packed TRTLLM attention metadata"):
        _impl().forward_cuda(q, k, v, metadata)


def test_packed_metadata_must_cover_inputs(monkeypatch):
    b, s, h, d = 1, 8, 8, 128
    q, k, v = (torch.zeros(b, s, h, d) for _ in range(3))
    cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)
    metadata = AttentionMetadata(
        extra={
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": 6,
            "max_seqlen_k": 6,
        }
    )
    monkeypatch.setattr(tg, "HAS_FLASHINFER", True)

    with pytest.raises(ValueError, match="must cover all Q/K/V tokens"):
        _impl().forward_cuda(q, k, v, metadata)


def test_propagate_calibration_045_schema(monkeypatch):
    import types

    from vllm_omni.diffusion.data import (
        AttentionConfig,
        AttentionSpec,
        OmniDiffusionConfig,
        TransformerConfig,
    )

    cfg = AttentionConfig(default=AttentionSpec(backend="TRTLLM_ATTN"))
    tf = TransformerConfig.from_dict(
        {
            "sparse_attention_config": {
                "config_groups": {
                    "group_0": {
                        "algorithm": "skip_softmax",
                        "targets": ["WanAttention"],
                        "ignore": ["blocks.0.attn2", "blocks.1.attn2"],
                        "threshold_scale_factor": {
                            "formula": "a * exp(b * target_sparsity)",
                            "coefficients": {"a": 2142.7, "b": 4.28},
                        },
                        "target_sparsity": 0.5,
                    }
                }
            }
        }
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.backends.trtllm_calibration.parse_secondary_expert_calibration",
        lambda *a, **k: None,
    )
    OmniDiffusionConfig._propagate_skip_softmax_calibration(
        types.SimpleNamespace(
            diffusion_attention_config=cfg,
            model="/x",
        ),
        tf,
    )
    spec, _ = cfg.resolve_with_source("self")
    cal = spec.skip_calibration["by_expert"]["transformer"]
    assert cal["a"] == pytest.approx(2142.7)
    assert cal["b"] == pytest.approx(4.28)
    assert cal["ignore"] == ["blocks.0.attn2", "blocks.1.attn2"]
    assert "transformer_2" not in spec.skip_calibration["by_expert"]


def test_target_sparsity_without_calibration_raises():
    import types

    from vllm_omni.diffusion.data import (
        AttentionConfig,
        AttentionSpec,
        OmniDiffusionConfig,
        TransformerConfig,
    )

    def _cfg(skip_softmax):
        return types.SimpleNamespace(
            diffusion_attention_config=AttentionConfig(
                default=AttentionSpec(backend="TRTLLM_ATTN", skip_softmax=skip_softmax)
            ),
            model="/models/no-such-calibration",
        )

    with pytest.raises(ValueError, match="no skip-softmax calibration"):
        OmniDiffusionConfig._propagate_skip_softmax_calibration(
            _cfg({"target_sparsity": 0.5}), TransformerConfig.from_dict({})
        )

    OmniDiffusionConfig._propagate_skip_softmax_calibration(_cfg({"threshold": 0.02}), TransformerConfig.from_dict({}))


def _sdpa_ref(q, k, v, scale):
    qp, kp, vp = (x.permute(0, 2, 1, 3) for x in (q, k, v))
    return F.scaled_dot_product_attention(qp, kp, vp, is_causal=False, scale=scale).permute(0, 2, 1, 3).float()


@requires_trtllm_attn
def test_bf16_dense_matches_sdpa():
    torch.manual_seed(0)
    B, S, H, D = 2, 256, 8, 128
    scale = 1.0 / math.sqrt(D)
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = _impl().forward_cuda(q, k, v, None).float()
    ref = _sdpa_ref(q, k, v, scale)
    rel = (out - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.01, f"BF16 dense rel err {rel:.4f} too high"


@requires_trtllm_attn
def test_bf16_packed_padding_matches_sdpa():
    torch.manual_seed(0)
    b, s, used, h, d = 1, 256, 192, 8, 128
    scale = 1.0 / math.sqrt(d)
    q, k, v = (torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    cu_seqlens = torch.tensor([0, used, s], dtype=torch.int32, device="cuda")
    metadata = AttentionMetadata(
        packed_padding=_packed_padding(cu_seqlens, cu_seqlens, used, used),
        extra={
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": used,
            "max_seqlen_k": used,
            "valid_kv_length": used,
        },
    )

    out = _impl().forward_cuda(q, k, v, metadata).float()
    ref_used = _sdpa_ref(q[:, :used], k[:, :used], v[:, :used], scale)
    used_rel = (out[:, :used] - ref_used).abs().mean() / ref_used.abs().mean()
    assert used_rel < 0.01, f"BF16 packed valid-token rel err {used_rel:.4f} too high"
    assert torch.count_nonzero(out[:, used:]) == 0


requires_sage = pytest.mark.skipif(
    not (_has_trtllm_attn() and tg._sage_quantize_fn() is not None),
    reason="requires Blackwell SM100+ GPU with flashinfer >= 0.6.16rc1 (trtllm_sage_attention_quantize)",
)


@requires_sage
@pytest.mark.parametrize("dtype_qk", ["fp8_e4m3", "int8"])
def test_sage_quant_matches_sdpa(dtype_qk):
    torch.manual_seed(0)
    B, S, H, D = 2, 256, 8, 128
    scale = 1.0 / math.sqrt(D)
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    impl = _impl(quant={"dtype_qk": dtype_qk, "q_block_size": 1, "k_block_size": 16})
    try:
        out = impl.forward_cuda(q, k, v, None)
    except RuntimeError as e:  # pragma: no cover - arch-dependent (int8 is SM100-only)
        if "Missing TRTLLM-GEN kernel" in str(e):
            pytest.skip(f"no trtllm-gen SAGE kernel for dtype_qk={dtype_qk} on this GPU arch")
        raise
    assert out.shape == (B, S, H, D) and torch.isfinite(out).all()
    ref = _sdpa_ref(q, k, v, scale)
    rel = (out.float() - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.1, f"SAGE({dtype_qk}) rel err {rel:.4f} too high"


@requires_sage
def test_sage_packed_non_aligned_length_matches_sdpa():
    torch.manual_seed(0)
    b, s, used, h, d = 1, 208, 194, 8, 128
    scale = 1.0 / math.sqrt(d)
    q, k, v = (torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    cu_seqlens = torch.tensor([0, used, s], dtype=torch.int32, device="cuda")
    metadata = AttentionMetadata(
        packed_padding=_packed_padding(cu_seqlens, cu_seqlens, used, used),
        extra={
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": used,
            "max_seqlen_k": used,
            "valid_kv_length": used,
        },
    )

    out = _impl(quant={"dtype_qk": "fp8_e4m3", "q_block_size": 1, "k_block_size": 16}).forward_cuda(q, k, v, metadata)
    assert torch.isfinite(out).all()
    ref = _sdpa_ref(q[:, :used], k[:, :used], v[:, :used], scale)
    rel = (out[:, :used].float() - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.2, f"SAGE packed valid-token rel err {rel:.4f} too high"
    assert torch.count_nonzero(out[:, used:]) == 0


@requires_sage
def test_sage_short_sequence_uses_dense_kernel():
    torch.manual_seed(0)
    b, s, h, d = 1, 14, 8, 128
    scale = 1.0 / math.sqrt(d)
    q, k, v = (torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    impl = _impl(quant={"dtype_qk": "fp8_e4m3", "q_block_size": 1, "k_block_size": 16})
    impl._sage_quantize_fn = lambda *args, **kwargs: pytest.fail("short sequence must not use SAGE")
    out = impl.forward_cuda(q, k, v)
    ref = _sdpa_ref(q, k, v, scale)
    rel = (out.float() - ref).abs().mean() / ref.abs().mean()
    assert torch.isfinite(out).all()
    assert rel < 0.01, f"SAGE short-sequence dense fallback rel err {rel:.4f} too high"


@requires_trtllm_attn
def test_skip_tiny_threshold_matches_dense():
    torch.manual_seed(0)
    B, S, H, D = 2, 512, 8, 128
    scale = 1.0 / math.sqrt(D)
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    out = _impl(skip_softmax_threshold=1e-9).forward_cuda(q, k, v, None).float()
    ref = _sdpa_ref(q, k, v, scale)
    rel = (out - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.02, f"skip(tiny) rel err {rel:.4f} should be ~dense"


@requires_trtllm_attn
def test_skip_threshold_engages():
    torch.manual_seed(0)
    B, S, H, D = 2, 512, 8, 128
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    dense = _impl().forward_cuda(q, k, v, None).float()
    skipped = _impl(skip_softmax_threshold=2.0).forward_cuda(q, k, v, None).float()
    assert torch.isfinite(skipped).all()
    assert (skipped - dense).abs().mean() / dense.abs().mean() > 0.02


@requires_trtllm_attn
def test_gqa_matches_sdpa():
    torch.manual_seed(0)
    B, S, Hq, Hkv, D = 1, 256, 64, 8, 128
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
    out = (
        TrtllmAttentionImpl(Hq, D, scale, causal=False, num_kv_heads=Hkv, backend_kwargs={})
        .forward_cuda(q, k, v, None)
        .float()
    )
    qp, kp, vp = (x.permute(0, 2, 1, 3) for x in (q, k, v))
    ref = (
        F.scaled_dot_product_attention(qp, kp, vp, is_causal=False, scale=scale, enable_gqa=True)
        .permute(0, 2, 1, 3)
        .float()
    )
    rel = (out - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.01, f"GQA rel err {rel:.4f} vs SDPA(GQA) too high"


@requires_trtllm_attn
def test_unsupported_head_dim_raises():
    q = k = v = torch.randn(1, 32, 8, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(Exception, match="[Uu]nsupported head dim"):
        _impl().forward_cuda(q, k, v, None)


def test_skip_end_to_end_config_path(monkeypatch):
    import types

    from vllm_omni.diffusion import forward_context as fc
    from vllm_omni.diffusion.data import (
        AttentionConfig,
        AttentionSpec,
        OmniDiffusionConfig,
        TransformerConfig,
    )

    cfg = AttentionConfig(
        default=AttentionSpec(
            backend="TRTLLM_ATTN",
            skip_softmax={"target_sparsity": 0.65, "disabled_until_timestep": 0.86},
        )
    )

    tf = TransformerConfig.from_dict(
        {"sparse_attention_config": {"threshold_scale_factor": {"prefill": {"a": 7.93, "b": 8.61}}}}
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.backends.trtllm_calibration.parse_secondary_expert_calibration",
        lambda *a, **k: None,
    )
    OmniDiffusionConfig._propagate_skip_softmax_calibration(
        types.SimpleNamespace(
            diffusion_attention_config=cfg,
            model="/x",
        ),
        tf,
    )
    spec, _ = cfg.resolve_with_source("self")
    cal = spec.skip_calibration["by_expert"]["transformer"]
    assert cal["a"] == 7.93 and cal["b"] == 8.61

    impl = TrtllmAttentionImpl(
        8, 128, 1.0 / math.sqrt(128), causal=False, num_kv_heads=8, backend_kwargs=spec.backend_kwargs()
    )
    impl.set_layer_calibration(cal["a"], cal["b"])
    expected = 7.93 * math.exp(8.61 * 0.65)

    ctx = fc.ForwardContext(denoise_timestep=0.3)
    monkeypatch.setattr(fc, "_forward_context", ctx)
    assert impl._resolve_skip_factor(4096) == pytest.approx(expected)

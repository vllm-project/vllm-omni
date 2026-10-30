# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for MindIE adapters; no NPU kernels are mocked in production."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention import layer as layer_mod
from vllm_omni.diffusion.attention.backends import flash_attn, rainfusion_attn
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
    VideoTokenLayout,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
_DEFAULT_SKIP_STEPS = frozenset({0, 1, 38, 39})


@pytest.fixture
def runtime(monkeypatch):
    mod = ModuleType("mindiesd")

    def execute(q, k, v, *, precision, layout, scale, q_rot=None, k_rot=None):
        return q

    setattr(mod, "quant_attention", Mock(side_effect=execute))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    for module in (flash_attn, layer_mod):
        monkeypatch.setattr(module, "current_omni_platform", SimpleNamespace(is_npu=lambda: True, device_name="npu"))
    monkeypatch.setattr(flash_attn, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(rainfusion_attn, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(rainfusion_attn, "is_forward_context_available", lambda: False)
    supports_precision = rainfusion_attn._mindiesd_supports_precision
    supports_precision.cache_clear()
    yield mod
    supports_precision.cache_clear()


def flash(layout="BSND"):
    return flash_attn.FlashAttentionImpl(
        num_heads=2,
        head_size=64,
        softmax_scale=0.37,
        qkv_layout=layout,
    )


def sparse(**kwargs):
    return rainfusion_attn.RainFusionAttentionImpl(
        num_heads=2,
        head_size=64,
        softmax_scale=0.37,
        qkv_layout="BSND",
        backend_kwargs={"sparsity": 0.8, **kwargs},
    )


def video_metadata(**extra):
    return AttentionMetadata(
        video_layout=VideoTokenLayout(prefix_len=0, latent_grid=(4, 32, 32), used_len=4096),
        extra={"max_seqlen_q": 4096, **extra},
    )


@pytest.mark.parametrize("method", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("layout", [None, "BSND", "BNSD"])
def test_exact_runtime_layout_scale_and_rotation(runtime, method, layout):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    if layout == "BNSD":
        q = q.transpose(1, 2)
    impl = flash(layout=layout)
    metadata = AttentionMetadata(extra={"kv_cache_dtype": method})
    assert impl.forward_npu(q, q, q, metadata) is q
    args, kwargs = runtime.quant_attention.call_args
    assert args[0] is q  # no unconditional transpose or extra quantization
    assert kwargs["layout"] == (layout or "BSND") and kwargs["scale"] == 0.37 and kwargs["precision"] == method
    assert "attn_mask" not in kwargs
    from vllm_omni.platforms.npu.quant.kv_quant_npu import get_quant_attention_rotation

    assert kwargs["q_rot"] is kwargs["k_rot"]
    torch.testing.assert_close(kwargs["q_rot"], get_quant_attention_rotation(q.device, q.dtype, 64))


def test_missing_runtime_requires_config_change(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    del runtime.quant_attention
    with pytest.raises(ImportError, match="diffusion_kv_cache_dtype"):
        flash().forward_npu(q, q, q, AttentionMetadata(extra={"kv_cache_dtype": "mxfp4"}))


def test_operator_failure_never_falls_back(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    runtime.quant_attention.side_effect = RuntimeError("operator failure")
    with pytest.raises(RuntimeError, match="operator failure"):
        flash().forward_npu(q, q, q, AttentionMetadata(extra={"kv_cache_dtype": "mxfp4"}))
    runtime.quant_attention.assert_called_once()


@pytest.mark.parametrize("method", ["fp8", "mxfp8", "mxfp4"])
def test_unsupported_shape_requires_config_change(runtime, method):
    q = torch.randn(1, 128, 2, 96, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="diffusion_kv_cache_dtype"):
        flash().forward_npu(q, q, q, AttentionMetadata(extra={"kv_cache_dtype": method}))
    runtime.quant_attention.assert_not_called()


def test_packed_metadata_never_reaches_quant_runtime(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    metadata = AttentionMetadata(extra={"kv_cache_dtype": "mxfp8", "cu_seqlens_q": torch.tensor([0, 64, 128])})
    with pytest.raises(ValueError, match="packed/varlen"):
        flash().forward_npu(q, q, q, metadata)
    runtime.quant_attention.assert_not_called()


def packed_padding(q_length, kv_length, *, total_q=128, total_kv=128):
    return PackedPaddingMetadata(
        q_length=q_length,
        kv_length=kv_length,
        cu_seqlens_q=torch.tensor([0, total_q], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, total_kv], dtype=torch.int32),
    )


def packed_metadata(method, q_length, kv_length, **extra):
    return AttentionMetadata(
        packed_padding=packed_padding(q_length, kv_length),
        extra={
            "kv_cache_dtype": method,
            "cu_seqlens_q": torch.tensor([0, 96, 128], dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, 96, 128], dtype=torch.int32),
            "max_seqlen_q": 96,
            "max_seqlen_k": 96,
            "npu_attn_varlen": True,
            **extra,
        },
    )


@pytest.mark.parametrize("method", ["fp8", "mxfp8"])
def test_packed_quant_trims_kv_to_valid_prefix(runtime, method):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    assert flash().forward_npu(q, k, v, packed_metadata(method, 128, 104)) is q
    args, kwargs = runtime.quant_attention.call_args
    # Padding rows of Q are never consumed, so Q keeps the packed length; the
    # padding document must not reach the softmax, so K/V lose it.
    assert args[0] is q
    assert args[1].shape == (1, 104, 2, 64) and args[2].shape == (1, 104, 2, 64)
    assert args[1].is_contiguous() and args[2].is_contiguous()
    torch.testing.assert_close(args[1], k[:, :104])
    torch.testing.assert_close(args[2], v[:, :104])
    assert kwargs["layout"] == "BSND" and kwargs["precision"] == method
    assert kwargs["q_rot"] is kwargs["k_rot"]


def test_packed_quant_aligned_sequence_is_not_trimmed(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    assert flash().forward_npu(q, q, q, packed_metadata("fp8", 128, 128)) is q
    args, _ = runtime.quant_attention.call_args
    assert args[0] is q and args[1] is q and args[2] is q


def test_packed_quant_bnsd_trims_dim2_and_materializes(runtime):
    q = torch.randn(1, 2, 128, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 128, 64, dtype=torch.bfloat16)
    flash(layout="BNSD").forward_npu(q, k, k, packed_metadata("fp8", 128, 96))
    args, kwargs = runtime.quant_attention.call_args
    assert args[0] is q
    assert args[1].shape == (1, 2, 96, 64) and args[2].shape == (1, 2, 96, 64)
    # A BNSD crop leaves row gaps; the quant preprocess must see contiguous input.
    assert args[1].is_contiguous() and args[2].is_contiguous()
    torch.testing.assert_close(args[1], k[:, :, :96])
    assert kwargs["layout"] == "BNSD"


def test_packed_quant_keeps_reference_kv_prefix(runtime):
    # Compact reference-KV reuse prepends cached rows to K/V: q_length excludes
    # them while kv_length already covers them, so neither side is trimmed.
    q = torch.randn(1, 80, 2, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 96, 2, 64, dtype=torch.bfloat16)
    metadata = AttentionMetadata(
        packed_padding=packed_padding(80, 96, total_q=80, total_kv=96),
        extra={"kv_cache_dtype": "mxfp8", "npu_attn_varlen": True},
    )
    flash().forward_npu(q, k, k, metadata)
    args, _ = runtime.quant_attention.call_args
    assert args[0] is q and args[1] is k


def test_packed_quant_rejects_mxfp4(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="mxfp4"):
        flash().forward_npu(q, q, q, packed_metadata("mxfp4", 128, 96))
    runtime.quant_attention.assert_not_called()


@pytest.mark.parametrize(
    "q_length,kv_length,error",
    [(128, 200, "kv_length"), (200, 128, "q_length"), (0, 96, "q_length")],
)
def test_packed_quant_rejects_lengths_outside_the_packed_sequence(runtime, q_length, kv_length, error):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    metadata = packed_metadata("fp8", 128, 96)
    metadata.packed_padding = packed_padding(q_length, kv_length)
    with pytest.raises(ValueError, match=error):
        flash().forward_npu(q, q, q, metadata)
    runtime.quant_attention.assert_not_called()


@pytest.mark.parametrize("contract", ["mask", "spans"])
def test_packed_quant_still_rejects_masks_and_piecewise(runtime, contract):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    metadata = packed_metadata("fp8", 128, 96)
    if contract == "mask":
        metadata.attn_mask = torch.ones(1, 128, dtype=torch.bool)
    else:
        metadata.full_attn_spans = [[(0, 128)]]
    with pytest.raises(ValueError, match="caller masks" if contract == "mask" else "piecewise"):
        flash().forward_npu(q, q, q, metadata)
    runtime.quant_attention.assert_not_called()


def test_packed_padding_without_legacy_packed_keys_is_accepted(runtime):
    # The crop keys on the producer-validated packed_padding view, not on the
    # legacy cu_seqlens/npu_attn_varlen markers.
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    metadata = AttentionMetadata(
        packed_padding=packed_padding(128, 96),
        extra={"kv_cache_dtype": "fp8"},
    )
    flash().forward_npu(q, q, q, metadata)
    args, _ = runtime.quant_attention.call_args
    assert args[1].shape == (1, 96, 2, 64)


def make_layer(runtime, monkeypatch, method="mxfp8", skip_steps=_DEFAULT_SKIP_STEPS):
    layer = object.__new__(layer_mod.Attention)
    torch.nn.Module.__init__(layer)
    layer.attention = flash()
    layer.attn_backend = flash_attn.FlashAttentionBackend
    layer._disable_kv_quant = False
    layer.layer_idx = 3
    cfg = SimpleNamespace(
        diffusion_kv_cache_dtype=method,
        diffusion_kv_cache_skip_step_indices=skip_steps,
        diffusion_kv_cache_skip_layer_indices=None,
        parallel_config=SimpleNamespace(ring_degree=1),
    )
    layer._init_kv_cache_quantization(cfg)
    return layer


@pytest.mark.parametrize("method", ["fp8", "mxfp8", "mxfp4"])
def test_global_quant_method_is_published_per_forward(runtime, monkeypatch, method):
    layer = make_layer(runtime, monkeypatch, method=method, skip_steps=None)
    assert layer._with_kv_cache_dtype(None).extra["kv_cache_dtype"] == method


@pytest.mark.parametrize("method", [None, "auto", "float"])
def test_optout_without_quant_policy_keeps_metadata_unchanged(runtime, monkeypatch, method):
    layer = make_layer(runtime, monkeypatch)
    layer._disable_kv_quant = True
    layer._init_kv_cache_quantization(
        SimpleNamespace(
            diffusion_kv_cache_dtype=method,
            diffusion_kv_cache_skip_step_indices=None,
            diffusion_kv_cache_skip_layer_indices=None,
            parallel_config=SimpleNamespace(ring_degree=1),
        )
    )
    source = video_metadata()
    assert layer._with_kv_cache_dtype(source) is source


def test_step_skip_is_request_local_and_cross_optout(runtime, monkeypatch):
    layer = make_layer(runtime, monkeypatch)
    ctx = SimpleNamespace(denoise_step_idx=0)
    monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(layer_mod, "get_forward_context", lambda: ctx)
    source = video_metadata()
    for step in list(range(40)) * 2:
        ctx.denoise_step_idx = step
        resolved = layer._with_kv_cache_dtype(source)
        assert resolved.extra["kv_cache_dtype"] == ("float" if step in (0, 1, 38, 39) else "mxfp8")
        assert resolved.video_layout is source.video_layout
    assert source.extra == {"max_seqlen_q": 4096}
    layer._disable_kv_quant = True
    assert layer._with_kv_cache_dtype(source).extra["kv_cache_dtype"] == "float"


def test_ring_attention_is_rejected(runtime, monkeypatch):
    layer = make_layer(runtime, monkeypatch)
    cfg = SimpleNamespace(diffusion_kv_cache_dtype="mxfp8", parallel_config=SimpleNamespace(ring_degree=2))
    with pytest.raises(ValueError, match="ring"):
        layer._init_kv_cache_quantization(cfg)


def test_sparse_tail_steps_and_dense_quant_dispatch(runtime, monkeypatch):
    ctx = SimpleNamespace(denoise_step_idx=49, total_denoise_steps=50)
    monkeypatch.setattr(rainfusion_attn, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(rainfusion_attn, "get_forward_context", lambda: ctx)
    impl = sparse(start_step=2, end_step=2)
    metadata = video_metadata(kv_cache_dtype="fp8")
    assert impl._resolve_plan(metadata) is None
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    impl.forward_npu(q, q, q, metadata)
    assert runtime.quant_attention.call_args.kwargs["precision"] == "fp8"


@pytest.mark.parametrize("method", ["fp8", "mxfp4"])
def test_sparse_quant_uses_public_runtime(runtime, monkeypatch, method):
    runtime.sparse_attention = Mock(side_effect=lambda q, k, v, **kwargs: q)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    q = torch.randn(1, 4224, 2, 64, dtype=torch.bfloat16)
    out = sparse().forward_npu(q, q, q, video_metadata(kv_cache_dtype=method))
    # The sparse path reuses the query tail as the output padding instead of
    # zero-filling a fresh buffer; rows past the 4096-row valid prefix are
    # never read back downstream (the model unpads by cu_seqlens), so pin the
    # passthrough rather than requiring zeros.
    assert out.shape == q.shape and torch.equal(out[:, 4096:], q[:, 4096:])
    kwargs = runtime.sparse_attention.call_args.kwargs
    assert (kwargs["precision"], kwargs["sparse_type"], kwargs["inner_precise"]) == (method, "rf_v3", 4)
    runtime.quant_attention.assert_not_called()


def test_sparse_custom_mask_stays_dense(runtime):
    metadata = AttentionMetadata(
        attn_mask=torch.ones(1, 4096, dtype=torch.bool),
        video_layout=VideoTokenLayout(prefix_len=0, latent_grid=(4, 32, 32)),
        extra={"max_seqlen_q": 4096},
    )
    assert sparse()._resolve_plan(metadata) is None


def test_sparse_model_padding_mask_is_equivalent_to_trimming(runtime):
    metadata = video_metadata()
    metadata.attn_mask = torch.arange(4224)[None] < 4096
    assert sparse()._resolve_plan(metadata).used_len == 4096


@pytest.mark.parametrize(
    "method,error",
    [
        ("mxfp4", RuntimeError("BSA operator failure")),
        ("mxfp8", ValueError("unsupported precision: mxfp8")),
    ],
)
def test_sparse_runtime_failure_never_retries(runtime, monkeypatch, method, error):
    runtime.sparse_attention = Mock(side_effect=error)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(type(error)) as exc_info:
        sparse().forward_npu(q, q, q, video_metadata(kv_cache_dtype=method))
    assert exc_info.value is error
    runtime.sparse_attention.assert_called_once()
    assert runtime.sparse_attention.call_args.kwargs["precision"] == method
    runtime.quant_attention.assert_not_called()


@pytest.mark.parametrize(
    "method,shape,supports_precision,error",
    [
        ("mxfp4", (1, 4096, 2, 64), False, "explicitly support precision"),
        ("fp8", (2, 4096, 2, 64), True, "batch size 1"),
        ("fp8", (1, 4096, 2, 32), True, "power-of-two head dimension"),
        ("mxfp4", (1, 4096, 2, 96), True, "power-of-two head dimension"),
    ],
)
def test_invalid_sparse_quant_requires_config_change(runtime, monkeypatch, method, shape, supports_precision, error):
    runtime.sparse_attention = Mock(side_effect=lambda q, k, v, **kwargs: q)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: supports_precision)
    q = torch.randn(shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=error):
        sparse().forward_npu(q, q, q, video_metadata(kv_cache_dtype=method))
    runtime.sparse_attention.assert_not_called()


def test_layer_selector_requires_index_except_cross_optout(runtime, monkeypatch):
    layer = make_layer(runtime, monkeypatch)
    layer.layer_idx = None
    config = SimpleNamespace(
        diffusion_kv_cache_dtype="mxfp8",
        diffusion_kv_cache_skip_step_indices=None,
        diffusion_kv_cache_skip_layer_indices={0, 3},
        parallel_config=SimpleNamespace(ring_degree=1),
    )
    with pytest.raises(ValueError, match="parseable transformer block index"):
        layer._init_kv_cache_quantization(config)
    layer._disable_kv_quant = True
    layer._init_kv_cache_quantization(config)
    assert layer._with_kv_cache_dtype(None).extra["kv_cache_dtype"] == "float"


@pytest.mark.parametrize("prefix", ["transformer.blocks.3.attn1", "transformer_2.blocks.3.attn1"])
@pytest.mark.parametrize("method", ["fp8", "mxfp4"])
def test_expert_layer_and_step_skip_remain_sparse(runtime, monkeypatch, prefix, method):
    def execute(q, k, v, *, precision="bf16", **kwargs):
        return q

    runtime.sparse_attention = Mock(wraps=execute)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    layer = make_layer(runtime, monkeypatch)
    layer.layer_idx = rainfusion_attn._try_extract_layer_index(prefix)
    assert layer.layer_idx == 3
    config = SimpleNamespace(
        diffusion_kv_cache_dtype=method,
        diffusion_kv_cache_skip_step_indices={0, 1, 38, 39},
        diffusion_kv_cache_skip_layer_indices={3},
        parallel_config=SimpleNamespace(ring_degree=1),
    )
    impl = sparse()
    layer.attention = impl
    layer.attn_backend = rainfusion_attn.RainFusionAttentionBackend
    layer._init_kv_cache_quantization(config)
    ctx = SimpleNamespace(denoise_step_idx=0)
    monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(layer_mod, "get_forward_context", lambda: ctx)
    source = video_metadata(kv_cache_dtype="mxfp8")
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    for layer_idx, step in ((3, 0), (3, 2), (4, 0), (4, 1), (4, 2), (4, 37), (4, 38), (4, 39), (4, 0)):
        expected = "bf16" if layer_idx == 3 or step in (0, 1, 38, 39) else method
        layer.layer_idx = layer_idx
        ctx.denoise_step_idx = step
        metadata = layer._with_kv_cache_dtype(source)
        impl.forward_npu(q, q, q, metadata)
        assert runtime.sparse_attention.call_args.kwargs["precision"] == expected
    runtime.quant_attention.assert_not_called()
    assert source.extra["kv_cache_dtype"] == "mxfp8"


def test_custom_attention_initialization_keeps_upstream_optout(monkeypatch):
    monkeypatch.setattr(layer_mod, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kw: object())
    monkeypatch.setattr(layer_mod, "NoParallelAttention", lambda: object())
    custom = torch.nn.Identity()
    layer = layer_mod.Attention(
        num_heads=2,
        head_size=64,
        softmax_scale=0.125,
        causal=False,
        custom_attention=custom,
        skip_sequence_parallel=True,
    )
    assert layer.attention is custom and layer._kv_cache_dtype is None


def test_legacy_global_fp8_uses_implicit_bsnd(runtime):
    q = torch.randn(1, 17, 2, 64, dtype=torch.bfloat16)
    impl = flash(layout=None)
    assert impl.forward_npu(q, q, q, AttentionMetadata(extra={"kv_cache_dtype": "fp8"})) is q
    assert runtime.quant_attention.call_args.kwargs["layout"] == "BSND"
    assert runtime.quant_attention.call_args.kwargs["precision"] == "fp8"

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
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata, VideoTokenLayout
from vllm_omni.diffusion.data import AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
_DEFAULT_SKIP_STEPS = frozenset({0, 1, 38, 39})


@pytest.fixture
def runtime(monkeypatch):
    mod = ModuleType("mindiesd")

    def execute(
        q,
        k,
        v,
        *,
        precision,
        layout,
        scale,
        q_rot=None,
        k_rot=None,
        fp8_fa_mode=None,
        mxfp4_scale_alg=None,
        mxfp4_dst_type_max=None,
    ):
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


def flash(layout="BSND", head_size=64, quant=None):
    return flash_attn.FlashAttentionImpl(
        num_heads=2,
        head_size=head_size,
        softmax_scale=0.37,
        qkv_layout=layout,
        backend_kwargs={"quant": quant} if quant is not None else None,
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
    assert not {"fp8_fa_mode", "mxfp4_scale_alg", "mxfp4_dst_type_max"} & kwargs.keys()
    from vllm_omni.platforms.npu.quant.kv_quant_npu import get_quant_attention_rotation

    assert kwargs["q_rot"] is kwargs["k_rot"]
    torch.testing.assert_close(kwargs["q_rot"], get_quant_attention_rotation(q.device, q.dtype, 64))


@pytest.mark.parametrize(
    "method,expected",
    [
        ("fp8", {"fp8_fa_mode": "C8V16_TILING512"}),
        ("mxfp8", {}),
        ("mxfp4", {"mxfp4_scale_alg": 0, "mxfp4_dst_type_max": 7.25}),
    ],
)
def test_quant_options_follow_actual_precision(runtime, method, expected):
    quant = {"fp8_fa_mode": "C8V16_TILING512", "mxfp4_scale_alg": 0, "mxfp4_dst_type_max": 7.25}
    original = quant.copy()
    impl = flash(head_size=128, quant=quant)
    query = torch.randn(1, 17, 2, 128, dtype=torch.bfloat16)
    impl.forward_npu(query, query, query, AttentionMetadata(extra={"kv_cache_dtype": method}))
    kwargs = runtime.quant_attention.call_args.kwargs
    assert {name: kwargs[name] for name in quant if name in kwargs} == expected
    assert quant == original


@pytest.mark.parametrize("method", [None, "auto", "float"])
def test_quant_options_do_not_enable_quantization(runtime, mocker, method):
    impl = flash(head_size=128, quant={"fp8_fa_mode": "C8V16_TILING512"})
    query = torch.randn(1, 17, 2, 128, dtype=torch.bfloat16)
    float_attention = mocker.patch.object(impl, "forward_fa_npu", return_value=query)
    impl.forward_npu(query, query, query, AttentionMetadata(extra={"kv_cache_dtype": method}))
    float_attention.assert_called_once()
    runtime.quant_attention.assert_not_called()


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


def packed_metadata(q_length, kv_length, total_q, total_kv, method="mxfp8"):
    cu_q = torch.tensor([0, q_length, total_q], dtype=torch.int32)
    cu_k = torch.tensor([0, kv_length, total_kv], dtype=torch.int32)
    return AttentionMetadata(
        packed_padding=PackedPaddingMetadata(q_length, kv_length, cu_q[:2], cu_k[:2]),
        extra={
            "kv_cache_dtype": method,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": q_length,
            "max_seqlen_k": kv_length,
            "valid_kv_length": kv_length,
            "npu_attn_varlen": True,
        },
    )


@pytest.mark.parametrize(
    "method,quant",
    [
        ("fp8", {}),
        ("fp8", {"fp8_fa_mode": "C8V16_TILING512"}),
        ("mxfp8", {}),
        ("mxfp4", {}),
        ("mxfp4", {"mxfp4_scale_alg": 0, "mxfp4_dst_type_max": 7.25}),
    ],
)
@pytest.mark.parametrize("q_length,kv_length,q_pad,kv_pad", [(5, 5, 3, 3), (8, 8, 0, 0), (3, 7, 2, 2), (5, 7, 0, 3)])
def test_quant_packed_padding_matches_masked_attention(runtime, method, quant, q_length, kv_length, q_pad, kv_pad):
    query = torch.randn(1, q_length + q_pad, 2, 128, dtype=torch.bfloat16)
    key = torch.randn(1, kv_length + kv_pad, 2, 128, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    metadata = packed_metadata(q_length, kv_length, query.shape[1], key.shape[1], method)
    impl = flash(head_size=128, quant=quant)

    def dense_attention(q, k, v, *, precision, layout, scale, q_rot, k_rot, **options):
        assert precision == method and layout == "BSND"
        assert options == quant
        assert q.shape[1] == q_length and k.shape[1] == v.shape[1] == kv_length
        assert q_rot is k_rot
        return (
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(), scale=scale
            )
            .transpose(1, 2)
            .to(q.dtype)
        )

    # Exercise prefix isolation with real floating attention; vendor quantization
    # accuracy requires separate NPU validation.
    runtime.quant_attention.side_effect = dense_attention
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2).float(),
            key.transpose(1, 2).float(),
            value.transpose(1, 2).float(),
            attn_mask=(torch.arange(key.shape[1]) < kv_length)[None, None, None, :],
            scale=impl.softmax_scale,
        )
        .transpose(1, 2)
        .to(query.dtype)
    )
    output = impl.forward_npu(query, key, value, metadata)
    assert output.shape == query.shape and output.dtype == query.dtype
    torch.testing.assert_close(output[:, :q_length], expected[:, :q_length])
    torch.testing.assert_close(output[:, q_length:], query[:, q_length:])

    # Padding must never affect valid attention outputs or quantization scales.
    query[:, q_length:] = 1000
    key[:, kv_length:] = -1000
    value[:, kv_length:] = 1000
    updated_output = impl.forward_npu(query, key, value, metadata)
    torch.testing.assert_close(updated_output[:, :q_length], output[:, :q_length])
    torch.testing.assert_close(updated_output[:, q_length:], query[:, q_length:])
    assert metadata.extra["cu_seqlens_q"].tolist() == [0, q_length, query.shape[1]]
    assert metadata.packed_padding.q_length == q_length


@pytest.mark.parametrize("q_length,kv_length", [(0, 5), (5, 0), (9, 5), (5, 9)])
def test_quant_packed_padding_rejects_out_of_bounds_lengths(runtime, q_length, kv_length):
    query = torch.randn(1, 8, 2, 64, dtype=torch.bfloat16)
    metadata = packed_metadata(q_length, kv_length, 8, 8)
    with pytest.raises(ValueError, match="packed padding lengths"):
        flash().forward_npu(query, query, query, metadata)
    runtime.quant_attention.assert_not_called()


@pytest.mark.parametrize("case", ["mask", "causal", "layout", "batch"])
def test_quant_packed_padding_preserves_unsupported_requests(runtime, case):
    query = torch.randn(2 if case == "batch" else 1, 8, 2, 64, dtype=torch.bfloat16)
    metadata = packed_metadata(5, 5, 8, 8, "fp8")
    impl = flash("BNSD" if case == "layout" else "BSND")
    if case == "mask":
        metadata.attn_mask = torch.ones(1, 8, dtype=torch.bool)
    elif case == "causal":
        impl.causal = True
    elif case == "layout":
        query = query.transpose(1, 2)
    with pytest.raises(ValueError):
        impl.forward_npu(query, query, query, metadata)
    runtime.quant_attention.assert_not_called()


def make_layer(runtime, monkeypatch, method="mxfp8", skip_steps=_DEFAULT_SKIP_STEPS, quant=None, head_size=64):
    layer = object.__new__(layer_mod.Attention)
    torch.nn.Module.__init__(layer)
    layer.attention = flash(head_size=head_size, quant=quant)
    layer.attn_backend = flash_attn.FlashAttentionBackend
    layer.attn_spec = AttentionSpec(backend="FLASH_ATTN", quant=quant)
    layer._schedule_candidates = {}
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


@pytest.mark.parametrize("skip_reason", ["step", "layer", "optout"])
@pytest.mark.parametrize("spec_precision", [False, True])
def test_fp8_options_preserve_float_skips(runtime, monkeypatch, mocker, skip_reason, spec_precision):
    quant = {"fp8_fa_mode": "C8V16_TILING512"}
    if spec_precision:
        quant["precision"] = "fp8"
    layer = make_layer(runtime, monkeypatch, method=None if spec_precision else "fp8", head_size=128, quant=quant)
    ctx = SimpleNamespace(denoise_step_idx=2)
    monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(layer_mod, "get_forward_context", lambda: ctx)
    query = torch.randn(1, 17, 2, 128, dtype=torch.bfloat16)
    float_attention = mocker.patch.object(layer.attention, "forward_fa_npu", return_value=query)

    layer.attention.forward_npu(query, query, query, layer._with_kv_cache_dtype(None))
    assert runtime.quant_attention.call_args.kwargs["fp8_fa_mode"] == "C8V16_TILING512"

    if skip_reason == "step":
        ctx.denoise_step_idx = 0
    elif skip_reason == "layer":
        layer._kv_cache_skip_layers = {layer.layer_idx}
    else:
        layer._disable_kv_quant = True
    layer.attention.forward_npu(query, query, query, layer._with_kv_cache_dtype(None))
    float_attention.assert_called_once()
    runtime.quant_attention.assert_called_once()


@pytest.mark.parametrize("global_dtype", [None, "auto", "float"])
@pytest.mark.parametrize(
    "method,options",
    [
        ("fp8", {"fp8_fa_mode": "C8V16_TILING512"}),
        ("mxfp8", {}),
        ("mxfp4", {"mxfp4_scale_alg": 0, "mxfp4_dst_type_max": 7.25}),
    ],
)
def test_spec_precision_enables_quant_without_global_dtype(runtime, monkeypatch, global_dtype, method, options):
    layer = make_layer(
        runtime,
        monkeypatch,
        method=global_dtype,
        skip_steps=None,
        head_size=128,
        quant={"precision": method, **options},
    )
    query = torch.randn(1, 17, 2, 128, dtype=torch.bfloat16)
    metadata = layer._with_kv_cache_dtype(None)
    layer.attention.forward_npu(query, query, query, metadata)
    kwargs = runtime.quant_attention.call_args.kwargs
    assert kwargs["precision"] == method
    assert {name: kwargs[name] for name in options} == options


@pytest.mark.parametrize("baseline_precision", [None, "fp8"])
def test_spec_precision_tracks_active_schedule(runtime, monkeypatch, mocker, baseline_precision):
    quant = {"precision": baseline_precision} if baseline_precision else None
    layer = make_layer(runtime, monkeypatch, method=None, skip_steps=None, head_size=128, quant=quant)
    layer._schedule_configured = True
    layer._schedule_candidates = {}
    for name, precision in (("fp8", "fp8"), ("mxfp4", "mxfp4"), ("float", None)):
        spec = AttentionSpec(backend="FLASH_ATTN", quant={"precision": precision} if precision else None)
        impl = flash(head_size=128, quant=spec.backend_kwargs()["quant"] if precision else None)
        layer._schedule_candidates[name] = layer_mod.ScheduleCandidate(
            backend_cls=flash_attn.FlashAttentionBackend, spec=spec, impl=impl, backend_explicit=True
        )
    layer._init_kv_cache_quantization(SimpleNamespace(diffusion_kv_cache_dtype=None))
    query = torch.randn(1, 17, 2, 128, dtype=torch.bfloat16)
    float_attention = mocker.patch.object(flash_attn.FlashAttentionImpl, "forward_fa_npu", return_value=query)
    source = video_metadata(kv_cache_dtype="mxfp8")
    for profile in ("mxfp4", "float", "fp8", None, "float"):
        monkeypatch.setattr(layer_mod, "get_attention_schedule_profile", lambda: profile)
        impl, _, _ = layer.effective_attention()
        metadata = layer._with_kv_cache_dtype(source)
        impl.forward_npu(query, query, query, metadata)
    assert [call.kwargs["precision"] for call in runtime.quant_attention.call_args_list] == (
        ["mxfp4", "fp8", "fp8"] if baseline_precision else ["mxfp4", "fp8"]
    )
    assert float_attention.call_count == (2 if baseline_precision else 3)
    assert source.extra["kv_cache_dtype"] == "mxfp8"


def test_spec_precision_rejects_unsupported_platform(runtime, monkeypatch):
    monkeypatch.setattr(layer_mod, "current_omni_platform", SimpleNamespace(device_name="cuda"))
    with pytest.raises(ValueError, match="does not support.*mxfp4.*cuda"):
        make_layer(runtime, monkeypatch, method=None, quant={"precision": "mxfp4"})


@pytest.mark.parametrize("method", [None, "auto", "float", "fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("quant", [None, {"fp8_fa_mode": "C8V16_TILING512"}])
def test_global_quant_method_keeps_original_policy(runtime, monkeypatch, mocker, method, quant):
    layer = make_layer(runtime, monkeypatch, method=method, skip_steps=None, quant=quant)
    mocker.patch.object(
        layer, "effective_attention", side_effect=AssertionError("global policy must not resolve a profile")
    )
    source = video_metadata(kv_cache_dtype="mxfp4")
    metadata = layer._with_kv_cache_dtype(source)
    assert metadata.extra.get("kv_cache_dtype") == (None if method in (None, "auto", "float") else method)
    assert metadata.video_layout is source.video_layout
    assert source.extra["kv_cache_dtype"] == "mxfp4"


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


@pytest.mark.parametrize("spec_precision", [False, True])
def test_ring_attention_is_rejected(runtime, monkeypatch, spec_precision):
    method = None if spec_precision else "mxfp8"
    layer = make_layer(runtime, monkeypatch, method=method, quant={"precision": "mxfp8"} if spec_precision else None)
    cfg = SimpleNamespace(diffusion_kv_cache_dtype=method, parallel_config=SimpleNamespace(ring_degree=2))
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

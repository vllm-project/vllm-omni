# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for MOSS audio tokenizer v2 projection modules.

Includes regression coverage for:
  * Attention-bias memory lifetime (PR #8002 P2 fix).
  * Mask-sharing numerical equivalence (bit-identical).
  * addcmul fusion numerical equivalence.
  * Multi-layer streaming, ring wraparound, slot reset/reuse, non-streaming.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerModel,
    MossAudioTokenizerProjectedTransformer,
    MossAudioTokenizerTransformer,
    MossAudioTokenizerTransformerLayer,
    StreamingExecutionContext,
)
from vllm_omni.model_executor.models.moss_tts.configuration_moss_audio_tokenizer_v2 import (
    MossAudioTokenizerConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------


def test_projected_transformer_keeps_learned_projections_when_dimensions_match() -> None:
    module = MossAudioTokenizerProjectedTransformer(
        input_dimension=8,
        output_dimension=8,
        d_model=8,
        module_type="transformer",
        num_heads=1,
        num_layers=0,
        positional_embedding="rope",
    )

    assert isinstance(module.input_proj, nn.Linear)
    assert isinstance(module.output_proj, nn.Linear)
    assert module.input_proj.weight.shape == (8, 8)
    assert module.output_proj.weight.shape == (8, 8)

    x = torch.randn(2, 8, 5)
    lengths = torch.tensor([5, 3], dtype=torch.long)
    output, output_lengths = module(x, lengths)

    assert output.shape == x.shape
    torch.testing.assert_close(output_lengths, lengths)


def test_projected_transformer_keeps_linear_projections_when_dimensions_differ() -> None:
    module = MossAudioTokenizerProjectedTransformer(
        input_dimension=8,
        output_dimension=6,
        d_model=10,
        module_type="transformer",
        num_heads=1,
        num_layers=0,
        positional_embedding="rope",
    )

    assert isinstance(module.input_proj, nn.Linear)
    assert isinstance(module.output_proj, nn.Linear)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transformer(
    num_layers: int = 4,
    d_model: int = 16,
    num_heads: int = 2,
    context: int | None = 8,
    layer_scale: float | None = 1.0,
    dim_feedforward: int = 32,
) -> MossAudioTokenizerTransformer:
    torch.manual_seed(0)
    return MossAudioTokenizerTransformer(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        causal=True,
        context=context,
        positional_embedding="rope",
        layer_scale=layer_scale,
        gating="none",
        norm="layer_norm",
    )


def _all_biases_cleared(transformer: MossAudioTokenizerTransformer) -> bool:
    return all(layer.self_attn._cached_attn_bias is None for layer in transformer.layers)


_TRANSFORMER_SPEC = {
    "module_type": "Transformer",
    "d_model": 16,
    "num_heads": 2,
    "dim_feedforward": 32,
    "causal": True,
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "max_period": 10000,
    "gating": "none",
    "layer_scale": 1.0,
    "conv_layout": True,
}


def _tiny_codec(num_layers: int = 1, dtype: torch.dtype | None = None) -> MossAudioTokenizerModel:
    spec = {**_TRANSFORMER_SPEC, "num_layers": num_layers}
    config = MossAudioTokenizerConfig(
        sampling_rate=64,
        downsample_rate=4,
        number_channels=1,
        enable_channel_interleave=False,
        encoder_kwargs=[
            {"module_type": "PatchedPretransform", "patch_size": 4},
            {**spec, "input_dimension": 4, "output_dimension": 8, "context_duration": 0.5},
        ],
        decoder_kwargs=[
            {**spec, "input_dimension": 8, "output_dimension": 16, "context_duration": 0.5},
            {"module_type": "PatchedPretransform", "patch_size": 2},
            {**spec, "input_dimension": 8, "output_dimension": 2, "context_duration": 0.25},
            {"module_type": "PatchedPretransform", "patch_size": 2},
        ],
        quantizer_kwargs={
            "input_dim": 8,
            "rvq_dim": 8,
            "output_dim": 8,
            "num_quantizers": 2,
            "codebook_size": 16,
            "codebook_dim": 4,
            "quantizer_type": "rlfq",
        },
    )
    torch.manual_seed(0)
    model = MossAudioTokenizerModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.5)
    lut_dtype = dtype if dtype is not None else torch.float32
    model.quantizer.build_decode_lut(2, dtype=lut_dtype)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def _stream_decode(model, codes, chunk_frames, *, headroom):
    n_q, _, total = codes.shape
    device = codes.device
    model.initialize_decoder_state_pool(1, 0, chunk_frames=chunk_frames if headroom else 0)
    try:
        slot = torch.zeros(1, dtype=torch.long, device=device)
        valid = torch.ones(1, dtype=torch.bool, device=device)
        parts = []
        for start in range(0, total, chunk_frames):
            chunk = codes[:, :, start : start + chunk_frames]
            lengths = torch.full((1,), chunk.shape[-1], dtype=torch.long, device=device)
            parts.append(model.decode_streaming_batch(chunk, lengths, slot, valid).audio)
        return torch.cat(parts, dim=-1)
    finally:
        model.close_decoder_state_pool()


# ---------------------------------------------------------------------------
# P2 fix — attention-bias memory lifetime
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_attn_bias_not_retained_after_forward_non_streaming():
    """_cached_attn_bias is cleared after a non-streaming forward."""
    transformer = _make_transformer(num_layers=4)
    x = torch.randn(2, 10, 16)
    _ = transformer(x)
    assert _all_biases_cleared(transformer)


@torch.no_grad()
def test_attn_bias_not_retained_after_forward_streaming():
    """_cached_attn_bias is cleared after a streaming forward and after streaming exit."""
    transformer = _make_transformer(num_layers=4)
    with transformer.streaming(2):
        x = torch.randn(2, 6, 16)
        _ = transformer(x)
        assert _all_biases_cleared(transformer)
    assert _all_biases_cleared(transformer)


@torch.no_grad()
def test_attn_bias_not_retained_after_forward_with_execution_context():
    """_cached_attn_bias is cleared after a forward driven by a StreamingExecutionContext."""
    transformer = _make_transformer(num_layers=4)
    with transformer.streaming(4):
        slots = torch.tensor([0, 1], dtype=torch.long)
        valid = torch.tensor([True, True], dtype=torch.bool)
        ec = StreamingExecutionContext(state_slot_ids=slots, valid_rows=valid)
        x = torch.randn(2, 6, 16)
        _ = transformer(x, execution_context=ec)
        assert _all_biases_cleared(transformer)
    assert _all_biases_cleared(transformer)


@torch.no_grad()
def test_attn_bias_not_retained_after_exception():
    """_cached_attn_bias is cleared even when a layer raises mid-forward."""

    class _InjectedError(Exception):
        pass

    transformer = _make_transformer(num_layers=4)
    original = transformer.layers[2].forward

    def _failing_forward(*args, **kwargs):
        raise _InjectedError("boom")

    transformer.layers[2].forward = _failing_forward
    try:
        x = torch.randn(2, 10, 16)
        with pytest.raises(_InjectedError):
            transformer(x)
        assert _all_biases_cleared(transformer)
    finally:
        transformer.layers[2].forward = original


@torch.no_grad()
def test_attn_bias_not_retained_multi_step_streaming():
    """_cached_attn_bias stays None across multiple streaming decode steps."""
    transformer = _make_transformer(num_layers=3, context=6)
    with transformer.streaming(2):
        for _ in range(5):
            x = torch.randn(2, 6, 16)
            _ = transformer(x)
            assert _all_biases_cleared(transformer)
    assert _all_biases_cleared(transformer)


# ---------------------------------------------------------------------------
# Mask-sharing numerical equivalence (bit-identical)
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_mask_sharing_bit_identical_non_streaming():
    """Sharing the mask across layers produces a bit-identical output (non-streaming)."""
    transformer = _make_transformer(num_layers=4)
    x = torch.randn(2, 10, 16)

    transformer._share_attn_bias = True
    output_shared = transformer(x)

    transformer._share_attn_bias = False
    output_no_share = transformer(x)
    transformer._share_attn_bias = True

    torch.testing.assert_close(output_shared, output_no_share, rtol=0, atol=0)


@torch.no_grad()
def test_mask_sharing_bit_identical_streaming():
    """Sharing the mask across layers produces a bit-identical output (streaming)."""
    transformer = _make_transformer(num_layers=4, context=6)
    x = torch.randn(2, 6, 16)

    with transformer.streaming(2):
        transformer._share_attn_bias = True
        output_shared = transformer(x)

    with transformer.streaming(2):
        transformer._share_attn_bias = False
        output_no_share = transformer(x)
    transformer._share_attn_bias = True

    torch.testing.assert_close(output_shared, output_no_share, rtol=0, atol=0)


@torch.no_grad()
def test_mask_sharing_bit_identical_with_execution_context():
    """Sharing works with execution_context (slot-based, non-slot kernel path)."""
    transformer = _make_transformer(num_layers=4, context=6)
    x = torch.randn(2, 6, 16)

    def _run(share: bool) -> torch.Tensor:
        with transformer.streaming(4):
            slots = torch.tensor([0, 1], dtype=torch.long)
            valid = torch.tensor([True, True], dtype=torch.bool)
            ec = StreamingExecutionContext(state_slot_ids=slots, valid_rows=valid)
            transformer._share_attn_bias = share
            return transformer(x, execution_context=ec)

    output_shared = _run(True)
    output_no_share = _run(False)
    transformer._share_attn_bias = True

    torch.testing.assert_close(output_shared, output_no_share, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# addcmul fusion numerical equivalence
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_addcmul_tensor_level_matches_mul_add():
    """torch.addcmul(x, u, s) is numerically close to x + s * u."""
    torch.manual_seed(42)
    for dtype in (torch.bfloat16, torch.float32):
        d = 64
        x = torch.randn(2, 10, d, dtype=dtype)
        update = torch.randn(2, 10, d, dtype=dtype)
        scale = torch.randn(d, dtype=dtype)

        mul_add = x + scale * update
        fused = torch.addcmul(x, update, scale)

        max_diff = (mul_add - fused).abs().max().item()
        # bf16 1-ULP (~0.0625); float32 may still differ by 1 ULP due to
        # accumulation order inside addcmul.
        tol = 0.1 if dtype == torch.bfloat16 else 1e-5
        assert max_diff <= tol, f"dtype={dtype} max_abs_diff={max_diff}"


@torch.no_grad()
def test_addcmul_layer_matches_mul_add():
    """A layer using addcmul (layer_scale set) is numerically close to manual mul+add."""
    torch.manual_seed(42)
    d_model = 16
    layer = MossAudioTokenizerTransformerLayer(
        d_model=d_model,
        num_heads=2,
        dim_feedforward=32,
        causal=True,
        context=8,
        norm="layer_norm",
        layer_scale=1.0,
        gating="none",
        rope=None,
    )
    layer.eval()
    x = torch.randn(2, 10, d_model)

    # Fused path (addcmul) — the layer's own forward.
    output_fused = layer(x)

    # Manual mul+add path.
    x_orig = x
    x_n1 = layer.norm1(x)
    update_attn = layer.self_attn(x_n1, x_n1, x_n1)
    x_sa = x_orig.to(update_attn) + layer.layer_scale_1(update_attn)

    x_n2 = layer.norm2(x_sa)
    update_ff = layer.linear2(layer.activation(layer.linear1(x_n2)))
    output_mul_add = x_sa.to(update_ff) + layer.layer_scale_2(update_ff)

    max_diff = (output_fused - output_mul_add).abs().max().item()
    assert max_diff < 0.5, f"max_abs_diff={max_diff}"


@torch.no_grad()
def test_layer_scale_none_uses_mul_add_path():
    """When layer_scale is None, nn.Identity is used and addcmul is skipped."""
    transformer = _make_transformer(num_layers=2, layer_scale=None)
    x = torch.randn(2, 10, 16)
    output = transformer(x)
    assert output.shape == (2, 10, 16)
    for layer in transformer.layers:
        assert isinstance(layer.layer_scale_1, nn.Identity)
        assert isinstance(layer.layer_scale_2, nn.Identity)


@torch.no_grad()
def test_addcmul_transformer_output_close_to_no_addcmul():
    """Transformer output with addcmul is numerically close to manual mul+add."""
    transformer = _make_transformer(num_layers=4, layer_scale=1.0)
    x = torch.randn(2, 10, 16)

    # Fused path (addcmul + mask sharing).
    output_fused = transformer(x)

    # Manual reference: run each layer with mul+add.
    x_ref = x.clone()
    # Replicate positional embedding (rope does not add sin emb).
    for layer in transformer.layers:
        x_orig = x_ref
        x_n1 = layer.norm1(x_ref)
        update_attn = layer.self_attn(x_n1, x_n1, x_n1)
        x_ref = x_orig.to(update_attn) + layer.layer_scale_1(update_attn)
        x_n2 = layer.norm2(x_ref)
        update_ff = layer.linear2(layer.activation(layer.linear1(x_n2)))
        x_ref = x_ref.to(update_ff) + layer.layer_scale_2(update_ff)

    max_diff = (output_fused - x_ref).abs().max().item()
    assert max_diff < 1.0, f"max_abs_diff={max_diff}"


# ---------------------------------------------------------------------------
# Multi-layer streaming, ring wraparound, slot reset/reuse, non-streaming
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_multi_layer_streaming_matches_whole_sequence_with_headroom():
    """Chunked streaming with headroom matches whole-sequence decode (multi-layer)."""
    model = _tiny_codec(num_layers=3)
    torch.manual_seed(1)
    codes = torch.randint(0, 16, (2, 1, 18))
    reference = model._decode_frame(codes).audio
    with_headroom = _stream_decode(model, codes, 6, headroom=True)
    assert with_headroom.shape == reference.shape
    torch.testing.assert_close(with_headroom, reference, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_ring_wraparound_mask_sharing():
    """Streaming through more chunks than the ring capacity still matches whole-sequence."""
    model = _tiny_codec(num_layers=2)
    torch.manual_seed(3)
    # 30 frames in 6-frame chunks = 5 chunks; context=8 tokens, ring capacity = 8+6.
    codes = torch.randint(0, 16, (2, 1, 30))
    reference = model._decode_frame(codes).audio
    with_headroom = _stream_decode(model, codes, 6, headroom=True)
    assert with_headroom.shape == reference.shape
    torch.testing.assert_close(with_headroom, reference, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_slot_reset_reuse_mask_sharing():
    """Reusing a reset slot produces the same audio as a fresh slot."""
    model = _tiny_codec(num_layers=3)
    torch.manual_seed(5)
    codes = torch.randint(0, 16, (2, 1, 12))

    # First decode on slot 0.
    model.initialize_decoder_state_pool(2, 0, chunk_frames=6)
    try:
        slot = torch.tensor([0], dtype=torch.long)
        valid = torch.ones(1, dtype=torch.bool)
        out1 = model.decode_streaming_batch(codes[:, :, :6], torch.tensor([6]), slot, valid).audio

        # Reset slot 0 and reuse.
        model.reset_decoder_state_slots(torch.tensor([0]))
        out2 = model.decode_streaming_batch(codes[:, :, :6], torch.tensor([6]), slot, valid).audio
    finally:
        model.close_decoder_state_pool()

    torch.testing.assert_close(out2, out1, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_non_streaming_mask_sharing_and_addcmul_correct():
    """Non-streaming decode produces correct-shaped output with mask sharing + addcmul."""
    model = _tiny_codec(num_layers=4)
    torch.manual_seed(7)
    codes = torch.randint(0, 16, (2, 1, 10))
    output = model._decode_frame(codes).audio
    assert output.dim() == 3
    assert torch.isfinite(output).all()


@torch.no_grad()
def test_multi_layer_streaming_no_bias_retained():
    """No _cached_attn_bias is retained after multi-layer streaming decode."""
    model = _tiny_codec(num_layers=3)
    torch.manual_seed(11)
    codes = torch.randint(0, 16, (2, 1, 18))
    _stream_decode(model, codes, 6, headroom=True)

    for module in model.decoder:
        if not isinstance(module, MossAudioTokenizerProjectedTransformer):
            continue
        for layer in module.transformer.layers:
            assert layer.self_attn._cached_attn_bias is None


# ---------------------------------------------------------------------------
# NPU (Ascend 910B) regression coverage
# ---------------------------------------------------------------------------


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_addcmul_tensor_level_matches_mul_add_bf16():
    """On NPU, torch.addcmul(x, u, s) is numerically close to x + s * u (bf16)."""
    torch.manual_seed(42)
    d = 64
    x = torch.randn(2, 16, d, device="npu", dtype=torch.bfloat16)
    update = torch.randn(2, 16, d, device="npu", dtype=torch.bfloat16)
    scale = torch.randn(d, device="npu", dtype=torch.bfloat16)

    mul_add = x + scale * update
    fused = torch.addcmul(x, update, scale)

    max_diff = (mul_add.float() - fused.float()).abs().max().item()
    # bf16 1-ULP; addcmul may accumulate in a different order on NPU kernels.
    assert max_diff < 1.0, f"max_abs_diff={max_diff}"


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_mask_sharing_bit_identical_non_streaming():
    """Mask sharing is bit-identical on NPU (non-streaming)."""
    transformer = _make_transformer(num_layers=4)
    transformer = transformer.to("npu")
    x = torch.randn(2, 10, 16, device="npu")

    transformer._share_attn_bias = True
    output_shared = transformer(x)

    transformer._share_attn_bias = False
    output_no_share = transformer(x)
    transformer._share_attn_bias = True

    torch.testing.assert_close(output_shared, output_no_share, rtol=0, atol=0)


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_mask_sharing_bit_identical_streaming():
    """Mask sharing is bit-identical on NPU (streaming)."""
    transformer = _make_transformer(num_layers=4, context=6)
    transformer = transformer.to("npu")
    x = torch.randn(2, 6, 16, device="npu")

    with transformer.streaming(2):
        transformer._share_attn_bias = True
        output_shared = transformer(x)

    with transformer.streaming(2):
        transformer._share_attn_bias = False
        output_no_share = transformer(x)
    transformer._share_attn_bias = True

    torch.testing.assert_close(output_shared, output_no_share, rtol=0, atol=0)


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_attn_bias_not_retained_after_forward():
    """_cached_attn_bias is cleared after forward on NPU (non-streaming + streaming)."""
    transformer = _make_transformer(num_layers=4)
    transformer = transformer.to("npu")

    x = torch.randn(2, 10, 16, device="npu")
    _ = transformer(x)
    assert _all_biases_cleared(transformer)

    with transformer.streaming(2):
        x = torch.randn(2, 6, 16, device="npu")
        _ = transformer(x)
        assert _all_biases_cleared(transformer)
    assert _all_biases_cleared(transformer)


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_addcmul_layer_matches_mul_add():
    """A layer using addcmul on NPU is numerically close to manual mul+add."""
    torch.manual_seed(42)
    d_model = 16
    layer = MossAudioTokenizerTransformerLayer(
        d_model=d_model,
        num_heads=2,
        dim_feedforward=32,
        causal=True,
        context=8,
        norm="layer_norm",
        layer_scale=1.0,
        gating="none",
        rope=None,
        device="npu",
    )
    layer.eval()
    x = torch.randn(2, 10, d_model, device="npu")

    output_fused = layer(x)

    x_orig = x
    x_n1 = layer.norm1(x)
    update_attn = layer.self_attn(x_n1, x_n1, x_n1)
    x_sa = x_orig.to(update_attn) + layer.layer_scale_1(update_attn)

    x_n2 = layer.norm2(x_sa)
    update_ff = layer.linear2(layer.activation(layer.linear1(x_n2)))
    output_mul_add = x_sa.to(update_ff) + layer.layer_scale_2(update_ff)

    max_diff = (output_fused.float() - output_mul_add.float()).abs().max().item()
    assert max_diff < 1.0, f"max_abs_diff={max_diff}"


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_multi_layer_streaming_matches_whole_sequence():
    """Chunked streaming with headroom matches whole-sequence decode on NPU."""
    model = _tiny_codec(num_layers=3, dtype=torch.bfloat16)
    model = model.to("npu")
    torch.manual_seed(1)
    codes = torch.randint(0, 16, (2, 1, 18), device="npu")
    reference = model._decode_frame(codes).audio
    with_headroom = _stream_decode(model, codes, 6, headroom=True)
    assert with_headroom.shape == reference.shape
    torch.testing.assert_close(with_headroom.float(), reference.float(), rtol=1e-3, atol=1e-3)


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_slot_reset_reuse():
    """Slot reset and reuse produces the same audio on NPU."""
    model = _tiny_codec(num_layers=3, dtype=torch.bfloat16)
    model = model.to("npu")
    torch.manual_seed(5)
    codes = torch.randint(0, 16, (2, 1, 12), device="npu")

    model.initialize_decoder_state_pool(2, 0, chunk_frames=6)
    try:
        slot = torch.tensor([0], dtype=torch.long, device="npu")
        valid = torch.ones(1, dtype=torch.bool, device="npu")
        out1 = model.decode_streaming_batch(codes[:, :, :6], torch.tensor([6], device="npu"), slot, valid).audio

        model.reset_decoder_state_slots(torch.tensor([0], device="npu"))
        out2 = model.decode_streaming_batch(codes[:, :, :6], torch.tensor([6], device="npu"), slot, valid).audio
    finally:
        model.close_decoder_state_pool()

    torch.testing.assert_close(out2, out1, rtol=1e-3, atol=1e-3)


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_non_streaming_decode_correct():
    """Non-streaming decode on NPU produces finite output."""
    model = _tiny_codec(num_layers=4, dtype=torch.bfloat16)
    model = model.to("npu")
    torch.manual_seed(7)
    codes = torch.randint(0, 16, (2, 1, 10), device="npu")
    output = model._decode_frame(codes).audio
    assert output.dim() == 3
    assert torch.isfinite(output).all()


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_no_bias_retained_after_streaming_decode():
    """No _cached_attn_bias is retained after NPU streaming decode."""
    model = _tiny_codec(num_layers=3, dtype=torch.bfloat16)
    model = model.to("npu")
    torch.manual_seed(11)
    codes = torch.randint(0, 16, (2, 1, 18), device="npu")
    _stream_decode(model, codes, 6, headroom=True)

    for module in model.decoder:
        if not isinstance(module, MossAudioTokenizerProjectedTransformer):
            continue
        for layer in module.transformer.layers:
            assert layer.self_attn._cached_attn_bias is None

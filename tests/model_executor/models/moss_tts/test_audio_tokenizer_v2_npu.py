# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend (910B) regression coverage for the MOSS audio tokenizer v2.

Mask-sharing and addcmul-fusion paths from PR #8002 are NPU-specific, so the
NPU tests live in their own module instead of inheriting the CPU ``pytestmark``
from :mod:`test_audio_tokenizer_v2`.  This keeps them out of the CPU test
shards and on the Ascend hardware CI.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model]


def _make_transformer(*args, **kwargs):
    from tests.model_executor.models.moss_tts.test_audio_tokenizer_v2 import _make_transformer as _cpu_make

    return _cpu_make(*args, **kwargs)


def _all_biases_cleared(transformer):
    from tests.model_executor.models.moss_tts.test_audio_tokenizer_v2 import _all_biases_cleared as _cpu_cleared

    return _cpu_cleared(transformer)


def _tiny_codec(*args, **kwargs):
    from tests.model_executor.models.moss_tts.test_audio_tokenizer_v2 import _tiny_codec as _cpu_codec

    return _cpu_codec(*args, **kwargs)


def _stream_decode(*args, **kwargs):
    from tests.model_executor.models.moss_tts.test_audio_tokenizer_v2 import _stream_decode as _cpu_dec

    return _cpu_dec(*args, **kwargs)


def _decode_with_manual_mul_add(model, codes):
    """Decode ``codes`` using the manual mul+add path instead of addcmul.

    ``unittest.mock.patch`` scopes the class-level swap to the ``with`` block so
    a parallel ``pytest-xdist`` worker cannot observe the stubs mid-run.
    """
    from unittest.mock import patch

    from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import MossAudioTokenizerTransformerLayer

    def _sa_ref(self, x, execution_context=None, attn_bias=None):
        update = self.self_attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            execution_context=execution_context,
            attn_bias=attn_bias,
        )
        return x.to(update.dtype) + self.layer_scale_1(update)

    def _ff_ref(self, x, execution_context=None):
        x_orig, x = x, self.norm2(x)
        if self.linear1 is not None:
            assert self.linear2 is not None
            update = self.linear2(self.activation(self.linear1(x)))
        else:
            update = self.gating(x)
        return x_orig.to(update.dtype) + self.layer_scale_2(update)

    with (
        patch.object(MossAudioTokenizerTransformerLayer, "_sa_block", _sa_ref),
        patch.object(MossAudioTokenizerTransformerLayer, "_ff_block", _ff_ref),
    ):
        return model._decode_frame(codes).audio


def _waveform_snr(reference, corrupt):
    error = reference.float() - corrupt.float()
    rms_ref = reference.float().pow(2).mean().sqrt().item()
    err_rms = error.pow(2).mean().sqrt().item()
    if err_rms == 0.0:
        return math.inf
    return 20.0 * math.log10(rms_ref / err_rms)


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
    # bf16 1-ULP, with addcmul allowed up to a few ULPs of reordering on NPU.
    # Tight enough that dropping the residual or scaling it wrong fails by ~1.
    assert max_diff < 0.05, f"max_abs_diff={max_diff}"


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
    from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import MossAudioTokenizerTransformerLayer

    torch.manual_seed(42)
    d_model = 16
    layer = MossAudioTokenizerTransformerLayer(
        d_model=d_model,
        num_heads=2,
        dim_feedforward=32,
        causal=True,
        context=8,
        norm="layer_norm",
        layer_scale=0.7,
        gating="none",
        rope=None,
        device="npu",
    )
    layer = layer.to(torch.bfloat16).eval()
    x = torch.randn(2, 10, d_model, device="npu", dtype=torch.bfloat16)

    output_fused = layer(x)

    x_orig = x
    x_n1 = layer.norm1(x)
    update_attn = layer.self_attn(x_n1, x_n1, x_n1)
    x_sa = x_orig.to(update_attn) + layer.layer_scale_1(update_attn)

    x_n2 = layer.norm2(x_sa)
    update_ff = layer.linear2(layer.activation(layer.linear1(x_n2)))
    output_mul_add = x_sa.to(update_ff) + layer.layer_scale_2(update_ff)

    torch.testing.assert_close(output_fused.float(), output_mul_add.float(), rtol=2e-2, atol=2e-2)


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_npu_multi_layer_streaming_matches_whole_sequence(dtype):
    """Chunked decode must preserve whole-sequence causal visibility on NPU."""
    model = _tiny_codec(num_layers=3, dtype=dtype)
    model = model.to("npu")
    torch.manual_seed(1)
    codes = torch.randint(0, 16, (2, 1, 18), device="npu")
    reference = model._decode_frame(codes).audio
    with_headroom = _stream_decode(model, codes, 6, headroom=True)
    assert with_headroom.shape == reference.shape
    torch.testing.assert_close(with_headroom.float(), reference.float(), rtol=1e-3, atol=1e-3)


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("frames", [1, 2])
def test_npu_shared_metadata_masks_invalid_and_future_keys(dtype: torch.dtype, frames: int) -> None:
    """Shared metadata and its layer-to-layer reuse must preserve causal visibility."""
    from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
        MossAudioTokenizerMultiheadAttention,
        StreamingExecutionContext,
        prepare_streaming_attention_metadata,
    )

    slots = torch.zeros(1, dtype=torch.long, device="npu")
    valid = torch.ones(1, dtype=torch.bool, device="npu")
    metadata = prepare_streaming_attention_metadata(slots, valid, frames, 4, 4, True)
    execution_context = StreamingExecutionContext(slots, valid, metadata)
    x = torch.ones(1, frames, 64, device="npu", dtype=dtype)
    if frames == 2:
        x[:, 1] = 3
    expected = torch.tensor([1.0] if frames == 1 else [1.0, 2.0], device="npu", dtype=dtype)
    shared_bias = None
    for _ in range(2):
        attention = MossAudioTokenizerMultiheadAttention(
            64, 1, causal=True, context=4, device="npu", dtype=dtype
        ).eval()
        # Q=K=0 gives uniform attention; V and the output projection are identities.
        attention.in_projs[0].weight.zero_()
        attention.in_projs[0].weight[128:].copy_(torch.eye(64, device="npu", dtype=dtype))
        attention.out_projs[0].weight.copy_(torch.eye(64, device="npu", dtype=dtype))
        state = attention._init_streaming_state(1)
        attention._streaming_state = state
        assert state.kv_cache is not None
        state.kv_cache.cache[0].zero_()
        state.kv_cache.cache[1].fill_(100)
        actual = attention(x, x, x, execution_context=execution_context, attn_bias=shared_bias)
        torch.testing.assert_close(actual, expected.view(1, frames, 1).expand_as(actual), rtol=0, atol=0)
        shared_bias = attention._cached_attn_bias


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_slot_reset_reuse():
    """A reset slot must not expose KV left by a different, longer input."""
    model = _tiny_codec(num_layers=3, dtype=torch.bfloat16)
    model = model.to("npu")
    torch.manual_seed(5)
    codes = torch.randint(0, 16, (2, 1, 24), device="npu")

    model.initialize_decoder_state_pool(2, 0, chunk_frames=6)
    try:
        slot = torch.tensor([0], dtype=torch.long, device="npu")
        valid = torch.ones(1, dtype=torch.bool, device="npu")
        out1 = model.decode_streaming_batch(codes[:, :, :6], torch.tensor([6], device="npu"), slot, valid).audio

        model.reset_decoder_state_slots(slot)
        for start in range(6, 24, 6):
            model.decode_streaming_batch(codes[:, :, start : start + 6], torch.tensor([6], device="npu"), slot, valid)
        model.reset_decoder_state_slots(slot)
        out2 = model.decode_streaming_batch(codes[:, :, :6], torch.tensor([6], device="npu"), slot, valid).audio
    finally:
        model.close_decoder_state_pool()

    torch.testing.assert_close(out2, out1, rtol=0, atol=0)


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_slot_reset_reuse_with_codec_graph() -> None:
    """Real codec graph replay must isolate reset request slots, including padding."""
    from vllm.config import VllmConfig

    from vllm_omni.platforms.npu.models.moss_tts_streaming_decode_wrapper import NPUGraphStreamingDecoderWrapper

    model = _tiny_codec(num_layers=3, dtype=torch.bfloat16).to("npu").eval()
    torch.manual_seed(5)
    codes = torch.randint(0, 16, (2, 1, 24), device="npu")
    slots = torch.zeros(1, dtype=torch.long, device="npu")
    model.initialize_decoder_state_pool(2, 1, chunk_frames=6)
    try:
        wrapper = NPUGraphStreamingDecoderWrapper(
            model, state_capacity=2, batch_sizes=[2], frame_sizes=[6], num_quantizers=2, vllm_config=VllmConfig()
        )
        wrapper.warmup(torch.device("npu:0"))
        first = wrapper.decode(codes[:, :, :6], slots)
        assert first is not None
        model.reset_decoder_state_slots(slots)
        for start in range(6, 24, 6):
            assert wrapper.decode(codes[:, :, start : start + 6], slots) is not None
        model.reset_decoder_state_slots(slots)
        second = wrapper.decode(codes[:, :, :6], slots)
        assert second is not None
        torch.testing.assert_close(second[0], first[0], rtol=0, atol=0)
        assert wrapper._graph_runner is not None
        assert wrapper._graph_runner.stats["failed"] == 0
        assert wrapper._graph_runner.stats["hits"] > 0
    finally:
        model.close_decoder_state_pool()


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
    from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import MossAudioTokenizerProjectedTransformer

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


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_addcmul_waveform_error_multilayer():
    """Fused decode waveform on NPU stays close to the manual mul+add path.

    Measured on this hardware: max_abs_diff 0.5 (single-element outlier),
    SNR 42 dB.  The SNR gate is what catches a broken residual — the fusion
    error sits below the bf16 quantization floor (bf16-vs-fp32 SNR ~27 dB on the
    real checkpoint), so bf16 is not expected to be bit-identical.
    """

    model = _tiny_codec(num_layers=4, dtype=torch.bfloat16)
    model = model.to("npu")
    torch.manual_seed(1)
    codes = torch.randint(0, 16, (2, 1, 16), device="npu")

    fused = model._decode_frame(codes).audio
    reference = _decode_with_manual_mul_add(model, codes)

    snr = _waveform_snr(reference, fused)
    assert snr > 35.0, f"SNR={snr:.2f} dB"


@hardware_test(res={"npu": "A3"}, num_cards=1)
@torch.no_grad()
def test_npu_audio_quality_equivalence_masking():
    """Mask sharing produces a bit-identical decoded waveform on NPU (SNR = inf)."""
    from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import MossAudioTokenizerProjectedTransformer

    model = _tiny_codec(num_layers=3, dtype=torch.bfloat16)
    model = model.to("npu")
    torch.manual_seed(2)
    codes = torch.randint(0, 16, (2, 1, 18), device="npu")

    def _decode(share: bool):
        for module in model.decoder:
            if isinstance(module, MossAudioTokenizerProjectedTransformer):
                module.transformer._share_attn_bias = share
        try:
            return model._decode_frame(codes).audio
        finally:
            for module in model.decoder:
                if isinstance(module, MossAudioTokenizerProjectedTransformer):
                    module.transformer._share_attn_bias = True

    shared = _decode(True)
    no_share = _decode(False)
    torch.testing.assert_close(shared, no_share, rtol=0, atol=0)
    assert _waveform_snr(no_share, shared) == math.inf

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav
from vllm_omni.model_executor.models.qwen3_tts.segmented_graph_wrapper import CUDAGraphDecoderWrapper
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    _CONV_CONTEXT_FRAME,
    _DOWNSTREAM_CONTEXT_FRAME,
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2DecoderTransformerModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Shared test helpers


def _decoder_stub(**kwargs):
    decoder = SimpleNamespace(**kwargs)
    decoder.batched_request_decode = Qwen3TTSTokenizerV2Decoder.batched_request_decode.__get__(decoder)
    return decoder


def _make_transformer_model() -> Qwen3TTSTokenizerV2DecoderTransformerModel:
    config = Qwen3TTSTokenizerV2DecoderConfig(
        hidden_size=32,
        latent_dim=16,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=32,
        upsample_rates=(2,),
        upsampling_ratios=(2,),
    )
    config._attn_implementation = "sdpa"
    return Qwen3TTSTokenizerV2DecoderTransformerModel._from_config(config).eval()


def _make_small_decoder() -> Qwen3TTSTokenizerV2Decoder:
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=32,
        hidden_size=16,
        latent_dim=16,
        codebook_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=32,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        sliding_window=72,
    )
    return Qwen3TTSTokenizerV2Decoder(config).eval()


def _run_downstream(decoder: Qwen3TTSTokenizerV2Decoder, hidden: torch.Tensor) -> torch.Tensor:
    for blocks in decoder.upsample:
        for block in blocks:
            hidden = block(hidden)
    for block in decoder.decoder:
        hidden = block(hidden)
    return hidden


class _RecordingDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(sliding_window=72)
        self.prefix_frames = None

    def decode_suffix(self, codes, caches, prefix_frames):
        self.prefix_frames = prefix_frames
        return codes


def _make_stateful_eager_decoder_pair() -> tuple[Qwen3TTSTokenizerV2Decoder, Qwen3TTSTokenizerV2Decoder]:
    batched_decoder = _make_small_decoder()
    per_request_decoder = copy.deepcopy(batched_decoder)
    batched_decoder._incremental_chunk_frames = 25
    per_request_decoder._incremental_chunk_frames = 25
    return batched_decoder, per_request_decoder


def test_dummy_forward_uses_exact_decode_during_outer_graph_capture(monkeypatch):
    decoder = _make_small_decoder()
    codes = torch.zeros(1, decoder.config.num_quantizers, 4, dtype=torch.long)
    cache = {"prefix_frames": 0, "_is_dummy_run": True}
    expected = torch.ones(1, 1, 7)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(decoder, "_forward_exact", lambda _codes: expected)

    def _unexpected_stateful_decode(*_args, **_kwargs):
        raise AssertionError("capturing dummy run must not initialize decoder state")

    monkeypatch.setattr(decoder, "_decode_xvec_first_chunk", _unexpected_stateful_decode)

    output = decoder(codes, cache)

    assert output is expected
    assert cache == {"prefix_frames": 0, "_is_dummy_run": True}


def test_code2wav_full_graph_dummy_forward_uses_exact_batched_decode(monkeypatch):
    decoder = _make_small_decoder()
    expected = torch.arange(7, dtype=torch.float32).view(1, 1, -1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(decoder, "_forward_exact", lambda _codes: expected)

    def _unexpected_stateful_decode(*_args, **_kwargs):
        raise AssertionError("FULL graph dummy capture must not initialize decoder state")

    monkeypatch.setattr(decoder, "_decode_xvec_first_chunk", _unexpected_stateful_decode)

    model = Qwen3TTSCode2Wav.__new__(Qwen3TTSCode2Wav)
    nn.Module.__init__(model)
    model.decoder = decoder
    model._async_chunk = True
    model._num_quantizers = decoder.config.num_quantizers
    model._total_upsample = decoder.total_upsample
    model._output_sample_rate = 24000
    model._decode_chunk_frames = 300
    model._decode_left_context_frames = 25
    model._decode_batch_max_size = 0
    model._decoder_state_cache = {}
    model._decoder_state_cache_warn_entries = 512
    model._logged_codec_stats = True
    model._logged_malformed_codec_lengths = set()
    model._batch_stats_enabled = False
    model._batch_stats_log_every = 0
    model._batch_stats_forwards = 0

    runtime_info = model.get_dummy_runtime_additional_information(1)
    output = model.forward(
        input_ids=torch.arange(8, dtype=torch.long),
        seq_token_counts=[8],
        runtime_additional_information=runtime_info,
    )

    torch.testing.assert_close(output.multimodal_outputs["model_outputs"][0], expected[0, 0])
    assert model._decoder_state_cache == {}


def _make_stateful_first_chunk_inputs(
    decoder: Qwen3TTSTokenizerV2Decoder,
    mode: str,
) -> tuple[torch.Tensor, list[int], list[dict], list[dict]]:
    if mode == "xvec":
        lengths = [1, 1]
        prefix_frames = [0, 0]
    else:
        lengths = [49, 73]
        prefix_frames = [48, 72]

    codes = torch.zeros(2, decoder.config.num_quantizers, max(lengths), dtype=torch.long)
    for row, length in enumerate(lengths):
        codes[row, :, :length] = torch.randint(
            0,
            decoder.config.codebook_size,
            (decoder.config.num_quantizers, length),
        )
    batched_caches = [{"prefix_frames": frames} for frames in prefix_frames]
    per_request_caches = copy.deepcopy(batched_caches)
    return codes, lengths, batched_caches, per_request_caches


def _decode_per_request(
    decoder: Qwen3TTSTokenizerV2Decoder,
    codes: torch.Tensor,
    lengths: list[int],
    caches: list[dict],
) -> torch.Tensor:
    return torch.cat(
        [
            decoder.chunked_decode(codes[row : row + 1, :, :length], caches=caches[row])
            for row, length in enumerate(lengths)
        ],
        dim=0,
    )


def _initialize_stateful_batch(
    mode: str,
) -> tuple[Qwen3TTSTokenizerV2Decoder, Qwen3TTSTokenizerV2Decoder, list[dict], list[dict]]:
    batched_decoder, per_request_decoder = _make_stateful_eager_decoder_pair()
    codes, lengths, batched_caches, per_request_caches = _make_stateful_first_chunk_inputs(batched_decoder, mode)
    batched_decoder.batched_chunked_decode(codes, lengths, caches=batched_caches)
    _decode_per_request(per_request_decoder, codes, lengths, per_request_caches)
    return batched_decoder, per_request_decoder, batched_caches, per_request_caches


# -----------------------------------------------------------------------------
# Result correctness: components, end-to-end incremental decode, rolling, and batching
# -----------------------------------------------------------------------------


def test_transformer_incremental_kv_matches_full_decode():
    torch.manual_seed(0)
    model = _make_transformer_model()
    prefix = torch.randn(1, 11, model.config.latent_dim)
    suffix = torch.randn(1, 6, model.config.latent_dim)

    prefix_output = model(inputs_embeds=prefix, use_cache=True)
    assert isinstance(prefix_output.past_key_values, DynamicCache)
    assert prefix_output.past_key_values.get_seq_length() == prefix.shape[1]
    suffix_output = model(
        inputs_embeds=suffix,
        past_key_values=prefix_output.past_key_values,
        use_cache=True,
    )
    full_output = model(inputs_embeds=torch.cat([prefix, suffix], dim=1))

    torch.testing.assert_close(
        suffix_output.last_hidden_state,
        full_output.last_hidden_state[:, prefix.shape[1] :, :],
        atol=1e-5,
        rtol=1e-4,
    )
    assert suffix_output.past_key_values.get_seq_length() == prefix.shape[1] + suffix.shape[1]


def test_transformer_left_padded_prefix_mask_matches_unpadded_incremental_decode():
    torch.manual_seed(0)
    model = _make_transformer_model()
    prefix = torch.randn(1, 11, model.config.latent_dim)
    suffix = torch.randn(1, 6, model.config.latent_dim)
    prefix_bucket = 16
    prefix_pad = prefix_bucket - prefix.shape[1]

    prefix_output = model(inputs_embeds=prefix, use_cache=True)
    suffix_output = model(
        inputs_embeds=suffix,
        past_key_values=prefix_output.past_key_values,
        use_cache=True,
    )

    padded_prefix = torch.cat([torch.zeros_like(prefix[:, :prefix_pad]), prefix], dim=1)
    prefix_mask = torch.cat(
        [
            torch.zeros(1, prefix_pad, dtype=torch.bool),
            torch.ones(1, prefix.shape[1], dtype=torch.bool),
        ],
        dim=1,
    )
    padded_prefix_output = model(
        inputs_embeds=padded_prefix,
        attention_mask=prefix_mask,
        use_cache=True,
    )
    assert padded_prefix_output.past_key_values.get_seq_length() == prefix_bucket
    padded_suffix_output = model(
        inputs_embeds=suffix,
        attention_mask=torch.cat(
            [prefix_mask, torch.ones(1, suffix.shape[1], dtype=torch.bool)],
            dim=1,
        ),
        past_key_values=padded_prefix_output.past_key_values,
        use_cache=True,
    )

    torch.testing.assert_close(
        padded_suffix_output.last_hidden_state,
        suffix_output.last_hidden_state,
        atol=1e-5,
        rtol=1e-4,
    )
    assert padded_suffix_output.past_key_values.get_seq_length() == prefix_bucket + suffix.shape[1]


def test_pre_conv_two_frame_context_matches_full_decode_and_one_frame_does_not():
    torch.manual_seed(2)
    decoder = _make_small_decoder()
    previous_frames = 19
    new_frames = 7
    hidden = torch.randn(1, decoder.config.codebook_dim, previous_frames + new_frames)

    with torch.no_grad():
        full = decoder.pre_conv(hidden)[..., -new_frames:]
        with_full_context = decoder.pre_conv(hidden[..., -(_CONV_CONTEXT_FRAME + new_frames) :])[..., -new_frames:]
        with_short_context = decoder.pre_conv(hidden[..., -((_CONV_CONTEXT_FRAME - 1) + new_frames) :])[
            ..., -new_frames:
        ]

    assert _CONV_CONTEXT_FRAME == 2
    torch.testing.assert_close(with_full_context, full, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(with_short_context, full, atol=1e-6, rtol=1e-5)


def test_downstream_twelve_frame_context_matches_full_decode_and_context_is_required():
    torch.manual_seed(3)
    decoder = _make_small_decoder()
    previous_frames = 20
    new_frames = 5
    hidden = torch.randn(1, decoder.config.latent_dim, previous_frames + new_frames)

    with torch.no_grad():
        full = _run_downstream(decoder, hidden)[..., -new_frames * decoder.total_upsample :]
        with_full_context = _run_downstream(
            decoder,
            hidden[..., -(_DOWNSTREAM_CONTEXT_FRAME + new_frames) :],
        )[..., -new_frames * decoder.total_upsample :]
        without_context = _run_downstream(
            decoder,
            hidden[..., -new_frames:],
        )[..., -new_frames * decoder.total_upsample :]

    assert _DOWNSTREAM_CONTEXT_FRAME == 12
    torch.testing.assert_close(with_full_context, full)
    assert (without_context - full).abs().max().item() > 1e-10


@pytest.mark.parametrize("mode", ["xvec", "icl"])
def test_incremental_request_audio_matches_full_decode(mode):
    torch.manual_seed(3)
    decoder = _make_small_decoder()
    decoder._incremental_chunk_frames = 25
    prefix_frames = 0 if mode == "xvec" else 10
    suffix_frames = 51
    codes = torch.randint(
        0,
        decoder.config.codebook_size,
        (1, decoder.config.num_quantizers, prefix_frames + suffix_frames),
    )
    suffix_codes = codes[..., prefix_frames:]
    caches = {"prefix_frames": prefix_frames}

    with torch.no_grad():
        first = decoder(codes[..., : prefix_frames + 1], caches=caches)
        second = decoder(suffix_codes[..., 1:26], caches=caches)
        third = decoder(suffix_codes[..., 26:51], caches=caches)
        incremental = torch.cat(
            [first[..., -decoder.total_upsample :], second, third],
            dim=-1,
        )
        full = decoder._forward_exact(codes)
        full_suffix = full[..., prefix_frames * decoder.total_upsample :]

    assert incremental.shape[-1] == suffix_frames * decoder.total_upsample
    assert full_suffix.shape == incremental.shape
    assert caches["decoder_prefix_frames"] == prefix_frames
    assert caches["suffix_frames"] == suffix_frames
    assert caches["past_key_values"].get_seq_length() == prefix_frames
    assert all(layer.is_initialized for layer in caches["past_key_values"].layers)
    torch.testing.assert_close(incremental, full_suffix, atol=1e-5, rtol=1e-4)


def test_single_frame_xvec_prefix_keeps_all_next_chunk_conv_frames():
    torch.manual_seed(4)
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=32,
        hidden_size=16,
        latent_dim=16,
        codebook_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=32,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        sliding_window=72,
    )
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    decoder._incremental_chunk_frames = 25
    codes = torch.randint(0, config.codebook_size, (1, config.num_quantizers, 26))
    caches = {"prefix_frames": 0}

    with torch.no_grad():
        decoder(codes[..., :1], caches=caches)
        incremental = decoder(codes[..., 1:], caches=caches)
        full = decoder._forward_exact(codes)

    assert caches["suffix_frames"] == 26
    assert caches["suffix_quantized"].shape[-1] == 26
    assert caches["suffix_conv"].shape[1] == 26
    torch.testing.assert_close(
        incremental,
        full[..., -25 * decoder.total_upsample :],
        atol=1e-5,
        rtol=1e-4,
    )


def test_xvec_rolling_matches_truncated_suffix_window():
    torch.manual_seed(5)
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=32,
        hidden_size=16,
        latent_dim=16,
        codebook_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=32,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        sliding_window=72,
    )
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    decoder._incremental_chunk_frames = 25
    codes = torch.randint(0, config.codebook_size, (1, config.num_quantizers, 101))
    caches = {"prefix_frames": 0}

    with torch.no_grad():
        decoder(codes[..., :1], caches=caches)
        decoder(codes[..., 1:26], caches=caches)
        decoder(codes[..., 26:51], caches=caches)
        decoder(codes[..., 51:76], caches=caches)
        rolling = decoder(codes[..., 76:101], caches=caches)
        truncated = decoder._forward_exact(codes[..., -(config.sliding_window + 25) :])

    assert caches["ref_hidden"].shape[-1] == 0
    assert caches["suffix_quantized"].shape[-1] == config.sliding_window
    assert caches["suffix_conv"].shape[1] == config.sliding_window - 2
    torch.testing.assert_close(
        rolling,
        truncated[..., -25 * decoder.total_upsample :],
        atol=1e-5,
        rtol=1e-4,
    )


def test_icl_rolling_matches_reference_plus_truncated_suffix_window():
    torch.manual_seed(6)
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=32,
        hidden_size=16,
        latent_dim=16,
        codebook_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=32,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        sliding_window=72,
    )
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    decoder._incremental_chunk_frames = 25
    prefix_frames = 10
    suffix_frames = 101
    codes = torch.randint(
        0,
        config.codebook_size,
        (1, config.num_quantizers, prefix_frames + suffix_frames),
    )
    prefix_codes = codes[..., :prefix_frames]
    suffix_codes = codes[..., prefix_frames:]
    caches = {"prefix_frames": prefix_frames}

    with torch.no_grad():
        decoder(codes[..., : prefix_frames + 1], caches=caches)
        decoder(suffix_codes[..., 1:26], caches=caches)
        decoder(suffix_codes[..., 26:51], caches=caches)
        decoder(suffix_codes[..., 51:76], caches=caches)
        rolling = decoder(suffix_codes[..., 76:101], caches=caches)
        truncated_codes = torch.cat(
            [prefix_codes, suffix_codes[..., -(config.sliding_window + 25) :]],
            dim=-1,
        )
        truncated = decoder._forward_exact(truncated_codes)

    assert caches["decoder_prefix_frames"] == prefix_frames
    assert caches["ref_hidden"].shape[-1] == prefix_frames
    assert caches["suffix_quantized"].shape[-1] == config.sliding_window
    assert caches["suffix_conv"].shape[1] == config.sliding_window - 2
    torch.testing.assert_close(
        rolling,
        truncated[..., -25 * decoder.total_upsample :],
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.mark.parametrize("mode", ["xvec", "icl"])
def test_batched_stateful_first_chunk_matches_per_request(mode):
    torch.manual_seed(7)
    batched_decoder, per_request_decoder = _make_stateful_eager_decoder_pair()
    codes, lengths, batched_caches, per_request_caches = _make_stateful_first_chunk_inputs(batched_decoder, mode)

    with torch.no_grad():
        batched = batched_decoder.batched_chunked_decode(codes, lengths, caches=batched_caches)
        per_request = _decode_per_request(per_request_decoder, codes, lengths, per_request_caches)

    torch.testing.assert_close(torch.stack(batched), per_request, atol=1e-5, rtol=1e-4)
    for batched_cache, per_request_cache in zip(batched_caches, per_request_caches, strict=True):
        batched_suffix_frames = batched_cache.get("suffix_frames", batched_cache["suffix_quantized"].shape[-1])
        per_request_suffix_frames = per_request_cache.get(
            "suffix_frames", per_request_cache["suffix_quantized"].shape[-1]
        )
        assert batched_suffix_frames == per_request_suffix_frames == 1
        torch.testing.assert_close(batched_cache["suffix_quantized"], per_request_cache["suffix_quantized"])
        torch.testing.assert_close(batched_cache["suffix_conv"], per_request_cache["suffix_conv"])


@pytest.mark.parametrize("mode", ["xvec", "icl"])
def test_batched_stateful_growing_suffix_matches_per_request(mode):
    torch.manual_seed(8)
    with torch.no_grad():
        batched_decoder, per_request_decoder, batched_caches, per_request_caches = _initialize_stateful_batch(mode)
        codes = torch.randint(0, batched_decoder.config.codebook_size, (2, batched_decoder.config.num_quantizers, 25))
        batched = batched_decoder.batched_chunked_decode(codes, [25, 25], caches=batched_caches)
        per_request = _decode_per_request(per_request_decoder, codes, [25, 25], per_request_caches)

    torch.testing.assert_close(torch.stack(batched), per_request, atol=1e-5, rtol=1e-4)
    for batched_cache, per_request_cache in zip(batched_caches, per_request_caches, strict=True):
        assert batched_cache["suffix_frames"] == per_request_cache["suffix_frames"] == 26
        torch.testing.assert_close(batched_cache["suffix_quantized"], per_request_cache["suffix_quantized"])
        torch.testing.assert_close(batched_cache["suffix_conv"], per_request_cache["suffix_conv"])


@pytest.mark.parametrize("mode", ["xvec", "icl"])
def test_batched_stateful_rolling_suffix_matches_per_request(mode):
    torch.manual_seed(9)
    with torch.no_grad():
        batched_decoder, per_request_decoder, batched_caches, per_request_caches = _initialize_stateful_batch(mode)
        for _ in range(3):
            codes = torch.randint(
                0,
                batched_decoder.config.codebook_size,
                (2, batched_decoder.config.num_quantizers, 25),
            )
            batched_decoder.batched_chunked_decode(codes, [25, 25], caches=batched_caches)
            _decode_per_request(per_request_decoder, codes, [25, 25], per_request_caches)

        rolling_codes = torch.randint(
            0,
            batched_decoder.config.codebook_size,
            (2, batched_decoder.config.num_quantizers, 25),
        )
        batched = batched_decoder.batched_chunked_decode(rolling_codes, [25, 25], caches=batched_caches)
        per_request = _decode_per_request(per_request_decoder, rolling_codes, [25, 25], per_request_caches)

    torch.testing.assert_close(torch.stack(batched), per_request, atol=1e-5, rtol=1e-4)
    for batched_cache, per_request_cache in zip(batched_caches, per_request_caches, strict=True):
        assert batched_cache["suffix_frames"] == per_request_cache["suffix_frames"] == 97
        assert batched_cache["suffix_quantized"].shape[-1] == batched_decoder.config.sliding_window
        assert batched_cache["suffix_conv"].shape[1] == batched_decoder.config.sliding_window - 2
        torch.testing.assert_close(batched_cache["suffix_quantized"], per_request_cache["suffix_quantized"])
        torch.testing.assert_close(batched_cache["suffix_conv"], per_request_cache["suffix_conv"])


# -----------------------------------------------------------------------------
# Warm-up, capture, replay, and cache initialization regressions
# -----------------------------------------------------------------------------


def test_segmented_capture_sizes_are_derived_from_decoder_and_chunk_config():
    decoder = _RecordingDecoder()
    decoder.config.sliding_window = 10

    wrapper = CUDAGraphDecoderWrapper(
        decoder,
        initial_chunk_frames=2,
        codec_chunk_frames=4,
        enabled=False,
    )

    assert wrapper.prefix_length == 10
    assert wrapper.icl_capture_sizes == [6, 10, 14]
    assert wrapper._icl_previous_frames_by_target == {6: 2, 10: 6, 14: 10}
    assert wrapper._xvec_previous_frames_by_target == {6: 2, 10: 6, 14: 10}


def test_stateless_capture_sizes_extend_defaults_with_configured_sizes():
    decoder = _RecordingDecoder()
    wrapper = CUDAGraphDecoderWrapper(
        decoder,
        async_chunk=False,
        decode_chunk_size=300,
        decode_left_context=25,
        stateless_capture_sizes=[97, 400],
        enabled=False,
    )

    assert wrapper.stateless_capture_sizes == [97, 150, 325, 400]


def test_capture_uses_cached_conv_lengths():
    decoder = _RecordingDecoder()
    decoder.config.sliding_window = 72
    wrapper = CUDAGraphDecoderWrapper(
        decoder,
        initial_chunk_frames=1,
        codec_chunk_frames=25,
        enabled=False,
    )

    assert [wrapper._cached_conv_frames(frames) for frames in (1, 26, 51, 72)] == [1, 26, 51, 70]


def test_icl_prefix_graph_cache_keeps_physical_prefix_length(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    batch_size = 1
    num_quantizers = 2
    prefix_length = 72
    initial_chunk_frames = 1
    physical_frames = prefix_length + initial_chunk_frames
    total_upsample = 1920

    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.icl_prefix_states = {
        batch_size: {
            "graph": type("_Graph", (), {"replay": lambda self: None})(),
            "input": {
                "codes": torch.zeros(batch_size, num_quantizers, physical_frames),
                "hidden_mask": torch.ones(batch_size, 1, physical_frames),
                "prefix_attention_mask": torch.ones(batch_size, prefix_length, dtype=torch.bool),
                "suffix_attention_mask": torch.ones(batch_size, physical_frames, dtype=torch.bool),
            },
            "output": torch.zeros(batch_size, 1, physical_frames * total_upsample),
            "cache": {
                "decoder_prefix_frames": prefix_length,
                "ref_hidden": torch.zeros(batch_size, 2, prefix_length),
                "ref_conv": torch.zeros(batch_size, prefix_length, 2),
                "prefix_hidden": torch.zeros(batch_size, prefix_length, 2),
                "past_key_values": object(),
                "suffix_quantized": torch.zeros(batch_size, num_quantizers, initial_chunk_frames),
                "suffix_conv": torch.zeros(batch_size, initial_chunk_frames, 2),
            },
        }
    }
    wrapper.prefix_length = prefix_length
    wrapper.initial_chunk_frames = initial_chunk_frames
    wrapper.capture_batch_sizes = [1]
    wrapper.decoder = type(
        "_Decoder",
        (),
        {
            "total_upsample": total_upsample,
            "_slice_dynamic_cache": staticmethod(lambda cache, row: cache),
        },
    )()
    wrapper._ensure_suffix_buffers = lambda caches: None

    caches = {"prefix_frames": 48}
    codes = torch.zeros(batch_size, num_quantizers, caches["prefix_frames"] + initial_chunk_frames)
    outputs = wrapper._decode_icl_prefix_batch([codes], [caches])

    assert outputs is not None
    output = outputs[0]
    assert caches["prefix_frames"] == 48
    assert caches["prefix_pad_frames"] == 24
    assert output.shape == (batch_size, 1, initial_chunk_frames * total_upsample)
    assert caches["decoder_prefix_frames"] == prefix_length


def test_xvec_dummy_cache_is_initialized_before_graph_capture():
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=32,
        hidden_size=16,
        latent_dim=16,
        codebook_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=32,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        sliding_window=72,
    )
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.decoder = SimpleNamespace(config=config)

    cache = wrapper._make_dummy_xvec_cache(2, torch.device("cpu"), torch.float32)["past_key_values"]

    assert cache.get_seq_length() == 0
    assert all(layer.is_initialized for layer in cache.layers)
    assert all(layer.keys.device.type == "cpu" for layer in cache.layers)


def test_batched_suffix_graph_updates_ping_pong_buffers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.codec_chunk_frames = 25
    wrapper._icl_previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper._xvec_previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper.capture_batch_sizes = [1, 2]
    key = ("icl", 2, 26)
    graph_output = torch.stack(
        [
            torch.arange(50, dtype=torch.float32).view(1, -1),
            torch.arange(50, dtype=torch.float32).view(1, -1) + 100,
        ]
    )
    graph_next_quantized = torch.stack([torch.full((2, 26), 1.0), torch.full((2, 26), 2.0)])
    graph_next_conv = torch.stack([torch.full((26, 3), 3.0), torch.full((26, 3), 4.0)])
    wrapper.combined_states = {
        key: {
            "graph": type("_Graph", (), {"replay": lambda self: None})(),
            "input": {
                "codes": torch.zeros(2, 2, 25),
                "quantized": torch.zeros(2, 2, 1),
                "conv": torch.zeros(2, 1, 3),
                "mask": torch.ones(2, 98, dtype=torch.bool),
            },
            "output": (graph_output, graph_next_quantized, graph_next_conv),
            "cache": {},
        }
    }
    wrapper.decoder = type("_Decoder", (), {"total_upsample": 2})()
    monkeypatch.setattr(wrapper, "_copy_request_caches_to_batch", lambda caches, static: None)

    caches = [
        {
            "suffix_quantized": torch.zeros(1, 2, 1),
            "suffix_conv": torch.zeros(1, 1, 3),
            "suffix_frames": 1,
        }
        for _ in range(2)
    ]
    codes = [torch.zeros(1, 2, 26) for _ in range(2)]

    outputs = wrapper._decode_suffix_batch("icl", 26, codes, caches, [25, 25])

    assert outputs is not None
    assert [cache["_suffix_buffer_index"] for cache in caches] == [1, 1]
    assert caches[0]["suffix_quantized"].data_ptr() == caches[0]["_suffix_quantized_buffers"][1].data_ptr()
    assert caches[1]["suffix_conv"].data_ptr() == caches[1]["_suffix_conv_buffers"][1].data_ptr()
    torch.testing.assert_close(caches[0]["suffix_quantized"], torch.full((1, 2, 26), 1.0))
    torch.testing.assert_close(caches[1]["suffix_quantized"], torch.full((1, 2, 26), 2.0))
    torch.testing.assert_close(caches[0]["suffix_conv"], torch.full((1, 26, 3), 3.0))
    torch.testing.assert_close(caches[1]["suffix_conv"], torch.full((1, 26, 3), 4.0))


def test_dummy_request_does_not_record_pre_capture_shape_fallback():
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.codec_chunk_frames = 25
    wrapper._icl_previous_frames_by_target = {26: 1}
    wrapper._xvec_previous_frames_by_target = {26: 1}
    wrapper.xvec_prefix_states = {}
    wrapper._stats_enabled = True
    wrapper._stats_total_requests = 0
    wrapper._stats_hit_requests = 0
    wrapper._stats_fallback_requests = 0
    wrapper._stats_replays = Counter()
    wrapper._stats_fallbacks = Counter()
    wrapper._maybe_log_stats = lambda: None
    wrapper.decoder = _decoder_stub(
        _decode_xvec_first_chunk=lambda codes, _cache: codes[:, :1, :],
    )

    wrapper._batched_request_decode(
        [torch.zeros(1, 2, 4)],
        [{"prefix_frames": 0, "_is_dummy_run": True}],
    )

    assert wrapper._stats_total_requests == 0
    assert wrapper._stats_fallback_requests == 0
    assert wrapper._stats_fallbacks == Counter()
    assert wrapper._suppress_stats is False


def test_xvec_first_chunk_replays_prefix_graph(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.codec_chunk_frames = 25
    wrapper.capture_batch_sizes = [1, 2]
    wrapper._icl_previous_frames_by_target = {26: 1}
    wrapper._xvec_previous_frames_by_target = {26: 1}
    replayed = []
    wrapper.xvec_prefix_states = {
        2: {
            "graph": SimpleNamespace(replay=lambda: replayed.append(True)),
            "input": {"codes": torch.zeros(2, 2, 1)},
            "output": torch.arange(4, dtype=torch.float32).view(2, 1, 2),
            "cache": {
                "ref_hidden": torch.zeros(2, 2, 0),
                "ref_conv": torch.zeros(2, 0, 3),
                "prefix_hidden": torch.zeros(2, 0, 3),
                "ref_upsample": torch.zeros(2, 3, 0),
                "ref_wav": torch.zeros(2, 1, 0),
                "suffix_quantized": torch.ones(2, 2, 1),
                "suffix_conv": torch.ones(2, 1, 3),
                "past_key_values": SimpleNamespace(layers=[]),
            },
        }
    }
    wrapper._record_graph_hit = lambda *_args: None
    wrapper._record_graph_fallback = lambda *_args: None
    wrapper._ensure_suffix_buffers = lambda cache: None
    wrapper.decoder = _decoder_stub(
        _is_suffix_cache_rolling=lambda previous, cached: False,
        _decode_xvec_first_chunk=lambda *_args: pytest.fail("unexpected eager fallback"),
        _slice_dynamic_cache=Qwen3TTSTokenizerV2Decoder._slice_dynamic_cache,
    )

    caches = [{"prefix_frames": 0}, {"prefix_frames": 0}]
    outputs = wrapper._batched_request_decode(
        [torch.full((1, 2, 1), 3), torch.full((1, 2, 1), 4)],
        caches,
    )

    assert replayed == [True]
    torch.testing.assert_close(wrapper.xvec_prefix_states[2]["input"]["codes"][0], torch.full((2, 1), 3.0))
    torch.testing.assert_close(wrapper.xvec_prefix_states[2]["input"]["codes"][1], torch.full((2, 1), 4.0))
    assert [cache["decoder_prefix_frames"] for cache in caches] == [0, 0]
    assert [cache["suffix_frames"] for cache in caches] == [1, 1]
    assert len(outputs) == 2


# -----------------------------------------------------------------------------
# Routing, phase grouping, fallback, and batch splitting
# -----------------------------------------------------------------------------


def test_eager_fallback_uses_dynamic_decoder_prefix_length():
    decoder = _RecordingDecoder()
    wrapper = CUDAGraphDecoderWrapper(decoder, enabled=False)
    codes = torch.zeros(1, 2, 2)

    output = wrapper._decode_request_fallback(
        codes,
        {
            "prefix_frames": 48,
            "decoder_prefix_frames": 48,
            "suffix_quantized": torch.zeros(1, 2, 1),
        },
    )

    assert output is codes
    assert decoder.prefix_frames == 48


def test_decode_modes_enforce_cache_contracts():
    decoder = _RecordingDecoder()
    async_wrapper = CUDAGraphDecoderWrapper(decoder, async_chunk=True, enabled=False)
    stateless_wrapper = CUDAGraphDecoderWrapper(decoder, async_chunk=False, enabled=False)
    codes = torch.zeros(1, 2, 1)

    with pytest.raises(ValueError, match="caches are required"):
        async_wrapper.chunked_decode_with_cudagraph(codes, caches=None)
    with pytest.raises(ValueError, match="caches must be None"):
        stateless_wrapper.chunked_decode_with_cudagraph(codes, caches={})


def test_single_request_xvec_first_chunk_uses_batched_graph_router(monkeypatch):
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.async_chunk = True
    wrapper.decoder = _decoder_stub(total_upsample=1)
    codes = torch.zeros(1, 2, 1)
    cache = {"prefix_frames": 0}
    calls = []

    def _batched_request_decode(codes_list, request_caches):
        calls.append((codes_list, request_caches))
        request_caches[0]["suffix_quantized"] = codes_list[0]
        return [codes_list[0][:, :1]]

    monkeypatch.setattr(wrapper, "_batched_request_decode", _batched_request_decode)

    output = wrapper.chunked_decode_with_cudagraph(codes, caches=cache)

    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0][0], codes)
    assert len(calls[0][1]) == 1
    assert calls[0][1][0] is cache
    torch.testing.assert_close(output, codes[:, :1])


def test_single_request_suffix_uses_batched_graph_router(monkeypatch):
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.async_chunk = True
    wrapper.decoder = _decoder_stub(total_upsample=1)
    codes = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    cache = {
        "prefix_frames": 0,
        "suffix_frames": 1,
        "suffix_quantized": torch.zeros(1, 2, 1),
    }
    calls = []

    def _batched_request_decode(codes_list, request_caches):
        calls.append((codes_list, request_caches))
        return [codes_list[0][:, :1]]

    monkeypatch.setattr(wrapper, "_batched_request_decode", _batched_request_decode)

    output = wrapper.chunked_decode_with_cudagraph(codes, caches=cache)

    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0][0], codes)
    assert calls[0][1][0] is cache
    torch.testing.assert_close(output, codes[:, :1])


def test_stateless_variable_lengths_share_capture_bucket(monkeypatch):
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.stateless_capture_sizes = [25]
    wrapper.capture_batch_sizes = [1, 2]
    wrapper.stateless_states = {(2, 25): {}}
    wrapper.decoder = _decoder_stub(total_upsample=2)
    calls = []

    def _decode_stateless(codes_chunk, *, clone_graph_output):
        calls.append((codes_chunk.clone(), clone_graph_output))
        return codes_chunk[:, :1].repeat_interleave(2, dim=-1).float()

    monkeypatch.setattr(wrapper, "_decode_stateless", _decode_stateless)
    codes = torch.arange(100).reshape(2, 2, 25)

    output = wrapper._batched_stateless_chunked_decode(
        codes,
        [25, 12],
        chunk_size=25,
        left_context_size=0,
        max_batch_size=0,
    )

    assert len(calls) == 1
    assert calls[0][0].shape == (2, 2, 25)
    assert calls[0][1] is False
    torch.testing.assert_close(output[0], codes[0, :1].repeat_interleave(2, dim=-1).float())
    torch.testing.assert_close(output[1], codes[1, :1, :12].repeat_interleave(2, dim=-1).float())


def test_batched_chunked_decode_groups_exact_phases(monkeypatch):
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.async_chunk = True
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.codec_chunk_frames = 25
    wrapper._icl_previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper._xvec_previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper.decoder = _decoder_stub(_is_suffix_cache_rolling=lambda previous, cached: previous >= 72 and cached == 72)
    calls: list[tuple[str, int, int]] = []

    def _icl_prefix(codes, caches):
        calls.append(("icl_prefix", 0, len(codes)))
        return [torch.full((1, 1, 1), 1.0) for _ in codes]

    def _suffix(mode, target, codes, caches, new_frames):
        calls.append((f"suffix:{mode}", target, len(codes)))
        return [torch.full((1, 1, 1), float(target)) for _ in codes]

    monkeypatch.setattr(wrapper, "_decode_icl_prefix_batch", _icl_prefix)
    monkeypatch.setattr(wrapper, "_decode_suffix_batch", _suffix)

    class _KV:
        @staticmethod
        def get_seq_length():
            return 72

    prefix_frames = 48
    codes_list = [torch.zeros(1, 2, prefix_frames + 1)]
    caches = [{"prefix_frames": prefix_frames}]
    for previous, _suffix_frames, cached_frames in (
        (1, 26, 1),
        (26, 51, 26),
        (51, 76, 51),
        (76, 97, 72),
    ):
        codes_list.append(torch.zeros(1, 2, 25))
        caches.append(
            {
                "prefix_frames": prefix_frames,
                "decoder_prefix_frames": 72,
                "ref_wav": torch.zeros(1, 1, 1),
                "past_key_values": _KV(),
                "suffix_frames": previous,
                "suffix_quantized": torch.zeros(1, 2, cached_frames),
            }
        )

    lengths = [codes.shape[-1] for codes in codes_list]
    padded_codes = torch.zeros(len(codes_list), 2, max(lengths))
    for row, codes in enumerate(codes_list):
        padded_codes[row, :, : codes.shape[-1]] = codes[0]
    outputs = wrapper.batched_chunked_decode_with_cudagraph(padded_codes, lengths, caches=caches)

    assert calls == [
        ("icl_prefix", 0, 1),
        ("suffix:icl", 26, 1),
        ("suffix:icl", 51, 1),
        ("suffix:icl", 76, 1),
        ("suffix:icl", 97, 1),
    ]
    assert [int(output.item()) for output in outputs] == [1, 26, 51, 76, 97]


def test_xvec_delta_chunks_route_through_all_reachable_graph_phases(monkeypatch):
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.async_chunk = True
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.codec_chunk_frames = 25
    wrapper._icl_previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper._xvec_previous_frames_by_target = {26: 1, 51: 26, 76: 51, 97: 72}
    wrapper.decoder = _decoder_stub(_is_suffix_cache_rolling=lambda previous, cached: previous >= 72 and cached == 72)
    calls: list[tuple[str, int]] = []

    def _suffix(mode, target, codes, caches, new_frames):
        calls.append((mode, target))
        return [torch.zeros(1, 1, 1) for _ in codes]

    monkeypatch.setattr(wrapper, "_decode_suffix_batch", _suffix)

    class _EmptyKV:
        @staticmethod
        def get_seq_length():
            return 0

    caches = [
        {
            "prefix_frames": 0,
            "decoder_prefix_frames": 0,
            "ref_wav": torch.zeros(1, 1, 0),
            "past_key_values": _EmptyKV(),
            "suffix_frames": previous,
            "suffix_quantized": torch.zeros(1, 2, cached_frames),
        }
        for previous, cached_frames in ((1, 1), (26, 26), (51, 51), (76, 72))
    ]
    lengths = [25, 25, 25, 25]
    codes = torch.zeros(4, 2, 25)

    wrapper.batched_chunked_decode_with_cudagraph(codes, lengths, caches=caches)

    assert calls == [("xvec", 26), ("xvec", 51), ("xvec", 76), ("xvec", 97)]


def test_graph_batch_overflow_splits_at_largest_bucket():
    assert CUDAGraphDecoderWrapper._split_for_graph_buckets(16, {1, 2, 4, 8}) == [(0, 8), (8, 16)]
    assert CUDAGraphDecoderWrapper._split_for_graph_buckets(10, {1, 2, 4, 8}) == [(0, 8), (8, 10)]


def test_missing_graph_phase_falls_back_instead_of_returning_empty(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.icl_prefix_states = {}
    wrapper.combined_states = {}
    wrapper._record_graph_fallback = lambda *_args: None

    codes = [torch.zeros(1, 2, 25)]
    caches = [{}]

    assert wrapper._decode_icl_prefix_batch(codes, caches) is None
    assert wrapper._decode_suffix_batch("xvec", 50, codes, caches, [25]) is None


def test_xvec_first_chunk_uses_prefixless_initializer(monkeypatch):
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = 1
    wrapper.codec_chunk_frames = 25
    wrapper._icl_previous_frames_by_target = {26: 1}
    wrapper._xvec_previous_frames_by_target = {26: 1}
    calls = []
    wrapper.decoder = _decoder_stub(
        _decode_xvec_first_chunk=lambda codes, cache: calls.append("xvec") or codes[:, :1, :],
    )

    output = wrapper._batched_request_decode(
        [torch.zeros(1, 2, 25)],
        [{"prefix_frames": 0}],
    )

    assert calls == ["xvec"]
    assert output[0].shape == (1, 1, 25)


def test_eager_backend_max_batch_size_splits_each_phase_group():
    decoder = _decoder_stub()
    batch_sizes = []

    def decode_xvec(codes, _caches):
        batch_sizes.append(len(codes))
        return [code[:, :1, :].clone() for code in codes]

    outputs = decoder.batched_request_decode(
        [torch.zeros(1, 2, 1) for _ in range(5)],
        [{"prefix_frames": 0} for _ in range(5)],
        prefix_length=72,
        initial_chunk_frames=1,
        codec_chunk_frames=25,
        transitions_by_mode={"icl": {26: 1}, "xvec": {26: 1}},
        decode_icl_prefix_batch=None,
        decode_xvec_prefix_batch=decode_xvec,
        decode_suffix_batch=None,
        decode_fallback=lambda *_args: pytest.fail("unexpected per-request fallback"),
        backend_max_batch_size=2,
    )

    assert batch_sizes == [2, 2, 1]
    assert len(outputs) == 5


def test_mixed_phase_batch_restores_outputs_and_caches_to_original_slots():
    decoder = _decoder_stub(
        _is_suffix_cache_rolling=lambda _previous, _cached: False,
    )

    class _KV:
        def __init__(self, length):
            self.length = length

        def get_seq_length(self):
            return self.length

    codes_list = [
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2, 25),
        torch.zeros(1, 2, 49),
        torch.zeros(1, 2, 25),
    ]
    caches = [
        {"slot": 0, "prefix_frames": 0},
        {
            "slot": 1,
            "prefix_frames": 0,
            "decoder_prefix_frames": 0,
            "past_key_values": _KV(0),
            "suffix_frames": 26,
            "suffix_quantized": torch.zeros(1, 2, 26),
        },
        {"slot": 2, "prefix_frames": 48},
        {
            "slot": 3,
            "prefix_frames": 48,
            "decoder_prefix_frames": 72,
            "past_key_values": _KV(72),
            "suffix_frames": 1,
            "suffix_quantized": torch.zeros(1, 2, 1),
        },
    ]

    def _outputs_for_slots(group_caches, backend):
        outputs = []
        for cache in group_caches:
            cache["backend"] = backend
            outputs.append(torch.full((1, 1, 1), float(cache["slot"])))
        return outputs

    outputs = decoder.batched_request_decode(
        codes_list,
        caches,
        prefix_length=72,
        initial_chunk_frames=1,
        codec_chunk_frames=25,
        transitions_by_mode={
            "icl": {26: 1, 51: 26, 76: 51, 97: 72},
            "xvec": {26: 1, 51: 26, 76: 51, 97: 72},
        },
        decode_icl_prefix_batch=lambda _codes, group_caches: _outputs_for_slots(group_caches, "icl_prefix"),
        decode_xvec_prefix_batch=lambda _codes, group_caches: _outputs_for_slots(group_caches, "xvec_first"),
        decode_suffix_batch=lambda mode, target, _codes, group_caches, _new_frames: _outputs_for_slots(
            group_caches, f"suffix:{mode}:{target}"
        ),
        decode_fallback=lambda *_args: pytest.fail("unexpected eager fallback"),
    )

    assert [int(output.item()) for output in outputs] == [0, 1, 2, 3]
    assert [cache["backend"] for cache in caches] == [
        "xvec_first",
        "suffix:xvec:51",
        "icl_prefix",
        "suffix:icl:26",
    ]


# -----------------------------------------------------------------------------
# Padding and output trimming regressions
# -----------------------------------------------------------------------------


def test_stateless_replay_pads_to_bucket_and_trims_output(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    class _Graph:
        def replay(self):
            return None

    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.async_chunk = False
    wrapper.enabled = True
    wrapper._warmed_up = True
    wrapper.stateless_capture_sizes = [4]
    wrapper.stateless_states = {
        (1, 4): {
            "graph": _Graph(),
            "input": {"codes": torch.full((1, 2, 4), -1, dtype=torch.long)},
            "output": torch.arange(4, dtype=torch.float32).view(1, 1, 4),
            "cache": None,
        }
    }
    wrapper.decoder = SimpleNamespace(total_upsample=1)
    codes = torch.tensor([[[1, 2], [3, 4]]])

    output = wrapper._decode_stateless(codes, clone_graph_output=True)

    torch.testing.assert_close(output, torch.tensor([[[0.0, 1.0]]]))
    torch.testing.assert_close(
        wrapper.stateless_states[(1, 4)]["input"]["codes"],
        torch.tensor([[[1, 2, 0, 0], [3, 4, 0, 0]]]),
    )


def test_stateful_eager_partial_final_chunk_is_right_trimmed():
    torch.manual_seed(10)
    with torch.no_grad():
        batched_decoder, per_request_decoder, batched_caches, per_request_caches = _initialize_stateful_batch("xvec")
        codes = torch.randint(
            0,
            batched_decoder.config.codebook_size,
            (2, batched_decoder.config.num_quantizers, 25),
        )
        lengths = [25, 12]
        batched = batched_decoder.batched_chunked_decode(codes, lengths, caches=batched_caches)
        per_request = [
            per_request_decoder.chunked_decode(
                codes[row : row + 1, :, :length],
                caches=per_request_caches[row],
            )
            for row, length in enumerate(lengths)
        ]

    full_samples = lengths[0] * batched_decoder.total_upsample
    partial_samples = lengths[1] * batched_decoder.total_upsample
    assert [output.shape[-1] for output in batched] == [full_samples, partial_samples]
    torch.testing.assert_close(batched[0], per_request[0][0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(batched[1], per_request[1][0], atol=1e-5, rtol=1e-4)

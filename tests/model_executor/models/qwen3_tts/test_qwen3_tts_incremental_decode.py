# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import (
    Qwen3TTSCode2Wav,
    _SegmentRhoTracker,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    _CONV_CONTEXT_FRAME,
    _DOWNSTREAM_CONTEXT_FRAME,
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2DecoderTransformerModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.usefixtures("single_threaded_cpu")]

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
    model._streaming_codec = None
    model._decoder_state_cache_warn_entries = 512
    model._logged_codec_stats = True
    model._logged_malformed_codec_lengths = set()
    model._batch_stats_enabled = False
    model._rho_stats = _SegmentRhoTracker(enabled=False)
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
    caches: dict[str, Any] = {"prefix_frames": prefix_frames}

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
    caches: dict[str, Any] = {"prefix_frames": 0}

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
    caches: dict[str, Any] = {"prefix_frames": 0}

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
    caches: dict[str, Any] = {"prefix_frames": prefix_frames}

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
# Routing, phase grouping, fallback, and batch splitting
# -----------------------------------------------------------------------------


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

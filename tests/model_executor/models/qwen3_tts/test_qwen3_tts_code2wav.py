# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import (
    Qwen3TTSCode2Wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NUM_QUANTIZERS = 2
_TOTAL_UPSAMPLE = 4
_OUTPUT_SAMPLE_RATE = 24000


class _FakeDecoder(nn.Module):
    def __init__(self, total_upsample: int = _TOTAL_UPSAMPLE):
        super().__init__()
        self.total_upsample = total_upsample
        self.decode_calls: list[dict[str, int]] = []
        self.batched_decode_calls: list[dict[str, object]] = []
        self.decode_codes: list[torch.Tensor] = []
        self.cudagraph_calls: list[dict[str, int | torch.device]] = []

    def to(self, *args, **kwargs):
        return self

    def chunked_decode(
        self,
        codes: torch.Tensor,
        *,
        caches=None,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        self.decode_codes.append(codes.detach().cpu().clone())
        self.decode_calls.append(
            {
                "chunk_size": chunk_size,
                "left_context_size": left_context_size,
                "codes_shape": tuple(codes.shape),
            }
        )
        batch = codes.shape[0]
        frames = codes.shape[-1]
        wav_len = frames * self.total_upsample + 6
        wav = torch.arange(wav_len, dtype=torch.float32).view(1, 1, -1)
        offsets = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1) * 1000
        return wav.expand(batch, 1, wav_len) + offsets

    def batched_chunked_decode(
        self,
        codes: torch.Tensor,
        lengths: list[int],
        *,
        caches=None,
        chunk_size: int = 300,
        left_context_size: int = 25,
        max_batch_size: int = 0,
    ) -> torch.Tensor:
        self.batched_decode_calls.append(
            {
                "chunk_size": chunk_size,
                "left_context_size": left_context_size,
                "max_batch_size": max_batch_size,
                "codes_shape": tuple(codes.shape),
                "lengths": tuple(lengths),
                "caches": caches,
            }
        )
        if caches is not None:
            outputs = []
            for row, (length, cache) in enumerate(zip(lengths, caches, strict=True)):
                output = self.chunked_decode(
                    codes[row : row + 1, :, :length],
                    caches=cache,
                    chunk_size=chunk_size,
                    left_context_size=left_context_size,
                )
                outputs.append(output[0])
            return outputs

        batch = codes.shape[0]
        frames = codes.shape[-1]
        wav_len = frames * self.total_upsample + 6
        wav = torch.arange(wav_len, dtype=torch.float32).view(1, 1, -1)
        offsets = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1) * 1000
        outputs = wav.expand(batch, 1, wav_len) + offsets
        return [outputs[row, :, : lengths[row] * self.total_upsample] for row in range(batch)]

    def enable_cudagraph(self, **kwargs):
        self.cudagraph_calls.append(kwargs)


def _fake_dec_config():
    return SimpleNamespace(
        num_quantizers=_NUM_QUANTIZERS,
        sliding_window=0,
    )


def _make_model(
    *,
    stage_connector_config=None,
    async_chunk: bool = False,
    device: torch.device | None = None,
) -> Qwen3TTSCode2Wav:
    dec_config = _fake_dec_config()
    tok_config = SimpleNamespace(
        decoder_config=dec_config,
        output_sample_rate=_OUTPUT_SAMPLE_RATE,
    )
    with (
        patch(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.Qwen3TTSTokenizerV2Config.from_pretrained",
            return_value=tok_config,
        ),
        patch(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.Qwen3TTSTokenizerV2Decoder._from_config",
            return_value=_FakeDecoder(),
        ),
    ):
        model = Qwen3TTSCode2Wav(
            vllm_config=SimpleNamespace(
                load_config=SimpleNamespace(),
                model_config=SimpleNamespace(
                    model="unused",
                    revision=None,
                    stage_connector_config=stage_connector_config,
                    async_chunk=async_chunk,
                ),
                device_config=SimpleNamespace(device=device or torch.device("cpu")),
            )
        )
    return model


def _load_weights_noop(model: Qwen3TTSCode2Wav) -> set[str]:
    class _FakeModelLoader:
        class Source:
            def __init__(self, **_: object):
                pass

        def __init__(self, _load_config: object):
            pass

        def _get_weights_iterator(self, _source: object):
            return iter(())

    class _FakeAutoWeightsLoader:
        def __init__(self, *_: object, **__: object):
            pass

        def load_weights(self, _weights: object) -> set[str]:
            return {"decoder.fake_weight"}

    with (
        patch(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.DefaultModelLoader",
            _FakeModelLoader,
        ),
        patch(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.AutoWeightsLoader",
            _FakeAutoWeightsLoader,
        ),
    ):
        return model.load_weights(iter(()))


def test_forward_uses_decoder_audio_contract_on_exact_frame_boundaries():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"meta": {"left_context_size": 2}}],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    expected = torch.arange(8, 24, dtype=torch.float32)
    torch.testing.assert_close(audio, expected)


def test_forward_uses_decoder_audio_contract_without_context():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"meta": {"left_context_size": 0}}],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    expected = torch.arange(24, dtype=torch.float32)
    torch.testing.assert_close(audio, expected)


@pytest.mark.parametrize("skipped_length", [0, 1], ids=["empty", "malformed"])
def test_stateless_batch_uses_original_request_index_for_left_context(skipped_length):
    model = _make_model()

    skipped_request = torch.zeros(skipped_length, dtype=torch.long)
    valid_request = torch.tensor([9, 8, 1, 2, 9, 8, 3, 4], dtype=torch.long)
    out = model.forward(
        input_ids=torch.cat([skipped_request, valid_request]),
        seq_token_counts=[skipped_length, valid_request.numel()],
        runtime_additional_information=[
            {
                "codes": {"audio": torch.zeros(skipped_length, dtype=torch.float32)},
                "meta": {"left_context_size": 0},
            },
            {
                "codes": {"audio": torch.zeros(8, dtype=torch.float32)},
                "meta": {"left_context_size": 2},
            },
        ],
    )
    wav_output = out.multimodal_outputs["model_outputs"]
    torch.testing.assert_close(wav_output[0], torch.empty(0))
    torch.testing.assert_close(wav_output[1], torch.arange(8, 16, dtype=torch.float32))


def test_stateless_forward_trims_reference_prefix_from_decoder_output():
    model = _make_model()

    def _incremental_decode(codes, lengths, *, caches, **_kwargs):
        return torch.arange(16, dtype=torch.float32).view(1, 1, -1)

    model.decoder.batched_chunked_decode = _incremental_decode
    out = model.forward(
        input_ids=torch.arange(8, dtype=torch.long),
        runtime_additional_information=[
            {
                "meta": {
                    "left_context_size": 2,
                    "ref_context_size": 2,
                    "ref_context_request_id": "rid",
                    "ref_context_included": True,
                }
            }
        ],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    torch.testing.assert_close(audio, torch.arange(8, 16, dtype=torch.float32))


def test_request_aware_forward_passes_only_delta_frames_to_decoder():
    model = _make_model(async_chunk=True)

    model.forward(
        input_ids=torch.arange(6, dtype=torch.long),
        runtime_additional_information=[{"meta": {"request_id": "rid", "left_context_size": 0}}],
    )
    model.forward(
        input_ids=torch.arange(4, dtype=torch.long),
        runtime_additional_information=[{"meta": {"request_id": "rid", "left_context_size": 0}}],
    )

    assert [tuple(codes.shape) for codes in model.decoder.decode_codes[-2:]] == [
        (1, _NUM_QUANTIZERS, 3),
        (1, _NUM_QUANTIZERS, 2),
    ]


def test_four_frame_first_chunk_uses_configured_initial_size_without_ramp():
    model = _make_model(
        async_chunk=True,
        stage_connector_config={
            "extra": {
                "codec_chunk_frames": 25,
                "initial_codec_chunk_frames": 4,
            }
        },
    )
    _load_weights_noop(model)

    model.forward(
        input_ids=torch.arange(8, dtype=torch.long),
        runtime_additional_information=[{"meta": {"request_id": "four-frame-request", "left_context_size": 0}}],
    )

    assert model.decoder._initial_codec_chunk_frames == 4
    assert model.decoder._incremental_chunk_ramp == []
    assert tuple(model.decoder.decode_codes[-1].shape) == (1, _NUM_QUANTIZERS, 4)


def test_connector_codec_chunking_does_not_override_decode_chunking():
    model = _make_model(
        async_chunk=True,
        stage_connector_config={
            "extra": {
                "codec_chunk_frames": 25,
                "codec_left_context_frames": 72,
            }
        },
    )

    loaded = _load_weights_noop(model)

    assert loaded == {"decoder.fake_weight"}
    assert model._decode_chunk_frames == 300
    assert model._decode_left_context_frames == 25

    model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"meta": {"request_id": "rid", "left_context_size": 0}}],
    )

    assert model.decoder.decode_calls[-1] == {
        "chunk_size": 300,
        "left_context_size": 25,
        "codes_shape": (1, _NUM_QUANTIZERS, 6),
    }


def test_decode_chunking_can_be_overridden_separately():
    model = _make_model(
        async_chunk=True,
        stage_connector_config={
            "extra": {
                "codec_chunk_frames": 25,
                "codec_left_context_frames": 72,
                "decode_chunk_frames": 400,
                "decode_left_context_frames": 17,
            }
        },
    )

    _load_weights_noop(model)

    assert model._decode_chunk_frames == 400
    assert model._decode_left_context_frames == 17


def test_malformed_codec_length_warning_is_rate_limited():
    model = _make_model()

    with patch("vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.logger.warning") as warning:
        out1 = model.forward(input_ids=torch.arange(1, dtype=torch.long))
        out2 = model.forward(input_ids=torch.arange(1, dtype=torch.long))

    assert warning.call_count == 1
    assert "not divisible by num_quantizers" in warning.call_args[0][0]
    assert "suppressing repeats" in warning.call_args[0][0]
    assert out1.multimodal_outputs["model_outputs"][0].numel() == 0
    assert out2.multimodal_outputs["model_outputs"][0].numel() == 0


def test_forward_emits_zero_samples_for_empty_codec_payload():
    """Consumer side of #4463.

    The empty-but-finished payload that ``talker2code2wav_full_payload`` now
    returns on a degenerate take carries a zero-length ``codes.audio``. Code2Wav
    must turn that into a 0-sample output (finishing the request) so the Stage-1
    wait gate releases, rather than stalling the pipeline to the connector
    timeout.
    """
    model = _make_model()

    out = model.forward(input_ids=torch.zeros(0, dtype=torch.long))

    assert out.multimodal_outputs["model_outputs"][0].numel() == 0


def test_dummy_runtime_information_provides_finished_request_state():
    model = _make_model(async_chunk=True)

    runtime_info = model.get_dummy_runtime_additional_information(2)

    assert len(runtime_info) == 2
    assert [info["meta"]["request_id"] for info in runtime_info] == [
        "__qwen3_tts_dummy_run__0",
        "__qwen3_tts_dummy_run__1",
    ]
    assert all(info["meta"]["finished"] for info in runtime_info)


def test_dummy_request_state_is_marked_for_graph_stats_suppression():
    model = _make_model(async_chunk=True)
    runtime_info = model.get_dummy_runtime_additional_information(1)

    model.forward(
        input_ids=torch.arange(4, dtype=torch.long),
        runtime_additional_information=runtime_info,
    )

    cache = model.decoder.batched_decode_calls[0]["caches"][0]
    assert cache["_is_dummy_run"] is True
    assert model._decoder_state_cache == {}


def test_decoder_state_cache_capacity_warns_without_evicting_or_failing():
    model = _make_model(async_chunk=True)
    assert model._decoder_state_cache_warn_entries == 512
    model._decoder_state_cache.update({f"active-{index}": {"prefix_frames": 0} for index in range(512)})

    with patch("vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.logger.warning_once") as warning:
        model.forward(
            input_ids=torch.arange(4, dtype=torch.long),
            runtime_additional_information=[{"meta": {"request_id": "new-request"}}],
        )

    warning.assert_called_once()
    assert len(model._decoder_state_cache) == 513
    assert "new-request" in model._decoder_state_cache


def test_decoder_state_cache_uses_scheduler_request_id_for_finished_cleanup():
    model = _make_model(async_chunk=True)

    model.forward(
        input_ids=torch.arange(4, dtype=torch.long),
        request_ids=["internal-request-id"],
        runtime_additional_information=[{"meta": {"request_id": "external-request-id"}}],
    )

    assert "internal-request-id" in model._decoder_state_cache
    assert "external-request-id" not in model._decoder_state_cache

    model.on_requests_finished(["internal-request-id"])

    assert model._decoder_state_cache == {}


def test_forward_batches_equal_length_requests_in_one_decoder_call():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        seq_token_counts=[6, 6],
        runtime_additional_information=[
            {"meta": {"left_context_size": 0}},
            {"meta": {"left_context_size": 1}},
        ],
    )

    assert model.decoder.decode_calls == []
    assert model.decoder.batched_decode_calls == [
        {
            "chunk_size": 300,
            "left_context_size": 25,
            "max_batch_size": 0,
            "codes_shape": (2, _NUM_QUANTIZERS, 3),
            "lengths": (3, 3),
            "caches": None,
        }
    ]
    audios = out.multimodal_outputs["model_outputs"]
    torch.testing.assert_close(audios[0], torch.arange(12, dtype=torch.float32))
    torch.testing.assert_close(audios[1], torch.arange(1004, 1012, dtype=torch.float32))


def test_forward_batches_request_aware_decoder_states():
    model = _make_model(async_chunk=True)
    calls = []

    def _batched_chunked_decode(codes, lengths, caches, **kwargs):
        calls.append((codes, lengths, caches))
        return [torch.arange(16, dtype=torch.float32).view(1, -1) + row * 100 for row in range(2)]

    model.decoder.batched_chunked_decode = _batched_chunked_decode
    per_request = torch.tensor([9, 8, 1, 2, 9, 8, 3, 4], dtype=torch.long)
    out = model.forward(
        input_ids=torch.cat([per_request, per_request]),
        seq_token_counts=[8, 8],
        runtime_additional_information=[
            {
                "meta": {
                    "left_context_size": 2,
                    "ref_context_size": 2,
                    "ref_context_request_id": request_id,
                    "request_id": request_id,
                    "ref_context_included": True,
                }
            }
            for request_id in ("a", "b")
        ],
    )

    assert len(calls) == 1
    assert tuple(calls[0][0].shape) == (2, 2, 4)
    assert calls[0][1] == [4, 4]
    assert [cache["prefix_frames"] for cache in calls[0][2]] == [2, 2]
    audios = out.multimodal_outputs["model_outputs"]
    torch.testing.assert_close(audios[0], torch.arange(16, dtype=torch.float32))
    torch.testing.assert_close(audios[1], torch.arange(100, 116, dtype=torch.float32))


def test_forward_allows_followup_delta_without_reference_prefix():
    model = _make_model(async_chunk=True)
    first_codes = torch.tensor([9, 8, 1, 2, 9, 8, 3, 4], dtype=torch.long)
    model.forward(
        input_ids=first_codes,
        runtime_additional_information=[
            {
                "meta": {
                    "request_id": "rid",
                    "ref_context_request_id": "rid",
                    "ref_context_size": 2,
                    "ref_context_included": True,
                }
            }
        ],
    )

    model.forward(
        input_ids=torch.tensor([5, 6, 7, 15, 16, 17], dtype=torch.long),
        runtime_additional_information=[
            {
                "meta": {
                    "request_id": "rid",
                    "ref_context_request_id": "rid",
                    "ref_context_size": 2,
                    "ref_context_included": False,
                }
            }
        ],
    )

    assert [tuple(codes.shape) for codes in model.decoder.decode_codes[-2:]] == [
        (1, _NUM_QUANTIZERS, 4),
        (1, _NUM_QUANTIZERS, 3),
    ]


def test_forward_passes_variable_length_padded_batch_to_decoder():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        seq_token_counts=[4, 8],
        runtime_additional_information=[
            {"meta": {"left_context_size": 0}},
            {"meta": {"left_context_size": 2}},
        ],
    )

    assert model.decoder.decode_calls == []
    assert model.decoder.batched_decode_calls == [
        {
            "chunk_size": 300,
            "left_context_size": 25,
            "max_batch_size": 0,
            "codes_shape": (2, _NUM_QUANTIZERS, 4),
            "lengths": (2, 4),
            "caches": None,
        }
    ]
    audios = out.multimodal_outputs["model_outputs"]
    torch.testing.assert_close(audios[0], torch.arange(8, dtype=torch.float32))
    torch.testing.assert_close(audios[1], torch.arange(1008, 1016, dtype=torch.float32))


def test_decode_chunking_override_is_passed_to_cudagraph():
    model = _make_model(
        async_chunk=True,
        device=torch.device("cuda"),
        stage_connector_config={
            "extra": {
                "codec_chunk_frames": 25,
                "codec_left_context_frames": 72,
                "decode_chunk_frames": 400,
                "decode_left_context_frames": 17,
            }
        },
    )

    _load_weights_noop(model)

    assert model.decoder.cudagraph_calls[-1] == {
        "capture_batch_sizes": None,
        "stateless_capture_sizes": None,
        "device": torch.device("cuda"),
        "codec_chunk_frames": 25,
        "codec_left_context_frames": 72,
        "initial_codec_chunk_frames": 1,
        "codec_chunk_ramp": None,
        "async_chunk": True,
        "decode_chunk_size": 400,
        "decode_left_context": 17,
    }


def test_cudagraph_batch_config_is_preserved():
    model = _make_model(
        async_chunk=True,
        device=torch.device("cuda"),
        stage_connector_config={
            "extra": {
                "decode_cudagraph_batch_sizes": [1, 2, 4, 8],
            }
        },
    )

    _load_weights_noop(model)

    call = model.decoder.cudagraph_calls[-1]
    assert call["capture_batch_sizes"] == [1, 2, 4, 8]


def test_stateless_cudagraph_capture_sizes_are_passed_from_config():
    model = _make_model(
        async_chunk=False,
        device=torch.device("cuda"),
        stage_connector_config={
            "extra": {
                "decode_cudagraph_capture_sizes": [97, 400],
            }
        },
    )

    _load_weights_noop(model)

    assert model.decoder.cudagraph_calls[-1]["stateless_capture_sizes"] == [97, 400]


def test_async_cudagraph_ignores_stateless_capture_sizes():
    model = _make_model(
        async_chunk=True,
        device=torch.device("cuda"),
        stage_connector_config={
            "extra": {
                "decode_cudagraph_capture_sizes": [97, 400],
            }
        },
    )

    _load_weights_noop(model)

    assert model.decoder.cudagraph_calls[-1]["stateless_capture_sizes"] is None


def test_decode_tf32_can_be_configured():
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        model = _make_model(
            async_chunk=True,
            device=torch.device("cuda"),
            stage_connector_config={
                "extra": {
                    "decode_enable_tf32": "true",
                }
            },
        )

        _load_weights_noop(model)

        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.set_float32_matmul_precision(old_matmul_precision)


def test_decode_batch_max_size_can_be_configured():
    model = _make_model(
        async_chunk=True,
        stage_connector_config={
            "extra": {
                "decode_batch_max_size": 10,
            }
        },
    )

    _load_weights_noop(model)

    assert model._decode_batch_max_size == 10


def test_invalid_decode_batch_max_size_is_rejected():
    model = _make_model(
        async_chunk=True,
        stage_connector_config={
            "extra": {
                "decode_batch_max_size": -1,
            }
        },
    )

    with pytest.raises(ValueError, match="decode_batch_max_size"):
        _load_weights_noop(model)


def test_invalid_decode_chunking_is_rejected():
    model = _make_model(
        async_chunk=True,
        stage_connector_config={
            "extra": {
                "decode_chunk_frames": 0,
            }
        },
    )

    with pytest.raises(ValueError, match="decode_chunk_frames=0"):
        _load_weights_noop(model)

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
        self.batched_decode_calls: list[dict[str, int]] = []
        self.decode_codes: list[torch.Tensor] = []
        self.cudagraph_calls: list[dict[str, int | torch.device]] = []

    def to(self, *args, **kwargs):
        return self

    def chunked_decode(
        self,
        codes: torch.Tensor,
        *,
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
            }
        )
        batch = codes.shape[0]
        frames = codes.shape[-1]
        wav_len = frames * self.total_upsample + 6
        wav = torch.arange(wav_len, dtype=torch.float32).view(1, 1, -1)
        offsets = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1) * 1000
        return wav.expand(batch, 1, wav_len) + offsets

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


def test_forward_trims_context_on_exact_frame_boundaries():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"meta": {"left_context_size": 2}}],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    expected = torch.arange(8, 24, dtype=torch.float32)
    torch.testing.assert_close(audio, expected)


def test_forward_trims_trailing_padding_without_context():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"meta": {"left_context_size": 0}}],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    expected = torch.arange(24, dtype=torch.float32)
    torch.testing.assert_close(audio, expected)


def test_forward_reuses_cached_ref_context_for_followup_chunk():
    model = _make_model()

    first_codes = torch.tensor(
        [
            9,
            8,
            1,
            2,
            9,
            8,
            3,
            4,
        ],
        dtype=torch.long,
    )
    model.forward(
        input_ids=first_codes,
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

    followup_codes = torch.tensor([5, 6, 7, 15, 16, 17], dtype=torch.long)
    model.forward(
        input_ids=followup_codes,
        runtime_additional_information=[
            {
                "meta": {
                    "left_context_size": 3,
                    "ref_context_size": 2,
                    "ref_context_request_id": "rid",
                    "ref_context_included": False,
                    "finished": torch.tensor(True),
                }
            }
        ],
    )

    torch.testing.assert_close(
        model.decoder.decode_codes[-1],
        torch.tensor(
            [
                [
                    [9, 8, 5, 6, 7],
                    [9, 8, 15, 16, 17],
                ]
            ],
            dtype=torch.long,
        ),
    )
    assert "rid" not in model._ref_context_cache


def test_forward_fails_fast_when_ref_context_cache_is_missing():
    model = _make_model()

    followup_codes = torch.tensor([5, 6, 7, 15, 16, 17], dtype=torch.long)
    with pytest.raises(ValueError, match="Missing Qwen3-TTS ref context cache"):
        model.forward(
            input_ids=followup_codes,
            runtime_additional_information=[
                {
                    "meta": {
                        "left_context_size": 3,
                        "ref_context_size": 2,
                        "ref_context_request_id": "rid",
                        "ref_context_included": False,
                    }
                }
            ],
        )


def test_ref_context_cache_evicts_lru_entries_when_requests_abort():
    model = _make_model()
    model._ref_context_cache_max_entries = 2

    first_codes = torch.tensor([9, 8, 1, 2, 9, 8, 3, 4], dtype=torch.long)
    for rid in ("a", "b", "c"):
        model.forward(
            input_ids=first_codes,
            runtime_additional_information=[
                {
                    "meta": {
                        "left_context_size": 2,
                        "ref_context_size": 2,
                        "ref_context_request_id": rid,
                        "ref_context_included": True,
                    }
                }
            ],
        )

    assert list(model._ref_context_cache) == ["b", "c"]
    assert model._ref_context_cache_bytes == sum(model._tensor_nbytes(t) for t in model._ref_context_cache.values())


def test_ref_context_cache_evicts_to_byte_cap():
    model = _make_model()
    model._ref_context_cache_max_entries = 100
    model._ref_context_cache_max_bytes = 33

    first_codes = torch.tensor([9, 8, 1, 2, 9, 8, 3, 4], dtype=torch.long)
    for rid in ("a", "b"):
        model.forward(
            input_ids=first_codes,
            runtime_additional_information=[
                {
                    "meta": {
                        "left_context_size": 2,
                        "ref_context_size": 2,
                        "ref_context_request_id": rid,
                        "ref_context_included": True,
                    }
                }
            ],
        )

    assert list(model._ref_context_cache) == ["b"]
    assert model._ref_context_cache_bytes == 32


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
        runtime_additional_information=[{"meta": {"left_context_size": 0}}],
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

    assert model.decoder.decode_calls == [
        {
            "chunk_size": 300,
            "left_context_size": 25,
            "codes_shape": (2, _NUM_QUANTIZERS, 3),
        }
    ]
    audios = out.multimodal_outputs["model_outputs"]
    torch.testing.assert_close(audios[0], torch.arange(12, dtype=torch.float32))
    torch.testing.assert_close(audios[1], torch.arange(1004, 1012, dtype=torch.float32))


def test_forward_uses_variable_length_chunk_batching_for_long_bucket_group():
    model = _make_model()
    model._decode_batch_bucket_frames = [4]
    model._decode_variable_chunk_batch_min_frames = 1

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
        }
    ]
    audios = out.multimodal_outputs["model_outputs"]
    torch.testing.assert_close(audios[0], torch.arange(8, dtype=torch.float32))
    torch.testing.assert_close(audios[1], torch.arange(1008, 1016, dtype=torch.float32))


def test_forward_bucket_batches_different_length_requests_and_trims_rows():
    model = _make_model()
    model._decode_batch_bucket_frames = [4]

    out = model.forward(
        input_ids=torch.arange(18, dtype=torch.long),
        seq_token_counts=[4, 6, 8],
        runtime_additional_information=[
            {"meta": {"left_context_size": 0}},
            {"meta": {"left_context_size": 1}},
            {"meta": {"left_context_size": 2}},
        ],
    )

    assert model.decoder.decode_calls == [
        {
            "chunk_size": 300,
            "left_context_size": 25,
            "codes_shape": (3, _NUM_QUANTIZERS, 4),
        }
    ]
    assert model.decoder.batched_decode_calls == []
    audios = out.multimodal_outputs["model_outputs"]
    torch.testing.assert_close(audios[0], torch.arange(8, dtype=torch.float32))
    torch.testing.assert_close(audios[1], torch.arange(1004, 1012, dtype=torch.float32))
    torch.testing.assert_close(audios[2], torch.arange(2008, 2016, dtype=torch.float32))


def test_forward_bucket_pads_only_to_group_max_frame_length():
    model = _make_model()
    model._decode_batch_bucket_frames = [8]

    out = model.forward(
        input_ids=torch.arange(10, dtype=torch.long),
        seq_token_counts=[4, 6],
        runtime_additional_information=[
            {"meta": {"left_context_size": 0}},
            {"meta": {"left_context_size": 0}},
        ],
    )

    assert model.decoder.decode_calls == [
        {
            "chunk_size": 300,
            "left_context_size": 25,
            "codes_shape": (2, _NUM_QUANTIZERS, 3),
        }
    ]
    assert model.decoder.batched_decode_calls == []
    audios = out.multimodal_outputs["model_outputs"]
    torch.testing.assert_close(audios[0], torch.arange(8, dtype=torch.float32))
    torch.testing.assert_close(audios[1], torch.arange(1000, 1012, dtype=torch.float32))


def test_forward_splits_bucket_groups_by_configured_max_batch_size():
    model = _make_model()
    model._decode_batch_bucket_frames = [4]
    model._decode_batch_max_size = 2

    model.forward(
        input_ids=torch.arange(18, dtype=torch.long),
        seq_token_counts=[4, 6, 8],
        runtime_additional_information=[
            {"meta": {"left_context_size": 0}},
            {"meta": {"left_context_size": 0}},
            {"meta": {"left_context_size": 0}},
        ],
    )

    assert model.decoder.decode_calls == [
        {
            "chunk_size": 300,
            "left_context_size": 25,
            "codes_shape": (2, _NUM_QUANTIZERS, 3),
        },
        {
            "chunk_size": 300,
            "left_context_size": 25,
            "codes_shape": (1, _NUM_QUANTIZERS, 4),
        },
    ]
    assert model.decoder.batched_decode_calls == []


def test_forward_does_not_pad_singleton_bucket_group():
    model = _make_model()
    model._decode_batch_bucket_frames = [4]

    model.forward(
        input_ids=torch.arange(4, dtype=torch.long),
        runtime_additional_information=[{"meta": {"left_context_size": 0}}],
    )

    assert model.decoder.decode_calls == [
        {
            "chunk_size": 300,
            "left_context_size": 25,
            "codes_shape": (1, _NUM_QUANTIZERS, 2),
        }
    ]


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
        "capture_sizes": None,
        "capture_batch_sizes": None,
        "extra_capture_shapes": None,
        "compile_shapes": None,
        "device": torch.device("cuda"),
        "codec_chunk_frames": 25,
        "codec_left_context_frames": 72,
        "decode_chunk_size": 400,
        "decode_left_context": 17,
    }


def test_cudagraph_capture_shapes_can_be_configured():
    model = _make_model(
        async_chunk=True,
        device=torch.device("cuda"),
        stage_connector_config={
            "extra": {
                "decode_cudagraph_capture_sizes": "97,325",
                "decode_cudagraph_batch_sizes": [1, 2, 4, 8],
                "decode_cudagraph_extra_capture_shapes": ["3:325", [5, 325]],
            }
        },
    )

    _load_weights_noop(model)

    call = model.decoder.cudagraph_calls[-1]
    assert call["capture_sizes"] == [97, 325]
    assert call["capture_batch_sizes"] == [1, 2, 4, 8]
    assert call["extra_capture_shapes"] == [(3, 325), (5, 325)]


def test_decode_compile_shapes_can_be_configured():
    model = _make_model(
        async_chunk=True,
        device=torch.device("cuda"),
        stage_connector_config={
            "extra": {
                "decode_compile_shapes": ["1:325", [1, 73]],
            }
        },
    )

    _load_weights_noop(model)

    call = model.decoder.cudagraph_calls[-1]
    assert call["compile_shapes"] == [(1, 73), (1, 325)]


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


def test_decode_batch_bucket_frames_can_be_configured():
    model = _make_model(
        async_chunk=True,
        stage_connector_config={
            "extra": {
                "decode_batch_bucket_frames": "73,169",
                "decode_batch_max_size": 10,
                "decode_variable_chunk_batch_min_frames": 512,
            }
        },
    )

    _load_weights_noop(model)

    assert model._decode_batch_bucket_frames == [73, 169]
    assert model._decode_batch_max_size == 10
    assert model._decode_variable_chunk_batch_min_frames == 512


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

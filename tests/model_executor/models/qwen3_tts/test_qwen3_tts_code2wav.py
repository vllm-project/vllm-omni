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
        self.decode_calls.append(
            {
                "chunk_size": chunk_size,
                "left_context_size": left_context_size,
            }
        )
        frames = codes.shape[-1]
        wav_len = frames * self.total_upsample + 6
        wav = torch.arange(wav_len, dtype=torch.float32)
        return wav.view(1, 1, -1)

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
        "device": torch.device("cuda"),
        "codec_chunk_frames": 25,
        "codec_left_context_frames": 72,
        "decode_chunk_size": 400,
        "decode_left_context": 17,
    }


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

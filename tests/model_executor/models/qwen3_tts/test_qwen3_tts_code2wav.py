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

    def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
        frames = codes.shape[-1]
        wav_len = frames * self.total_upsample + 6
        wav = torch.arange(wav_len, dtype=torch.float32)
        return wav.view(1, 1, -1)


def _fake_dec_config():
    return SimpleNamespace(
        num_quantizers=_NUM_QUANTIZERS,
        sliding_window=0,
    )


def _make_model() -> Qwen3TTSCode2Wav:
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
                model_config=SimpleNamespace(model="unused"),
                device_config=SimpleNamespace(device=torch.device("cpu")),
            )
        )
    return model


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

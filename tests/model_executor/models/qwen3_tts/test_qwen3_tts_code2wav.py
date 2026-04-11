# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeDecoder(nn.Module):
    def __init__(self, total_upsample: int = 4):
        super().__init__()
        self.total_upsample = total_upsample

    def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
        frames = codes.shape[-1]
        wav_len = frames * self.total_upsample + 6
        wav = torch.arange(wav_len, dtype=torch.float32)
        return wav.view(1, 1, -1)


def _make_model() -> Qwen3TTSCode2Wav:
    model = Qwen3TTSCode2Wav(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model="unused"),
            device_config=SimpleNamespace(device=torch.device("cpu")),
        )
    )
    model._decoder = _FakeDecoder()
    model._num_quantizers = 2
    model._output_sample_rate = 24000
    model._total_upsample = 4
    model._ensure_speech_tokenizer_loaded = lambda: None
    return model


def test_forward_trims_context_on_exact_frame_boundaries():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"left_context_size": 2}],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    expected = torch.arange(8, 24, dtype=torch.float32)
    torch.testing.assert_close(audio, expected)


def test_forward_trims_trailing_padding_without_context():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"left_context_size": 0}],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    expected = torch.arange(24, dtype=torch.float32)
    torch.testing.assert_close(audio, expected)

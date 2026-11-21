# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Token2Wav emits a config-derived sampling rate under ``SAMPLE_RATE_KEY``."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

# Token2Wav imports the optional `step-audio2` extras (flashcosyvoice / s3tokenizer
# / hyperpyyaml) at module load. Skip instead of erroring at collection time on CI
# images that do not install them (e.g. the CPU core_model runner); a genuine
# import error (e.g. vLLM API drift) still surfaces. See
# examples/offline_inference/step_audio2/README.md.
try:
    from vllm_omni.model_executor.models.step_audio2 import step_audio2_token2wav as token2wav_module
    from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
        StepAudio2Token2WavForConditionalGeneration,
    )
except ModuleNotFoundError as exc:
    pytest.skip(
        f"Step-Audio2 Token2Wav extras not installed ({exc.name})",
        allow_module_level=True,
    )

from vllm_omni.model_executor.models.step_audio2.step_audio2_constants import DEFAULT_TOKEN2WAV_CONFIG
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# An unusual rate proves the emission reads the config instead of a literal.
UNUSUAL_RATE = 22050


class _DummyModelConfig:
    def __init__(self) -> None:
        self.hf_config = SimpleNamespace(
            token2wav_path="/tmp/fake-token2wav",
            token2wav_float16=False,
        )
        self.model = "/tmp/fake-model"

    def get_hidden_size(self) -> int:
        return 8


def _build_model() -> StepAudio2Token2WavForConditionalGeneration:
    vllm_config = SimpleNamespace(
        model_config=_DummyModelConfig(),
        device_config=SimpleNamespace(device="cpu"),
    )
    return StepAudio2Token2WavForConditionalGeneration(vllm_config=vllm_config)


def test_sync_path_emits_config_derived_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _build_model()

    monkeypatch.setattr("os.path.exists", lambda _: True)
    monkeypatch.setattr(
        model.token2wav,
        "forward",
        lambda *args, **kwargs: torch.zeros(1, 16, dtype=torch.float32),
    )
    monkeypatch.setattr(
        token2wav_module,
        "DEFAULT_TOKEN2WAV_CONFIG",
        replace(DEFAULT_TOKEN2WAV_CONFIG, sample_rate=UNUSUAL_RATE),
    )

    out = model.forward(
        input_ids=torch.tensor([1, 2, 3]),
        positions=torch.tensor([0, 1, 2]),
        runtime_additional_information=[{}],
    )

    payload = out.multimodal_outputs
    assert SAMPLE_RATE_KEY in payload
    assert get_sampling_rate(payload) == UNUSUAL_RATE


def test_async_empty_chunk_emits_sample_rate() -> None:
    model = _build_model()

    out = model._forward_async_chunk(
        input_ids=torch.tensor([], dtype=torch.int64),
        runtime_additional_information=[{"meta": {"left_context_size": 1}}],
    )

    assert get_sampling_rate(out.multimodal_outputs) == DEFAULT_TOKEN2WAV_CONFIG.sample_rate

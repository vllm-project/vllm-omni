# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline E2E smoke test for Whisper ASR transcription."""

from __future__ import annotations

import os

import pytest
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("WHISPER_TEST_MODEL", "openai/whisper-large-v3-turbo")
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

_OMNI_RUNNER_PARAMS = [
    pytest.param(
        (MODEL, get_deploy_config_path("whisper.yaml"), {"async_chunk": False}),
        id="whisper_large_v3_turbo",
    ),
]

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.parametrize("omni_runner", _OMNI_RUNNER_PARAMS, indirect=True),
]


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_whisper_offline_transcription(omni_runner: OmniRunner) -> None:
    """Transcribing the public mary_had_lamb audio asset yields expected text."""
    audio, sr = AudioAsset("mary_had_lamb").audio_and_sample_rate
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    outputs = list(
        omni_runner.omni.generate(
            [{"prompt": WHISPER_PROMPT, "multi_modal_data": {"audio": (audio, sr)}}],
            [sampling_params],
        )
    )
    assert len(outputs) == 1
    out = outputs[0]
    req_out = out.request_output if hasattr(out, "request_output") and out.request_output is not None else out
    assert req_out.outputs
    text = req_out.outputs[0].text.strip().lower()
    assert "mary had a little lamb" in text, f"Unexpected transcript: {text!r}"

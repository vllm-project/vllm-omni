# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for MOSS-TTS-Nano single-stage pipeline."""

from __future__ import annotations

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from vllm_omni import Omni

MODEL_NAME = "OpenMOSS-Team/MOSS-TTS-Nano"
SAMPLE_RATE = 48000

DEFAULT_SAMPLING = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    top_k=50,
    max_tokens=4096,
    seed=42,
    detokenize=False,
)


def _build_request(
    text: str,
    voice: str = "Junhao",
    mode: str = "voice_clone",
    max_new_frames: int = 100,  # short for tests
    seed: int = 42,
) -> dict:
    return {
        "prompt": "<|im_start|>assistant\n",
        "additional_information": {
            "text": [text],
            "voice": [voice],
            "mode": [mode],
            "max_new_frames": [max_new_frames],
            "seed": [seed],
        },
    }


def _collect_audio(omni: Omni, request: dict) -> tuple[torch.Tensor, int]:
    """Run a single request and return (waveform, sample_rate)."""
    for stage_outputs in omni.generate(request, DEFAULT_SAMPLING):
        for req_output in stage_outputs.request_output:
            mm = req_output.outputs[0].multimodal_output
            assert mm is not None, "Expected multimodal_output to be non-None"
            audio = mm.get("audio")
            sr = mm.get("sr")
            assert audio is not None, "Expected 'audio' key in multimodal_output"
            assert isinstance(audio, torch.Tensor), f"audio should be Tensor, got {type(audio)}"
            return audio.cpu(), int(sr.item()) if sr is not None else SAMPLE_RATE
    raise AssertionError("No stage outputs received")


@pytest.fixture(scope="module")
def omni_engine():
    """Module-scoped Omni engine to avoid re-loading the model for each test."""
    return Omni(model=MODEL_NAME, stage_init_timeout=180)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_nano_english(omni_engine):
    """English TTS produces non-empty 48 kHz stereo audio."""
    req = _build_request("Hello, this is a test of MOSS-TTS-Nano.", voice="Ava")
    audio, sr = _collect_audio(omni_engine, req)

    assert sr == SAMPLE_RATE, f"Expected sample_rate={SAMPLE_RATE}, got {sr}"
    assert audio.numel() > 0, "Audio tensor should not be empty"
    assert not torch.all(audio == 0), "Audio should not be all-zeros (silence)"


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_nano_chinese(omni_engine):
    """Chinese TTS produces non-empty audio."""
    req = _build_request("你好，这是语音合成测试。", voice="Junhao")
    audio, sr = _collect_audio(omni_engine, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert not torch.all(audio == 0)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_nano_deterministic(omni_engine):
    """Same seed produces identical waveforms."""
    req = _build_request("Reproducible output test.", voice="Adam", seed=123)
    audio1, _ = _collect_audio(omni_engine, req)
    audio2, _ = _collect_audio(omni_engine, req)

    assert audio1.shape == audio2.shape, "Waveform shapes should match with same seed"
    assert torch.allclose(audio1, audio2, atol=1e-4), "Waveforms should match with same seed"


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_nano_batch(omni_engine):
    """Batch of two requests returns audio for each."""
    requests = [
        _build_request("First request.", voice="Ava"),
        _build_request("Second request.", voice="Bella"),
    ]
    results = []
    for stage_outputs in omni_engine.generate(requests, [DEFAULT_SAMPLING] * 2):
        for req_output in stage_outputs.request_output:
            mm = req_output.outputs[0].multimodal_output
            assert mm is not None
            results.append(mm["audio"].cpu())

    assert len(results) == 2, f"Expected 2 outputs, got {len(results)}"
    for i, audio in enumerate(results):
        assert audio.numel() > 0, f"Audio {i} is empty"


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_nano_voice_presets(omni_engine):
    """Multiple built-in voice presets all produce valid audio."""
    voices = ["Junhao", "Ava", "Bella"]
    for voice in voices:
        req = _build_request(f"Testing voice {voice}.", voice=voice)
        audio, sr = _collect_audio(omni_engine, req)
        assert audio.numel() > 0, f"Voice {voice} produced empty audio"
        assert sr == SAMPLE_RATE

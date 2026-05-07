# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for MOSS-TTS-Nano single-stage pipeline.

Every request needs a reference audio clip. We test upstream's
recommended ``voice_clone`` mode (no transcript needed). We fetch the
upstream sample (assets/audio/zh_1.wav, ~50 KB) once per test session
and reuse it for all cases.
"""

from __future__ import annotations

import os
import urllib.request

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from vllm_omni import Omni

MODEL_NAME = "OpenMOSS-Team/MOSS-TTS-Nano"
SAMPLE_RATE = 48000
REF_AUDIO_URL = "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS-Nano/main/assets/audio/zh_1.wav"

DEFAULT_SAMPLING = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    top_k=50,
    max_tokens=4096,
    seed=42,
    detokenize=False,
)


@pytest.fixture(scope="session")
def ref_audio_path(tmp_path_factory) -> str:
    """Download the upstream reference clip once per test session.

    Bounded with a 30 s timeout (urlretrieve has no timeout kwarg, so we use
    urlopen + write ourselves). Fetch failures are hard failures, not skips,
    so a broken network path can't silently mask regressions. Set
    ``MOSS_TTS_NANO_SKIP_ON_NET_FAIL=1`` to opt into skipping for air-gapped
    environments.
    """
    cache_dir = tmp_path_factory.mktemp("moss_tts_nano_ref")
    target = cache_dir / "zh_1.wav"
    try:
        with urllib.request.urlopen(REF_AUDIO_URL, timeout=30) as resp:
            data = resp.read()
        target.write_bytes(data)
    except Exception as e:
        msg = f"Cannot fetch upstream reference clip {REF_AUDIO_URL}: {e}"
        if os.environ.get("MOSS_TTS_NANO_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)
    if not target.exists() or os.path.getsize(target) == 0:
        pytest.fail(f"Reference clip empty after download: {target}")
    return str(target)


def _build_request(
    text: str,
    prompt_audio_path: str,
    prompt_text: str | None = None,
    mode: str = "voice_clone",
    max_new_frames: int = 100,  # short for tests
    seed: int = 42,
) -> dict:
    """Build a MOSS-TTS-Nano offline request.

    Upstream forbids ``prompt_text`` in ``voice_clone`` mode; only forward
    it when explicitly supplied (and typically with ``mode='continuation'``).
    """
    additional: dict = {
        "text": [text],
        "mode": [mode],
        "prompt_audio_path": [prompt_audio_path],
        "max_new_frames": [max_new_frames],
        "seed": [seed],
    }
    if prompt_text is not None:
        additional["prompt_text"] = [prompt_text]
    return {
        "prompt": "<|im_start|>assistant\n",
        "additional_information": additional,
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
def test_moss_tts_nano_english(omni_engine, ref_audio_path):
    """English voice_clone mode (no prompt_text) produces non-empty 48 kHz audio."""
    req = _build_request("Hello, this is a short voice cloning demo for testing.", ref_audio_path)
    audio, sr = _collect_audio(omni_engine, req)

    assert sr == SAMPLE_RATE, f"Expected sample_rate={SAMPLE_RATE}, got {sr}"
    assert audio.numel() > 0, "Audio tensor should not be empty"
    assert not torch.all(audio == 0), "Audio should not be all-zeros (silence)"


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_nano_chinese(omni_engine, ref_audio_path):
    """Chinese voice_clone mode produces non-empty audio."""
    req = _build_request("你好，这是语音合成测试。", ref_audio_path)
    audio, sr = _collect_audio(omni_engine, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert not torch.all(audio == 0)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_nano_deterministic(omni_engine, ref_audio_path):
    """Same seed produces identical waveforms."""
    req = _build_request("Reproducible output test.", ref_audio_path, seed=123)
    audio1, _ = _collect_audio(omni_engine, req)
    audio2, _ = _collect_audio(omni_engine, req)

    assert audio1.shape == audio2.shape, "Waveform shapes should match with same seed"
    assert torch.allclose(audio1, audio2, atol=1e-4), "Waveforms should match with same seed"


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_nano_batch(omni_engine, ref_audio_path):
    """Batch of two requests returns audio for each."""
    requests = [
        _build_request("First request.", ref_audio_path),
        _build_request("Second request.", ref_audio_path),
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

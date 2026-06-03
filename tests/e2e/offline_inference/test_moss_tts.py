# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for MOSS-TTS two-stage pipeline.

Tests the MossTTSDelayModel path using MOSS-VoiceGenerator (1.7B) — the
smallest MossTTSDelayModel variant — so the test runs on a single A10G or L4
without requiring an 80 GB GPU.

A second test class covers MOSS-TTS-Realtime (1.7B, MossTTSRealtime) to
exercise the local-transformer generation path.

Both classes use a module-scoped engine so the model is loaded once per test
module execution.  Do NOT add a second engine fixture in this file — that
triggers mid-module teardown races (see skill invariant I4).
"""

from __future__ import annotations

import os
import urllib.request

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from vllm_omni import Omni

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Selected by the nightly "TTS · Function Test" (-m "full_model and L4 and tts").
# Without the tts + run-level marks these tests are collected but never run.
pytestmark = [pytest.mark.full_model, pytest.mark.tts]

SAMPLE_RATE = 24_000  # All MOSS-TTS full variants output 24 kHz

REF_AUDIO_URL = "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS/main/assets/audio/reference_zh_1.wav"

_DEFAULT_SAMPLING = SamplingParams(
    temperature=1.7,
    top_p=0.8,
    top_k=25,
    max_tokens=512,  # short for tests (~20 s at 24 kHz, ~12.5 tokens/s)
    seed=42,
    detokenize=False,
)

# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ref_audio_path(tmp_path_factory) -> str:
    """Download the upstream reference clip once per session.

    Set ``MOSS_TTS_SKIP_ON_NET_FAIL=1`` to skip in air-gapped environments.
    """
    cache_dir = tmp_path_factory.mktemp("moss_tts_ref")
    target = cache_dir / "zh_1.wav"
    try:
        with urllib.request.urlopen(REF_AUDIO_URL, timeout=30) as resp:
            target.write_bytes(resp.read())
    except Exception as exc:
        msg = f"Cannot fetch reference clip {REF_AUDIO_URL}: {exc}"
        if os.environ.get("MOSS_TTS_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)
    if not target.exists() or target.stat().st_size == 0:
        pytest.fail(f"Reference clip empty after download: {target}")
    return str(target)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_request(text: str, ref_audio_path: str, seed: int = 42) -> dict:
    return {
        "prompt": "<|im_start|>assistant\n",
        "additional_information": {
            "task_type": ["voice_clone"],
            "text": [text],
            "mode": ["voice_clone"],
            "prompt_audio_path": [ref_audio_path],
            "seed": [seed],
        },
    }


def _collect_audio(omni: Omni, request: dict) -> tuple[torch.Tensor, int]:
    """Run one request; return (waveform_cpu, sample_rate)."""
    for stage_outputs in omni.generate(request, _DEFAULT_SAMPLING):
        for req_output in stage_outputs.request_output:
            mm = req_output.outputs[0].multimodal_output
            assert mm is not None, "Expected non-None multimodal_output"
            # Never use ``a or b`` on tensor values — a multi-element tensor
            # raises on truthiness. Check ``is None`` explicitly and handle the
            # pre-consolidation list form.
            audio = mm.get("audio")
            if audio is None:
                audio = mm.get("model_outputs")
            sr = mm.get("sr")
            assert audio is not None, "Expected 'audio' or 'model_outputs' in multimodal_output"
            if isinstance(audio, list):
                audio = torch.cat(
                    [t.reshape(-1) for t in audio if isinstance(t, torch.Tensor) and t.numel() > 0],
                    dim=0,
                )
            assert isinstance(audio, torch.Tensor), f"Expected Tensor, got {type(audio)}"
            return audio.reshape(-1).cpu(), int(sr.item()) if sr is not None else SAMPLE_RATE
    raise AssertionError("No stage outputs received")


# ---------------------------------------------------------------------------
# MossTTSDelayModel — MOSS-VoiceGenerator 1.7B
# (tests the delay-pattern generation path on a small GPU)
# ---------------------------------------------------------------------------

_DELAY_MODEL = "OpenMOSS-Team/MOSS-VoiceGenerator"


@pytest.fixture(scope="module")
def delay_engine():
    return Omni(model=_DELAY_MODEL, stage_init_timeout=300)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_english(delay_engine, ref_audio_path):
    """MossTTSDelayModel: English voice_clone produces non-empty 24 kHz audio."""
    req = _build_request("Hello, this is a MOSS-TTS voice cloning test.", ref_audio_path)
    audio, sr = _collect_audio(delay_engine, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_chinese(delay_engine, ref_audio_path):
    """MossTTSDelayModel: Chinese input produces non-empty audio."""
    req = _build_request("你好，这是语音合成测试。", ref_audio_path)
    audio, sr = _collect_audio(delay_engine, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert not torch.all(audio == 0)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_deterministic(delay_engine, ref_audio_path):
    """MossTTSDelayModel: same seed yields identical waveforms."""
    req = _build_request("Reproducibility check.", ref_audio_path, seed=99)
    audio1, _ = _collect_audio(delay_engine, req)
    audio2, _ = _collect_audio(delay_engine, req)

    assert audio1.shape == audio2.shape, "Shapes differ across identical seeds"
    assert torch.allclose(audio1, audio2, atol=1e-4), "Waveforms differ across identical seeds"


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_batch(delay_engine, ref_audio_path):
    """MossTTSDelayModel: batch of two requests each returns non-empty audio."""
    requests = [
        _build_request("First sentence.", ref_audio_path),
        _build_request("Second sentence.", ref_audio_path),
    ]
    results: list[torch.Tensor] = []
    for stage_outputs in delay_engine.generate(requests, [_DEFAULT_SAMPLING] * 2):
        for req_output in stage_outputs.request_output:
            mm = req_output.outputs[0].multimodal_output
            assert mm is not None
            audio = mm.get("audio")
            if audio is None:
                audio = mm.get("model_outputs")
            assert audio is not None
            if isinstance(audio, list):
                audio = torch.cat(
                    [t.reshape(-1) for t in audio if isinstance(t, torch.Tensor) and t.numel() > 0],
                    dim=0,
                )
            results.append(audio.reshape(-1).cpu())

    assert len(results) == 2, f"Expected 2 outputs, got {len(results)}"
    for i, audio in enumerate(results):
        assert audio.numel() > 0, f"Audio[{i}] is empty"


# ---------------------------------------------------------------------------
# MossTTSRealtime — MOSS-TTS-Realtime 1.7B
# (tests the local-transformer generation path)
# ---------------------------------------------------------------------------

_REALTIME_MODEL = "OpenMOSS-Team/MOSS-TTS-Realtime"


@pytest.fixture(scope="module")
def realtime_engine():
    return Omni(model=_REALTIME_MODEL, stage_init_timeout=300)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_realtime_english(realtime_engine, ref_audio_path):
    """MossTTSRealtime: English voice_clone produces non-empty 24 kHz audio."""
    req = _build_request("This is a real-time TTS streaming test.", ref_audio_path)
    audio, sr = _collect_audio(realtime_engine, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_realtime_deterministic(realtime_engine, ref_audio_path):
    """MossTTSRealtime: same seed yields identical waveforms."""
    req = _build_request("Determinism check for realtime model.", ref_audio_path, seed=7)
    audio1, _ = _collect_audio(realtime_engine, req)
    audio2, _ = _collect_audio(realtime_engine, req)

    assert audio1.shape == audio2.shape
    assert torch.allclose(audio1, audio2, atol=1e-4)

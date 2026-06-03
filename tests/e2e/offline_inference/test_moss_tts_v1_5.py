# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference test for MOSS-TTS-v1.5 (MossTTSDelay-8B).

MOSS-TTS-v1.5 is a continued-training upgrade of MOSS-TTS 1.0 with the SAME
``MossTTSDelay`` architecture and API, so it reuses the existing talker/codec
path unchanged — this test just pins that the 8B v1.5 checkpoint loads and
produces audio.

The model is 8B, so this lives in its own file (one module-scoped engine per
file, per skill invariant I4) and is gated behind an 80 GB GPU; the small-GPU
MossTTSDelay coverage stays in ``test_moss_tts.py`` (1.7B VoiceGenerator).
"""

from __future__ import annotations

import os
import urllib.request

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from vllm_omni import Omni

# H100-gated (8B). Picked up by the nightly H100 omni job
# (-m "full_model and H100 and omni"); the tts mark keeps it consistent with
# the other MOSS-TTS tests and any tts-scoped sweeps.
pytestmark = [pytest.mark.full_model, pytest.mark.tts]

SAMPLE_RATE = 24_000
REF_AUDIO_URL = "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS/main/assets/audio/reference_zh_1.wav"
_MODEL = "OpenMOSS-Team/MOSS-TTS-v1.5"

_DEFAULT_SAMPLING = SamplingParams(
    temperature=1.7,
    top_p=0.8,
    top_k=25,
    max_tokens=512,  # short for tests (~20 s at 24 kHz, ~12.5 tokens/s)
    seed=42,
    detokenize=False,
)


@pytest.fixture(scope="session")
def ref_audio_path(tmp_path_factory) -> str:
    """Download the upstream reference clip once per session.

    Set ``MOSS_TTS_SKIP_ON_NET_FAIL=1`` to skip in air-gapped environments.
    """
    target = tmp_path_factory.mktemp("moss_tts_v15_ref") / "zh_1.wav"
    try:
        with urllib.request.urlopen(REF_AUDIO_URL, timeout=30) as resp:
            target.write_bytes(resp.read())
    except Exception as exc:  # noqa: BLE001
        msg = f"Cannot fetch reference clip {REF_AUDIO_URL}: {exc}"
        if os.environ.get("MOSS_TTS_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)
    if not target.exists() or target.stat().st_size == 0:
        pytest.fail(f"Reference clip empty after download: {target}")
    return str(target)


@pytest.fixture(scope="module")
def v15_engine():
    return Omni(model=_MODEL, stage_init_timeout=600)


def _build_request(text: str, ref_audio_path: str) -> dict:
    return {
        "prompt": "<|im_start|>assistant\n",
        "additional_information": {
            "task_type": ["voice_clone"],
            "text": [text],
            "mode": ["voice_clone"],
            "prompt_audio_path": [ref_audio_path],
            "seed": [42],
        },
    }


def _collect_audio(omni: Omni, request: dict) -> tuple[torch.Tensor, int]:
    for stage_outputs in omni.generate(request, _DEFAULT_SAMPLING):
        for req_output in stage_outputs.request_output:
            mm = req_output.outputs[0].multimodal_output
            assert mm is not None, "Expected non-None multimodal_output"
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


@pytest.mark.omni
@hardware_test(res={"cuda": "H100"})
def test_moss_tts_v15_voice_clone(v15_engine, ref_audio_path):
    """MOSS-TTS-v1.5 (8B): voice_clone produces non-empty 24 kHz audio."""
    req = _build_request("Hello, this is a MOSS-TTS v1.5 voice cloning test.", ref_audio_path)
    audio, sr = _collect_audio(v15_engine, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference test for MOSS-TTS-v1.5 (MossTTSDelay-8B).

MOSS-TTS-v1.5 is a continued-training upgrade of MOSS-TTS 1.0 with the SAME
``MossTTSDelay`` architecture and API. The 8B checkpoint is H100-gated; small-GPU
MossTTSDelay coverage stays in ``test_moss_tts_expansion.py``.
"""

from __future__ import annotations

import os
import urllib.request

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "OpenMOSS-Team/MOSS-TTS-v1.5"
DEPLOY_CONFIG = get_deploy_config_path("moss_tts.yaml")
_OMNI_RUNNER_PARAM = (
    MODEL,
    DEPLOY_CONFIG,
    {"stage_init_timeout": 600, "trust_remote_code": True},
)

pytestmark = [
    pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/4643"),
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]

SAMPLE_RATE = 24_000
REF_AUDIO_URL = "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS/main/assets/audio/reference_zh_1.wav"

_DEFAULT_SAMPLING = SamplingParams(
    temperature=1.7,
    top_p=0.8,
    top_k=25,
    max_tokens=512,
    seed=42,
    detokenize=False,
)


@pytest.fixture(scope="session")
def ref_audio_path(tmp_path_factory) -> str:
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


def _sampling_for(omni_runner: OmniRunner) -> SamplingParams | list[SamplingParams]:
    omni = omni_runner.omni
    if omni.num_stages == 1:
        return _DEFAULT_SAMPLING
    params = omni_runner.get_default_sampling_params_list()
    params[0] = _DEFAULT_SAMPLING
    return params


def _collect_audio(omni_runner: OmniRunner, request: dict) -> tuple[torch.Tensor, int]:
    for stage_outputs in omni_runner.omni.generate(request, _sampling_for(omni_runner)):
        mm = stage_outputs.multimodal_output
        if not mm:
            continue
        audio = mm.get("audio")
        if audio is None:
            audio = mm.get("model_outputs")
        if audio is None:
            continue
        if isinstance(audio, list):
            audio = torch.cat(
                [t.reshape(-1) for t in audio if isinstance(t, torch.Tensor) and t.numel() > 0],
                dim=0,
            )
        if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
            continue
        sr = mm.get("sr")
        return audio.reshape(-1).cpu(), int(sr.item()) if sr is not None else SAMPLE_RATE
    raise AssertionError("No stage outputs received")


@hardware_test(res={"cuda": "H100"})
def test_moss_tts_v15_voice_clone(omni_runner: OmniRunner, ref_audio_path) -> None:
    """MOSS-TTS-v1.5 (8B): voice_clone produces non-empty 24 kHz audio."""
    req = _build_request("Hello, this is a MOSS-TTS v1.5 voice cloning test.", ref_audio_path)
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"

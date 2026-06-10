# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for the SongGen single-stage pipeline.

SongGen turns lyrics plus a music-style description into a 16 kHz mono song in
one auto-regressive pass (the 1.3B AR LM and the X-Codec decoder both run
inside ``SongGenForGeneration``). These mirror the offline example in
``examples/offline_inference/text_to_speech/songgen/end2end.py``.

The model and its ``songgen`` package dependency are large, so these tests are
gated behind the ``full_model`` / ``tts`` markers and only run in the model CI
lane (the deploy config targets a single 80 GB GPU).
"""

from __future__ import annotations

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni import Omni

MODEL_NAME = "LiuZH-19/SongGen_mixed_pro"
STAGE_CONFIG = get_deploy_config_path("songgen.yaml")

# (model, stage_configs_path) for the ``omni_runner`` indirect parametrize.
_OMNI_RUNNER_PARAM = (
    MODEL_NAME,
    STAGE_CONFIG,
)

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]

SAMPLE_RATE = 16000

DEFAULT_SAMPLING = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    top_k=50,
    max_tokens=4096,
    seed=42,
    detokenize=False,
)


def _build_request(lyrics: str, description: str = "a pop song", seed: int = 42) -> dict:
    """Build a SongGen offline request (lyrics + style description)."""
    return {
        "prompt": "<|im_start|>assistant\n",
        "additional_information": {
            "lyrics": [lyrics],
            "text_description": [description],
            "seed": [seed],
        },
    }


def _collect_audio(omni: Omni, request: dict) -> tuple[torch.Tensor, int]:
    """Run a single request and return (waveform, sample_rate)."""
    for stage_outputs in omni.generate(request, DEFAULT_SAMPLING):
        req_output = stage_outputs.request_output
        if req_output is not None:
            mm = req_output.outputs[0].multimodal_output
            assert mm is not None, "Expected multimodal_output to be non-None"
            audio = mm.get("audio")
            sr = mm.get("sr")
            assert audio is not None, "Expected 'audio' key in multimodal_output"
            assert isinstance(audio, torch.Tensor), f"audio should be Tensor, got {type(audio)}"
            return audio.cpu(), int(sr.item()) if sr is not None else SAMPLE_RATE
    raise AssertionError("No stage outputs received")


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_songgen_text_to_song(omni_runner: OmniRunner) -> None:
    """Lyrics + description produce non-empty 16 kHz audio."""
    req = _build_request("Under the moonlight, we dance through the night.")
    audio, sr = _collect_audio(omni_runner.omni, req)

    assert sr == SAMPLE_RATE, f"Expected sample_rate={SAMPLE_RATE}, got {sr}"
    assert audio.numel() > 0, "Audio tensor should not be empty"
    assert not torch.all(audio == 0), "Audio should not be all-zeros (silence)"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_songgen_batch(omni_runner: OmniRunner) -> None:
    """Batch of two requests returns audio for each."""
    requests = [
        _build_request("First verse under a quiet sky."),
        _build_request("Second verse as the morning breaks."),
    ]
    results = []
    # Single-stage model (num_stages=1): one sampling param for all requests.
    for stage_outputs in omni_runner.omni.generate(requests, [DEFAULT_SAMPLING]):
        req_output = stage_outputs.request_output
        if req_output is not None:
            mm = req_output.outputs[0].multimodal_output
            assert mm is not None
            results.append(mm["audio"].cpu())

    assert len(results) == 2, f"Expected 2 outputs, got {len(results)}"
    for i, audio in enumerate(results):
        assert audio.numel() > 0, f"Audio {i} is empty"

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable Audio 3 offline e2e: real weights, smoke tests.

This file mirrors tests/e2e/offline_inference/test_stable_audio_expansion.py for SA3.

Status: **scaffold** — tests are skipped while the port from
https://github.com/Stability-AI/stable-audio-3 is in progress
(see vllm_omni/diffusion/models/stable_audio_3/ for the implementation status).

To activate: once `StableAudio3DiTModel` and `SAMEAutoencoder` have working
implementations, remove the `pytest.mark.skip` from the tests below.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.helpers import skip_if_gated_repo_inaccessible  # noqa: F401  (used once SA3 weights are public)
from tests.helpers.assertions import assert_audio_valid
from tests.helpers.mark import hardware_test
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

_MODEL_REPO = "stabilityai/stable-audio-3-medium"

_SAMPLE_RATE = 44100
_SHORT_CLIP_DURATION_S = 2.0
_LONG_CLIP_DURATION_S = 60.0  # SA3-distinctive: validates variable-length latents at scale


def _generate_short_sa3_clip(
    omni: Omni,
    *,
    prompt: str = "A brief synth chord",
    audio_end_in_s: float = _SHORT_CLIP_DURATION_S,
    num_inference_steps: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Run a minimal SA3 generation and return audio as (batch, channels, samples)."""
    outputs = omni.generate(
        prompts={"prompt": prompt, "negative_prompt": "Low quality."},
        sampling_params_list=OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            guidance_scale=7.0,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
            num_outputs_per_prompt=1,
            extra_args={"audio_start_in_s": 0.0, "audio_end_in_s": audio_end_in_s},
        ),
    )

    assert outputs is not None
    first_output = outputs[0]
    assert first_output.final_output_type == "audio"
    assert hasattr(first_output, "request_output") and first_output.request_output

    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput)
    assert req_out.final_output_type == "audio"
    assert hasattr(req_out, "multimodal_output") and req_out.multimodal_output
    audio = req_out.multimodal_output.get("audio")
    assert isinstance(audio, np.ndarray)
    return audio


@pytest.mark.skip(reason="SA3 DiT + SAME autoencoder port not yet complete (issue #3787)")
@hardware_test(res={"cuda": "L4"})
def test_stable_audio_3_short_clip_smoke() -> None:
    """SA3 short-clip smoke test — checks basic text→audio works end-to-end."""
    omni = Omni(model=_MODEL_REPO)
    audio = _generate_short_sa3_clip(omni)

    # Shape: SA3 outputs stereo at 44.1 kHz.
    expected_samples = int(_SHORT_CLIP_DURATION_S * _SAMPLE_RATE)
    assert audio.shape[-2] == 2, f"expected stereo (2 channels), got shape {audio.shape}"
    assert abs(audio.shape[-1] - expected_samples) < _SAMPLE_RATE, (
        f"expected ~{expected_samples} samples for {_SHORT_CLIP_DURATION_S}s, got {audio.shape[-1]}"
    )
    assert_audio_valid(audio, sample_rate=_SAMPLE_RATE)


@pytest.mark.skip(reason="SA3 DiT + SAME autoencoder port not yet complete (issue #3787)")
@hardware_test(res={"cuda": "L4"})
def test_stable_audio_3_variable_length() -> None:
    """SA3 variable-length test — verifies latents scale to requested duration.

    This is SA3-distinctive vs SA Open 1.0 (which has fixed sample_size).
    """
    omni = Omni(model=_MODEL_REPO)

    short = _generate_short_sa3_clip(omni, audio_end_in_s=5.0, num_inference_steps=4)
    long_ = _generate_short_sa3_clip(omni, audio_end_in_s=_LONG_CLIP_DURATION_S, num_inference_steps=4)

    # Longer requested duration → more samples (not the same trimmed window).
    assert long_.shape[-1] > short.shape[-1] * 5, (
        f"variable-length latents not engaged: short={short.shape}, long={long_.shape}"
    )
    assert_audio_valid(long_, sample_rate=_SAMPLE_RATE)


# TODO(stable-audio-3): once port is done, add:
#   - test_stable_audio_3_long_clip (380s, validates SAME chunked decode + VRAM cap)
#   - test_stable_audio_3_cfg (CFG on/off output divergence)
#   - test_stable_audio_3_hsdp (parallelism phase 10d)
#   - test_stable_audio_3_cache_dit (cache-dit acceleration phase 9)

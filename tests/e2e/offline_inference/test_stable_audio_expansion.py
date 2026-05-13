# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable Audio offline e2e: real weights, FP8 + TeaCache (single job to save GPU).

NOTE: This test instantiates Omni directly instead of using the omni_runner
fixture (introduced in PR #2711) because the fixture's parametrize interface
only accepts (model, stage_config_path) and does not support extra kwargs like
quantization, cache_backend, or cache_config.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.helpers.assertions import assert_audio_valid
from tests.helpers.mark import hardware_test
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

_SAMPLE_RATE = 44100
_CLIP_DURATION_S = 2.0


def generate_stable_audio_short_clip(
    omni: Omni,
    *,
    audio_start_in_s: float = 0.0,
    audio_end_in_s: float = 2.0,
    num_inference_steps: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Run a minimal Stable Audio generation and return audio as (batch, channels, samples)."""
    outputs = omni.generate(
        prompts={
            "prompt": "The sound of a dog barking",
            "negative_prompt": "Low quality.",
        },
        sampling_params_list=OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            guidance_scale=7.0,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
            num_outputs_per_prompt=1,
            extra_args={
                "audio_start_in_s": audio_start_in_s,
                "audio_end_in_s": audio_end_in_s,
            },
        ),
    )

    assert outputs is not None
    first_output = outputs[0]
    # Audio-output diffusion pipelines (those with ``support_audio_output = True``) now have
    # ``final_output_type="audio"`` set on the outer stage metadata as well as the inner request.
    assert first_output.final_output_type == "audio"
    assert hasattr(first_output, "request_output") and first_output.request_output

    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput)
    assert req_out.final_output_type == "audio"
    assert hasattr(req_out, "multimodal_output") and req_out.multimodal_output
    audio = req_out.multimodal_output.get("audio")
    assert isinstance(audio, np.ndarray)
    return audio


@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "xpu": "B60"})
def test_stable_audio_quantization_and_teacache() -> None:
    """Stable Audio Open on real Hub weights with FP8 + TeaCache (covers former L2 smoke + L4 features).

    CI should provide ``HF_TOKEN`` if the checkpoint is gated.
    """
    # ``model_class_name`` must be passed explicitly: the default-stage-cfg
    # factory in ``async_omni_engine.py`` reads it out of ``kwargs`` when
    # deciding ``final_output_type`` (#2077), and at construction time the
    # auto-resolution from ``model_index.json`` has not run yet. AudioX's
    # offline test follows the same pattern.
    m = Omni(
        model="stabilityai/stable-audio-open-1.0",
        model_class_name="StableAudioPipeline",
        quantization="fp8",
        cache_backend="tea_cache",
        cache_config={"rel_l1_thresh": 0.2},
    )
    try:
        audio = generate_stable_audio_short_clip(m)
        assert_audio_valid(
            audio,
            sample_rate=_SAMPLE_RATE,
            channels=2,
            duration_s=_CLIP_DURATION_S,
        )
    finally:
        m.close()

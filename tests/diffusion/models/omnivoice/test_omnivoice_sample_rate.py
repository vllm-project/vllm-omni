# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniVoice diffusion output carries the audio sample rate."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import (
    OmniVoicePipeline,
    get_omnivoice_post_process_func,
)
from vllm_omni.diffusion.output_formatter import (
    format_diffusion_outputs,
    normalize_diffusion_postprocess_output,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.sample_rate import DeprecatedSampleRateKeyWarning, get_sampling_rate
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _request() -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt="speak",
        request_id="req-1",
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=1,
            num_outputs_per_prompt=1,
            resolution=512,
        ),
    )


def test_pipeline_exposes_audio_sample_rate() -> None:
    # The output formatter resolves the rate from the class without a pipeline
    # instance, so the advertised ClassVar must match the config default.
    # __init__ rejects checkpoints whose config disagrees (auk/stable_audio
    # pattern), which is what keeps the static advertisement honest.
    assert OmniVoicePipeline.audio_sample_rate == OmniVoiceConfig().sample_rate == 24000


def test_postprocess_output_carries_rate() -> None:
    # Drive the registered post-process func and the formatter against the
    # real registry-resolved pipeline class (no monkeypatching): a raw audio
    # tensor with no metadata must still come out carrying the rate.
    od_config = SimpleNamespace(model_class_name="OmniVoicePipeline")
    post_process = get_omnivoice_post_process_func(od_config)
    audio = post_process(torch.zeros(1, 8))
    postprocess_output = normalize_diffusion_postprocess_output(audio)

    [result] = format_diffusion_outputs(
        request=_request(),
        od_config=od_config,
        diffusion_output=DiffusionOutput(output=audio),
        output_data=audio,
        postprocess_output=postprocess_output,
    )

    multimodal_output = result.multimodal_output
    assert result.final_output_type == "audio"
    assert "audio" in multimodal_output
    # The formatter still emits the legacy diffusion spelling; PR 10 moves it
    # to SAMPLE_RATE_KEY. Until then the rate resolves through the shared
    # reader, with the deprecation warning the migration contract requires.
    assert multimodal_output["audio_sample_rate"] == OmniVoicePipeline.audio_sample_rate
    with pytest.warns(DeprecatedSampleRateKeyWarning):
        assert get_sampling_rate(multimodal_output) == OmniVoicePipeline.audio_sample_rate

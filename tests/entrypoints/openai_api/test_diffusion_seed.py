# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.protocol.images import ImageGenerationRequest
from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("seed", [-(2**63), -1, 0, 2**63 - 1])
@pytest.mark.parametrize(
    ("request_type", "request_fields"),
    [
        (OpenAICreateSpeechRequest, {"input": "test"}),
        (ImageGenerationRequest, {"prompt": "test"}),
        (VideoGenerationRequest, {"prompt": "test"}),
    ],
)
def test_api_seed_is_accepted_by_diffusion_request(request_type, request_fields, seed):
    api_request = request_type(**request_fields, seed=seed)
    diffusion_request = OmniDiffusionRequest(
        prompt="test",
        sampling_params=OmniDiffusionSamplingParams(seed=api_request.seed),
        request_id="seed-test",
    )

    assert diffusion_request.sampling_params.seed == seed

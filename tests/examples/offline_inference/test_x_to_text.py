# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from examples.offline_inference.x_to_text.x_to_text import (
    _configure_sampling_params,
    _extract_text,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.example]


def test_extract_text_supports_ar_and_diffusion_outputs() -> None:
    ar_output = SimpleNamespace(
        request_output=SimpleNamespace(outputs=[SimpleNamespace(text="AR text")]),
        multimodal_output={},
    )
    diffusion_output = SimpleNamespace(
        request_output=None,
        multimodal_output={"text": "diffusion text"},
    )

    assert _extract_text([ar_output]) == "AR text"
    assert _extract_text([diffusion_output]) == "diffusion text"


def test_configure_sampling_params_maps_diffusion_text_decoder_args() -> None:
    params = OmniDiffusionSamplingParams(extra_args={"existing": True})

    _configure_sampling_params(
        params,
        max_tokens=512,
        temperature=0.8,
        top_p=0.9,
        seed=7,
        stop_token_ids=None,
        extra_body_params=frozenset({"max_think_tokens", "text_temperature", "do_sample"}),
    )

    assert params.seed == 7
    assert params.extra_args == {
        "existing": True,
        "max_think_tokens": 512,
        "text_temperature": 0.8,
        "do_sample": True,
    }

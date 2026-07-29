# SPDX-License-Identifier: Apache-2.0
"""Regression tests for diffusion sampling params coercion."""

from __future__ import annotations

from typing import Any

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sampling_params(*, seed: int | None = None, extra_args: dict[str, Any] | None = None) -> SamplingParams:
    return SamplingParams(max_tokens=1, seed=seed, extra_args=extra_args)


def test_from_params_returns_existing_omni_params() -> None:
    params = OmniDiffusionSamplingParams(num_inference_steps=20)

    converted = OmniDiffusionSamplingParams.from_params(params)

    assert converted is params


def test_from_params_converts_sampling_params_seed_and_known_extra_args() -> None:
    params = _sampling_params(
        seed=1234,
        extra_args={
            "num_inference_steps": 50,
            "height": 1024,
            "width": 768,
            "guidance_scale": 7.5,
        },
    )

    converted = OmniDiffusionSamplingParams.from_params(params)

    assert converted.seed == 1234
    assert converted.num_inference_steps == 50
    assert converted.height == 1024
    assert converted.width == 768
    assert converted.guidance_scale == 7.5
    assert converted.extra_args == {}


def test_from_params_preserves_unknown_extra_args() -> None:
    params = _sampling_params(
        extra_args={
            "height": 512,
            "pipeline_specific": "kept",
        },
    )

    converted = OmniDiffusionSamplingParams.from_params(params)

    assert converted.height == 512
    assert converted.extra_args == {"pipeline_specific": "kept"}


def test_from_params_rejects_unsupported_types() -> None:
    with pytest.raises(TypeError, match="Diffusion stage requires OmniDiffusionSamplingParams"):
        OmniDiffusionSamplingParams.from_params({"height": 512})

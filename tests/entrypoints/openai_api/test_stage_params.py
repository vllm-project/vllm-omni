# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm import SamplingParams

from vllm_omni.entrypoints.openai.stage_params import (
    build_stage_sampling_params_list,
    get_default_sampling_params_list,
    resolve_stage_sampling_params,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def test_resolve_stage_sampling_params_clones_stage_default():
    default = SamplingParams(temperature=0.2, seed=11)

    resolved = resolve_stage_sampling_params(
        SimpleNamespace(stage_type="llm"),
        0,
        [default],
    )

    assert resolved is not default
    assert resolved.temperature == 0.2
    assert resolved.seed == 11

    resolved.seed = 99
    assert default.seed == 11


def test_resolve_stage_sampling_params_uses_diffusion_fallback_when_default_missing():
    request_params = OmniDiffusionSamplingParams(height=768, width=1024, seed=7)

    resolved = resolve_stage_sampling_params(
        SimpleNamespace(stage_type="diffusion"),
        1,
        [],
        diffusion_params=request_params,
    )

    assert resolved is not request_params
    assert resolved.height == 768
    assert resolved.width == 1024
    assert resolved.seed == 7


def test_build_stage_sampling_params_list_can_replace_diffusion_defaults():
    request_params = OmniDiffusionSamplingParams(height=512, width=512)
    diffusion_default = OmniDiffusionSamplingParams(height=1024, width=1024)
    llm_default = SamplingParams(temperature=0.1)
    stages = [
        SimpleNamespace(stage_type="llm"),
        SimpleNamespace(stage_type="diffusion"),
        SimpleNamespace(stage_type="diffusion"),
    ]

    resolved = build_stage_sampling_params_list(
        stages,
        [llm_default, diffusion_default],
        diffusion_params=request_params,
        replace_diffusion_params=True,
    )

    assert resolved[0] is not llm_default
    assert resolved[0].temperature == 0.1
    assert resolved[1] is not request_params
    assert resolved[2] is not request_params
    assert resolved[1] is not resolved[2]
    assert resolved[1].height == 512
    assert resolved[2].height == 512
    assert resolved[1].width == 512
    assert resolved[2].width == 512


def test_get_default_sampling_params_list_reads_engine_defaults_only():
    assert get_default_sampling_params_list(SimpleNamespace()) == []
    assert get_default_sampling_params_list(SimpleNamespace(default_sampling_params_list=None)) == []

    defaults = [SamplingParams(max_tokens=8)]
    resolved = get_default_sampling_params_list(SimpleNamespace(default_sampling_params_list=defaults))
    assert resolved == defaults
    assert resolved is not defaults

# SPDX-License-Identifier: Apache-2.0
"""Regression tests for diffusion sampling params coercion."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from typing import Any

import msgspec
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


@pytest.mark.parametrize("quality", ["lossless", "high"])
def test_quality_accepts_supported_request_levels(quality: str) -> None:
    params = OmniDiffusionSamplingParams(quality=quality)

    assert params.quality == quality


def test_quality_defaults_to_model_owned_policy() -> None:
    params = OmniDiffusionSamplingParams()
    assert params.quality is None


def test_quality_rejects_unsupported_request_level() -> None:
    with pytest.raises(ValueError, match="quality must be one of"):
        OmniDiffusionSamplingParams(quality="medium")


def test_from_params_converts_sampling_params_seed_and_known_extra_args() -> None:
    params = _sampling_params(
        seed=1234,
        extra_args={
            "num_inference_steps": 50,
            "height": 1024,
            "width": 768,
            "guidance_scale": 7.5,
            "quality": "high",
        },
    )

    converted = OmniDiffusionSamplingParams.from_params(params)

    assert converted.seed == 1234
    assert converted.num_inference_steps == 50
    assert converted.height == 1024
    assert converted.width == 768
    assert converted.guidance_scale == 7.5
    assert converted.quality == "high"
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


@pytest.mark.parametrize("schedule", [None, [], [{"start": 3, "end": None, "profile": "quantized"}]])
def test_attention_schedule_typed_and_extra_args_inputs_match(schedule):
    typed = OmniDiffusionSamplingParams(attention_schedule=copy.deepcopy(schedule))
    extra = {"attention_schedule": copy.deepcopy(schedule), "pipeline_specific": "kept"}
    saved = copy.deepcopy(extra)
    nested = OmniDiffusionSamplingParams(extra_args=extra)
    plain = OmniDiffusionSamplingParams.from_params(_sampling_params(extra_args=extra))

    assert typed.attention_schedule == nested.attention_schedule == plain.attention_schedule
    assert nested.extra_args == plain.extra_args == {"pipeline_specific": "kept"}
    assert extra == saved
    assert (typed.attention_schedule is None) == (schedule is None)
    if schedule == []:
        assert typed.attention_schedule == ()


def test_attention_schedule_copies_input_and_survives_transport():
    source = [{"start": 3, "end": 6, "profile": "sparse"}]
    params = OmniDiffusionSamplingParams(attention_schedule=source)
    source[0]["start"] = 0
    source.clear()

    assert params.attention_schedule[0].start == 3
    for restored in (
        params.clone(),
        OmniDiffusionSamplingParams(**msgspec.msgpack.decode(msgspec.msgpack.encode(params))),
        OmniDiffusionSamplingParams(**json.loads(json.dumps(asdict(params)))),
    ):
        assert restored.attention_schedule == params.attention_schedule
        assert restored is not params


def test_attention_schedule_accepts_equal_duplicate_sources():
    source = [{"start": 3, "end": 6, "profile": "sparse"}]
    params = OmniDiffusionSamplingParams(attention_schedule=source, extra_args={"attention_schedule": source})
    assert params.attention_schedule[0].profile == "sparse"
    assert "attention_schedule" not in params.extra_args


def test_attention_schedule_rejects_conflicting_duplicate_sources():
    extra = {"attention_schedule": [{"start": 3, "end": 6, "profile": "sparse"}]}
    with pytest.raises(ValueError, match="conflicting.*attention_schedule"):
        OmniDiffusionSamplingParams(attention_schedule=[], extra_args=extra)


@pytest.mark.parametrize(
    "typed", [None, [], [{"start": 3, "end": 6, "profile": "sparse"}]], ids=["inherit", "disabled", "ranges"]
)
def test_null_extra_args_schedule_keeps_the_typed_value(typed):
    # null means the same as an omitted key: it neither conflicts with nor clears the typed value.
    params = OmniDiffusionSamplingParams(
        attention_schedule=copy.deepcopy(typed), extra_args={"attention_schedule": None, "other": 1}
    )

    assert params.attention_schedule == OmniDiffusionSamplingParams(attention_schedule=typed).attention_schedule
    assert params.extra_args == {"other": 1}


def test_attention_schedule_rejects_request_profile_definitions():
    with pytest.raises(TypeError, match="attention_schedule"):
        OmniDiffusionSamplingParams(extra_args={"attention_schedule": {"profiles": {"new": {}}}})

# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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

from vllm_omni.diffusion.attention.schedule import InvalidAttentionScheduleError
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("streaming_max_output_bytes", 0),
        ("streaming_max_output_bytes", True),
        ("streaming_max_output_bytes", 1.5),
        ("streaming_idle_timeout_s", 0),
        ("streaming_idle_timeout_s", True),
        ("streaming_idle_timeout_s", float("inf")),
        ("streaming_idle_timeout_s", float("nan")),
    ],
)
def test_streaming_resource_limits_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        OmniDiffusionSamplingParams(**{field: value})


# Both schedule fields go through absorb_attention_schedule_extra_args; each has one non-empty example.
_SCHEDULES = {
    "attention_schedule": [{"start": 3, "end": 6, "profile": "sparse"}],
    "attention_sigma_schedule": [{"low": 0.2, "high": 0.6, "profile": "sparse"}],
}
_FIELDS = pytest.mark.parametrize("field", sorted(_SCHEDULES))


@_FIELDS
@pytest.mark.parametrize("kind", ["inherit", "disabled", "set"])
def test_attention_schedule_typed_and_extra_args_inputs_match(field, kind):
    schedule = {"inherit": None, "disabled": [], "set": _SCHEDULES[field]}[kind]
    typed = OmniDiffusionSamplingParams(**{field: copy.deepcopy(schedule)})
    extra = {field: copy.deepcopy(schedule), "pipeline_specific": "kept"}
    saved = copy.deepcopy(extra)
    nested = OmniDiffusionSamplingParams(extra_args=extra)
    plain = OmniDiffusionSamplingParams.from_params(_sampling_params(extra_args=extra))

    assert getattr(typed, field) == getattr(nested, field) == getattr(plain, field)
    assert nested.extra_args == plain.extra_args == {"pipeline_specific": "kept"}
    assert extra == saved
    assert (getattr(typed, field) is None) == (schedule is None)
    if schedule == []:
        assert getattr(typed, field) == ()


@_FIELDS
def test_attention_schedule_copies_input_and_survives_transport(field):
    source = copy.deepcopy(_SCHEDULES[field])
    params = OmniDiffusionSamplingParams(**{field: source})
    expected = OmniDiffusionSamplingParams(**{field: _SCHEDULES[field]})
    source[0]["profile"] = "changed"
    source.clear()

    assert getattr(params, field) == getattr(expected, field)
    for restored in (
        params.clone(),
        OmniDiffusionSamplingParams(**msgspec.msgpack.decode(msgspec.msgpack.encode(params))),
        OmniDiffusionSamplingParams(**json.loads(json.dumps(asdict(params)))),
    ):
        assert getattr(restored, field) == getattr(params, field)
        assert restored is not params


@_FIELDS
def test_attention_schedule_accepts_equal_duplicate_sources(field):
    source = _SCHEDULES[field]
    params = OmniDiffusionSamplingParams(**{field: source}, extra_args={field: source})
    assert getattr(params, field)[0].profile == "sparse"
    assert field not in params.extra_args


@_FIELDS
def test_attention_schedule_rejects_conflicting_duplicate_sources(field):
    with pytest.raises(ValueError, match=f"conflicting {field} values"):
        OmniDiffusionSamplingParams(**{field: []}, extra_args={field: _SCHEDULES[field]})


@_FIELDS
@pytest.mark.parametrize("kind", ["inherit", "disabled", "set"])
def test_null_extra_args_schedule_keeps_the_typed_value(field, kind):
    # null means the same as an omitted key: it neither conflicts with nor clears the typed value.
    typed = {"inherit": None, "disabled": [], "set": _SCHEDULES[field]}[kind]
    params = OmniDiffusionSamplingParams(**{field: copy.deepcopy(typed)}, extra_args={field: None, "other": 1})

    assert getattr(params, field) == getattr(OmniDiffusionSamplingParams(**{field: typed}), field)
    assert params.extra_args == {"other": 1}


def test_attention_schedule_rejects_request_profile_definitions():
    with pytest.raises(InvalidAttentionScheduleError, match="attention_schedule"):
        OmniDiffusionSamplingParams(extra_args={"attention_schedule": {"profiles": {"new": {}}}})

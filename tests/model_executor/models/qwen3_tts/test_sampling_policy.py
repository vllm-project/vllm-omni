# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request temperature inheritance and explicit residual overrides."""

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.qwen3_tts.sampling_policy import get_mtp_sampling_overrides

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("temperature", "extra_args", "expected"),
    [
        (0.0, {}, {"do_sample": False, "temperature": 0.0}),
        (0.9, {}, None),
        (0.3, {"temperature": 0.3}, {"do_sample": True, "temperature": 0.3}),
    ],
)
def test_main_temperature_inheritance(temperature, extra_args, expected):
    assert get_mtp_sampling_overrides(SimpleNamespace(temperature=temperature, extra_args=extra_args)) == expected


def test_explicit_residual_policy_overrides_temperature_without_mutating_input():
    explicit = {"do_sample": True, "temperature": 0.7, "top_k": 20}
    params = SimpleNamespace(temperature=0.0, extra_args={"temperature": 0.0, "subtalker_sampling_params": explicit})
    result = get_mtp_sampling_overrides(params)
    assert result is not None
    assert result == explicit
    assert result is not explicit
    result["top_k"] = 5
    assert explicit == {"do_sample": True, "temperature": 0.7, "top_k": 20}


def test_empty_residual_policy_opts_out_of_greedy_inheritance():
    params = SimpleNamespace(temperature=0.0, extra_args={"subtalker_sampling_params": {}})
    assert get_mtp_sampling_overrides(params) == {}

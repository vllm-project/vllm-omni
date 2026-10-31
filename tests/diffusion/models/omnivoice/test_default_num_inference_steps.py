# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniVoice fills its default step count before step-scheduler admission (#9475)."""

import json
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import get_omnivoice_pre_process_func
from vllm_omni.diffusion.registry import _DIFFUSION_PRE_PROCESS_FUNCS
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def pre_process(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_step": 16}))
    return get_omnivoice_pre_process_func(SimpleNamespace(model=str(tmp_path)))


def _request(**sampling_kwargs):
    return SimpleNamespace(sampling_params=OmniDiffusionSamplingParams(**sampling_kwargs))


def test_unset_steps_are_rejected_by_step_scheduler_without_pre_process():
    with pytest.raises(AssertionError):
        StepScheduler._get_total_steps(None, _request())


def test_unset_steps_default_to_model_num_step(pre_process):
    request = pre_process(_request())

    assert request.sampling_params.num_inference_steps == 16
    assert StepScheduler._get_total_steps(None, request) == 16


def test_explicit_steps_are_preserved(pre_process):
    request = pre_process(_request(num_inference_steps=8))

    assert request.sampling_params.num_inference_steps == 8


def test_explicit_timesteps_are_not_overridden(pre_process):
    request = pre_process(_request(timesteps=[3, 2, 1]))

    assert request.sampling_params.num_inference_steps is None


@pytest.mark.parametrize("model_class_name", ["OmniVoice", "OmniVoicePipeline"])
def test_pre_process_is_registered_for_both_class_names(model_class_name):
    # The HF checkpoint declares architectures=["OmniVoice"], so the engine
    # resolves that name; the hook must be found under it, not only the alias.
    assert _DIFFUSION_PRE_PROCESS_FUNCS[model_class_name] == "get_omnivoice_pre_process_func"

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniVoice fills its default step count before step-scheduler admission (#9475)."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import TransformerConfig
from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import get_omnivoice_pre_process_func
from vllm_omni.diffusion.registry import _DIFFUSION_PRE_PROCESS_FUNCS
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _od_config(**model_config):
    return SimpleNamespace(tf_model_config=TransformerConfig.from_dict(model_config))


@pytest.fixture
def pre_process():
    return get_omnivoice_pre_process_func(_od_config(num_step=16))


def _request(**sampling_kwargs):
    return SimpleNamespace(sampling_params=OmniDiffusionSamplingParams(**sampling_kwargs))


def test_unset_steps_are_rejected_by_step_scheduler_without_pre_process():
    with pytest.raises(AssertionError):
        StepScheduler._get_total_steps(None, _request())


def test_unset_steps_default_to_model_num_step(pre_process):
    request = pre_process(_request())

    assert request.sampling_params.num_inference_steps == 16
    assert StepScheduler._get_total_steps(None, request) == 16


def test_missing_num_step_falls_back_to_config_default():
    request = get_omnivoice_pre_process_func(_od_config())(_request())

    assert request.sampling_params.num_inference_steps == OmniVoiceConfig().num_step


def test_default_is_resolved_at_hook_creation_without_model_path():
    # od_config has no ``model``: the hook must not touch disk or the Hub at admission.
    request = get_omnivoice_pre_process_func(_od_config(num_step=8))(_request())

    assert request.sampling_params.num_inference_steps == 8


def test_explicit_steps_are_preserved(pre_process):
    request = pre_process(_request(num_inference_steps=8))

    assert request.sampling_params.num_inference_steps == 8


def test_explicit_timesteps_are_not_overridden(pre_process):
    request = pre_process(_request(timesteps=[3, 2, 1]))

    assert request.sampling_params.num_inference_steps is None


def test_sigmas_only_is_admitted_without_default(pre_process):
    request = pre_process(_request(sigmas=[1.0, 0.5, 0.0]))

    assert request.sampling_params.num_inference_steps is None
    assert StepScheduler._get_total_steps(None, request) == 3


@pytest.mark.parametrize("model_class_name", ["OmniVoice", "OmniVoicePipeline"])
def test_pre_process_is_registered_for_both_class_names(model_class_name):
    # The HF checkpoint declares architectures=["OmniVoice"], so the engine
    # resolves that name; the hook must be found under it, not only the alias.
    assert _DIFFUSION_PRE_PROCESS_FUNCS[model_class_name] == "get_omnivoice_pre_process_func"

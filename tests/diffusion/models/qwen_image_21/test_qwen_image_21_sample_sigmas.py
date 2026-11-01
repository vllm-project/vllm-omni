# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU lock for the model-level ``sample_sigmas`` sampling grid (diffusers PR #14950).

Qwen-Image-2.1-Turbo ships a preset sigma grid in model_index.json. Priority:
request-level sigmas > config-level sample_sigmas > linspace fallback, and the
grid length decides the step count whenever the request carries no explicit
sigmas.
"""

import json

import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from torch import nn

from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import (
    QwenImage21Pipeline,
    get_qwen_image_21_pre_process_func,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

SAMPLE_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


def _toy_pipeline(sample_sigmas: list[float] | None) -> QwenImage21Pipeline:
    """A pipeline shell with the attributes prepare_timesteps reads, no weights."""
    pipeline = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.device = torch.device("cpu")
    pipeline.sample_sigmas = sample_sigmas
    return pipeline


def _write_model_index(tmp_path, payload: dict) -> str:
    (tmp_path / "model_index.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(tmp_path)


# ---------------------------------------------------------------------------
# enrich_config
# ---------------------------------------------------------------------------


def test_enrich_config_extracts_sample_sigmas_from_model_index(tmp_path):
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    model = _write_model_index(
        tmp_path,
        {"_class_name": "QwenImage21Pipeline", "sample_sigmas": SAMPLE_SIGMAS},
    )
    config = OmniDiffusionConfig(model=model)
    config.enrich_config()

    assert config.model_class_name == "QwenImage21Pipeline"
    assert config.extras["sample_sigmas"] == SAMPLE_SIGMAS


def test_enrich_config_without_sample_sigmas_leaves_extras_untouched(tmp_path):
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    model = _write_model_index(tmp_path, {"_class_name": "QwenImage21Pipeline"})
    config = OmniDiffusionConfig(model=model)
    config.enrich_config()

    assert "sample_sigmas" not in config.extras


def test_enrich_config_does_not_override_explicit_extras_sample_sigmas(tmp_path):
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    explicit = [1.0, 0.5]
    model = _write_model_index(
        tmp_path,
        {"_class_name": "QwenImage21Pipeline", "sample_sigmas": SAMPLE_SIGMAS},
    )
    config = OmniDiffusionConfig(model=model, extras={"sample_sigmas": explicit})
    config.enrich_config()

    assert config.extras["sample_sigmas"] == explicit


# ---------------------------------------------------------------------------
# prepare_timesteps priority
# ---------------------------------------------------------------------------


def test_prepare_timesteps_request_sigmas_win_over_sample_sigmas():
    pipeline = _toy_pipeline(SAMPLE_SIGMAS)
    request_sigmas = [1.0, 0.75, 0.5]

    _, num_inference_steps = pipeline.prepare_timesteps(
        num_inference_steps=50, sigmas=request_sigmas, image_seq_len=256
    )

    assert num_inference_steps == len(request_sigmas)
    torch.testing.assert_close(
        pipeline.scheduler.sigmas[: len(request_sigmas)],
        torch.tensor(request_sigmas, dtype=torch.float32),
    )


def test_prepare_timesteps_sample_sigmas_decide_step_count_over_num_inference_steps():
    pipeline = _toy_pipeline(SAMPLE_SIGMAS)

    _, num_inference_steps = pipeline.prepare_timesteps(num_inference_steps=50, sigmas=None, image_seq_len=256)

    # The config grid wins even though the caller passed num_inference_steps=50.
    assert num_inference_steps == len(SAMPLE_SIGMAS)
    torch.testing.assert_close(
        pipeline.scheduler.sigmas[: len(SAMPLE_SIGMAS)],
        torch.tensor(SAMPLE_SIGMAS, dtype=torch.float32),
    )


def test_prepare_timesteps_linspace_fallback_without_sample_sigmas():
    pipeline = _toy_pipeline(None)

    _, num_inference_steps = pipeline.prepare_timesteps(num_inference_steps=10, sigmas=None, image_seq_len=256)

    assert num_inference_steps == 10
    torch.testing.assert_close(
        pipeline.scheduler.sigmas[:10],
        torch.linspace(1.0, 1 / 10, 10, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# default_num_inference_steps
# ---------------------------------------------------------------------------


def test_default_num_inference_steps_is_sample_sigmas_length():
    assert _toy_pipeline(SAMPLE_SIGMAS).default_num_inference_steps == len(SAMPLE_SIGMAS)


def test_default_num_inference_steps_is_none_without_sample_sigmas():
    assert _toy_pipeline(None).default_num_inference_steps is None


# ---------------------------------------------------------------------------
# pre-process injection (step-scheduler total-steps resolution)
# ---------------------------------------------------------------------------


def _make_pre_process_func(tmp_path, extras: dict):
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "config.json").write_text(json.dumps({}), encoding="utf-8")
    od_config = OmniDiffusionConfig(model=str(tmp_path), extras=extras)
    return get_qwen_image_21_pre_process_func(od_config)


def _make_request(sigmas: list[float] | None = None, extra_args: dict | None = None):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

    return OmniDiffusionRequest(
        prompt=OmniTextPrompt(prompt="a cat"),
        sampling_params=OmniDiffusionSamplingParams(sigmas=sigmas, extra_args=extra_args or {}),
        request_id="req-0",
    )


def test_pre_process_injects_sample_sigmas_when_request_has_none(tmp_path):
    pre_process = _make_pre_process_func(tmp_path, {"sample_sigmas": SAMPLE_SIGMAS})

    request = pre_process(_make_request())

    assert request.sampling_params.sigmas == SAMPLE_SIGMAS


def test_pre_process_keeps_explicit_request_sigmas(tmp_path):
    pre_process = _make_pre_process_func(tmp_path, {"sample_sigmas": SAMPLE_SIGMAS})

    request = pre_process(_make_request(sigmas=[1.0, 0.5]))

    assert request.sampling_params.sigmas == [1.0, 0.5]


def test_pre_process_leaves_sigmas_none_without_sample_sigmas(tmp_path):
    pre_process = _make_pre_process_func(tmp_path, {})

    request = pre_process(_make_request())

    assert request.sampling_params.sigmas is None


# ---------------------------------------------------------------------------
# serving extra_body path: whitelisted sigmas arrive in ``extra_args``
# ---------------------------------------------------------------------------


def test_pre_process_hoists_sigmas_from_extra_args_over_sample_sigmas(tmp_path):
    """extra-body sigmas (serving layer puts them in extra_args) win over the grid."""
    pre_process = _make_pre_process_func(tmp_path, {"sample_sigmas": SAMPLE_SIGMAS})

    request = pre_process(_make_request(extra_args={"sigmas": [1.0, 0.75, 0.5]}))

    assert request.sampling_params.sigmas == [1.0, 0.75, 0.5]


def test_pre_process_typed_sigmas_win_over_extra_args(tmp_path):
    pre_process = _make_pre_process_func(tmp_path, {"sample_sigmas": SAMPLE_SIGMAS})

    request = pre_process(_make_request(sigmas=[1.0, 0.5], extra_args={"sigmas": [0.9, 0.1]}))

    assert request.sampling_params.sigmas == [1.0, 0.5]


def test_pre_process_extra_args_sigmas_coerced_to_float(tmp_path):
    pre_process = _make_pre_process_func(tmp_path, {})

    request = pre_process(_make_request(extra_args={"sigmas": [1, 0.5]}))

    assert request.sampling_params.sigmas == [1.0, 0.5]
    assert all(isinstance(s, float) for s in request.sampling_params.sigmas)

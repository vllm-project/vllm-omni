# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Entrypoints move a request ``attention_schedule`` from extra params onto the typed sampling field."""

import json
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pytest_mock import MockerFixture
from vllm.entrypoints.openai.models.protocol import BaseModelPath

from tests.entrypoints.openai_api.test_image_server import MockGenerationResult
from tests.entrypoints.openai_api.test_video_server import FakeAsyncOmni
from vllm_omni.diffusion.attention.schedule import (
    AttentionScheduleRange,
    AttentionSigmaWindow,
    InvalidAttentionScheduleError,
)
from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.diffusion_request_utils import apply_normalized_diffusion_request_extra_args
from vllm_omni.entrypoints.openai.models.serving import _DiffusionServingModels
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
from vllm_omni.errors import client_error_metadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
_STAGE_DEFAULT = [{"start": 0, "end": 2, "profile": "sparse"}]
_REPLACEMENT = [{"start": 1, "end": 3, "profile": "dense"}]
_STAGE_DEFAULT_RANGES = (AttentionScheduleRange(start=0, end=2, profile="sparse"),)
_REPLACEMENT_RANGES = (AttentionScheduleRange(start=1, end=3, profile="dense"),)
_OVERRIDES = [([], ()), (_REPLACEMENT, _REPLACEMENT_RANGES), (None, _STAGE_DEFAULT_RANGES)]
_OVERRIDE_IDS = ["disable", "replace", "null-keeps-stage-default"]
_INVALID = [
    [{"start": 0, "end": 2, "profile": "sparse"}, {"start": 1, "end": 3, "profile": "dense"}],
    [{"start": True, "end": 2, "profile": "sparse"}],
]
_INVALID_IDS = ["overlapping", "bool-step"]


@pytest.fixture
def client(mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    app = FastAPI()
    app.state.api_server_count = 1
    app.include_router(router)
    app.state.openai_serving_video = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=FakeAsyncOmni(),
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    with TestClient(app) as test_client:
        yield test_client


def _post_sync(client, extra_params):
    return client.post(
        "/v1/videos/sync",
        data={"prompt": "A boat on a lake.", "extra_params": json.dumps(extra_params)},
    )


def test_sync_video_moves_schedule_into_typed_field(client):
    response = _post_sync(
        client,
        {"attention_schedule": [{"start": 0, "end": 2, "profile": "sparse"}], "flow_shift": 5.0},
    )

    assert response.status_code == 200
    captured = client.app.state.openai_serving_video._engine_client.captured_sampling_params_list[0]
    assert captured.attention_schedule == (AttentionScheduleRange(start=0, end=2, profile="sparse"),)
    assert "attention_schedule" not in captured.extra_args
    assert captured.extra_args["flow_shift"] == 5.0


@pytest.mark.parametrize(("value", "expected"), _OVERRIDES, ids=_OVERRIDE_IDS)
def test_sync_video_request_schedule_replaces_stage_default(client, value, expected):
    engine = client.app.state.openai_serving_video._engine_client
    engine.default_sampling_params_list = [OmniDiffusionSamplingParams(attention_schedule=_STAGE_DEFAULT)]

    response = _post_sync(client, {"attention_schedule": value})

    assert response.status_code == 200
    captured = engine.captured_sampling_params_list[0]
    assert captured.attention_schedule == expected
    assert "attention_schedule" not in captured.extra_args


@pytest.mark.parametrize("schedule", _INVALID, ids=_INVALID_IDS)
def test_sync_video_rejects_invalid_schedule_before_generation(client, schedule):
    response = _post_sync(client, {"attention_schedule": schedule})

    assert response.status_code == 400
    assert "attention_schedule" in response.json()["detail"]
    assert client.app.state.openai_serving_video._engine_client.captured_sampling_params_list is None


@pytest.mark.parametrize("value", [None, [], [{"low": 0.3, "high": 1.0, "profile": "dense"}]])
def test_sync_video_sigma_override_moves_to_typed_field(client, value):
    engine = client.app.state.openai_serving_video._engine_client
    inherited = (AttentionSigmaWindow(0.0, 0.3, "sparse"),)
    engine.default_sampling_params_list = [OmniDiffusionSamplingParams(attention_sigma_schedule=inherited)]
    response = _post_sync(client, {"attention_sigma_schedule": value, "flow_shift": 5.0})
    assert response.status_code == 200
    captured = engine.captured_sampling_params_list[0]
    expected = inherited if value is None else () if value == [] else (AttentionSigmaWindow(0.3, 1.0, "dense"),)
    assert captured.attention_sigma_schedule == expected
    assert "attention_sigma_schedule" not in captured.extra_args
    assert captured.extra_args["flow_shift"] == 5.0


def test_sync_video_rejects_invalid_sigma_before_generation(client):
    response = _post_sync(
        client,
        {
            "attention_sigma_schedule": [
                {"low": 0.0, "high": 0.5, "profile": "sparse"},
                {"low": 0.3, "high": 1.0, "profile": "dense"},
            ],
        },
    )
    assert response.status_code == 400
    assert "attention_sigma_schedule" in response.json()["detail"]
    assert client.app.state.openai_serving_video._engine_client.captured_sampling_params_list is None


class _ImageEngine:
    def __init__(self):
        self.captured_sampling_params_list: list[Any] | None = None

    async def generate(self, **kwargs):
        self.captured_sampling_params_list = kwargs["sampling_params_list"]
        yield MockGenerationResult([Image.new("RGB", (16, 16), color="blue")])


@pytest.fixture
def image_client():
    engine = _ImageEngine()
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = engine
    app.state.diffusion_engine = engine
    app.state.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    app.state.openai_serving_models = _DiffusionServingModels(
        [BaseModelPath(name="Qwen/Qwen-Image", model_path="Qwen/Qwen-Image")]
    )
    app.state.args = Namespace(
        default_sampling_params='{"0": {"num_inference_steps":4, "guidance_scale":7.5, "generator_device":"cpu"}}',
        max_generated_image_size=1024 * 1792,
    )
    return TestClient(app)


def _post_image(client, extra_params):
    return client.post("/v1/images/generations", json={"prompt": "A red fox.", "extra_params": extra_params})


def test_image_generation_moves_schedule_into_typed_field(image_client):
    response = _post_image(image_client, {"attention_schedule": _REPLACEMENT, "other": 1})

    assert response.status_code == 200
    captured = image_client.app.state.engine_client.captured_sampling_params_list[0]
    assert captured.attention_schedule == _REPLACEMENT_RANGES
    assert "attention_schedule" not in captured.extra_args
    assert captured.extra_args["other"] == 1


@pytest.mark.parametrize("schedule", _INVALID, ids=_INVALID_IDS)
def test_image_generation_rejects_invalid_schedule_before_generation(image_client, schedule):
    response = _post_image(image_client, {"attention_schedule": schedule})

    assert response.status_code == 400
    assert "attention_schedule" in response.json()["detail"]
    assert image_client.app.state.engine_client.captured_sampling_params_list is None


@pytest.mark.parametrize(("value", "expected"), _OVERRIDES, ids=_OVERRIDE_IDS)
def test_chat_diffusion_extra_args_replace_stage_default_schedule(value, expected):
    params = OmniDiffusionSamplingParams(attention_schedule=_STAGE_DEFAULT, extra_args={"stage": 1})

    apply_normalized_diffusion_request_extra_args(params, {"attention_schedule": value, "other": 2})

    assert params.attention_schedule == expected
    assert params.extra_args == {"stage": 1, "other": 2}


@pytest.mark.parametrize("schedule", _INVALID, ids=_INVALID_IDS)
def test_chat_diffusion_extra_args_reject_invalid_schedule_as_client_error(schedule):
    params = OmniDiffusionSamplingParams()

    with pytest.raises(InvalidAttentionScheduleError, match="attention_schedule") as excinfo:
        apply_normalized_diffusion_request_extra_args(params, {"attention_schedule": schedule})

    assert client_error_metadata(excinfo.value) == (400, "invalid_attention_schedule")

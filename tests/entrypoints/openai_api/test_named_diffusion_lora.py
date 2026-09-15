# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Named diffusion LoRA registration, serving resolution and request isolation."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from starlette.datastructures import State
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.models.protocol import LoRAModulePath

from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.lora import build_diffusion_lora_registry
from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_video_output_stream import OmniStreamingVideoOutputHandler
from vllm_omni.entrypoints.openai.utils import parse_lora_request
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.utils import stable_lora_int_id

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ADAPTER_NAME = "h3-turbo"
ADAPTER_PATH = "/server/models/turbo.safetensors"


def test_registration_does_not_activate_or_load_adapter():
    registry = build_diffusion_lora_registry([LoRAModulePath(name=ADAPTER_NAME, path=ADAPTER_PATH)])
    assert registry == {ADAPTER_NAME: ADAPTER_PATH}
    assert parse_lora_request(None, registry) == (None, None)


def test_duplicate_names_fail_registration():
    with pytest.raises(ValueError, match="Duplicate diffusion LoRA name"):
        build_diffusion_lora_registry(
            [LoRAModulePath(name=ADAPTER_NAME, path=path) for path in (ADAPTER_PATH, "/another/turbo.safetensors")]
        )


@pytest.mark.parametrize("name,path", [("", ADAPTER_PATH), (ADAPTER_NAME, "")])
def test_empty_registration_rejected(name, path):
    with pytest.raises(ValueError, match="requires a name and path"):
        build_diffusion_lora_registry([LoRAModulePath(name=name, path=path)])


def test_identical_request_resolves_each_servers_path():
    body = {"name": ADAPTER_NAME, "scale": 0.7, "int_id": None}
    for path in (ADAPTER_PATH, "/different/server/turbo.safetensors"):
        registry = build_diffusion_lora_registry([LoRAModulePath(name=ADAPTER_NAME, path=path)])
        request, scale = parse_lora_request(body, registry)
        explicit_request, _ = parse_lora_request({"name": ADAPTER_NAME, "path": path})
        assert request.lora_path == path
        assert request.lora_int_id == explicit_request.lora_int_id == stable_lora_int_id(path)
        assert scale == 0.7
    assert body == {"name": ADAPTER_NAME, "scale": 0.7, "int_id": None}


@pytest.mark.parametrize("name_key", ["name", "lora_name", "adapter"])
def test_named_selection_aliases(name_key):
    request, scale = parse_lora_request({name_key: ADAPTER_NAME, "lora_scale": 0.5}, {ADAPTER_NAME: ADAPTER_PATH})
    assert request.lora_name == ADAPTER_NAME
    assert request.lora_path == ADAPTER_PATH
    assert scale == 0.5


@pytest.mark.parametrize(
    "body,match",
    [
        ({"name": "missing"}, "unknown LoRA name"),
        ({"name": ADAPTER_NAME, "path": "/wrong/adapter"}, "path conflicts"),
        ({"name": ADAPTER_NAME, "int_id": 1}, "int_id conflicts"),
        ({"name": ADAPTER_NAME, "int_id": {}}, "int_id must be an integer"),
        ({"name": ADAPTER_NAME, "int_id": float("inf")}, "int_id must be an integer"),
        ({"name": "legacy", "path": "/legacy/adapter", "int_id": []}, "int_id must be an integer"),
        ({"name": ADAPTER_NAME, "scale": {}}, "scale must be a finite number"),
        ({"name": ADAPTER_NAME, "scale": []}, "scale must be a finite number"),
        ({"name": ADAPTER_NAME, "scale": "nan"}, "scale must be a finite number"),
        ({"name": ADAPTER_NAME, "scale": float("inf")}, "scale must be a finite number"),
    ],
)
def test_invalid_registered_selection_rejected(body, match):
    with pytest.raises(ValueError, match=match):
        parse_lora_request(body, {ADAPTER_NAME: ADAPTER_PATH})


def test_registered_explicit_path_uses_same_id():
    request, _ = parse_lora_request(
        {"name": ADAPTER_NAME, "local_path": ADAPTER_PATH, "int_id": stable_lora_int_id(ADAPTER_PATH)},
        {ADAPTER_NAME: ADAPTER_PATH},
    )
    assert request.lora_int_id == stable_lora_int_id(ADAPTER_PATH)


def test_unregistered_explicit_path_keeps_legacy_id():
    request, scale = parse_lora_request(
        {"name": "legacy", "local_path": "/legacy/adapter", "int_id": 17, "scale": 0.25},
        {ADAPTER_NAME: ADAPTER_PATH},
    )
    assert request.lora_path == "/legacy/adapter"
    assert request.lora_int_id == 17
    assert scale == 0.25


@pytest.mark.asyncio
async def test_pure_diffusion_initialization_shares_registry(monkeypatch):
    monkeypatch.setattr(api_server.openai_app_state, "_get_vllm_config", AsyncMock(return_value=None))
    # Audio constructors require unrelated model and speaker-storage state.
    monkeypatch.setattr(api_server.OmniOpenAIServingAudioGenerate, "for_diffusion", Mock())
    monkeypatch.setattr(api_server.OmniOpenAIServingSpeech, "for_diffusion", Mock())
    state = State()
    engine = SimpleNamespace(stage_configs=[SimpleNamespace(stage_type="diffusion")])
    args = Namespace(
        model="MiniMaxAI/MiniMax-H3",
        served_model_name=None,
        enable_log_requests=False,
        disable_log_stats=True,
        lora_modules=[LoRAModulePath(name=ADAPTER_NAME, path=ADAPTER_PATH)],
    )
    await api_server.omni_init_app_state(engine, state, args)
    try:
        registry = state.diffusion_lora_modules
        assert registry == {ADAPTER_NAME: ADAPTER_PATH}
        assert state.openai_serving_chat._diffusion_lora_modules is registry
        assert state.openai_serving_chat_batch._diffusion_lora_modules is registry
        assert state.openai_serving_video._lora_modules is registry
        assert state.openai_streaming_video_output._lora_modules is registry
    finally:
        state.openai_serving_video.shutdown()


@pytest.fixture
def chat_handler():
    engine = SimpleNamespace(generate=Mock(), stage_configs=[SimpleNamespace(stage_type="diffusion")])
    return OmniOpenAIServingChat.for_diffusion(engine, "test-model", lora_modules={ADAPTER_NAME: ADAPTER_PATH})


def test_chat_image_preparation_resolves_name_and_preserves_base(chat_handler):
    _, _, turbo, _ = chat_handler._prepare_diffusion_image_request(
        prompt="test", extra_body={"lora": {"name": ADAPTER_NAME, "scale": 1.0}}
    )
    _, _, base, _ = chat_handler._prepare_diffusion_image_request(prompt="test")
    assert turbo.lora_request.lora_path == ADAPTER_PATH
    assert turbo.lora_scale == 1.0
    assert base.lora_request is None


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["chat", "images"])
async def test_unknown_chat_lora_returns_client_error_before_generation(chat_handler, entrypoint):
    if entrypoint == "images":
        result = await chat_handler.generate_diffusion_images(prompt="test", extra_body={"lora": {"name": "missing"}})
    else:
        result = await chat_handler.create_chat_completion(
            ChatCompletionRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                lora={"name": "missing"},
            )
        )
    assert result.error.code == 400
    assert "unknown LoRA name" in result.error.message
    chat_handler._diffusion_engine.generate.assert_not_called()


def test_multistage_chat_resolves_named_lora(chat_handler):
    engine = SimpleNamespace(
        stage_configs=[SimpleNamespace(stage_type="diffusion")],
        default_sampling_params_list=[OmniDiffusionSamplingParams()],
    )
    _, params = chat_handler._build_multistage_generation_inputs(
        engine=engine,
        prompt="test",
        extra_body={"lora": {"name": ADAPTER_NAME, "scale": 0.5}},
        reference_images=[],
        gen_params=OmniDiffusionSamplingParams(),
    )
    assert params[0].lora_request.lora_path == ADAPTER_PATH
    assert params[0].lora_scale == 0.5
    assert engine.default_sampling_params_list[0].lora_request is None


@pytest.mark.asyncio
async def test_streaming_video_resolves_name_without_mutating_defaults():
    engine = SimpleNamespace(default_sampling_params_list=[OmniDiffusionSamplingParams()])
    handler = OmniStreamingVideoOutputHandler(engine, lora_modules={ADAPTER_NAME: ADAPTER_PATH})
    _, turbo, _ = await handler._build_prompt_and_sampling_params(
        VideoGenerationRequest(prompt="test", lora={"name": ADAPTER_NAME, "scale": 0.5})
    )
    _, base, _ = await handler._build_prompt_and_sampling_params(VideoGenerationRequest(prompt="test"))
    assert turbo.lora_request.lora_path == ADAPTER_PATH
    assert turbo.lora_scale == 0.5
    assert base.lora_request is None
    assert engine.default_sampling_params_list[0].lora_request is None
    with pytest.raises(HTTPException) as exc:
        await handler._build_prompt_and_sampling_params(VideoGenerationRequest(prompt="test", lora={"name": "missing"}))
    assert exc.value.status_code == 400

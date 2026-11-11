# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI app-state accessors.

This module owns small helpers that read initialized serving objects or engine
configuration from ``request.app.state``."""

from http import HTTPStatus
from typing import Any, cast

from fastapi import HTTPException, Request, WebSocket
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.batch_serving import OmniOpenAIServingChatBatch
from vllm_omni.entrypoints.openai.serving_audio_generate import OmniOpenAIServingAudioGenerate
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_policy import (
    get_image_stage_configs,
    get_route_handler,
    image_generation_error,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo

ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


async def _get_vllm_config(engine_client: EngineClient) -> Any:
    if hasattr(engine_client, "get_vllm_config"):
        return await engine_client.get_vllm_config()
    return getattr(engine_client, "vllm_config", None)


def Omnivideo(request: Request) -> OmniOpenAIServingVideo | None:
    return cast(OmniOpenAIServingVideo | None, get_route_handler(request.app.state, "/v1/videos"))


def Omnichat(request: Request) -> OmniOpenAIServingChat | None:
    return cast(OmniOpenAIServingChat | None, get_route_handler(request.app.state, "/v1/chat/completions"))


def OmniBatchChat(request: Request) -> OmniOpenAIServingChatBatch | None:
    return cast(OmniOpenAIServingChatBatch | None, get_route_handler(request.app.state, "/v1/chat/completions/batch"))


def Omnispeech(request: Request) -> OmniOpenAIServingSpeech | None:
    return cast(OmniOpenAIServingSpeech | None, get_route_handler(request.app.state, "/v1/audio/speech"))


def OmniAudioGenerate(request: Request) -> OmniOpenAIServingAudioGenerate | None:
    return cast(OmniOpenAIServingAudioGenerate | None, get_route_handler(request.app.state, "/v1/audio/generate"))


def _is_served_model(request: Request | WebSocket, model_name: str, default_model_name: str | None) -> bool:
    """Check public aliases without changing the default response name.

    Standalone handlers may have only a default name; initialized OpenAI apps
    use the same base-model names as ``/v1/models``.
    """
    serving_models = getattr(request.app.state, "openai_serving_models", None)
    base_model_paths = getattr(serving_models, "base_model_paths", None)
    if base_model_paths:
        return any(path.name == model_name for path in base_model_paths)
    return model_name == default_model_name


def _get_engine_and_model(raw_request: Request):
    # Get engine client (AsyncOmni) from app state
    engine_client: EngineClient | AsyncOmni | None = getattr(raw_request.app.state, "engine_client", None)
    if engine_client is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Multi-stage engine not initialized. Start server with a multi-stage omni model.",
        )

    # Check if there's an image generation stage (classical diffusion, or a
    # legacy generation-LLM DiT stage that emits images).
    # Prefer app state (compat layer populated at startup), then fall back to
    # the engine client's stage configs for refactored AsyncOmni paths.
    normalized_stage_configs = get_image_stage_configs(raw_request.app.state, engine_client)
    error = image_generation_error(normalized_stage_configs)
    if error is not None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail=error,
        )

    # Get server's loaded model name
    serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
    base_model_paths = getattr(serving_models, "base_model_paths", None) if serving_models else None
    if base_model_paths:
        model_name = base_model_paths[0].name
    else:
        model_name = "unknown"

    return engine_client, model_name, normalized_stage_configs


def _get_diffusion_od_config_from_state(state: State, engine_client: object) -> object | None:
    diffusion_engine = getattr(state, "diffusion_engine", None) or engine_client
    get_diffusion_od_config = getattr(diffusion_engine, "get_diffusion_od_config", None)
    return (
        get_diffusion_od_config() if callable(get_diffusion_od_config) else getattr(diffusion_engine, "od_config", None)
    )


def _get_diffusion_od_config(raw_request: Request, engine_client: object) -> object | None:
    return _get_diffusion_od_config_from_state(raw_request.app.state, engine_client)

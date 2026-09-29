# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI app-state accessors.

This module owns small helpers that read initialized serving objects or engine
configuration from ``request.app.state``."""

from http import HTTPStatus
from typing import Any

from fastapi import HTTPException, Request
from vllm.engine.protocol import EngineClient

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.batch_serving import OmniOpenAIServingChatBatch
from vllm_omni.entrypoints.openai.serving_audio_generate import OmniOpenAIServingAudioGenerate
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
from vllm_omni.entrypoints.openai.utils import get_stage_type

ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


async def _get_vllm_config(engine_client: EngineClient) -> Any:
    if hasattr(engine_client, "get_vllm_config"):
        return await engine_client.get_vllm_config()
    return getattr(engine_client, "vllm_config", None)


def Omnivideo(request: Request) -> OmniOpenAIServingVideo | None:
    return request.app.state.openai_serving_video


def Omnichat(request: Request) -> OmniOpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def OmniBatchChat(request: Request) -> OmniOpenAIServingChatBatch | None:
    return request.app.state.openai_serving_chat_batch


def Omnispeech(request: Request) -> OmniOpenAIServingSpeech | None:
    return request.app.state.openai_serving_speech


def OmniAudioGenerate(request: Request) -> OmniOpenAIServingAudioGenerate | None:
    return getattr(request.app.state, "openai_serving_audio_generate", None)


def _get_engine_and_model(raw_request: Request):
    # Get engine client (AsyncOmni) from app state
    engine_client: EngineClient | AsyncOmni | None = getattr(raw_request.app.state, "engine_client", None)
    if engine_client is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Multi-stage engine not initialized. Start server with a multi-stage omni model.",
        )

    # Check if there's a diffusion stage.
    # Prefer app state (compat layer populated at startup), then fall back to
    # the engine client's stage configs for refactored AsyncOmni paths.
    stage_configs = getattr(raw_request.app.state, "stage_configs", None)
    if not stage_configs:
        stage_configs = getattr(engine_client, "stage_configs", None)
    if not stage_configs:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Stage configs not found. Start server with a multi-stage omni model.",
        )

    normalized_stage_configs = list(stage_configs)
    has_diffusion_stage = any(get_stage_type(stage_cfg) == "diffusion" for stage_cfg in normalized_stage_configs)

    if not has_diffusion_stage:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="No diffusion stage found in multi-stage pipeline.",
        )

    # Get server's loaded model name
    serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
    base_model_paths = getattr(serving_models, "base_model_paths", None) if serving_models else None
    if base_model_paths:
        model_name = base_model_paths[0].name
    else:
        model_name = "unknown"

    return engine_client, model_name, normalized_stage_configs


def _get_diffusion_od_config(raw_request: Request, engine_client: Any) -> Any:
    diffusion_engine = getattr(raw_request.app.state, "diffusion_engine", None) or engine_client
    get_diffusion_od_config = getattr(diffusion_engine, "get_diffusion_od_config", None)
    return (
        get_diffusion_od_config() if callable(get_diffusion_od_config) else getattr(diffusion_engine, "od_config", None)
    )

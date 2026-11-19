# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Initialized serving policy shared by route guards and discovery.

These helpers only read frontend state. They do not import serving handlers,
execute requests, resolve model configuration, or contact an engine worker.
Callers retain ownership of when and how a rejection is reported.
"""

from collections.abc import Sequence

from starlette.datastructures import State

from vllm_omni.entrypoints.openai.protocol.capabilities import Exposure
from vllm_omni.entrypoints.openai.utils import get_stage_type, is_image_generation_stage, is_video_generation_pipeline

_ROUTE_HANDLERS = {
    "/v1/chat/completions": "openai_serving_chat",
    "/v1/chat/completions/batch": "openai_serving_chat_batch",
    "/v1/audio/speech": "openai_serving_speech",
    "/v1/audio/generate": "openai_serving_audio_generate",
    "/v1/videos": "openai_serving_video",
    "/v1/videos/sync": "openai_serving_video",
    "/v1/realtime/video": "openai_streaming_video_output",
    "/v1/audio/speech/batch": "openai_serving_speech",
    "/v1/audio/speech/stream": "openai_streaming_speech",
    "/v1/video/chat/stream": "openai_streaming_video",
    "/v1/duplex": "openai_serving_duplex",
    "/v1/realtime": "openai_realtime_handler",
    "/v1/realtime/robot/openpi": "openai_serving_realtime_robot",
}


def get_route_handler(state: State, path: str) -> object | None:
    return getattr(state, _ROUTE_HANDLERS[path], None)


def handler_exposure(state: State, path: str) -> Exposure:
    if path not in _ROUTE_HANDLERS:
        return "unknown"
    return "enabled" if get_route_handler(state, path) is not None else "disabled"


def get_ming3_chat_handler(state: State) -> object | None:
    chat = get_route_handler(state, "/v1/chat/completions")
    return chat if getattr(chat, "_ming3_speech_service", None) is not None else None


def get_image_stage_configs(state: State, engine_client: object) -> list[object]:
    # The request compatibility layer takes precedence over the engine view.
    return list(getattr(state, "stage_configs", None) or getattr(engine_client, "stage_configs", None) or ())


def image_generation_error(stages: Sequence[object]) -> str | None:
    if not stages:
        return "Stage configs not found. Start server with a multi-stage omni model."
    if not any(is_image_generation_stage(stage) for stage in stages):
        return "No image generation stage found in multi-stage pipeline."
    return None


def image_edit_error(stages: Sequence[object]) -> str | None:
    if not any(get_stage_type(stage) == "diffusion" for stage in stages):
        return "No diffusion stage found in multi-stage pipeline."
    return None


def requires_image_chat(stages: Sequence[object]) -> bool:
    return len(stages) > 1


def get_video_stage_configs(stages: Sequence[object] | None, engine_client: object) -> list[object]:
    return list(stages or getattr(engine_client, "stage_configs", None) or ())


def video_generation_error(stages: list[object]) -> str | None:
    if not stages:
        return "Stage configs not found. Start server with an omni diffusion model."
    if not is_video_generation_pipeline(stages):
        return "No final video output stage found in video generation pipeline."
    return None


def single_api_exposure(state: State) -> Exposure:
    count = getattr(state, "api_server_count", None)
    if type(count) is not int or count < 1:
        return "unknown"
    return "enabled" if count == 1 else "disabled"


def duplex_admission_error(state: State) -> str | None:
    count = getattr(state, "api_server_count", None)
    if count is None or (type(count) is int and count == 1):
        return None
    if type(count) is not int or count < 1:
        return "Duplex sessions require initialized API worker topology."
    engine_client = getattr(state, "engine_client", None)
    config = getattr(engine_client, "duplex_session_config", None)
    capabilities = getattr(engine_client, "duplex_capabilities", None)
    if getattr(config, "replica_policy", None) != "least_sessions":
        return "Duplex sessions with multiple API workers require replica_policy=least_sessions."
    if getattr(capabilities, "supports_session_resume", None) is not False:
        # Session credentials and replay state belong to each frontend;
        # the shared socket cannot route reconnects back to their owner.
        return "Duplex session resume routing across API workers is not supported."
    return None

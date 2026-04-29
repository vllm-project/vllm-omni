# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import base64
import dataclasses
import io
import json
import multiprocessing
import multiprocessing.forkserver as forkserver
import os

# Image generation API imports
import random
import time
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from numbers import Integral
from typing import Annotated, Any, Literal, cast

import httpx
import numpy as np
import vllm.envs as envs
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field
from starlette.datastructures import State
from starlette.routing import Route
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.launcher import serve_http, terminate_if_errored
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.mcp.tool_server import DemoToolServer, MCPToolServer, ToolServer
from vllm.entrypoints.openai.api_server import build_app as build_openai_app
from vllm.entrypoints.openai.api_server import setup_server as setup_openai_server
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

# yapf conflicts with isort for this block
# yapf: disable
# yapf: enable
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.realtime.serving import OpenAIServingRealtime
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.openai.server_utils import get_uvicorn_log_config
from vllm.entrypoints.openai.speech_to_text.serving import (
    OpenAIServingTranscription,
    OpenAIServingTranslation,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.classify.serving import ServingClassification
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding as OpenAIServingEmbedding
from vllm.entrypoints.pooling.pooling.serving import OpenAIServingPooling
from vllm.entrypoints.pooling.score.serving import ServingScores
from vllm.entrypoints.serve.disagg.serving import ServingTokens

# vLLM moved `base` from openai.basic.api_router to serve.instrumentator.basic.
# Keep a fallback for older/newer upstream layouts during rebase windows.
from vllm.entrypoints.serve.instrumentator.basic import base
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.entrypoints.utils import (
    create_error_response,
    load_aware_call,
    process_lora_modules,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.tasks import POOLING_TASKS
from vllm.tool_parsers import ToolParserManager
from vllm.utils import random_uuid
from vllm.utils.system_utils import decorate_logs
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError
from vllm_omni.entrypoints.openai.image_api_utils import (
    SUPPORTED_LAYERED_RESOLUTIONS,
    encode_image_base64,
    parse_size,
    validate_layered_layers,
)
from vllm_omni.entrypoints.openai.protocol.audio import (
    BatchSpeechRequest,
    OpenAICreateAudioGenerateRequest,
    OpenAICreateSpeechRequest,
)
from vllm_omni.entrypoints.openai.protocol.images import (
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from vllm_omni.entrypoints.openai.protocol.videos import (
    SecondStr,
    SizeStr,
    VideoDeleteResponse,
    VideoError,
    VideoGenerationRequest,
    VideoGenerationStatus,
    VideoListResponse,
    VideoResponse,
)
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection
from vllm_omni.entrypoints.openai.serving_audio_generate import OmniOpenAIServingAudioGenerate
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_speech_stream import OmniStreamingSpeechHandler
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo, ReferenceImage
from vllm_omni.entrypoints.openai.serving_video_stream import OmniStreamingVideoHandler
from vllm_omni.entrypoints.openai.stage_params import (
    build_stage_sampling_params_list,
    get_default_sampling_params_list,
)
from vllm_omni.entrypoints.openai.storage import STORAGE_MANAGER
from vllm_omni.entrypoints.openai.stores import VIDEO_STORE, VIDEO_TASKS
from vllm_omni.entrypoints.openai.utils import get_stage_type, parse_lora_request
from vllm_omni.entrypoints.openai.video_api_utils import decode_input_reference
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

logger = init_logger(__name__)
router = APIRouter()

MAX_UINT32_SEED = 2**32 - 1
profiler_router = APIRouter()


def _should_enable_profiler_endpoints(stage_configs: list | None) -> bool:
    """Check if any stage has profiler_config set in its engine_args."""
    if not stage_configs:
        return False
    for stage in stage_configs:
        engine_args = stage.get("engine_args") if isinstance(stage, dict) else getattr(stage, "engine_args", None)
        if engine_args is None:
            continue
        profiler_config = (
            engine_args.get("profiler_config")
            if isinstance(engine_args, dict)
            else getattr(engine_args, "profiler_config", None)
        )
        if profiler_config is not None:
            profiler = (
                profiler_config.get("profiler")
                if isinstance(profiler_config, dict)
                else getattr(profiler_config, "profiler", None)
            )
            if profiler is not None:
                return True
    return False


class ProfileRequest(BaseModel):
    """Request model for profiling endpoints."""

    stages: list[int] | None = Field(
        default=None,
        description="List of stage IDs to profile. If None, profiles all stages.",
    )


def _remove_route_from_router(
    router: APIRouter,
    path: str,
    methods: set[str] | None = None,
) -> None:
    methods_set = {method.upper() for method in methods} if methods else None
    for route in list(router.routes):
        if getattr(route, "path", None) != path:
            continue
        if methods_set is not None:
            route_methods = {method.upper() for method in (getattr(route, "methods", None) or set())}
            if not (route_methods & methods_set):
                continue
        router.routes.remove(route)


ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


async def _get_vllm_config(engine_client: EngineClient) -> Any:
    if hasattr(engine_client, "get_vllm_config"):
        return await engine_client.get_vllm_config()
    return getattr(engine_client, "vllm_config", None)


def _remove_route_from_app(app, path: str, methods: set[str] | None = None):
    """Remove a route from the app by path and optionally by methods.

    OMNI: used to override upstream /v1/chat/completions with omni behavior.
    """
    routes_to_remove = []
    for route in app.routes:
        if isinstance(route, Route) and route.path == path:
            if methods is None or (hasattr(route, "methods") and route.methods & methods):
                routes_to_remove.append(route)

    for route in routes_to_remove:
        app.routes.remove(route)


def _register_omni_exception_handlers(app) -> None:
    """Override upstream vLLM exception handlers with Omni-aware versions.

    The upstream ``engine_error_handler`` is designed for ``AsyncLLM`` (single
    EngineCore process).  Omni uses a multi-stage orchestrator with different
    health semantics, so we register our own handlers that:

    - Log multi-stage diagnostic info (orchestrator liveness, per-stage health)
      when an ``EngineDeadError`` is caught.
    - Call ``terminate_if_errored``
    - Return an OpenAI-compatible error JSON response.
    """

    async def omni_engine_error_handler(
        req: Request,
        exc: EngineDeadError | EngineGenerateError,
    ):
        request_id = _get_request_id_from_request(req)

        if req.app.state.args.log_error_stack:
            logger.exception("Engine Exception caught. Request id: %s", request_id)

        return _create_engine_error_json_response(req, exc)

    app.exception_handler(EngineGenerateError)(omni_engine_error_handler)
    app.exception_handler(EngineDeadError)(omni_engine_error_handler)


def _get_request_id_from_request(req: Request) -> str | None:
    return req.state.request_metadata.request_id if hasattr(req.state, "request_metadata") else None


def _build_engine_error_payload(
    exc: EngineDeadError | EngineGenerateError,
    *,
    request_id: str | None,
) -> tuple[dict[str, Any], int]:
    err = create_error_response(exc)
    payload = err.model_dump()
    error_body = payload.get("error", {})

    error_body["request_id"] = request_id
    error_body["error_stage_id"] = getattr(exc, "error_stage_id", None)

    return payload, err.error.code


def _create_engine_error_json_response(
    req: Request,
    exc: EngineDeadError | EngineGenerateError,
) -> JSONResponse:
    request_id = _get_request_id_from_request(req)
    error_stage_id = getattr(exc, "error_stage_id", None)
    engine = req.app.state.engine_client

    if isinstance(exc, EngineDeadError):
        # Log Omni-specific diagnostic information for dead engines.
        orchestrator_alive = engine.engine.is_alive() if hasattr(engine, "engine") else "N/A"
        logger.error(
            "EngineDeadError: orchestrator_alive=%s, errored=%s, request_id=%s, error_stage_id=%s",
            orchestrator_alive,
            engine.errored,
            request_id,
            error_stage_id,
        )

    terminate_if_errored(
        server=req.app.state.server,
        engine=engine,
    )

    payload, status_code = _build_engine_error_payload(exc, request_id=request_id)
    return JSONResponse(content=payload, status_code=status_code)


class _DiffusionServingModels:
    """Minimal OpenAIServingModels implementation for diffusion-only servers.

    vLLM's /v1/models route expects `app.state.openai_serving_models` to expose
    `show_available_models()`. In pure diffusion mode we don't initialize the
    full OpenAIServingModels (it depends on LLM-specific processors), so we
    provide a lightweight fallback.
    """

    class _NullModelConfig:
        def __getattr__(self, name):
            return None

    class _Unsupported:
        def __init__(self, name: str):
            self.name = name

        def __call__(self, *args, **kwargs):
            raise NotImplementedError(f"{self.name} is not supported in diffusion mode")

        def __getattr__(self, attr):
            raise NotImplementedError(f"{self.name}.{attr} is not supported in diffusion mode")

    def __init__(self, base_model_paths: list[BaseModelPath]) -> None:
        self._base_model_paths = base_model_paths
        self.model_config = self._NullModelConfig()

    @property
    def base_model_paths(self) -> list[BaseModelPath]:
        return self._base_model_paths

    def is_base_model(self, model_name: str) -> bool:
        return any(p.name == model_name for p in self._base_model_paths)

    def __getattr__(self, name):
        return self._Unsupported(name)

    async def show_available_models(self) -> ModelList:
        return ModelList(
            data=[
                ModelCard(
                    id=base_model.name,
                    root=base_model.model_path,
                    permission=[ModelPermission()],
                )
                for base_model in self._base_model_paths
            ]
        )


# Server entry points


async def omni_run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server.

    Unified entry point that automatically handles both LLM and Diffusion models
    through AsyncOmni, which manages multi-stage pipelines.
    """
    # Suppress Pydantic serialization warnings globally for multimodal content
    # (e.g., when ChatMessage.content is a list instead of str)
    import warnings as warnings_module

    warnings_module.filterwarnings("ignore", message=".*Pydantic.*serialization.*", category=UserWarning)
    warnings_module.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*", category=UserWarning)

    # Add process-specific prefix to stdout and stderr.
    decorate_logs("APIServer")

    listen_address, sock = setup_openai_server(args)

    # Unified use of omni_run_server_worker, AsyncOmni automatically handles LLM and Diffusion models
    await omni_run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def omni_run_server_worker(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)
    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        from vllm.reasoning import ReasoningParserManager

        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    # Load logging config for uvicorn if specified
    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_omni(
        args,
        client_config=client_config,
    ) as engine_client:
        supported_tasks: tuple[str, ...]
        if hasattr(engine_client, "get_supported_tasks"):
            supported_tasks = tuple(await engine_client.get_supported_tasks())
        else:
            supported_tasks = ("generate",)
        if not supported_tasks:
            # Only default to "generate" when get_supported_tasks is not implemented;
            # TTS-only models intentionally return an empty set.
            if not hasattr(engine_client, "get_supported_tasks"):
                supported_tasks = ("generate",)

        # OMNI: Pass supported_tasks to build_app (required by upstream vLLM)
        app = build_openai_app(args, supported_tasks)
        # OMNI: Remove upstream routes that we override with omni-specific handlers
        _remove_route_from_app(app, "/v1/chat/completions", {"POST"})
        _remove_route_from_app(app, "/v1/models", {"GET"})  # Remove upstream /v1/models to use omni's handler
        app.include_router(router)

        # OMNI: Override upstream exception handlers with Omni-aware versions
        # that understand the multi-stage orchestrator lifecycle.
        _register_omni_exception_handlers(app)

        await omni_init_app_state(engine_client, app.state, args)

        # Conditionally register profiler endpoints based on stage YAML configs
        stage_configs = engine_client.stage_configs if hasattr(engine_client, "stage_configs") else None
        if _should_enable_profiler_endpoints(stage_configs):
            logger.warning("Profiler endpoints are enabled. This should ONLY be used for local development!")
            app.include_router(profiler_router)

        vllm_config = await _get_vllm_config(engine_client)

        # Check if pure diffusion mode (vllm_config will be None)
        is_pure_diffusion = vllm_config is None
        if is_pure_diffusion:
            logger.info(
                "Starting vLLM API server (pure diffusion mode) on %s",
                listen_address,
            )
        else:
            logger.info(
                "Starting vLLM API server %d on %s",
                vllm_config.parallel_config._api_process_rank,
                listen_address,
            )
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            ssl_ciphers=args.ssl_ciphers,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        state = getattr(app, "state", None)
        serving_speech = getattr(state, "openai_serving_speech", None) if state is not None else None
        if serving_speech is not None:
            serving_speech.shutdown()
        sock.close()


@asynccontextmanager
async def build_async_omni(
    args: Namespace,
    *,
    disable_frontend_multiprocessing: bool | None = None,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    """Build an AsyncOmni instance from command-line arguments.

    Creates an async context manager that yields an AsyncOmni instance
    configured from the provided arguments. Handles forkserver setup if
    needed and ensures proper cleanup on exit.

    Args:
        args: Parsed command-line arguments containing model and configuration
        disable_frontend_multiprocessing: Optional flag to disable frontend
            multiprocessing
        client_config: Optional client configuration dictionary

    Yields:
        EngineClient instance (AsyncOmni) ready for use
    """
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        # The executor is expected to be mp.
        # Pre-import heavy modules in the forkserver process
        logger.debug("Setup forkserver with pre-imports")
        multiprocessing.set_start_method("forkserver")
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
        logger.debug("Forkserver setup complete!")

    # Context manager to handle async_omni lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    async with build_async_omni_from_stage_config(
        args,
        disable_frontend_multiprocessing=disable_frontend_multiprocessing,
    ) as async_omni:
        yield async_omni


@asynccontextmanager
async def build_async_omni_from_stage_config(
    args: Namespace,
    *,
    disable_frontend_multiprocessing: bool = False,
) -> AsyncIterator[EngineClient]:
    """Create AsyncOmni from stage configuration.

    Creates an AsyncOmni instance either in-process or using multiprocess
    RPC. Loads stage configurations from the model or from a specified path.

    Args:
        args: Parsed command-line arguments containing model and stage configs
        disable_frontend_multiprocessing: Flag to disable frontend multiprocessing
            for compatibility with existing CLI options
        client_config: Optional client configuration dictionary

    Yields:
        EngineClient instance (AsyncOmni) ready for use

    Note:
        Stage configurations are loaded from args.stage_configs_path if provided,
        otherwise from the model's default configuration.
    """

    if disable_frontend_multiprocessing:
        logger.warning("Ignoring --disable-frontend-multiprocessing for AsyncOmni runtime.")

    async_omni: EngineClient | None = None

    try:
        kwargs = vars(args).copy()
        kwargs.pop("model", None)
        async_omni = AsyncOmni(model=args.model, **kwargs)

        # # Don't keep the dummy data in memory
        # await async_llm.reset_mm_cache()

        yield async_omni
    finally:
        if async_omni:
            async_omni.shutdown()


async def omni_init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
) -> None:
    """Initialize the FastAPI application state for omni API server.

    Sets up the application state with model information, request logger,
    and other server configuration needed for handling API requests.
    Automatically detects pure diffusion mode (single diffusion stage) and
    handles it appropriately.

    Args:
        engine_client: Engine client instance (AsyncOmni)
        state: FastAPI application state object to initialize
        args: Parsed command-line arguments
    """
    # Get vllm_config from engine_client (following 0.14.0 pattern)
    vllm_config = await _get_vllm_config(engine_client)

    # Detect if it's pure Diffusion mode (single stage and is Diffusion)
    is_pure_diffusion = False
    if hasattr(engine_client, "stage_configs") and engine_client.stage_configs:
        stage_configs = engine_client.stage_configs
        if len(stage_configs) == 1:
            stage_type = get_stage_type(stage_configs[0])
            if stage_type == "diffusion":
                is_pure_diffusion = True
                logger.info("Detected pure diffusion mode (single diffusion stage)")

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    base_model_paths = [BaseModelPath(name=name, model_path=args.model) for name in served_model_names]
    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.args = args

    # For omni models
    state.stage_configs = engine_client.stage_configs if hasattr(engine_client, "stage_configs") else None
    model_name = served_model_names[0] if served_model_names else args.model

    # Pure Diffusion mode: use simplified initialization logic
    if is_pure_diffusion:
        state.vllm_config = None
        state.diffusion_engine = engine_client
        state.openai_serving_models = _DiffusionServingModels(base_model_paths)
        # OMNI: tokenization endpoints are not supported in pure diffusion mode.
        state.openai_serving_tokenization = None

        # Use for_diffusion method to create chat handler
        state.openai_serving_chat = OmniOpenAIServingChat.for_diffusion(
            diffusion_engine=engine_client,  # type: ignore
            model_name=model_name,
        )

        # audio related
        state.openai_serving_speech = None
        state.openai_serving_audio_generate = OmniOpenAIServingAudioGenerate.for_diffusion(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            model_name=model_name,
        )

        # video related
        diffusion_stage_configs = engine_client.stage_configs if hasattr(engine_client, "stage_configs") else None
        state.openai_serving_video = OmniOpenAIServingVideo.for_diffusion(
            diffusion_engine=engine_client,  # type: ignore
            model_name=model_name,
            stage_configs=diffusion_stage_configs,
        )

        state.openai_serving_speech = OmniOpenAIServingSpeech.for_diffusion(
            diffusion_engine=engine_client,
            model_name=model_name,
            stage_configs=diffusion_stage_configs,
        )
        state.openai_streaming_speech = None
        state.openai_streaming_video = None

        state.enable_server_load_tracking = getattr(args, "enable_server_load_tracking", False)
        state.server_load_metrics = 0
        logger.info("Pure diffusion API server initialized for model: %s", model_name)
        return

    # LLM or multi-stage mode: use standard initialization logic
    if vllm_config is None:
        # Try to get vllm_config from engine_client
        vllm_config = await _get_vllm_config(engine_client)
        if vllm_config is None:
            logger.warning("vllm_config is None, some features may not work correctly")

    state.vllm_config = vllm_config

    # Get supported tasks
    supported_tasks: set[str] = {"generate"}
    if hasattr(engine_client, "get_supported_tasks"):
        supported_tasks = set(await engine_client.get_supported_tasks())
    logger.info("Supported tasks: %s", supported_tasks)

    resolved_chat_template = load_chat_template(args.chat_template)

    if args.tool_server == "demo":
        tool_server: ToolServer | None = DemoToolServer()
        assert isinstance(tool_server, DemoToolServer)
        await tool_server.init_and_validate()
    elif args.tool_server:
        tool_server = MCPToolServer()
        await tool_server.add_tool_server(args.tool_server)
    else:
        tool_server = None

    # Merge default_mm_loras into the static lora_modules
    default_mm_loras = (
        vllm_config.lora_config.default_mm_loras
        if vllm_config is not None and vllm_config.lora_config is not None
        else {}
    )
    lora_modules = process_lora_modules(args.lora_modules, default_mm_loras)

    # Ensure input_processor, io_processor, and model_config exist for OpenAIServingModels compatibility
    if (
        not hasattr(engine_client, "input_processor")
        or engine_client.input_processor is None
        or not hasattr(engine_client, "io_processor")
        or engine_client.io_processor is None
        or not hasattr(engine_client, "model_config")
        or engine_client.model_config is None
    ):
        if vllm_config is not None:
            # Try to initialize processors if vllm_config is available
            try:
                from vllm.plugins.io_processors import get_io_processor
                from vllm.v1.engine.input_processor import InputProcessor

                tokenizer = await engine_client.get_tokenizer()
                if tokenizer is not None:
                    # Initialize input_processor
                    if not hasattr(engine_client, "input_processor") or engine_client.input_processor is None:
                        engine_client.input_processor = InputProcessor(
                            vllm_config=vllm_config,
                        )
                        logger.info("Initialized input_processor for AsyncOmni")

                    # Initialize model_config
                    if not hasattr(engine_client, "model_config") or engine_client.model_config is None:
                        engine_client.model_config = vllm_config.model_config
                        logger.info("Initialized model_config for AsyncOmni")

                    # Initialize io_processor
                    if not hasattr(engine_client, "io_processor") or engine_client.io_processor is None:
                        model_config = (
                            engine_client.model_config
                            if hasattr(engine_client, "model_config")
                            else vllm_config.model_config
                        )
                        io_processor_plugin = model_config.io_processor_plugin
                        renderer = getattr(engine_client, "renderer", None)
                        if renderer is None:
                            from vllm.renderers import renderer_from_config

                            renderer = renderer_from_config(vllm_config)
                            engine_client.renderer = renderer
                        engine_client.io_processor = get_io_processor(vllm_config, renderer, io_processor_plugin)
                        logger.info("Initialized io_processor for AsyncOmni")
                else:
                    logger.warning("Cannot initialize processors: tokenizer is None. OpenAIServingModels may fail.")
            except Exception as e:
                logger.warning(
                    "Failed to initialize processors for AsyncOmni: %s. OpenAIServingModels may fail.",
                    e,
                )
        else:
            logger.warning("Cannot initialize processors: vllm_config is None. OpenAIServingModels may fail.")

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=lora_modules,
    )
    await state.openai_serving_models.init_static_loras()

    state.openai_serving_render = OpenAIServingRender(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        io_processor=engine_client.io_processor,
        model_registry=state.openai_serving_models.registry,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )

    state.openai_serving_responses = (
        OpenAIServingResponses(
            engine_client,
            state.openai_serving_models,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            tool_server=tool_server,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_log_outputs=args.enable_log_outputs,
        )
        if "generate" in supported_tasks
        else None
    )
    state.openai_serving_chat = (
        OmniOpenAIServingChat(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            default_chat_template_kwargs=args.default_chat_template_kwargs,
            trust_request_chat_template=args.trust_request_chat_template,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_log_outputs=args.enable_log_outputs,
            enable_log_deltas=args.enable_log_deltas,
        )
        if "generate" in supported_tasks
        else None
    )
    # Warm up chat template processing to avoid first-request latency
    if state.openai_serving_chat is not None:
        state.openai_serving_chat.warmup()

    state.openai_serving_completion = (
        OpenAIServingCompletion(
            engine_client,
            state.openai_serving_models,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "generate" in supported_tasks
        else None
    )
    state.openai_serving_pooling = (
        OpenAIServingPooling(
            engine_client,
            state.openai_serving_models,
            state.openai_serving_render,
            supported_tasks=tuple(supported_tasks),
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
        )
        if any(task in POOLING_TASKS for task in supported_tasks)
        else None
    )
    state.openai_serving_embedding = (
        OpenAIServingEmbedding(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
        )
        if "embed" in supported_tasks
        else None
    )
    state.openai_serving_classification = (
        ServingClassification(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
        )
        if "classify" in supported_tasks
        else None
    )
    state.openai_serving_scores = (
        ServingScores(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            score_template=resolved_chat_template,
            log_error_stack=args.log_error_stack,
        )
        if any(t in supported_tasks for t in ("embed", "score", "token_embed"))
        else None
    )
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        state.openai_serving_models,
        state.openai_serving_render,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
    )
    state.openai_serving_transcription = (
        OpenAIServingTranscription(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "transcription" in supported_tasks
        else None
    )
    state.openai_serving_translation = (
        OpenAIServingTranslation(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "transcription" in supported_tasks
        else None
    )
    state.anthropic_serving_messages = (
        AnthropicServingMessages(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            default_chat_template_kwargs=args.default_chat_template_kwargs,
        )
        if "generate" in supported_tasks
        else None
    )
    state.serving_tokens = (
        ServingTokens(
            engine_client,
            state.openai_serving_models,
            state.openai_serving_render,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_log_outputs=args.enable_log_outputs,
            force_no_detokenize=args.tokens_only,
        )
        if "generate" in supported_tasks
        else None
    )

    state.openai_serving_speech = OmniOpenAIServingSpeech(
        engine_client, state.openai_serving_models, request_logger=request_logger, model_name=model_name
    )

    state.openai_serving_audio_generate = OmniOpenAIServingAudioGenerate(
        engine_client, state.openai_serving_models, request_logger=request_logger, model_name=model_name
    )

    state.openai_streaming_speech = OmniStreamingSpeechHandler(
        speech_service=state.openai_serving_speech,
    )
    state.openai_streaming_video = (
        OmniStreamingVideoHandler(
            chat_service=state.openai_serving_chat,
            engine_client=engine_client,
        )
        if state.openai_serving_chat is not None
        else None
    )
    state.openai_serving_realtime = OpenAIServingRealtime(
        engine_client=engine_client,
        models=state.openai_serving_models,
        request_logger=request_logger,
    )

    state.openai_serving_video = OmniOpenAIServingVideo(
        engine_client,
        model_name=served_model_names[0] if served_model_names else None,
        stage_configs=state.stage_configs,
    )

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0
    state.sleeping_stages = set()


def Omnivideo(request: Request) -> OmniOpenAIServingVideo | None:
    return request.app.state.openai_serving_video


def Omnichat(request: Request) -> OmniOpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def Omnispeech(request: Request) -> OmniOpenAIServingSpeech | None:
    return request.app.state.openai_serving_speech


def OmniAudioGenerate(request: Request) -> OmniOpenAIServingAudioGenerate | None:
    return getattr(request.app.state, "openai_serving_audio_generate", None)


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, "")
    handler = Omnichat(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "openai_serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Chat Completions API",
            )
        return base_server.create_error_response(message="The model does not support Chat Completions API")
    try:
        generator = await handler.create_chat_completion(request, raw_request)
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as e:
        logger.exception("Chat completion failed: %s", e)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(),
            status_code=generator.error.code if generator.error else 400,
        )

    elif isinstance(generator, ChatCompletionResponse):
        # Completely bypass Pydantic serialization warnings for multimodal content
        # by converting to dict first, then serializing with warnings suppressed
        import json as json_lib
        import warnings as warnings_module

        # Temporarily suppress ALL Pydantic UserWarnings during serialization
        with warnings_module.catch_warnings():
            warnings_module.filterwarnings("ignore", category=UserWarning)
            warnings_module.filterwarnings("ignore", message=".*Pydantic.*", category=UserWarning)
            try:
                # Use serialize_as_any=True to bypass type checking
                response_dict = generator.model_dump(mode="json", serialize_as_any=True, warnings="none")
                return JSONResponse(
                    content=response_dict,
                    headers=metrics_header(metrics_header_format),
                )
            except Exception:
                # Fallback: convert to JSON string and parse back to avoid any serialization issues
                try:
                    response_json = generator.model_dump_json(warnings="none", serialize_as_any=True)
                    response_dict = json_lib.loads(response_json)
                    return JSONResponse(
                        content=response_dict,
                        headers=metrics_header(metrics_header_format),
                    )
                except Exception:
                    # Last resort: regular dump with warnings suppressed
                    with warnings_module.catch_warnings():
                        warnings_module.filterwarnings("ignore", category=UserWarning)
                        return JSONResponse(
                            content=generator.model_dump(mode="json", warnings="none"),
                            headers=metrics_header(metrics_header_format),
                        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


_remove_route_from_router(router, "/v1/audio/speech", {"POST"})


@router.post(
    "/v1/audio/speech",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"audio/*": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_speech(request: OpenAICreateSpeechRequest, raw_request: Request):
    """Generate speech audio from text using the loaded TTS model.

    Args:
        request: Speech synthesis request in OpenAI-compatible format.
        raw_request: Raw FastAPI request for accessing app state.

    Returns:
        The generated audio response, or an OpenAI-style error payload when
        the request cannot be fulfilled.

    Raises:
        HTTPException: If the server does not support speech generation or the
        synthesis request fails unexpectedly.
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "openai_serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Speech API",
            )
        return base_server.create_error_response(message="The model does not support Speech API")
    try:
        result = await handler.create_speech(request, raw_request)
        if isinstance(result, ErrorResponse):
            return JSONResponse(
                content=result.model_dump(),
                status_code=result.error.code if result.error else 400,
            )
        return result
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


@router.post(
    "/v1/audio/speech/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_speech_batch(request: BatchSpeechRequest, raw_request: Request):
    handler = Omnispeech(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "openai_serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Speech API",
            )
        return base_server.create_error_response(message="The model does not support Speech API")
    try:
        result = await handler.create_speech_batch(request)
        if isinstance(result, ErrorResponse):
            return JSONResponse(
                content=result.model_dump(),
                status_code=result.error.code if result.error else 400,
            )
        return JSONResponse(content=result.model_dump())
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


@router.post(
    "/v1/audio/generate",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"audio/*": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_audio_generate(request: OpenAICreateAudioGenerateRequest, raw_request: Request):
    handler = OmniAudioGenerate(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "openai_serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Audio Generate API",
            )
        return base_server.create_error_response(message="The model does not support Audio Generate API")
    try:
        result = await handler.create_audio_generate(request, raw_request)
        if isinstance(result, ErrorResponse):
            return JSONResponse(
                content=result.model_dump(),
                status_code=result.error.code if result.error else 400,
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


@router.get(
    "/v1/audio/voices",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def list_voices(raw_request: Request):
    """List available TTS voices exposed by the loaded speech model.

    Args:
        raw_request: Raw FastAPI request for accessing app state.

    Returns:
        A JSON payload containing the sorted set of supported speaker names, or
        an OpenAI-style error response when the current server configuration
        does not support the Speech API.
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(message="The model does not support Speech API")

    # Get all speakers (both model built-in and uploaded)
    speakers = sorted(handler.supported_speakers) if handler.supported_speakers else []

    # Get uploaded speakers details
    uploaded_speakers = []
    if hasattr(handler, "uploaded_speakers"):
        for voice_name, info in handler.uploaded_speakers.items():
            voice_entry = {
                "name": info.get("name", voice_name),
                "consent": info.get("consent", ""),
                "created_at": info.get("created_at", 0),
                "file_size": info.get("file_size", 0),
                "mime_type": info.get("mime_type", ""),
                "embedding_source": info.get("embedding_source", "audio"),
                "embedding_dim": info.get("embedding_dim"),
            }
            if info.get("ref_text"):
                voice_entry["ref_text"] = info["ref_text"]
            if info.get("speaker_description"):
                voice_entry["speaker_description"] = info["speaker_description"]
            uploaded_speakers.append(voice_entry)

    return JSONResponse(content={"voices": speakers, "uploaded_voices": uploaded_speakers})


@router.post(
    "/v1/audio/voices",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def upload_voice(
    raw_request: Request,
    audio_sample: UploadFile | None = File(None),
    speaker_embedding: str | None = Form(None),
    consent: str = Form(...),
    name: str = Form(...),
    ref_text: str | None = Form(None),
    speaker_description: str | None = Form(None),
):
    """Upload a new voice for voice cloning.

    Accepts either an audio file or a pre-computed speaker embedding vector.
    These are mutually exclusive: provide one or the other.

    When using ``audio_sample``, the server stores the audio and extracts the
    speaker embedding on first use (Base task models only).

    When using ``speaker_embedding``, pass a JSON-encoded list of floats
    (1024-dim for 0.6B, 2048-dim for 1.7B). The voice is stored as a
    safetensors file and is immediately ready for use.

    Args:
        audio_sample: Audio file (max 10MB). Mutually exclusive with speaker_embedding.
        speaker_embedding: JSON-encoded float list. Mutually exclusive with audio_sample.
        consent: Consent recording ID
        name: Name for the new voice
        ref_text: Optional transcript of the audio for ICL (in-context
            learning) mode. When provided, voice clone requests using this
            voice will produce higher quality results.
        speaker_description: Optional free-form description of the voice
            (e.g. "warm speaker", "energetic narrator").
        raw_request: Raw FastAPI request

    Returns:
        JSON response with voice information
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(message="The model does not support Speech API")

    try:
        if speaker_embedding is not None and audio_sample is not None:
            return base(raw_request).create_error_response(
                message="'audio_sample' and 'speaker_embedding' are mutually exclusive"
            )
        if speaker_embedding is not None:
            result = await handler.upload_voice_embedding(speaker_embedding, consent, name)
        elif audio_sample is not None:
            result = await handler.upload_voice(
                audio_sample,
                consent,
                name,
                ref_text=ref_text,
                speaker_description=speaker_description,
            )
        else:
            return base(raw_request).create_error_response(
                message="Either 'audio_sample' or 'speaker_embedding' must be provided"
            )

        return JSONResponse(content={"success": True, "voice": result})

    except ValueError as e:
        return base(raw_request).create_error_response(message=str(e))
    except Exception as e:
        logger.exception(f"Failed to upload voice: {e}")
        return base(raw_request).create_error_response(message=f"Failed to upload voice: {str(e)}")


@router.delete(
    "/v1/audio/voices/{name}",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def delete_voice(name: str, raw_request: Request):
    """Delete an uploaded voice.

    Deletes the voice sample and associated metadata. Also removes any
    cached voice clone prompts for this voice.

    Args:
        name: Name of the voice to delete
        raw_request: Raw FastAPI request

    Returns:
        JSON response indicating success or failure
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(message="The model does not support Speech API")

    try:
        # Delete the voice
        success = await handler.delete_voice(name)
        if not success:
            return JSONResponse(
                content={"success": False, "error": f"Voice '{name}' not found"},
                status_code=HTTPStatus.NOT_FOUND.value,
            )

        return JSONResponse(content={"success": True, "message": f"Voice '{name}' deleted successfully"})

    except ValueError as e:
        return base(raw_request).create_error_response(message=str(e))
    except Exception as e:
        logger.exception(f"Failed to delete voice '{name}': {e}")
        return base(raw_request).create_error_response(message=f"Failed to delete voice: {str(e)}")


@router.websocket("/v1/audio/speech/stream")
async def streaming_speech(websocket: WebSocket):
    """WebSocket endpoint for streaming text input TTS.

    Accepts text incrementally, splits at sentence boundaries, and
    returns audio per sentence. See serving_speech_stream.py for protocol.
    """
    handler = getattr(websocket.app.state, "openai_streaming_speech", None)
    if handler is None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "message": "Streaming speech is not available",
            }
        )
        await websocket.close()
        return
    await handler.handle_session(websocket)


@router.websocket("/v1/video/chat/stream")
async def streaming_video_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming video input chat."""
    handler = getattr(websocket.app.state, "openai_streaming_video", None)
    if handler is None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "message": "Streaming video chat is not available",
            }
        )
        await websocket.close()
        return
    await handler.handle_session(websocket)


@router.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket):
    """WebSocket endpoint for OpenAI-style realtime interactions."""
    engine_client = getattr(websocket.app.state, "engine_client", None)
    if engine_client is not None and getattr(engine_client, "async_chunk", False):
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "error": (
                    "The /v1/realtime API is not supported when async_chunk is enabled on the server. "
                    "Use a stage configuration with async_chunk disabled and restart the server before using "
                    "this endpoint."
                ),
                "code": "unsupported",
            }
        )
        await websocket.close()
        return
    serving = getattr(websocket.app.state, "openai_serving_realtime", None)
    if serving is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "error": "Realtime API is not available", "code": "unsupported"})
        await websocket.close()
        return
    connection = RealtimeConnection(websocket, serving)
    await connection.handle_connection()


# Health and Model endpoints for diffusion mode


# Remove existing health endpoint if present (from vllm imports)
# to ensure our handler takes precedence
_remove_route_from_router(router, "/health")


@router.get("/health")
async def health(raw_request: Request) -> JSONResponse:
    """Health check endpoint that works for both LLM and diffusion modes.

    Returns 200 OK if the server is healthy, 503 if the engine is dead.
    Mirrors vLLM upstream's /health which catches EngineDeadError -> 503.
    """
    engine_client = getattr(raw_request.app.state, "engine_client", None) or getattr(
        raw_request.app.state, "diffusion_engine", None
    )
    if engine_client is None:
        return JSONResponse(
            content={"status": "unhealthy", "reason": "No engine initialized"},
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        )

    try:
        await engine_client.check_health()
        return JSONResponse(content={"status": "healthy"})
    except EngineDeadError:
        return JSONResponse(
            content={"status": "unhealthy"},
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        )


# Remove existing models endpoint if present (from vllm imports)
# to ensure our handler takes precedence
_remove_route_from_router(router, "/v1/models")


@router.get("/v1/models")
async def show_available_models(raw_request: Request) -> JSONResponse:
    """Show available models for both LLM and diffusion modes.

    Delegates to state.openai_serving_models which is set to either
    OpenAIServingModels (LLM) or _DiffusionServingModels (pure diffusion).
    """
    handler = getattr(raw_request.app.state, "openai_serving_models", None)
    if handler is not None:
        models = await handler.show_available_models()
        return JSONResponse(content=models.model_dump())
    return JSONResponse(content={"object": "list", "data": []})


# Image generation API endpoints


@router.post(
    "/v1/images/generations",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": ImageGenerationResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def generate_images(request: ImageGenerationRequest, raw_request: Request):
    """Generate images from text prompts using diffusion models.

    OpenAI DALL-E compatible endpoint for text-to-image generation.
    Only supports multi-stage omni mode with diffusion stages.

    Args:
        request: Image generation request with prompt and parameters
        raw_request: Raw FastAPI request for accessing app state

    Returns:
        ImageGenerationResponse with generated images as base64 PNG

    Raises:
        HTTPException: For validation errors, missing engine, or generation failures
    """
    # Get engine client (AsyncOmni) from app state
    engine_client, model_name, stage_configs = _get_engine_and_model(raw_request)

    if request.model is not None and request.model != model_name:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(f"Model mismatch: request specifies '{request.model}' but server is running '{model_name}'."),
        )

    try:
        # Unify request construction for any multi-stage pipeline to avoid
        # divergence between /v1/images and /v1/chat/completions.
        if len(stage_configs) > 1:
            chat_handler = getattr(raw_request.app.state, "openai_serving_chat", None)
            if chat_handler is None:
                logger.warning("openai_serving_chat is not initialized for multi-stage /v1/images/generations")
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    detail="openai_serving_chat is not initialized for multi-stage image generation.",
                )

            effective_seed = request.seed if request.seed is not None else random.randint(0, MAX_UINT32_SEED)
            extra_body: dict[str, Any] = {
                "seed": effective_seed,
                "num_outputs_per_prompt": request.n,
            }
            if request.size is not None:
                parse_size(request.size)
                width, height = parse_size(request.size)
                app_state_args = getattr(raw_request.app.state, "args", None)
                _check_max_generated_image_size(app_state_args, width, height)
                extra_body["size"] = request.size
            if request.negative_prompt is not None:
                extra_body["negative_prompt"] = request.negative_prompt
            if request.num_inference_steps is not None:
                extra_body["num_inference_steps"] = request.num_inference_steps
            if request.guidance_scale is not None:
                extra_body["guidance_scale"] = request.guidance_scale
            if request.true_cfg_scale is not None:
                extra_body["true_cfg_scale"] = request.true_cfg_scale
            if request.generator_device is not None:
                extra_body["generator_device"] = request.generator_device
            if request.lora is not None:
                # Keep /images validation semantics: invalid LoRA should fail with 400.
                _parse_lora_request(request.lora)
                extra_body["lora"] = request.lora

            generation_result = await chat_handler.generate_diffusion_images(
                prompt=request.prompt,
                extra_body=extra_body,
                request_id=f"img_gen-{random_uuid()}",
            )
            if isinstance(generation_result, ErrorResponse):
                return JSONResponse(
                    status_code=generation_result.error.code if generation_result.error else 400,
                    content=generation_result.model_dump(),
                )
            flat_images, _, _ = generation_result
            image_data = [ImageData(b64_json=encode_image_base64(img), revised_prompt=None) for img in flat_images]
            return ImageGenerationResponse(created=int(time.time()), data=image_data)

        # Build params - pass through user values directly
        prompt: OmniTextPrompt = {"prompt": request.prompt}
        if request.negative_prompt is not None:
            prompt["negative_prompt"] = request.negative_prompt
        gen_params = OmniDiffusionSamplingParams(num_outputs_per_prompt=request.n)
        extra_args = {}
        if request.use_system_prompt is not None:
            extra_args["use_system_prompt"] = request.use_system_prompt
        if request.system_prompt is not None:
            extra_args["system_prompt"] = request.system_prompt
        if extra_args:
            gen_params.extra_args = extra_args
        # Parse per-request LoRA (compatible with chat's extra_body.lora shape).
        lora_request, lora_scale = _parse_lora_request(request.lora)
        _update_if_not_none(gen_params, "lora_request", lora_request)
        _update_if_not_none(gen_params, "lora_scale", lora_scale)

        # Parse and add size if provided
        width, height = None, None
        if request.size:
            width, height = parse_size(request.size)
            size_str = f"{width}x{height}"
        else:
            size_str = "model default"

        # Keep AR stage target grid in sync with requested output size.
        # GLM-Image consumes target_h/target_w via mm_processor_kwargs.
        if width is not None and height is not None:
            prompt["mm_processor_kwargs"] = {
                "target_h": height,
                "target_w": width,
            }
            # Backward-compatible fallback for processors reading top-level fields.
            prompt["height"] = height
            prompt["width"] = width
        app_state_args = getattr(raw_request.app.state, "args", None)
        _check_max_generated_image_size(app_state_args, width, height)

        _update_if_not_none(gen_params, "width", width)
        _update_if_not_none(gen_params, "height", height)

        # 3.3 Add optional parameters ONLY if provided
        _update_if_not_none(gen_params, "num_inference_steps", request.num_inference_steps)
        _update_if_not_none(gen_params, "guidance_scale", request.guidance_scale)
        _update_if_not_none(gen_params, "true_cfg_scale", request.true_cfg_scale)
        # If seed is not provided, generate a random one to ensure
        # a proper generator is initialized in the backend.
        # This fixes issues where using the default global generator
        # might produce blurry images in some environments.
        _update_if_not_none(
            gen_params, "seed", request.seed if request.seed is not None else random.randint(0, MAX_UINT32_SEED)
        )
        _update_if_not_none(gen_params, "generator_device", request.generator_device)
        _update_if_not_none(gen_params, "layers", request.layers)

        request_id = f"img_gen-{random_uuid()}"
        raw_request.state.request_metadata = RequestResponseMetadata(request_id=request_id)

        logger.info(f"Generating {request.n} image(s) {size_str}")

        # Generate images using AsyncOmni (multi-stage mode)
        result = await _generate_with_async_omni(
            engine_client=engine_client,
            gen_params=gen_params,
            stage_configs=stage_configs,
            prompt=prompt,
            request_id=request_id,
        )

        if result is None:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No output generated from multi-stage pipeline.",
            )

        # Extract images from result
        images = _extract_images_from_result(result)

        logger.info(f"Successfully generated {len(images)} image(s)")

        # Determine output format (default to png)
        output_format = _choose_output_format(request.output_format or "png", None)

        # Encode images to base64 with the specified format
        image_data = [
            ImageData(b64_json=_encode_image_base64_with_compression(img, format=output_format), revised_prompt=None)
            for img in images
        ]

        response_kwargs = {
            "created": int(time.time()),
            "data": image_data,
            "output_format": output_format,
        }
        if request.size:
            response_kwargs["size"] = size_str
        return ImageGenerationResponse(**response_kwargs)

    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e))
    except Exception as e:
        logger.exception(f"Image generation failed: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Image generation failed: {str(e)}"
        )


@router.post(
    "/v1/images/edits",
    responses={
        HTTPStatus.OK.value: {"model": ImageGenerationResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def edit_images(
    raw_request: Request,
    image: list[UploadFile] | None = File(None),
    image_array: list[UploadFile] | None = File(None, alias="image[]"),
    url: list[str] | None = Form(None),
    url_array: list[str] | None = Form(None, alias="url[]"),
    prompt: str = Form(...),
    model: str = Form(None),
    n: int = Form(1),
    size: str = Form("auto"),
    response_format: str = Form("b64_json"),
    output_format: str | None = Form("png"),
    background: str | None = Form("auto"),
    output_compression: Annotated[int, Form(ge=0, le=100)] = 100,
    user: str | None = Form(None),  # unused now
    # vllm-omni extensions for image editing
    mask_image: str | UploadFile | None = None,
    reference_image: str | UploadFile | None = None,
    # vllm-omni extensions for diffusion control
    negative_prompt: str | None = Form(None),
    num_inference_steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    strength: float | None = Form(None),
    true_cfg_scale: float | None = Form(None),
    seed: int | None = Form(None),
    generator_device: str | None = Form(None),
    # vllm-omni extension for per-request LoRA.
    lora: str | None = Form(None),  # Json string
    # vllm-omni extension for layered models (e.g., Qwen-Image-Layered)
    layers: int | None = Form(None),
    resolution: int | None = Form(None),  # See SUPPORTED_LAYERED_RESOLUTIONS
) -> ImageGenerationResponse:
    """
    OpenAI-compatible image edit endpoint.
    """
    # 1. get engine and model
    engine_client, model_name, stage_configs = _get_engine_and_model(raw_request)
    if model is not None and model != model_name:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(f"Model mismatch: request specifies '{model}' but server is running '{model_name}'."),
        )
    # 2. get output format & compression
    output_format = _choose_output_format(output_format, background)
    if response_format != "b64_json":
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Only response_format 'b64_json' is supported now.",
        )
    try:
        # 2. Build prompt & images params
        prompt: OmniTextPrompt = {"prompt": prompt}
        if negative_prompt is not None:
            prompt["negative_prompt"] = negative_prompt
        input_images_list = []
        images = image or image_array
        urls = url or url_array
        if images:
            input_images_list.extend(images)
        if urls:
            input_images_list.extend(urls)
        if not input_images_list:
            raise HTTPException(status_code=422, detail="Field 'image' or 'url' is required")
        # Reject oversized multi-image edit requests before fetching or decoding
        # any inputs. This keeps over-limit URL requests from burning network,
        # CPU, and memory on work that will be rejected anyway.
        max_input_images = _get_max_edit_input_images(raw_request, engine_client)
        if max_input_images is not None and len(input_images_list) > max_input_images:
            detail = (
                "Received multiple input images. Only a single image is supported by this model."
                if max_input_images == 1
                else (
                    f"Received {len(input_images_list)} input images. "
                    f"At most {max_input_images} images are supported by this model."
                )
            )
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=detail,
            )
        pil_images = await _load_input_images(input_images_list)
        prompt["multi_modal_data"] = {}
        prompt["multi_modal_data"]["image"] = pil_images

        if mask_image is not None:
            loaded = await _load_input_images([mask_image])
            prompt["multi_modal_data"]["mask_image"] = loaded[0]

        if reference_image is not None:
            loaded = await _load_input_images([reference_image])
            prompt["multi_modal_data"]["reference_image"] = loaded[0]

        # 3 Build sample params
        gen_params = OmniDiffusionSamplingParams()
        # 3.0 Init with system default values
        app_state_args = getattr(raw_request.app.state, "args", None)
        default_sample_param = getattr(app_state_args, "default_sampling_params", None)
        # Currently only have one diffusion stage.
        diffusion_stage_ids = [i for i, cfg in enumerate(stage_configs) if get_stage_type(cfg) == "diffusion"]
        if not diffusion_stage_ids:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                detail="No diffusion stage found in multi-stage pipeline.",
            )
        diffusion_stage_id = diffusion_stage_ids[0]
        apply_stage_default_sampling_params(
            default_sample_param,
            gen_params,
            str(diffusion_stage_id),
        )
        _update_if_not_none(gen_params, "num_outputs_per_prompt", n)
        # 3.1 Parse per-request LoRA (compatible with chat's extra_body.lora shape).
        lora_dict = _get_lora_from_json_str(lora)
        lora_request, lora_scale = _parse_lora_request(lora_dict)
        _update_if_not_none(gen_params, "lora_request", lora_request)
        _update_if_not_none(gen_params, "lora_scale", lora_scale)
        # 3.2 Validate resolution if provided
        if resolution is not None and resolution not in SUPPORTED_LAYERED_RESOLUTIONS:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Invalid resolution {resolution}. Supported resolutions: {SUPPORTED_LAYERED_RESOLUTIONS}.",
            )
        # 3.2.1 Validate layers if provided
        try:
            layers = validate_layered_layers(layers)
        except ValueError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=str(e),
            ) from e
        # 3.2.2 Check for conflicting size and resolution parameters
        if resolution is not None and size.lower() != "auto":
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Cannot specify both 'resolution' and 'size'. "
                "Use 'resolution' with size='auto', or use 'size' without 'resolution'.",
            )

        # 3.3 Parse and add size if provided
        width, height = None, None
        if size.lower() == "auto":
            if resolution is None:
                # No resolution specified, use input image size
                width, height = pil_images[0].size
            # else: let pipeline calculate dimensions based on resolution
        else:
            width, height = parse_size(size)

        _check_max_generated_image_size(app_state_args, width, height, resolution)

        size_str = f"{width}x{height}" if width is not None and height is not None else "auto"

        # Keep AR stage target grid in sync with requested output size.
        # GLM-Image consumes target_h/target_w via mm_processor_kwargs.
        if width is not None and height is not None:
            prompt["mm_processor_kwargs"] = {
                "target_h": height,
                "target_w": width,
            }
            # Backward-compatible fallback for processors reading top-level fields.
            prompt["height"] = height
            prompt["width"] = width

        _update_if_not_none(gen_params, "width", width)
        _update_if_not_none(gen_params, "height", height)

        # 3.4 Add optional parameters ONLY if provided
        _update_if_not_none(gen_params, "num_inference_steps", num_inference_steps)
        _update_if_not_none(gen_params, "guidance_scale", guidance_scale)
        _update_if_not_none(gen_params, "strength", strength)
        _update_if_not_none(gen_params, "true_cfg_scale", true_cfg_scale)
        # If seed is not provided, generate a random one to ensure
        # a proper generator is initialized in the backend.
        # This fixes issues where using the default global generator
        # might produce blurry images in some environments.
        _update_if_not_none(gen_params, "seed", seed if seed is not None else random.randint(0, MAX_UINT32_SEED))
        _update_if_not_none(gen_params, "generator_device", generator_device)
        _update_if_not_none(gen_params, "layers", layers)
        _update_if_not_none(gen_params, "resolution", resolution)

        # 4. Generate images
        request_id = f"img_edit-{random_uuid()}"
        raw_request.state.request_metadata = RequestResponseMetadata(request_id=request_id)
        logger.info(f"Generating {n} image(s) {size_str}")

        if len(stage_configs) > 1:
            # Multi-stage pipeline (e.g. GLM-Image AR+Diffusion): route through
            # the chat handler so the AR stage gets correct max_tokens and
            # target_h/w (same path as /v1/images/generations).
            chat_handler = getattr(raw_request.app.state, "openai_serving_chat", None)
            if chat_handler is None:
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    detail="openai_serving_chat is not initialized for multi-stage image editing.",
                )

            # Encode input images to base64 for generate_diffusion_images.
            import base64
            import io as _io

            ref_b64_list: list[str] = []
            for _img in pil_images:
                buf = _io.BytesIO()
                _img.save(buf, format="PNG")
                ref_b64_list.append(base64.b64encode(buf.getvalue()).decode())

            effective_seed = seed if seed is not None else random.randint(0, MAX_UINT32_SEED)
            extra_body: dict[str, Any] = {
                "seed": effective_seed,
                "num_outputs_per_prompt": n,
            }
            if width is not None:
                extra_body["width"] = width
            if height is not None:
                extra_body["height"] = height
            if negative_prompt is not None:
                extra_body["negative_prompt"] = negative_prompt
            if num_inference_steps is not None:
                extra_body["num_inference_steps"] = num_inference_steps
            if guidance_scale is not None:
                extra_body["guidance_scale"] = guidance_scale
            if strength is not None:
                extra_body["strength"] = strength
            if true_cfg_scale is not None:
                extra_body["true_cfg_scale"] = true_cfg_scale
            if layers is not None:
                extra_body["layers"] = layers
            if resolution is not None:
                extra_body["resolution"] = resolution
            if lora is not None:
                # Validate LoRA, then pass through.
                lora_dict = _get_lora_from_json_str(lora)
                _parse_lora_request(lora_dict)
                extra_body["lora"] = lora_dict

            prompt_text = prompt.get("prompt", "")
            generation_result = await chat_handler.generate_diffusion_images(
                prompt=prompt_text,
                extra_body=extra_body,
                reference_images=ref_b64_list,
                request_id=request_id,
            )
            if isinstance(generation_result, ErrorResponse):
                raise HTTPException(
                    status_code=generation_result.error.code if generation_result.error else 400,
                    detail=generation_result.message,
                )
            images, _, _ = generation_result
        else:
            # Single-stage diffusion: use the direct path.
            result = await _generate_with_async_omni(
                engine_client=engine_client,
                gen_params=gen_params,
                stage_configs=stage_configs,
                prompt=prompt,
                request_id=request_id,
            )
            images = _extract_images_from_result(result)

        logger.info(f"Successfully generated {len(images)} image(s)")

        # Encode images to base64
        image_data = [
            ImageData(
                b64_json=_encode_image_base64_with_compression(
                    img, format=output_format, output_compression=output_compression
                ),
                revised_prompt=None,
            )
            for img in images
        ]

        return ImageGenerationResponse(
            created=int(time.time()),
            data=image_data,
            output_format=output_format,
            size=size_str,
        )

    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e))
    except Exception as e:
        logger.exception(f"Image edit failed: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Image edit failed: {str(e)}")


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


def _get_max_edit_input_images(raw_request: Request, engine_client: Any) -> int | None:
    od_config = _get_diffusion_od_config(raw_request, engine_client)
    if od_config is None:
        # Preserve the existing compatibility behavior when the diffusion
        # config is not exposed on the serving surface.
        return None

    supports_multimodal_inputs = getattr(od_config, "supports_multimodal_inputs", None)
    if not isinstance(supports_multimodal_inputs, bool):
        # Older serving surfaces and mocked engines may expose a placeholder
        # object instead of a real diffusion config. Treat that as "unknown"
        # so existing single-image flows keep working.
        return None

    if not supports_multimodal_inputs:
        return 1

    max_input_images = getattr(od_config, "max_multimodal_image_inputs", None)
    if max_input_images is None:
        return None
    if isinstance(max_input_images, bool) or not isinstance(max_input_images, Integral):
        return None
    if max_input_images < 1:
        return None
    return int(max_input_images)


def _get_lora_from_json_str(lora_body):
    if lora_body is None:
        return None
    try:
        lora_dict = json.loads(lora_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid LoRA JSON string")

    if not isinstance(lora_dict, dict):
        raise HTTPException(status_code=400, detail="LoRA must be a JSON object")

    return lora_dict


def _parse_lora_request(lora_body: dict[str, Any]):
    try:
        return parse_lora_request(lora_body)
    except ValueError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(e),
        ) from e


async def _generate_with_async_omni(
    engine_client: AsyncOmni | Any,
    gen_params: Any,
    stage_configs: list[Any],
    **kwargs,
):
    engine_client = cast(AsyncOmni, engine_client)
    result = None
    normalized_stage_configs = list(stage_configs)
    if not normalized_stage_configs:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Stage configs not found. Start server with a multi-stage omni model.",
        )
    sampling_params_list = build_stage_sampling_params_list(
        normalized_stage_configs,
        get_default_sampling_params_list(engine_client),
        diffusion_params=gen_params,
        replace_diffusion_params=True,
    )

    async for output in engine_client.generate(
        sampling_params_list=sampling_params_list,
        **kwargs,
    ):
        result = output

    if result is None:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="No output generated from multi-stage pipeline.",
        )
    return result


def _check_max_generated_image_size(
    app_state_args: Any,
    width: int | None,
    height: int | None,
    resolution: int | None = None,
) -> None:
    """Raise 400 if the requested image size exceeds --max-generated-image-size."""
    max_generated_image_size = getattr(app_state_args, "max_generated_image_size", None)
    # Check max_generated_image_size
    if max_generated_image_size is None:
        return
    if width is not None and height is not None:
        if width * height > max_generated_image_size:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Requested image size {width}x{height} exceeds the maximum allowed "
                f"size of {max_generated_image_size} pixels.",
            )
    elif resolution is not None:
        # When resolution is set, the output size is resolution * resolution
        if resolution * resolution > max_generated_image_size:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Requested resolution {resolution} (max {resolution}x{resolution} pixels) "
                f"exceeds the maximum allowed size of {max_generated_image_size} pixels.",
            )


def _update_if_not_none(object: Any, key: str, val: Any) -> None:
    if val is not None:
        setattr(object, key, val)


def _normalize_image(image: Any) -> Any:
    """Normalize a single image output to a PIL-compatible format."""
    if isinstance(image, Image.Image):
        return image
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Unsupported image type: {type(image)}")
    if not np.issubdtype(image.dtype, np.integer) and not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"Unsupported dtype: {image.dtype}")
    if isinstance(image, np.ndarray):
        while image.ndim > 3:
            image = image[0]
        if image.min() < 0:
            if image.min() < -1.01 or image.max() > 1.01:
                logger.warning(
                    f"Image float range [{image.min():.2f}, {image.max():.2f}] outside expected [-1, 1]. "
                    f"Clipping to [-1, 1] before normalization."
                )
            image = np.clip(image, -1.0, 1.0) * 0.5 + 0.5
        elif image.max() > 1.01:
            logger.warning(
                f"Image float range [{image.min():.2f}, {image.max():.2f}] outside expected [0, 1]. "
                f"Clipping to [0, 1] before normalization."
            )
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        image = Image.fromarray(image)
    return image


def _extract_images_from_result(result: Any) -> list[Any]:
    images = []
    if hasattr(result, "images") and result.images:
        images = result.images
    elif hasattr(result, "request_output"):
        request_output = result.request_output
        if isinstance(request_output, dict) and request_output.get("images"):
            images = request_output["images"]
        elif hasattr(request_output, "images") and request_output.images:
            images = request_output.images
    # Handle when generate more than one image
    if images and isinstance(images[0], np.ndarray) and images[0].shape[0] > 1 and images[0].ndim == 5:
        # Unwrap batch: (N, T, H, W, C) -> [img1, img2, ...]
        images = list(images[0])
    # Flatten nested lists (e.g., from layered models like Qwen-Image-Layered).
    # Note: This only flattens one level deep. Deeper nesting is not supported.
    flattened = []
    for img in images:
        if isinstance(img, list):
            flattened.extend(img)
        else:
            flattened.append(img)
    return [_normalize_image(img) for img in flattened]


async def _load_input_images(
    inputs: list[str],
) -> list[Image.Image]:
    """
    convert to PIL.Image.Image list
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    images: list[Image.Image] = []

    for inp in inputs:
        # 1. URL + base64
        if isinstance(inp, str) and inp.startswith("data:image"):
            try:
                _, b64_data = inp.split(",", 1)
                image_bytes = base64.b64decode(b64_data)
                img = Image.open(io.BytesIO(image_bytes))
                images.append(img)
            except Exception as e:
                raise ValueError(f"Invalid base64 image: {e}")

        # 2. URL
        elif isinstance(inp, str) and inp.startswith("http"):
            async with httpx.AsyncClient(timeout=60) as client:
                try:
                    resp = await client.get(inp)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content))
                    images.append(img)
                except Exception as e:
                    raise ValueError(f"Failed to download image from URL {inp}: {e}")

        # 3. UploadFile
        elif hasattr(inp, "file"):
            try:
                img_data = await inp.read()
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            except Exception as e:
                raise ValueError(f"Failed to open uploaded file: {e}")
        else:
            raise ValueError(f"Unsupported input: {inp}")

    if not images:
        raise ValueError("No valid input images found")

    return images


def _choose_output_format(output_format: str | None, background: str | None) -> str:
    # Normalize and choose extension
    fmt = (output_format or "").lower()
    if fmt in {"jpg", "png", "webp", "jpeg"}:
        return fmt
    # If transparency requested, prefer png
    if (background or "auto").lower() == "transparent":
        return "png"
    # Default
    return "jpeg"


def _prepare_image_for_output_format(image: Image.Image, format: str) -> Image.Image:
    fmt = format.lower()
    if fmt not in {"jpg", "jpeg"}:
        return image

    if image.mode == "RGB":
        return image

    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        alpha_image = image.convert("RGBA")
        flattened = Image.new("RGB", alpha_image.size, (255, 255, 255))
        flattened.paste(alpha_image, mask=alpha_image.getchannel("A"))
        return flattened

    return image.convert("RGB")


def _encode_image_base64_with_compression(
    image: Image.Image, format: str = "png", output_compression: int = 100
) -> str:
    """Encode PIL Image to a base64 image string.

    Args:
        image: PIL Image object
        format: Output image format (e.g., "PNG", "JPEG", "WEBP")
        output_compression: Compression level (0-100%), 100 for best quality
    Returns:
        Base64-encoded image as string
    """
    buffer = io.BytesIO()
    image = _prepare_image_for_output_format(image, format)
    save_kwargs = {}
    if format in ("jpg", "jpeg", "webp"):
        save_kwargs["quality"] = output_compression
    elif format == "png":
        save_kwargs["compress_level"] = max(0, min(9, 9 - output_compression // 11))  # Map 0-100 to 9-0

    image.save(buffer, format=format, **save_kwargs)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def apply_stage_default_sampling_params(
    default_params_json: str | None,
    sampling_params: Any,
    stage_key: str,
) -> None:
    """
    Update a stage's sampling parameters with vLLM-Omni defaults.

    Args:
        default_params_json: JSON string of stage-keyed default parameters
        sampling_params: The sampling parameters object to update
        stage_key: The stage ID/key in the pipeline
    """
    if default_params_json is not None:
        default_params_dict = json.loads(default_params_json)
        if stage_key in default_params_dict:
            stage_defaults = default_params_dict[stage_key]
            for param_name, param_value in stage_defaults.items():
                if hasattr(sampling_params, param_name):
                    setattr(sampling_params, param_name, param_value)


def _resolve_video_runtime_context(raw_request: Request) -> tuple[str | None, list[Any] | None]:
    app_model_name = None
    serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
    if serving_models and getattr(serving_models, "base_model_paths", None):
        base_paths = serving_models.base_model_paths
        if base_paths:
            app_model_name = base_paths[0].name

    app_stage_configs = getattr(raw_request.app.state, "stage_configs", None)
    return app_model_name, app_stage_configs


def _parse_form_json(value: str | None, expected_type: type | None = None) -> Any:
    if value is None or value == "":
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Invalid JSON in form field.",
        ) from exc
    if expected_type is not None and not isinstance(parsed, expected_type):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"Invalid JSON in form field: expected {expected_type.__name__}, got {type(parsed).__name__}.",
        )
    return parsed


def video_response_from_request(model_name: str, req: VideoGenerationRequest) -> VideoResponse:
    resp = VideoResponse(
        model=model_name,
        status=VideoGenerationStatus.QUEUED,
        size=req.size,
        prompt=req.prompt,
    )
    resp.seconds = str(req.seconds or resp.seconds)
    return resp


def _status_code_for_video_failure(error: VideoError | None) -> int:
    if error is None:
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    if isinstance(error.code, int):
        if 400 <= error.code < 600:
            return error.code
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    if error.code == "HTTPException":
        status_text, _, _ = error.message.partition(":")
        try:
            status_code = int(status_text)
        except ValueError:
            return HTTPStatus.INTERNAL_SERVER_ERROR.value
        if 400 <= status_code < 600:
            return status_code
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    if error.code == "EngineDeadError":
        return HTTPStatus.INTERNAL_SERVER_ERROR.value
    if error.code == "EngineGenerateError":
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    return HTTPStatus.INTERNAL_SERVER_ERROR.value


def _video_error_from_exception(exc: Exception) -> VideoError:
    if isinstance(exc, HTTPException):
        message = str(exc.detail) if exc.detail else str(exc)
        return VideoError(code=exc.status_code, message=message)

    if isinstance(exc, (EngineGenerateError, EngineDeadError)):
        err = create_error_response(exc)
        return VideoError(code=err.error.code, message=err.error.message)

    return VideoError(
        code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        message=str(exc),
    )


def _cleanup_video(video_id: str, output_path: str | None):
    try:
        if output_path is not None:
            os.remove(output_path)
    except OSError:
        logger.warning("Failed to cleanup partial video file '%s' for id=%s", output_path, video_id)


async def _run_video_generation_job(
    handler: OmniOpenAIServingVideo,
    request: VideoGenerationRequest,
    video_id: str,
    reference_image: ReferenceImage | None = None,
    app_state: Any | None = None,
) -> None:
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        logger.warning("Video job %s missing before generation task started; skipping", video_id)
        return

    await VIDEO_STORE.update_fields(video_id, {"status": VideoGenerationStatus.IN_PROGRESS})
    started_at = time.perf_counter()
    output_path = None
    try:
        video_bytes, stage_durations, peak_memory_mb = await handler.generate_video_bytes(
            request, video_id, reference_image=reference_image
        )

        file_name = f"{video_id}.{job.file_extension}"
        output_path = await STORAGE_MANAGER.save(video_bytes, file_name)
        logger.info("Video request %s persisted %s output file.", video_id, output_path)

        await VIDEO_STORE.update_fields(
            video_id,
            {
                "status": VideoGenerationStatus.COMPLETED,
                "progress": 100,
                "file_name": file_name,
                "completed_at": int(time.time()),
                "inference_time_s": time.perf_counter() - started_at,
                "stage_durations": stage_durations,
                "peak_memory_mb": peak_memory_mb,
            },
        )
    except (EngineGenerateError, EngineDeadError) as exc:
        logger.exception("Video generation failed (engine error) for id=%s", video_id)

        _cleanup_video(video_id, output_path)
        await VIDEO_STORE.update_fields(
            video_id,
            {
                "status": VideoGenerationStatus.FAILED,
                "completed_at": int(time.time()),
                "error": _video_error_from_exception(exc),
                "inference_time_s": time.perf_counter() - started_at,
            },
        )
        # Background tasks can't propagate exceptions to FastAPI handlers.
        # Actively signal shutdown when the engine is dead.
        if app_state is not None and isinstance(exc, EngineDeadError):
            terminate_if_errored(
                server=app_state.server,
                engine=app_state.engine_client,
            )
    except Exception as exc:
        logger.exception("Video generation failed for id=%s", video_id)

        _cleanup_video(video_id, output_path)
        await VIDEO_STORE.update_fields(
            video_id,
            {
                "status": VideoGenerationStatus.FAILED,
                "completed_at": int(time.time()),
                "error": _video_error_from_exception(exc),
                "inference_time_s": time.perf_counter() - started_at,
            },
        )
    except asyncio.CancelledError:
        _cleanup_video(video_id, output_path)
        await VIDEO_STORE.pop(video_id)
        raise


VIDEO_SYNC_TIMEOUT_S = 600.0


async def _parse_video_form(
    raw_request: Request,
    prompt: str = Form(...),
    input_reference: UploadFile | None = File(default=None),
    image_reference: str | None = Form(default=None),
    model: str | None = Form(default=None),
    seconds: SecondStr | None = Form(default=None),
    size: SizeStr | None = Form(default=None),
    user: str | None = Form(default=None),
    width: int | None = Form(default=None),
    height: int | None = Form(default=None),
    num_frames: int | None = Form(default=None),
    fps: int | None = Form(default=None),
    num_inference_steps: int | None = Form(default=None),
    guidance_scale: float | None = Form(default=None),
    guidance_scale_2: float | None = Form(default=None),
    boundary_ratio: float | None = Form(default=None),
    flow_shift: float | None = Form(default=None),
    true_cfg_scale: float | None = Form(default=None),
    seed: int | None = Form(default=None),
    negative_prompt: str | None = Form(default=None),
    enable_frame_interpolation: bool | None = Form(default=None),
    frame_interpolation_exp: int | None = Form(default=None, ge=1),
    frame_interpolation_scale: float | None = Form(default=None, gt=0.0),
    frame_interpolation_model_path: str | None = Form(default=None),
    lora: str | None = Form(default=None),
    extra_params: str | None = Form(default=None),
) -> tuple[VideoGenerationRequest, "OmniOpenAIServingVideo", str, ReferenceImage | None]:
    """FastAPI dependency that parses video form data, validates inputs,
    resolves the handler, and decodes any reference image.

    Used by both ``POST /v1/videos`` (async) and ``POST /v1/videos/sync``.
    """
    input_reference_bytes = await input_reference.read() if input_reference is not None else None
    parsed_image_reference = _parse_form_json(image_reference)

    if parsed_image_reference is not None and input_reference_bytes is not None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Provide either input_reference or image_reference, not both.",
        )

    request_data: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "seconds": seconds,
        "size": size,
        "image_reference": parsed_image_reference,
        "user": user,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "guidance_scale_2": guidance_scale_2,
        "boundary_ratio": boundary_ratio,
        "flow_shift": flow_shift,
        "true_cfg_scale": true_cfg_scale,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "enable_frame_interpolation": enable_frame_interpolation,
        "frame_interpolation_exp": frame_interpolation_exp,
        "frame_interpolation_scale": frame_interpolation_scale,
        "frame_interpolation_model_path": frame_interpolation_model_path,
        "lora": _parse_form_json(lora, expected_type=dict),
        "extra_params": _parse_form_json(extra_params, expected_type=dict),
    }
    request_data = {k: v for k, v in request_data.items() if v is not None}
    request = VideoGenerationRequest(**request_data)

    handler = Omnivideo(raw_request)
    if handler is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Video generation handler not initialized.",
        )
    logger.info("Video generation handler: %s", type(handler).__name__)
    try:
        app_model_name, app_stage_configs = _resolve_video_runtime_context(raw_request)
        effective_model_name = handler.model_name or app_model_name or request.model or "unknown"
        if request.model is not None and effective_model_name is not None and request.model != effective_model_name:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=(
                    f"Model mismatch: request specifies '{request.model}' but server is running "
                    f"'{effective_model_name}'."
                ),
            )
        handler.set_stage_configs_if_missing(app_stage_configs)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Video generation setup failed: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"Video generation setup failed: {str(e)}",
        )

    try:
        image_data = await decode_input_reference(request.image_reference, input_reference_bytes)
    except InvalidInputReferenceError as exc:
        raise HTTPException(400, detail=str(exc) or "Invalid input reference.") from exc

    reference_image = ReferenceImage(data=image_data) if image_data is not None else None
    return request, handler, effective_model_name, reference_image


@router.post(
    "/v1/videos",
    responses={
        HTTPStatus.OK.value: {"model": VideoResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def create_video(
    raw_request: Request,
    ctx: tuple[VideoGenerationRequest, OmniOpenAIServingVideo, str, ReferenceImage | None] = Depends(_parse_video_form),
) -> VideoResponse:
    """Create an asynchronous video generation job.

    Accepts multipart form-data (see ``_parse_video_form`` for parameters),
    persists a queued job record, and starts generation in the background.
    """
    request, handler, effective_model_name, reference_image = ctx
    ref = video_response_from_request(effective_model_name, request)
    await VIDEO_STORE.upsert(ref.id, ref)
    task = asyncio.create_task(
        _run_video_generation_job(handler, request, ref.id, reference_image, app_state=raw_request.app.state)
    )
    await VIDEO_TASKS.upsert(ref.id, task)
    return ref


@router.post(
    "/v1/videos/sync",
    responses={
        HTTPStatus.OK.value: {"content": {"video/mp4": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def create_video_sync(
    raw_request: Request,
    ctx: tuple[VideoGenerationRequest, OmniOpenAIServingVideo, str, ReferenceImage | None] = Depends(_parse_video_form),
) -> Response:
    """Synchronous video generation endpoint.

    Accepts the same form parameters as ``POST /v1/videos`` but blocks until
    generation completes and returns raw video bytes (``video/mp4``) directly.
    Designed for benchmark and testing scenarios.

    Metadata is returned via response headers ``X-Request-Id``,
    ``X-Model``, and ``X-Inference-Time-S``.
    """
    request, handler, effective_model_name, reference_image = ctx
    request_id = f"video_sync-{random_uuid()}"
    raw_request.state.request_metadata = RequestResponseMetadata(request_id=request_id)
    started_at = time.perf_counter()
    try:
        video_bytes, stage_durations, peak_memory_mb = await asyncio.wait_for(
            handler.generate_video_bytes(request, request_id, reference_image=reference_image),
            timeout=VIDEO_SYNC_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=HTTPStatus.GATEWAY_TIMEOUT.value,
            detail=f"Video generation timed out after {VIDEO_SYNC_TIMEOUT_S}s.",
        )
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Sync video generation failed for request_id=%s", request_id)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"Video generation failed: {str(exc)}",
        ) from exc
    inference_time_s = time.perf_counter() - started_at

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "X-Request-Id": request_id,
            "X-Model": effective_model_name,
            "X-Inference-Time-S": f"{inference_time_s:.3f}",
            "X-Stage-Durations": json.dumps(stage_durations, separators=(",", ":")),
            "X-Peak-Memory-MB": f"{peak_memory_mb:.3f}",
        },
    )


@router.get("/v1/videos", response_model=VideoListResponse)
async def list_videos(
    after: str | None = None,
    limit: int | None = Query(None, ge=0, le=100),
    order: Annotated[Literal["asc", "desc"], Query()] = "desc",
):
    """List stored video generation jobs.

    Args:
        after: Optional cursor indicating the last seen video ID.
        limit: Optional maximum number of jobs to return.
        order: Sort order for the returned jobs by creation time.

    Returns:
        A ``VideoListResponse`` containing paginated job metadata and cursor
        information.
    """
    jobs = await VIDEO_STORE.list_values()
    jobs.sort(key=lambda j: j.created_at, reverse=order == "desc")

    if after is not None:
        idx = next((i for i, job in enumerate(jobs) if job.id == after), None)
        jobs = [] if idx is None else jobs[idx + 1 :]

    has_more = False
    if limit is not None:
        has_more = len(jobs) > limit
        jobs = jobs[:limit]

    first_id, last_id = None, None
    if len(jobs) > 0:
        first_id = jobs[0].id
        last_id = jobs[-1].id

    return VideoListResponse(data=jobs, has_more=has_more, first_id=first_id, last_id=last_id)


@router.get("/v1/videos/{video_id}", response_model=None)
async def retrieve_video(video_id: str) -> VideoResponse | JSONResponse:
    """Retrieve metadata for a previously created video job.

    Args:
        video_id: Identifier returned by ``POST /v1/videos``.

    Returns:
        The stored ``VideoResponse`` for the requested job.

    Raises:
        HTTPException: If the video job does not exist.
    """
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    if job.status == VideoGenerationStatus.FAILED:
        status_code = _status_code_for_video_failure(job.error)
        content = job.model_dump(mode="json")
        if content.get("error") is not None:
            content["error"]["code"] = status_code
        return JSONResponse(
            content=content,
            status_code=status_code,
        )
    return job


@router.delete("/v1/videos/{video_id}")
async def delete_video(video_id: str) -> VideoDeleteResponse:
    """Delete a stored video job and any generated output.

    If the job is still queued or running, this endpoint first attempts to
    cancel the in-flight generation task before removing the stored metadata.

    Args:
        video_id: Identifier of the video job to delete.

    Returns:
        A ``VideoDeleteResponse`` confirming the job was removed.

    Raises:
        HTTPException: If the video job does not exist, cancellation is still
        in progress, or output is not yet ready for a completed job.
    """
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")

    if job.status in (VideoGenerationStatus.QUEUED, VideoGenerationStatus.IN_PROGRESS):
        task = await VIDEO_TASKS.get(video_id)
        if task is not None:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=409, detail="Cancellation in progress. Please try again later.")
            except asyncio.CancelledError:
                pass

            await VIDEO_STORE.pop(video_id)
            return VideoDeleteResponse(id=job.id, deleted=True)
    elif job.status is VideoGenerationStatus.FAILED:
        if job.file_name is not None:
            try:
                await STORAGE_MANAGER.delete(job.file_name)
            except Exception:
                logger.warning("Failed to delete stored artifact for failed video job %s", video_id, exc_info=True)

        await VIDEO_STORE.pop(video_id)
        return VideoDeleteResponse(id=job.id, deleted=True)

    if job.file_name is None:
        raise HTTPException(status_code=409, detail="Video output not yet available. Please try again later.")

    await STORAGE_MANAGER.delete(job.file_name)
    await VIDEO_STORE.pop(video_id)
    return VideoDeleteResponse(id=job.id, deleted=True)


@router.get("/v1/videos/{video_id}/content")
async def download_video(video_id: str) -> FileResponse:
    """Download the generated file for a completed video job.

    Args:
        video_id: Identifier of the video job whose output should be returned.

    Returns:
        A ``FileResponse`` streaming the generated video file from local
        storage.

    Raises:
        HTTPException: If the job does not exist, is still in progress, or the
        generated file is missing from disk.
    """
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")

    if job.status == VideoGenerationStatus.FAILED:
        raise HTTPException(status_code=422, detail="Video generation failed. Check job status for error details.")
    if not job.file_name:
        raise HTTPException(status_code=404, detail="Generation is still in-progress")

    full_path = STORAGE_MANAGER.get_full_file_path(job.file_name)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Generated video file not found on disk")

    return FileResponse(path=full_path, media_type=job.media_type, filename=job.file_name)


@profiler_router.post("/start_profile")
async def start_profile(raw_request: Request, request: ProfileRequest | None = None):
    """Start profiling for the engine.

    Args:
        request: Optional request body with stages to profile.
            - stages: List of stage IDs to profile. If None, profiles all stages.

    Example:
        POST /start_profile
        {"stages": [0, 1]}  # Profile only stages 0 and 1
    """
    try:
        stages = request.stages if request else None
        logger.info("Starting profiler for stages: %s", stages if stages else "all")
        engine_client = raw_request.app.state.engine_client
        result = await engine_client.start_profile(stages=stages)
        logger.info("Profiler started.")
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Failed to start profiler: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Failed to start profiler: {str(e)}"
        )


@profiler_router.post("/stop_profile")
async def stop_profile(raw_request: Request, request: ProfileRequest | None = None):
    """Stop profiling for the engine.

    Args:
        request: Optional request body with stages to stop profiling.
            - stages: List of stage IDs to stop profiling. If None, stops all stages.

    Example:
        POST /stop_profile
        {"stages": [0, 1]}  # Stop profiling only stages 0 and 1
    """
    try:
        stages = request.stages if request else None
        logger.info("Stopping profiler for stages: %s", stages if stages else "all")
        engine_client = raw_request.app.state.engine_client
        result = await engine_client.stop_profile(stages=stages)
        logger.info("Profiler stopped.")
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Failed to stop profiler: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Failed to stop profiler: {str(e)}"
        )


class OmniSleepRequest(BaseModel):
    stage_ids: list[int]
    level: int = 2


class OmniWakeupRequest(BaseModel):
    stage_ids: list[int]


@router.post("/v1/omni/sleep")
async def omni_sleep(request: OmniSleepRequest, raw_request: Request):
    engine_client = raw_request.app.state.engine_client
    sleeping_set = raw_request.app.state.sleeping_stages
    if not hasattr(engine_client, "sleep"):
        raise HTTPException(status_code=501, detail="Engine does not support sleep")
    acks = await engine_client.sleep(stage_ids=request.stage_ids, level=request.level)
    for sid in request.stage_ids:
        sleeping_set.add(sid)
    return {"status": "SUCCESS", "acks": [dataclasses.asdict(a) if dataclasses.is_dataclass(a) else a for a in acks]}


@router.post("/v1/omni/wakeup")
async def omni_wakeup(request: OmniWakeupRequest, raw_request: Request):
    engine_client = raw_request.app.state.engine_client
    sleeping_set = raw_request.app.state.sleeping_stages
    if not any(sid in sleeping_set for sid in request.stage_ids):
        return {"status": "SKIPPED", "reason": "Target stages are not sleeping."}
    if not hasattr(engine_client, "wake_up"):
        raise HTTPException(status_code=501, detail="Engine does not support wake_up")
    acks = await engine_client.wake_up(stage_ids=request.stage_ids)
    for sid in request.stage_ids:
        if sid in sleeping_set:
            sleeping_set.remove(sid)
    return {"status": "SUCCESS", "acks": [dataclasses.asdict(a) if dataclasses.is_dataclass(a) else a for a in acks]}


if __name__ == "__main__":
    import argparse
    import asyncio

    from vllm.entrypoints.openai.cli_args import make_arg_parser

    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    parser = argparse.ArgumentParser(description="vLLM-Omni OpenAI-Compatible REST API server")
    parser = make_arg_parser(parser)
    registered_flags = set()
    for action in parser._actions:
        registered_flags.update(action.option_strings)
    if "--omni" not in registered_flags:
        parser.add_argument("--omni", action="store_true", default=False, help="Enable vLLM-Omni mode.")
    if "--enable-sleep-mode" not in registered_flags:
        parser.add_argument(
            "--enable-sleep-mode", action="store_true", default=False, help="Enable GPU memory pool for sleep mode."
        )
    nullify_stage_engine_defaults(parser)
    args = parser.parse_args()
    if not hasattr(args, "model_tag"):
        setattr(args, "model_tag", args.model)
    if hasattr(args, "model_tag") and args.model_tag is None:
        args.model_tag = args.model
    asyncio.run(omni_run_server(args))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI-compatible API server bootstrap for vLLM-Omni.

This module owns app construction, server startup, app-state initialization,
and route bodies that have not yet moved to endpoint-owned modules."""

import asyncio
import dataclasses
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
from typing import Annotated, Any, Literal

import vllm.envs as envs
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from PIL import Image
from starlette.datastructures import State
from starlette.types import ASGIApp, Receive, Scope, Send
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.chat_utils import ChatTemplateConfig, load_chat_template
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.mcp.tool_server import DemoToolServer, MCPToolServer, ToolServer
from vllm.entrypoints.openai.api_server import build_app as build_openai_app
from vllm.entrypoints.openai.api_server import setup_server as setup_openai_server
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser

# yapf conflicts with isort for this block
# yapf: disable
# yapf: enable
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.pooling.classify.serving import ServingClassification
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding as OpenAIServingEmbedding
from vllm.entrypoints.pooling.pooling.serving import ServingPooling
from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens

# vLLM moved `base` from openai.basic.api_router to serve.instrumentator.basic.
# Keep a fallback for older/newer upstream layouts during rebase windows.
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    process_lora_modules,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.orca_metrics import metrics_header
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.serve.utils.server_utils import get_uvicorn_log_config
from vllm.entrypoints.speech_to_text.realtime.serving import OpenAIServingRealtime
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.entrypoints.speech_to_text.translation.serving import (
    OpenAIServingTranslation,
)
from vllm.logger import init_logger
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tasks import POOLING_TASKS
from vllm.tool_parsers import ToolParserManager
from vllm.utils import random_uuid
from vllm.utils.system_utils import decorate_logs
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.config.endpoint_policy import (
    shutdown_unsupported_routes,
)
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai import app_state as openai_app_state
from vllm_omni.entrypoints.openai.app_state import (
    ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL,
    OmniAudioGenerate,
    OmniBatchChat,
    Omnichat,
    Omnispeech,
    _get_engine_and_model,
)
from vllm_omni.entrypoints.openai.batch_serving import OmniOpenAIServingChatBatch
from vllm_omni.entrypoints.openai.chat_template import _load_model_chat_template_json
from vllm_omni.entrypoints.openai.diffusion import (
    MAX_UINT32_SEED,
    _generate_with_async_omni,
    apply_stage_default_sampling_params,
)
from vllm_omni.entrypoints.openai.duplex_capability import should_enable_duplex_endpoint
from vllm_omni.entrypoints.openai.errors import (
    _create_speech_error_json_response,
    _error_response_to_json_response,
)
from vllm_omni.entrypoints.openai.image_api_utils import (
    SUPPORTED_LAYERED_RESOLUTIONS,
    encode_image_base64_with_compression,
    parse_size,
    validate_layered_layers,
)
from vllm_omni.entrypoints.openai.images.helpers import (
    _build_hunyuan_edit_extra_args,
    _check_max_generated_image_size,
    _choose_output_format,
    _extract_images_from_result,
    _get_max_edit_input_images,
    _load_input_images,
    _update_if_not_none,
)
from vllm_omni.entrypoints.openai.lora import _get_lora_from_json_str, _parse_lora_request
from vllm_omni.entrypoints.openai.models import serving as openai_models_serving
from vllm_omni.entrypoints.openai.protocol.audio import (
    BatchSpeechRequest,
    OpenAICreateAudioGenerateRequest,
    OpenAICreateSpeechRequest,
)
from vllm_omni.entrypoints.openai.protocol.images import (
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ResponseFormat,
)
from vllm_omni.entrypoints.openai.protocol.videos import (
    VideoDeleteResponse,
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
from vllm_omni.entrypoints.openai.serving_video import (
    OmniOpenAIServingVideo,
    ReferenceAudio,
    ReferenceImage,
    ReferenceVideo,
)
from vllm_omni.entrypoints.openai.serving_video_output_stream import OmniStreamingVideoOutputHandler
from vllm_omni.entrypoints.openai.serving_video_stream import create_streaming_video_handler
from vllm_omni.entrypoints.openai.storage import STORAGE_MANAGER, FileStorageHandle
from vllm_omni.entrypoints.openai.stores import VIDEO_STORE, VIDEO_TASKS
from vllm_omni.entrypoints.openai.utils import get_stage_type
from vllm_omni.entrypoints.openai.video.generation.helpers import (
    VIDEO_SYNC_TIMEOUT_S,
    _cleanup_video_references,
    _parse_video_form,
    _run_video_generation_job,
    _status_code_for_video_failure,
    video_response_from_request,
)
from vllm_omni.entrypoints.openpi.serving import ServingRealtimeRobotOpenPI
from vllm_omni.entrypoints.serve.omni_control.protocol import OmniSleepRequest, OmniWakeupRequest
from vllm_omni.entrypoints.serve.profile.protocol import ProfileRequest
from vllm_omni.entrypoints.serve.profile.utils import _should_enable_profiler_endpoints
from vllm_omni.entrypoints.serve.utils.errors import (
    _create_engine_error_json_response,
    _register_omni_exception_handlers,
)
from vllm_omni.entrypoints.serve.utils.routes import (
    _remove_route_from_router,
    remove_route_from_app,
)
from vllm_omni.entrypoints.utils import PureDiffusionLauncherAdapter
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.utils.forced_aligner import build_forced_aligner_config
from vllm_omni.utils.tracking_parser import TrackingArgumentParser, TrackingNamespace

logger = init_logger(__name__)
router = APIRouter()

profiler_router = APIRouter()


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
    decorate_logs("APIServer", skip_if_decorated=True)

    listen_address, sock = setup_openai_server(args, reuse_port=False)

    # Unified use of omni_run_server_worker, AsyncOmni automatically handles LLM and Diffusion models
    await omni_run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def omni_run_server_worker(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)
    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        from vllm.reasoning import ReasoningParserManager

        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    # In vllm-omni's multi-process architecture, create_engine_config() runs
    # in the subprocess (StageEngineCoreProc), not in the API server process.
    # Propagate reasoning_parser from CLI args into structured_outputs_config
    # here so that OmniOpenAIServingChat receives the correct value.
    # (Upstream vLLM does this in EngineArgs.create_engine_config().)
    if hasattr(args, "reasoning_parser") and args.reasoning_parser:
        if not args.structured_outputs_config.reasoning_parser:
            args.structured_outputs_config.reasoning_parser = args.reasoning_parser

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
        remove_route_from_app(app, "/v1/chat/completions", {"POST"})
        remove_route_from_app(app, "/v1/chat/completions/batch", {"POST"})
        remove_route_from_app(app, "/v1/models", {"GET"})  # Remove upstream /v1/models to use omni's handler
        app.include_router(router)

        # OMNI: Override upstream exception handlers with Omni-aware versions
        # that understand the multi-stage orchestrator lifecycle.
        _register_omni_exception_handlers(app)

        await omni_init_app_state(engine_client, app.state, args)

        # After initializing the app state, shut down any endpoints that are model specific
        if hasattr(engine_client, "endpoint_restrictions"):
            shutdown_unsupported_routes(app, engine_client.endpoint_restrictions)
        else:
            logger.warning("engine client has no endpoint restrictions attribute")

        # Start background processes
        await STORAGE_MANAGER.start()

        # Conditionally register profiler endpoints based on stage YAML configs
        stage_configs = engine_client.stage_configs if hasattr(engine_client, "stage_configs") else None
        if _should_enable_profiler_endpoints(stage_configs):
            logger.warning("Profiler endpoints are enabled. This should ONLY be used for local development!")
            remove_route_from_app(app, "/start_profile", frozenset({"POST"}))
            remove_route_from_app(app, "/stop_profile", frozenset({"POST"}))
            app.include_router(profiler_router)

        vllm_config = await openai_app_state._get_vllm_config(engine_client)

        # Check if pure diffusion mode (vllm_config will be None)
        is_pure_diffusion = vllm_config is None
        if is_pure_diffusion:
            logger.info(
                "Starting vLLM API server (pure diffusion mode) on %s",
                listen_address,
            )
            # The vLLM 0.27 launcher's shutdown path reads
            # engine_client.vllm_config.shutdown_timeout; AsyncOmni.vllm_config
            # is None for pure diffusion (no comprehension stage), which would
            # crash handle_shutdown and hang teardown. Wrap app.state.engine_client
            # with a shim that only overrides vllm_config and forwards everything
            # else (get_vllm_config still returns None for pure-diffusion detection).
            app.state.engine_client = PureDiffusionLauncherAdapter(
                engine_client,
                shutdown_timeout=getattr(args, "shutdown_timeout", 0),
            )
        else:
            logger.info(
                "Starting vLLM API server %d on %s",
                vllm_config.parallel_config._api_process_rank,
                listen_address,
            )

        class _TimestampMiddleware:
            """Pure-ASGI outermost wrapper that stamps HTTP request arrival time.

            Wraps the fully-built Starlette app as an outer ASGI layer so no
            Starlette internals (user_middleware, middleware_stack, etc.) are
            touched. Websocket and lifespan scopes pass through unchanged.
            """

            def __init__(self, inner: ASGIApp) -> None:
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
                if scope["type"] == "http":
                    scope.setdefault("state", {})
                    scope["state"]["request_timestamp"] = time.time()
                await self._inner(scope, receive, send)

        shutdown_task = await serve_http(
            _TimestampMiddleware(app),
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

        try:
            await shutdown_task
        finally:
            state = getattr(app, "state", None)
            serving_video = getattr(state, "openai_serving_video", None) if state is not None else None
            if serving_video is not None:
                serving_video.shutdown()
            serving_speech = getattr(state, "openai_serving_speech", None) if state is not None else None
            if serving_speech is not None:
                serving_speech.shutdown()
            sock.close()


@asynccontextmanager
async def build_async_omni(
    args: TrackingNamespace,
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
    args: TrackingNamespace,
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
        Stage configurations are loaded from ``args.deploy_config`` when provided,
        otherwise from the model's default configuration.
    """

    if disable_frontend_multiprocessing:
        logger.warning("Ignoring --disable-frontend-multiprocessing for AsyncOmni runtime.")

    async_omni: EngineClient | None = None

    # Pre-load the model config so HuggingFace registers `transformers_modules`
    # in this process — only when the user has explicitly opted in via
    # `--trust-remote-code`. Stage workers consume the same flag through their
    # deploy config, but the API server process also needs the dynamic modules
    # for ZMQ pickle deserialization of stage outputs that reference them
    # (e.g. trust_remote_code models like MiniCPM-o).
    if getattr(args, "trust_remote_code", False) and getattr(args, "model", None):
        try:
            import os

            from transformers import AutoConfig

            # Hide GPUs so the custom config code doesn't allocate CUDA memory.
            saved = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            try:
                AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            finally:
                if saved is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = saved
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        except Exception as e:
            logger.debug("Pre-loading transformers_modules failed: %s", e)

    try:
        kwargs = args.get_explicit_kwargs_dict()
        model = kwargs.pop("model", None) or args.model
        kwargs.setdefault("log_stats", not args.disable_log_stats)
        async_omni = AsyncOmni(model=model, **kwargs)

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
    vllm_config = await openai_app_state._get_vllm_config(engine_client)

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
    state.sleeping_stages = set()

    # For omni models
    state.stage_configs = engine_client.stage_configs if hasattr(engine_client, "stage_configs") else None
    model_name = served_model_names[0] if served_model_names else args.model

    # Pure Diffusion mode: use simplified initialization logic
    if is_pure_diffusion:
        state.vllm_config = None
        state.diffusion_engine = engine_client
        state.openai_serving_models = openai_models_serving._DiffusionServingModels(base_model_paths)
        # OMNI: tokenization endpoints are not supported in pure diffusion mode.
        state.serving_tokenization = None

        # Use for_diffusion method to create chat handler
        state.openai_serving_chat = OmniOpenAIServingChat.for_diffusion(
            diffusion_engine=engine_client,  # type: ignore
            model_name=model_name,
        )
        state.openai_serving_chat_batch = OmniOpenAIServingChatBatch.for_diffusion(
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
        state.openai_streaming_video_output = OmniStreamingVideoOutputHandler(
            engine_client=engine_client,
            model_name=model_name,
            stage_configs=diffusion_stage_configs,
        )

        state.openai_serving_speech = OmniOpenAIServingSpeech.for_diffusion(
            diffusion_engine=engine_client,
            model_name=model_name,
            stage_configs=diffusion_stage_configs,
            allowed_local_media_path=getattr(args, "allowed_local_media_path", ""),
            allowed_media_domains=getattr(args, "allowed_media_domains", None),
        )
        state.openai_serving_duplex = None
        state.openai_streaming_speech = None
        state.openai_streaming_video = None
        state.openai_serving_realtime_robot = ServingRealtimeRobotOpenPI.create_policy_server(
            engine_client=engine_client,
            model_name=model_name,
        )

        state.enable_server_load_tracking = getattr(args, "enable_server_load_tracking", False)
        state.server_load_metrics = 0
        logger.info("Pure diffusion API server initialized for model: %s", model_name)
        return

    # LLM or multi-stage mode: use standard initialization logic
    if vllm_config is None:
        # Try to get vllm_config from engine_client
        vllm_config = await openai_app_state._get_vllm_config(engine_client)
        if vllm_config is None:
            logger.warning("vllm_config is None, some features may not work correctly")

    state.vllm_config = vllm_config

    # Get supported tasks
    supported_tasks: set[str] = {"generate"}
    if hasattr(engine_client, "get_supported_tasks"):
        supported_tasks = set(await engine_client.get_supported_tasks())
    logger.info("Supported tasks: %s", supported_tasks)

    resolved_chat_template = load_chat_template(args.chat_template)
    if resolved_chat_template is None:
        try:
            tokenizer = await engine_client.get_tokenizer()
        except Exception as exc:
            logger.debug("Could not inspect tokenizer chat_template before serving init: %s", exc)
            tokenizer = None
        if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
            resolved_chat_template = _load_model_chat_template_json(args.model)

    chat_template_config = ChatTemplateConfig(
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
    )

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

    # Ensure `input_processor` and `model_config` exist on the engine client
    # for OpenAIServingModels compatibility.
    #
    # vLLM 0.20 dropped the `io_processor` kwarg from OpenAIServingRender and
    # neither `vllm.entrypoints.openai.*` nor `vllm.entrypoints.serve.*` reads
    # `engine_client.io_processor` anymore, so we no longer need to back-fill
    # it here. AsyncOmni still sets `self.io_processor` in its own __init__
    # for any vllm-omni internal callers that rely on it.
    if (
        not hasattr(engine_client, "input_processor")
        or engine_client.input_processor is None
        or not hasattr(engine_client, "model_config")
        or engine_client.model_config is None
    ):
        if vllm_config is not None:
            try:
                from vllm.v1.engine.input_processor import InputProcessor

                tokenizer = await engine_client.get_tokenizer()
                if tokenizer is not None:
                    if not hasattr(engine_client, "input_processor") or engine_client.input_processor is None:
                        engine_client.input_processor = InputProcessor(
                            vllm_config=vllm_config,
                        )
                        logger.info("Initialized input_processor for AsyncOmni")

                    if not hasattr(engine_client, "model_config") or engine_client.model_config is None:
                        engine_client.model_config = vllm_config.model_config
                        logger.info("Initialized model_config for AsyncOmni")
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

    # NOTE: kept aligned with upstream `init_app_state`:
    # Use OnlineRenderer (replaced OpenAIServingRender which was removed upstream).
    state.online_renderer = OnlineRenderer(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )

    state.openai_serving_responses = (
        OpenAIServingResponses(
            engine_client,
            state.openai_serving_models,
            state.online_renderer,
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

    _chat_kwargs = dict(
        engine_client=engine_client,
        models=state.openai_serving_models,
        response_role=args.response_role,
        online_renderer=state.online_renderer,
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

    state.openai_serving_chat = OmniOpenAIServingChat(**_chat_kwargs) if "generate" in supported_tasks else None
    state.openai_serving_chat_batch = (
        OmniOpenAIServingChatBatch(**_chat_kwargs) if "generate" in supported_tasks else None
    )

    # Warm up chat template processing to avoid first-request latency
    # Upstream f5ffc59b6a removed OpenAIServingChat.warmup() and moved the
    # warmup onto the renderer (OnlineRenderer.warmup()); mirror upstream.
    state.online_renderer.warmup()

    state.openai_serving_completion = (
        OpenAIServingCompletion(
            engine_client,
            state.openai_serving_models,
            online_renderer=state.online_renderer,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "generate" in supported_tasks
        else None
    )
    state.openai_serving_pooling = (
        ServingPooling(
            engine_client,
            state.openai_serving_models,
            supported_tasks=tuple(supported_tasks),
            request_logger=request_logger,
            chat_template_config=chat_template_config,
        )
        if any(task in POOLING_TASKS for task in supported_tasks)
        else None
    )
    state.openai_serving_embedding = (
        OpenAIServingEmbedding(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template_config=chat_template_config,
        )
        if "embed" in supported_tasks
        else None
    )
    state.openai_serving_classification = (
        ServingClassification(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template_config=chat_template_config,
        )
        if "classify" in supported_tasks
        else None
    )
    state.openai_serving_scores = (
        ServingScores(
            engine_client,
            state.openai_serving_models,
            supported_tasks=tuple(supported_tasks),
            request_logger=request_logger,
            chat_template_config=chat_template_config,
            log_error_stack=args.log_error_stack,
        )
        if any(t in supported_tasks for t in ("embed", "score", "token_embed"))
        else None
    )
    state.serving_tokenization = ServingTokenization(
        state.openai_serving_models,
        state.online_renderer,
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
            online_renderer=state.online_renderer,
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
            state.online_renderer,
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
        engine_client,
        state.openai_serving_models,
        request_logger=request_logger,
        model_name=model_name,
        forced_aligner_enabled=build_forced_aligner_config(
            getattr(args, "forced_aligner", None),
            getattr(args, "forced_aligner_config", None),
        )
        is not None,
    )

    # Warm up speech pipeline (CUDA Graph capture, torch.compile) so the first
    # real user request is fast instead of paying a 100s compilation tax.
    await state.openai_serving_speech.warmup()

    state.openai_serving_audio_generate = OmniOpenAIServingAudioGenerate(
        engine_client, state.openai_serving_models, request_logger=request_logger, model_name=model_name
    )

    state.openai_streaming_speech = OmniStreamingSpeechHandler(
        speech_service=state.openai_serving_speech,
    )
    state.openai_streaming_video = (
        create_streaming_video_handler(
            chat_service=state.openai_serving_chat,
            engine_client=engine_client,
        )
        if state.openai_serving_chat is not None
        else None
    )
    state.openai_serving_duplex = None
    if state.openai_serving_chat is not None and should_enable_duplex_endpoint(
        state.stage_configs,
        config_path=getattr(args, "deploy_config", None),
    ):
        from vllm_omni.experimental.fullduplex.openai.serving import OmniDuplexSessionHandler

        state.openai_serving_duplex = OmniDuplexSessionHandler(
            chat_service=state.openai_serving_chat,
            duplex_session_config=getattr(engine_client, "duplex_session_config", None),
            serving_runtime_adapter_path=getattr(engine_client, "duplex_serving_adapter_path", None),
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
    state.openai_serving_realtime_robot = None

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


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
        base_server = getattr(raw_request.app.state, "serving_tokenization", None)
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


@router.post(
    "/v1/chat/completions/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_batch_chat_completion(request: BatchChatCompletionRequest, raw_request: Request):
    handler = OmniBatchChat(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Chat Completions API",
            )
        return base_server.create_error_response(message="The model does not support Chat Completions API")
    try:
        result = await handler.create_batch_chat_completion(request, raw_request)
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as e:
        logger.exception("Batched chat completion failed: %s", e)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e

    if isinstance(result, ErrorResponse):
        return JSONResponse(
            content=result.model_dump(),
            status_code=result.error.code if result.error else 400,
        )
    return JSONResponse(content=result.model_dump(mode="json"))


_remove_route_from_router(router, "/v1/audio/speech", {"POST"})


@router.post(
    "/v1/audio/speech",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"audio/*": {}, "text/event-stream": {}}},
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
        base_server = getattr(raw_request.app.state, "serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Speech API",
            )
        err = base_server.create_error_response(
            message="The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )
        return _error_response_to_json_response(err, status_code=HTTPStatus.NOT_FOUND)
    try:
        result = await handler.create_speech(request, raw_request)
        if isinstance(result, ErrorResponse):
            return _error_response_to_json_response(result)
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
        base_server = getattr(raw_request.app.state, "serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Speech API",
            )
        err = base_server.create_error_response(
            message="The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )
        return _error_response_to_json_response(err, status_code=HTTPStatus.NOT_FOUND)
    try:
        result = await handler.create_speech_batch(request)
        if isinstance(result, ErrorResponse):
            return _error_response_to_json_response(result)
        # exclude_none so optional per-item fields are omitted rather than
        # serialized as null: errored items drop `usage`/`audio_data`/`media_type`,
        # successful items drop `error`. Matches the documented batch response shape.
        return JSONResponse(content=result.model_dump(exclude_none=True))
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
        base_server = getattr(raw_request.app.state, "serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Audio Generate API",
            )
        err = base_server.create_error_response(
            message="The model does not support Audio Generate API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )
        return _error_response_to_json_response(err, status_code=HTTPStatus.NOT_FOUND)
    try:
        result = await handler.create_audio_generate(request, raw_request)
        if isinstance(result, ErrorResponse):
            return _error_response_to_json_response(result)
        return result
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
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
        return _create_speech_error_json_response(
            raw_request,
            "The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )

    speakers = sorted(handler._get_available_voices())

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
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
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
        return _create_speech_error_json_response(
            raw_request,
            "The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )

    try:
        if speaker_embedding is not None and audio_sample is not None:
            return _create_speech_error_json_response(
                raw_request, "'audio_sample' and 'speaker_embedding' are mutually exclusive"
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
            return _create_speech_error_json_response(
                raw_request, "Either 'audio_sample' or 'speaker_embedding' must be provided"
            )

        return JSONResponse(content={"success": True, "voice": result})

    except ValueError as e:
        return _create_speech_error_json_response(raw_request, str(e))
    except Exception as e:
        logger.exception(f"Failed to upload voice: {e}")
        return _create_speech_error_json_response(
            raw_request,
            f"Failed to upload voice: {str(e)}",
            err_type="InternalServerError",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@router.delete(
    "/v1/audio/voices/{name}",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
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
        return _create_speech_error_json_response(
            raw_request,
            "The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )

    try:
        # Delete the voice
        success = await handler.delete_voice(name)
        if not success:
            return _create_speech_error_json_response(
                raw_request,
                f"Voice '{name}' not found",
                err_type="NotFoundError",
                status_code=HTTPStatus.NOT_FOUND,
            )

        return JSONResponse(content={"success": True, "message": f"Voice '{name}' deleted successfully"})

    except ValueError as e:
        return _create_speech_error_json_response(raw_request, str(e))
    except Exception as e:
        logger.exception(f"Failed to delete voice '{name}': {e}")
        return _create_speech_error_json_response(
            raw_request,
            f"Failed to delete voice: {str(e)}",
            err_type="InternalServerError",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@router.websocket("/v1/audio/speech/stream")
async def streaming_speech(websocket: WebSocket):
    """WebSocket endpoint for streaming text input TTS.

    Accepts text incrementally and returns audio for the buffered text on
    input.done, which flushes without closing the connection. See
    serving_speech_stream.py for protocol.
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


@router.websocket("/v1/realtime/video")
async def streaming_video_output(websocket: WebSocket):
    """WebSocket endpoint for streaming generated video output chunks."""
    handler = getattr(websocket.app.state, "openai_streaming_video_output", None)
    if handler is None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "message": "Streaming video generation is not available",
            }
        )
        await websocket.close()
        return
    await handler.handle_session(websocket)


@router.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket):
    """WebSocket endpoint for OpenAI-style realtime interactions."""
    duplex_handler = getattr(websocket.app.state, "openai_serving_duplex", None)
    duplex_query = websocket.query_params.get("duplex")
    use_duplex_realtime = (
        duplex_handler is not None and isinstance(duplex_query, str) and duplex_query.lower() in {"1", "true", "on"}
    )
    if use_duplex_realtime and duplex_handler is not None:
        await duplex_handler.handle_realtime_session(websocket)
        return

    serving = getattr(websocket.app.state, "openai_serving_realtime", None)
    if serving is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "error": "Realtime API is not available", "code": "unsupported"})
        await websocket.close()
        return
    connection = RealtimeConnection(websocket, serving)
    await connection.handle_connection()


@router.websocket("/v1/realtime/robot/openpi")
async def realtime_robot_openpi(websocket: WebSocket):
    """WebSocket endpoint for robot policy inference via OpenPI messages."""
    from vllm_omni.entrypoints.openpi.connection import (
        RobotRealtimeConnection,
    )

    serving = getattr(websocket.app.state, "openai_serving_realtime_robot", None)
    if serving is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "error": "Robot policy not available", "code": "unsupported"})
        await websocket.close()
        return
    connection = RobotRealtimeConnection(websocket, serving)
    await connection.handle_connection()


@router.websocket("/v1/duplex")
async def duplex_websocket(websocket: WebSocket):
    """WebSocket endpoint for vLLM-Omni duplex session control."""
    handler = getattr(websocket.app.state, "openai_serving_duplex", None)
    if handler is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "error": "Duplex API is not available", "code": "unsupported"})
        await websocket.close()
        return
    await handler.handle_session(websocket)


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


def _build_image_generation_response(
    *,
    images: list[Image.Image],
    request: ImageGenerationRequest,
    stage_durations: Any,
    peak_memory_mb: Any,
) -> ImageGenerationResponse | StreamingResponse:
    """Encode generated images and apply the requested response format."""
    output_format = _choose_output_format(request.output_format or "png", None)
    image_data = [
        ImageData(
            b64_json=encode_image_base64_with_compression(image, format=output_format),
            revised_prompt=None,
        )
        for image in images
    ]
    response_kwargs: dict[str, Any] = {
        "created": int(time.time()),
        "data": image_data,
        "output_format": output_format,
        "metrics": {
            "stage_durations": stage_durations or None,
            "peak_memory_mb": float(peak_memory_mb) if peak_memory_mb else None,
        },
    }
    if request.size is not None:
        response_kwargs["size"] = request.size
    response = ImageGenerationResponse(**response_kwargs)
    if request.response_format == ResponseFormat.FILE:
        return response.stream_response()
    return response


@router.post(
    "/v1/images/generations",
    dependencies=[Depends(validate_json_request)],
    response_model=None,
    responses={
        HTTPStatus.OK.value: {"model": ImageGenerationResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def generate_images(
    request: ImageGenerationRequest, raw_request: Request
) -> ImageGenerationResponse | StreamingResponse:
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
    request_timestamp = float(getattr(raw_request.state, "request_timestamp", time.time()))
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
            if request.flow_shift is not None:
                extra_body["flow_shift"] = request.flow_shift
            if request.extra_params is not None:
                extra_body["extra_params"] = request.extra_params
            if request.generator_device is not None:
                extra_body["generator_device"] = request.generator_device
            if request.lora is not None:
                # Keep /images validation semantics: invalid LoRA should fail with 400.
                _parse_lora_request(request.lora)
                extra_body["lora"] = request.lora
            if request.bot_task is not None:
                extra_body["bot_task"] = request.bot_task
            if request.use_system_prompt is not None:
                extra_body["use_system_prompt"] = request.use_system_prompt
            if request.system_prompt is not None:
                extra_body["system_prompt"] = request.system_prompt

            generation_result = await chat_handler.generate_diffusion_images(
                prompt=request.prompt,
                extra_body=extra_body,
                request_id=f"img_gen-{random_uuid()}",
                raw_request=raw_request,
                arrival_time=request_timestamp,
            )
            if isinstance(generation_result, ErrorResponse):
                return JSONResponse(
                    status_code=generation_result.error.code if generation_result.error else 400,
                    content=generation_result.model_dump(),
                )
            flat_images, stage_durations, peak_memory_mb, _ = generation_result
            return _build_image_generation_response(
                images=flat_images,
                request=request,
                stage_durations=stage_durations,
                peak_memory_mb=peak_memory_mb,
            )

        # Build params - pass through user values directly
        prompt: OmniTextPrompt = {"prompt": request.prompt, "modalities": ["image"]}
        if request.negative_prompt is not None:
            prompt["negative_prompt"] = request.negative_prompt
        gen_params = OmniDiffusionSamplingParams(num_outputs_per_prompt=request.n)
        extra_args = dict(request.extra_params or {})
        if request.use_system_prompt is not None:
            extra_args["use_system_prompt"] = request.use_system_prompt
        if request.system_prompt is not None:
            extra_args["system_prompt"] = request.system_prompt
        if request.bot_task is not None:
            extra_args["bot_task"] = request.bot_task
        if request.flow_shift is not None:
            extra_args["flow_shift"] = request.flow_shift
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

        logger.debug(f"Generating {request.n} image(s) {size_str}")

        # Generate images using AsyncOmni.
        result = await _generate_with_async_omni(
            engine_client=engine_client,
            gen_params=gen_params,
            stage_configs=stage_configs,
            prompt=prompt,
            request_id=request_id,
            arrival_time=request_timestamp,
        )

        if result is None:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No output generated from multi-stage pipeline.",
            )

        # Extract images from result
        images = _extract_images_from_result(result)

        logger.debug(f"Successfully generated {len(images)} image(s)")

        stage_durations = getattr(result, "stage_durations", None)
        peak_memory_mb = getattr(result, "peak_memory_mb", None)
        return _build_image_generation_response(
            images=images,
            request=request,
            stage_durations=stage_durations,
            peak_memory_mb=peak_memory_mb,
        )

    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except HTTPException:
        raise
    except OmniClientError as e:
        logger.info("Client error during image generation: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
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
    stream: bool = Form(False),
    user: str | None = Form(None),  # unused now
    # vllm-omni extensions for image editing
    mask_image: str | UploadFile | None = None,
    reference_image: str | UploadFile | None = None,
    # vllm-omni extensions for diffusion control
    negative_prompt: str | None = Form(None),
    num_inference_steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    guidance_scale_2: float | None = Form(None),
    strength: float | None = Form(None),
    true_cfg_scale: float | None = Form(None),
    seed: int | None = Form(None),
    generator_device: str | None = Form(None),
    # vllm-omni extension for per-request LoRA.
    lora: str | None = Form(None),  # Json string
    # vllm-omni extension for layered models (e.g., Qwen-Image-Layered)
    layers: int | None = Form(None),
    resolution: int | None = Form(None),  # See SUPPORTED_LAYERED_RESOLUTIONS
    # /v1/images/edits is always IT2I; only the prompting knobs are exposed.
    bot_task: str | None = Form(None),
    sys_type: str | None = Form(None),
    system_prompt: str | None = Form(None),
    return_stage_metrics: bool | None = Form(None),
) -> ImageGenerationResponse:
    """
    OpenAI-compatible image edit endpoint.
    """

    # 1. get engine and model
    request_timestamp = float(getattr(raw_request.state, "request_timestamp", time.time()))
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
    if stream and len(stage_configs) <= 1:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="stream=true is only supported for multi-stage image editing pipelines.",
        )
    try:
        # 2. Build prompt & images params
        cot_output = None
        prompt: OmniTextPrompt = {"prompt": prompt, "modalities": ["image"]}
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
        # Match the offline path: RGB normalize when the caller opts into
        # Hunyuan-aware behavior. RGBA/P uploads otherwise diverge from offline.
        normalize_edit_images_rgb = bot_task is not None or sys_type is not None
        pil_images = await _load_input_images(input_images_list, normalize_rgb=normalize_edit_images_rgb)
        prompt["multi_modal_data"] = {}
        prompt["multi_modal_data"]["image"] = pil_images

        if mask_image is not None:
            # Mask role is different (alpha channel matters); never normalize.
            loaded = await _load_input_images([mask_image], normalize_rgb=False)
            prompt["multi_modal_data"]["mask_image"] = loaded[0]

        if reference_image is not None:
            loaded = await _load_input_images([reference_image], normalize_rgb=normalize_edit_images_rgb)
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
        size_was_auto = size.lower() == "auto"
        if size_was_auto:
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
        _update_if_not_none(gen_params, "guidance_scale_2", guidance_scale_2)
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

        extra_args = dict(getattr(gen_params, "extra_args", {}) or {})
        edit_extra_args = _build_hunyuan_edit_extra_args(
            bot_task=bot_task,
            sys_type=sys_type,
            system_prompt=system_prompt,
        )
        extra_args.update(edit_extra_args)
        if extra_args:
            gen_params.extra_args = extra_args

        # 4. Generate images
        request_id = f"img_edit-{random_uuid()}"
        raw_request.state.request_metadata = RequestResponseMetadata(request_id=request_id)
        logger.debug(f"Generating {n} image(s) {size_str}")

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
            # size="auto" resolves width/height from input image; forwarding
            # those would override AR-driven `<img_ratio_*>` token selection.
            if not size_was_auto:
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
            if guidance_scale_2 is not None:
                extra_body["guidance_scale_2"] = guidance_scale_2
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
            if bot_task is not None:
                extra_body["bot_task"] = bot_task
            if sys_type is not None:
                extra_body["sys_type"] = sys_type
            if system_prompt is not None:
                extra_body["system_prompt"] = system_prompt
            if return_stage_metrics is not None:
                extra_body["return_stage_metrics"] = return_stage_metrics

            prompt_text = prompt.get("prompt", "")
            generation_result = await chat_handler.generate_diffusion_images(
                prompt=prompt_text,
                extra_body=extra_body,
                reference_images=ref_b64_list,
                request_id=request_id,
                arrival_time=request_timestamp,
                stream=stream,
                model=model_name,
                output_format=output_format,
                output_compression=output_compression,
                size=size_str,
                raw_request=raw_request,
            )
            if stream and not isinstance(generation_result, ErrorResponse):
                return StreamingResponse(
                    content=generation_result,
                    media_type="text/event-stream",
                )
            if isinstance(generation_result, ErrorResponse):
                raise HTTPException(
                    status_code=generation_result.error.code if generation_result.error else 400,
                    detail=generation_result.message,
                )
            images, stage_durations, peak_memory_mb, cot_output = generation_result
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
            stage_durations = getattr(result, "stage_durations", None)
            peak_memory_mb = getattr(result, "peak_memory_mb", None)

        logger.debug(f"Successfully generated {len(images)} image(s)")

        # Encode images to base64
        image_data = [
            ImageData(
                b64_json=encode_image_base64_with_compression(
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
            cot_output=cot_output,
            metrics={
                "stage_durations": stage_durations or None,
                "peak_memory_mb": float(peak_memory_mb) if peak_memory_mb else None,
            },
        )

    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except HTTPException:
        raise
    except OmniClientError as e:
        logger.info("Client error during image edit: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e))
    except Exception as e:
        logger.exception(f"Image edit failed: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Image edit failed: {str(e)}")


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
    ctx: tuple[
        VideoGenerationRequest,
        OmniOpenAIServingVideo,
        str,
        ReferenceImage | None,
        ReferenceVideo | None,
        ReferenceAudio | None,
    ] = Depends(_parse_video_form),
) -> VideoResponse:
    """Create an asynchronous video generation job.

    Accepts multipart form-data (see ``_parse_video_form`` for parameters),
    persists a queued job record, and starts generation in the background.
    """
    request, handler, effective_model_name, reference_image, reference_video, reference_audio = ctx
    ref = video_response_from_request(effective_model_name, request)
    await VIDEO_STORE.upsert(ref.id, ref)
    task = asyncio.create_task(
        _run_video_generation_job(
            handler,
            request,
            ref.id,
            reference_image,
            reference_video,
            reference_audio,
            app_state=raw_request.app.state,
        )
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
    ctx: tuple[
        VideoGenerationRequest,
        OmniOpenAIServingVideo,
        str,
        ReferenceImage | None,
        ReferenceVideo | None,
        ReferenceAudio | None,
    ] = Depends(_parse_video_form),
) -> Response:
    """Synchronous video generation endpoint.

    Accepts the same form parameters as ``POST /v1/videos`` but blocks until
    generation completes and returns raw video bytes (``video/mp4``) directly.
    Designed for benchmark and testing scenarios.

    Metadata is returned via response headers ``X-Request-Id``,
    ``X-Model``, and ``X-Inference-Time-S``.
    """
    request, handler, effective_model_name, reference_image, reference_video, reference_audio = ctx
    request_id = f"video_sync-{random_uuid()}"
    raw_request.state.request_metadata = RequestResponseMetadata(request_id=request_id)
    started_at = time.perf_counter()
    try:
        video_bytes, stage_durations, peak_memory_mb, _action = await asyncio.wait_for(
            handler.generate_video_bytes(
                request,
                request_id,
                reference_image=reference_image,
                reference_video=reference_video,
                reference_audio=reference_audio,
            ),
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
    except OmniClientError as exc:
        logger.info("Client error during sync video generation: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        logger.exception("Sync video generation failed for request_id=%s", request_id)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"Video generation failed: {str(exc)}",
        ) from exc
    finally:
        _cleanup_video_references(reference_video, reference_audio)
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
                await STORAGE_MANAGER.delete(video_id)
            except Exception:
                logger.warning("Failed to delete stored artifact for failed video job %s", video_id, exc_info=True)

        await VIDEO_STORE.pop(video_id)
        return VideoDeleteResponse(id=job.id, deleted=True)

    if job.file_name is None:
        raise HTTPException(status_code=409, detail="Video output not yet available. Please try again later.")

    await STORAGE_MANAGER.delete(video_id)
    await VIDEO_STORE.pop(video_id)
    return VideoDeleteResponse(id=job.id, deleted=True)


@router.get("/v1/videos/{video_id}/content")
async def download_video(video_id: str) -> Response:
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

    file_handle = await STORAGE_MANAGER.open(video_id)
    if file_handle is None:
        raise HTTPException(status_code=404, detail="Generated video file not found on disk")

    file_name = job.file_name or f"{video_id}.{job.file_extension}"
    if isinstance(file_handle, FileStorageHandle):
        response = FileResponse(path=file_handle.path, media_type=job.media_type, filename=file_name)
    else:
        raise HTTPException(
            status_code=500, detail=f"Server generated an unsupported file storage handle for file id {video_id}"
        )

    return response


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
        await engine_client.start_profile(stages=stages)
        logger.info("Profiler started.")
        return JSONResponse(content={"status": "SUCCESS"})
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
        await engine_client.stop_profile(stages=stages)
        logger.info("Profiler stopped.")
        return JSONResponse(content={"status": "SUCCESS"})
    except Exception as e:
        logger.exception("Failed to stop profiler: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Failed to stop profiler: {str(e)}"
        )


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
    parser = TrackingArgumentParser(description="vLLM-Omni OpenAI-Compatible REST API server")
    parser = make_arg_parser(parser)
    # Ensure that passing --omni won't crash the server.
    # NOTE: the value here does not matter since we are always running the Omni server
    # when __main__ is called, i.e., --omni is only used when called through the entrypoints.
    parser.add_argument("--omni", action="store_true", default=False)
    args = parser.parse_args()
    # sync args.model to model_tag, because if we pass the model positionally,
    # args.model will be the default from vLLM's ModelConfig (currently
    # Qwen/Qwen3-0.6B) and crash cryptically.
    if args.model_tag is not None:
        args.model = args.model_tag
    asyncio.run(omni_run_server(args))

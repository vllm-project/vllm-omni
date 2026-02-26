# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import io
import json
import multiprocessing
import multiprocessing.forkserver as forkserver
import os

# Image generation API imports
import random
import time
import uuid
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Annotated, Any, cast

import httpx
import vllm.envs as envs
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from starlette.datastructures import State
from starlette.routing import Route
from vllm import SamplingParams
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.mcp.tool_server import DemoToolServer, MCPToolServer, ToolServer
from vllm.entrypoints.openai.api_server import build_app as build_openai_app
from vllm.entrypoints.openai.api_server import setup_server as setup_openai_server

# vLLM moved `base` from openai.basic.api_router to serve.instrumentator.basic.
# Keep a fallback for older/newer upstream layouts during rebase windows.
try:
    from vllm.entrypoints.serve.instrumentator.basic import base
except ModuleNotFoundError:
    from vllm.entrypoints.openai.basic.api_router import base
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
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.openai.server_utils import get_uvicorn_log_config
from vllm.entrypoints.openai.speech_to_text.serving import (
    OpenAIServingTranscription,
    OpenAIServingTranslation,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.classify.serving import ServingClassification
from vllm.entrypoints.pooling.embed.serving import OpenAIServingEmbedding
from vllm.entrypoints.pooling.pooling.serving import OpenAIServingPooling
from vllm.entrypoints.pooling.score.serving import ServingScores
from vllm.entrypoints.serve.disagg.serving import ServingTokens
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.entrypoints.utils import (
    load_aware_call,
    process_lora_modules,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.tasks import POOLING_TASKS
from vllm.tool_parsers import ToolParserManager
from vllm.utils.system_utils import decorate_logs

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.image_api_utils import (
    encode_image_base64,
    parse_size,
)
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.protocol.images import (
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from vllm_omni.entrypoints.openai.protocol.videos import (
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams, OmniTextPrompt
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

logger = init_logger(__name__)
router = APIRouter()


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


class _DiffusionServingModels:
    """Minimal OpenAIServingModels implementation for diffusion-only servers.

    vLLM's /v1/models route expects `app.state.openai_serving_models` to expose
    `show_available_models()`. In pure diffusion mode we don't initialize the
    full OpenAIServingModels (it depends on LLM-specific processors), so we
    provide a lightweight fallback.
    """

    def __init__(self, base_model_paths: list[BaseModelPath]) -> None:
        self._base_model_paths = base_model_paths

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
            supported_tasks = ("generate",)

        # OMNI: Pass supported_tasks to build_app (required by upstream vLLM)
        app = build_openai_app(args, supported_tasks)
        # OMNI: Remove upstream routes that we override with omni-specific handlers
        _remove_route_from_app(app, "/v1/chat/completions", {"POST"})
        _remove_route_from_app(app, "/v1/models", {"GET"})  # Remove upstream /v1/models to use omni's handler
        app.include_router(router)

        await omni_init_app_state(engine_client, app.state, args)

        vllm_config = await engine_client.get_vllm_config()

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
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
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
            multiprocessing (deprecated in V1)
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
            (deprecated in V1)
        client_config: Optional client configuration dictionary

    Yields:
        EngineClient instance (AsyncOmni) ready for use

    Note:
        Stage configurations are loaded from args.stage_configs_path if provided,
        otherwise from the model's default configuration.
    """

    # V1 AsyncLLM.
    if disable_frontend_multiprocessing:
        logger.warning("V1 is enabled, but got --disable-frontend-multiprocessing.")

    async_omni: EngineClient | None = None

    try:
        # Convert args Namespace to kwargs dict for AsyncOmni to use
        kwargs = vars(args).copy()
        # Remove model as it will be passed separately
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
    vllm_config = await engine_client.get_vllm_config()

    # Detect if it's pure Diffusion mode (single stage and is Diffusion)
    is_pure_diffusion = False
    if hasattr(engine_client, "stage_configs") and engine_client.stage_configs:
        stage_configs = engine_client.stage_configs
        if len(stage_configs) == 1:
            stage_type = stage_configs[0].get("stage_type", "llm")
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

    # Pure Diffusion mode: use simplified initialization logic
    if is_pure_diffusion:
        model_name = served_model_names[0] if served_model_names else args.model
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

        diffusion_stage_configs = engine_client.stage_configs if hasattr(engine_client, "stage_configs") else None
        state.openai_serving_video = OmniOpenAIServingVideo.for_diffusion(
            diffusion_engine=engine_client,  # type: ignore
            model_name=model_name,
            stage_configs=diffusion_stage_configs,
        )

        state.enable_server_load_tracking = getattr(args, "enable_server_load_tracking", False)
        state.server_load_metrics = 0
        logger.info("Pure diffusion API server initialized for model: %s", model_name)
        return

    # LLM or multi-stage mode: use standard initialization logic
    if vllm_config is None:
        # Try to get vllm_config from engine_client
        vllm_config = await engine_client.get_vllm_config()
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

                from vllm_omni.engine.input_processor import OmniInputProcessor

                tokenizer = await engine_client.get_tokenizer()
                if tokenizer is not None:
                    # Initialize input_processor
                    # OMNI: OmniInputProcessor creates tokenizer internally from vllm_config
                    if not hasattr(engine_client, "input_processor") or engine_client.input_processor is None:
                        engine_client.input_processor = OmniInputProcessor(
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
                        engine_client.io_processor = get_io_processor(vllm_config, io_processor_plugin)
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

    state.openai_serving_responses = (
        OpenAIServingResponses(
            engine_client,
            state.openai_serving_models,
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
            log_error_stack=args.log_error_stack,
        )
        if "generate" in supported_tasks
        else None
    )
    state.openai_serving_chat = (
        OmniOpenAIServingChat(
            engine_client,
            state.openai_serving_models,
            args.response_role,
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
            log_error_stack=args.log_error_stack,
        )
        if "generate" in supported_tasks
        else None
    )
    # Warm up chat template processing to avoid first-request latency
    if state.openai_serving_chat is not None:
        await state.openai_serving_chat.warmup()

    state.openai_serving_completion = (
        OpenAIServingCompletion(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            log_error_stack=args.log_error_stack,
        )
        if "generate" in supported_tasks
        else None
    )
    state.openai_serving_pooling = (
        OpenAIServingPooling(
            engine_client,
            state.openai_serving_models,
            supported_tasks=supported_tasks,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
            log_error_stack=args.log_error_stack,
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
            log_error_stack=args.log_error_stack,
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
            log_error_stack=args.log_error_stack,
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
        if ("embed" in supported_tasks or "score" in supported_tasks)
        else None
    )
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        log_error_stack=args.log_error_stack,
    )
    state.openai_serving_transcription = (
        OpenAIServingTranscription(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            log_error_stack=args.log_error_stack,
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
            log_error_stack=args.log_error_stack,
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
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "generate" in supported_tasks
        else None
    )
    state.serving_tokens = (
        ServingTokens(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            log_error_stack=args.log_error_stack,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_log_outputs=args.enable_log_outputs,
            force_no_detokenize=args.tokens_only,
        )
        if "generate" in supported_tasks
        else None
    )

    state.openai_serving_speech = OmniOpenAIServingSpeech(
        engine_client, state.openai_serving_models, request_logger=request_logger
    )

    state.openai_serving_video = OmniOpenAIServingVideo(
        engine_client,
        model_name=served_model_names[0] if served_model_names else None,
        stage_configs=state.stage_configs,
    )

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


def Omnivideo(request: Request) -> OmniOpenAIServingVideo | None:
    return request.app.state.openai_serving_video


def Omnichat(request: Request) -> OmniOpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def Omnispeech(request: Request) -> OmniOpenAIServingSpeech | None:
    return request.app.state.openai_serving_speech


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
        return await handler.create_speech(request, raw_request)
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
    """List available TTS voices/speakers from the loaded model."""
    handler = Omnispeech(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(message="The model does not support Speech API")

    speakers = sorted(handler.supported_speakers) if handler.supported_speakers else []
    return JSONResponse(content={"voices": speakers})


# Health and Model endpoints for diffusion mode


# Remove existing health endpoint if present (from vllm imports)
# to ensure our handler takes precedence
_remove_route_from_router(router, "/health")


@router.get("/health")
async def health(raw_request: Request) -> JSONResponse:
    """Health check endpoint that works for both LLM and diffusion modes.

    Returns 200 OK if the server is healthy.
    For LLM mode: delegates to engine_client health check
    For diffusion mode: checks if diffusion_engine is running
    """
    # Check if we're in diffusion mode
    diffusion_engine = getattr(raw_request.app.state, "diffusion_engine", None)
    if diffusion_engine is not None:
        # Diffusion mode health check
        if hasattr(diffusion_engine, "is_running") and diffusion_engine.is_running:
            return JSONResponse(content={"status": "healthy"})
        return JSONResponse(
            content={"status": "unhealthy", "reason": "Diffusion engine is not running"},
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        )

    # LLM mode - delegate to engine_client
    engine_client = getattr(raw_request.app.state, "engine_client", None)
    if engine_client is not None:
        await engine_client.check_health()
        return JSONResponse(content={"status": "healthy"})

    return JSONResponse(
        content={"status": "unhealthy", "reason": "No engine initialized"},
        status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
    )


# Remove existing models endpoint if present (from vllm imports)
# to ensure our handler takes precedence
_remove_route_from_router(router, "/v1/models")


@router.get("/v1/models")
async def show_available_models(raw_request: Request) -> JSONResponse:
    """Show available models endpoint that works for both LLM and diffusion modes.

    Returns model information in OpenAI-compatible format.
    """
    # Check if we're in diffusion mode
    diffusion_model_name = getattr(raw_request.app.state, "diffusion_model_name", None)
    if diffusion_model_name is not None:
        # Diffusion mode - return the loaded model
        return JSONResponse(
            content={
                "object": "list",
                "data": [
                    {
                        "id": diffusion_model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "vllm-omni",
                        "permission": [],
                    }
                ],
            }
        )

    # LLM mode - delegate to openai_serving_models
    openai_serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
    if openai_serving_models is not None:
        models = await openai_serving_models.show_available_models()
        return JSONResponse(content=models.model_dump())

    return JSONResponse(
        content={"object": "list", "data": []},
    )


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
async def generate_images(request: ImageGenerationRequest, raw_request: Request) -> ImageGenerationResponse:
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
    engine_client, model_name, stage_types = _get_engine_and_model(raw_request)

    # Validate model field (warn if mismatch, don't error)
    if request.model is not None and request.model != model_name:
        logger.warning(
            f"Model mismatch: request specifies '{request.model}' but "
            f"server is running '{model_name}'. Using server model."
        )

    try:
        # Build params - pass through user values directly
        prompt: OmniTextPrompt = {"prompt": request.prompt}
        if request.negative_prompt is not None:
            prompt["negative_prompt"] = request.negative_prompt
        gen_params = OmniDiffusionSamplingParams(num_outputs_per_prompt=request.n)

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
            gen_params, "seed", request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        )
        _update_if_not_none(gen_params, "generator_device", request.generator_device)

        request_id = f"img_gen_{uuid.uuid4().hex}"

        logger.info(f"Generating {request.n} image(s) {size_str}")

        # Generate images using AsyncOmni (multi-stage mode)
        result = await _generate_with_async_omni(
            engine_client=engine_client,
            gen_params=gen_params,
            stage_types=stage_types,
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

        # Encode images to base64
        image_data = [ImageData(b64_json=encode_image_base64(img), revised_prompt=None) for img in images]

        return ImageGenerationResponse(
            created=int(time.time()),
            data=image_data,
        )

    except HTTPException:
        # Re-raise HTTPExceptions as-is
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
    # vllm-omni extensions for diffusion control
    negative_prompt: str | None = Form(None),
    num_inference_steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    true_cfg_scale: float | None = Form(None),
    seed: int | None = Form(None),
    generator_device: str | None = Form(None),
    # vllm-omni extension for per-request LoRA.
    lora: str | None = Form(None),  # Json string
) -> ImageGenerationResponse:
    """
    OpenAI-compatible image edit endpoint.
    """
    # 1. get engine and model
    engine_client, model_name, stage_types = _get_engine_and_model(raw_request)
    if model is not None and model != model_name:
        logger.warning(
            f"Model mismatch: request specifies '{model}' but server is running '{model_name}'. Using server model."
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
        pil_images = await _load_input_images(input_images_list)
        prompt["multi_modal_data"] = {}
        prompt["multi_modal_data"]["image"] = pil_images

        # 3 Build sample params
        gen_params = OmniDiffusionSamplingParams()
        # 3.0 Init with system default values
        app_state_args = getattr(raw_request.app.state, "args", None)
        default_sample_param = getattr(app_state_args, "default_sampling_params", None)
        # Currently only have one diffusion stage
        diffusion_stage_id = [i for i, t in enumerate(stage_types) if t == "diffusion"][0]
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
        # 3.2 Parse and add size if provided
        max_generated_image_size = getattr(app_state_args, "max_generated_image_size", None)
        width, height = None, None
        if size.lower() == "auto":
            width, height = pil_images[0].size  # Use first image size
        else:
            width, height = parse_size(size)
        if max_generated_image_size is not None and (width * height > max_generated_image_size):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Requested image size {width}x{height} exceeds the maximum allowed "
                f"size of {max_generated_image_size} pixels.",
            )

        size_str = f"{width}x{height}"
        _update_if_not_none(gen_params, "width", width)
        _update_if_not_none(gen_params, "height", height)

        # 3.3 Add optional parameters ONLY if provided
        _update_if_not_none(gen_params, "num_inference_steps", num_inference_steps)
        _update_if_not_none(gen_params, "guidance_scale", guidance_scale)
        _update_if_not_none(gen_params, "true_cfg_scale", true_cfg_scale)
        # If seed is not provided, generate a random one to ensure
        # a proper generator is initialized in the backend.
        # This fixes issues where using the default global generator
        # might produce blurry images in some environments.
        _update_if_not_none(gen_params, "seed", seed if seed is not None else random.randint(0, 2**32 - 1))
        _update_if_not_none(gen_params, "generator_device", generator_device)

        # 4. Generate images using AsyncOmni (multi-stage mode)
        request_id = f"img_edit_{int(time.time())}"
        logger.info(f"Generating {n} image(s) {size_str}")
        result = await _generate_with_async_omni(
            engine_client=engine_client,
            gen_params=gen_params,
            stage_types=stage_types,
            prompt=prompt,
            request_id=request_id,
        )

        # 5. Extract images from result
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

    except HTTPException:
        # Re-raise HTTPExceptions as-is
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
    if engine_client is None or not hasattr(engine_client, "stage_list"):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Multi-stage engine not initialized. Start server with a multi-stage omni model.",
        )

    # Check if there's a diffusion stage
    stage_configs = getattr(raw_request.app.state, "stage_configs", None)
    if not stage_configs:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Stage configs not found. Start server with a multi-stage omni model.",
        )

    # Check for diffusion stage and collect stage types
    has_diffusion_stage = False
    stage_types: list[str] = []
    for stage in stage_configs:
        # Handle both dict and OmegaConf objects
        stage_type = None
        if isinstance(stage, dict):
            stage_type = stage.get("stage_type", "llm")
        elif hasattr(stage, "get"):
            stage_type = stage.get("stage_type", "llm")
        elif hasattr(stage, "stage_type"):
            stage_type = stage.stage_type
        else:
            # Fallback: try to access as dict-like
            try:
                stage_type = stage["stage_type"] if "stage_type" in stage else "llm"
            except (TypeError, KeyError):
                stage_type = "llm"

        if stage_type == "diffusion":
            has_diffusion_stage = True
        stage_types.append(stage_type)

    if not has_diffusion_stage:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="No diffusion stage found in multi-stage pipeline.",
        )

    # Get server's loaded model name
    serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
    if serving_models and hasattr(serving_models, "base_model_paths") and serving_models.base_model_paths:
        model_name = serving_models.base_model_paths[0].name
    else:
        model_name = "unknown"

    return engine_client, model_name, stage_types


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
    if lora_body is not None:
        if not isinstance(lora_body, dict):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Invalid lora field: expected an object.",
            )
        lora_name = lora_body.get("name") or lora_body.get("lora_name") or lora_body.get("adapter")
        lora_path = (
            lora_body.get("local_path")
            or lora_body.get("path")
            or lora_body.get("lora_path")
            or lora_body.get("lora_local_path")
        )
        lora_scale = lora_body.get("scale")
        if lora_scale is None:
            lora_scale = lora_body.get("lora_scale")
        lora_int_id = lora_body.get("int_id")
        if lora_int_id is None:
            lora_int_id = lora_body.get("lora_int_id")
        if lora_int_id is None and lora_path:
            lora_int_id = stable_lora_int_id(str(lora_path))

        if not lora_name or not lora_path:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Invalid lora object: both name and path are required.",
            )

        return LoRARequest(str(lora_name), int(lora_int_id), str(lora_path)), lora_scale
    return None, None


async def _generate_with_async_omni(
    engine_client: AsyncOmni | Any,
    gen_params: Any,
    stage_types: list[str],
    **kwargs,
):
    engine_client = cast(AsyncOmni, engine_client)
    result = None
    stage_list = getattr(engine_client, "stage_list", None)
    if isinstance(stage_list, list):
        default_params_list: list[OmniSamplingParams] | None = getattr(
            engine_client, "default_sampling_params_list", None
        )
        if not isinstance(default_params_list, list):
            default_params_list = [
                OmniDiffusionSamplingParams() if st == "diffusion" else SamplingParams() for st in stage_types
            ]
        else:
            default_params_list = list(default_params_list)
        if len(default_params_list) != len(stage_types):
            default_params_list = (
                default_params_list
                + [OmniDiffusionSamplingParams() if st == "diffusion" else SamplingParams() for st in stage_types]
            )[: len(stage_types)]

        sampling_params_list: list[OmniSamplingParams] = []
        for idx, stage_type in enumerate(stage_types):
            if stage_type == "diffusion":
                sampling_params_list.append(gen_params)
            else:
                base_params = default_params_list[idx]
                sampling_params_list.append(base_params)

        async for output in engine_client.generate(
            sampling_params_list=sampling_params_list,
            **kwargs,
        ):
            result = output
    else:
        result = await engine_client.generate(
            sampling_params_list=[gen_params],
            **kwargs,
        )

    if result is None:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="No output generated from multi-stage pipeline.",
        )
    return result


def _update_if_not_none(object: any, key: str, val: any) -> None:
    if val is not None:
        setattr(object, key, val)


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
    return images


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


def _encode_image_base64_with_compression(
    image: Image.Image, format: str = "png", output_compression: int = 100
) -> str:
    """Encode PIL Image to base64 PNG string.

    Args:
        image: PIL Image object
        format: Output image format (e.g., "PNG", "JPEG", "WEBP")
        output_compression: Compression level (0-100%), 100 for best quality
    Returns:
        Base64-encoded image as string
    """
    buffer = io.BytesIO()
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
    sampling_params: any,
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


async def _run_video_generation(
    request: VideoGenerationRequest,
    raw_request: Request,
    *,
    input_reference_bytes: bytes | None = None,
) -> VideoGenerationResponse:
    handler = Omnivideo(raw_request)
    if handler is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Video generation handler not initialized.",
        )
    logger.info("Video generation handler: %s", type(handler).__name__)
    try:
        return await handler.generate_videos(request, raw_request, input_reference_bytes=input_reference_bytes)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Video generation failed: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"Video generation failed: {str(e)}",
        )


def _parse_form_json(value: str | None) -> Any:
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Invalid JSON in form field.",
        ) from exc


@router.post(
    "/v1/videos",
    responses={
        HTTPStatus.OK.value: {"model": VideoGenerationResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def create_video(
    raw_request: Request,
    prompt: str = Form(...),
    input_reference: UploadFile | None = File(default=None),
    model: str | None = Form(default=None),
    n: int | None = Form(default=None),
    seconds: int | None = Form(default=None),
    size: str | None = Form(default=None),
    response_format: str | None = Form(default=None),
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
    lora: str | None = Form(default=None),
) -> VideoGenerationResponse:
    """OpenAI-style video create endpoint (multipart form-data)."""
    input_reference_bytes = await input_reference.read() if input_reference is not None else None

    request_data: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "seconds": seconds,
        "size": size,
        "user": user,
        "response_format": response_format,
        "input_reference": None,
        "n": n,
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
        "lora": _parse_form_json(lora),
    }
    request_data = {k: v for k, v in request_data.items() if v is not None}
    request = VideoGenerationRequest(**request_data)
    return await _run_video_generation(request, raw_request, input_reference_bytes=input_reference_bytes)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import multiprocessing.forkserver as forkserver
import os

# Image generation API imports
import time
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any

import vllm.envs as envs
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import State
from starlette.routing import Route
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.anthropic.serving_messages import AnthropicServingMessages
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.api_server import (
    base,
    build_app,
    load_log_config,
    router,
    setup_server,
)
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
)

# yapf conflicts with isort for this block
# yapf: disable
# yapf: enable
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.openai.serving_responses import OpenAIServingResponses
from vllm.entrypoints.openai.serving_transcription import (
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
from vllm.entrypoints.tool_server import DemoToolServer, MCPToolServer, ToolServer
from vllm.entrypoints.utils import (
    load_aware_call,
    process_chat_template,
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
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

logger = init_logger(__name__)

ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def _remove_route_from_router(router_obj, path: str, methods: set[str] | None = None):
    """Remove a route from the router by path and optionally by methods.

    This is needed because vllm's api_server registers routes when imported,
    and we need to override some routes (like /v1/chat/completions) with
    omni-specific implementations.
    """
    routes_to_remove = []
    for route in router_obj.routes:
        if isinstance(route, Route) and route.path == path:
            if methods is None or (hasattr(route, "methods") and route.methods & methods):
                routes_to_remove.append(route)

    for route in routes_to_remove:
        router_obj.routes.remove(route)


# Remove vllm's /v1/chat/completions route so we can register our own omni version
_remove_route_from_router(router, "/v1/chat/completions", {"POST"})


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

    listen_address, sock = setup_server(args)

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
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_omni(
        args,
        client_config=client_config,
    ) as engine_client:
        app = build_app(args)

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

        # Use for_diffusion method to create chat handler
        state.openai_serving_chat = OmniOpenAIServingChat.for_diffusion(
            diffusion_engine=engine_client,  # type: ignore
            model_name=model_name,
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

    resolved_chat_template = await process_chat_template(
        args.chat_template,
        engine_client,
        vllm_config.model_config if vllm_config is not None else None,
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
                    if not hasattr(engine_client, "input_processor") or engine_client.input_processor is None:
                        engine_client.input_processor = OmniInputProcessor(
                            vllm_config=vllm_config,
                            tokenizer=tokenizer,
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

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


def Omnichat(request: Request) -> OmniOpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def Omnispeech(request: Request) -> OmniOpenAIServingSpeech | None:
    return request.app.state.openai_serving_speech


# Remove the original /v1/chat/completions route before registering our own
# This prevents duplicate route registration warnings in FastAPI logs.
for route in router.routes[:]:
    if hasattr(route, "path") and route.path == "/v1/chat/completions":
        router.routes.remove(route)
        break


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
        return base(raw_request).create_error_response(message="The model does not support Chat Completions API")
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
        return base(raw_request).create_error_response(message="The model does not support Speech API")
    try:
        return await handler.create_speech(request, raw_request)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


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
    engine_client: EngineClient | None = getattr(raw_request.app.state, "engine_client", None)
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

    # Validate model field (warn if mismatch, don't error)
    if request.model is not None and request.model != model_name:
        logger.warning(
            f"Model mismatch: request specifies '{request.model}' but "
            f"server is running '{model_name}'. Using server model."
        )

    try:
        # Build params - pass through user values directly
        gen_params = {
            "prompt": request.prompt,
            "num_outputs_per_prompt": request.n,
        }

        # Parse per-request LoRA (compatible with chat's extra_body.lora shape).
        if request.lora is not None:
            if not isinstance(request.lora, dict):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="Invalid lora field: expected an object.",
                )
            lora_body = request.lora
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

            gen_params["lora_request"] = LoRARequest(str(lora_name), int(lora_int_id), str(lora_path))
            if lora_scale is not None:
                gen_params["lora_scale"] = float(lora_scale)

        # Parse and add size if provided
        if request.size:
            width, height = parse_size(request.size)
            gen_params["height"] = height
            gen_params["width"] = width
            size_str = f"{width}x{height}"
        else:
            size_str = "model default"

        # Add optional parameters ONLY if provided
        if request.num_inference_steps is not None:
            gen_params["num_inference_steps"] = request.num_inference_steps
        if request.negative_prompt is not None:
            gen_params["negative_prompt"] = request.negative_prompt
        if request.guidance_scale is not None:
            gen_params["guidance_scale"] = request.guidance_scale
        if request.true_cfg_scale is not None:
            gen_params["true_cfg_scale"] = request.true_cfg_scale
        if request.seed is not None:
            gen_params["seed"] = request.seed
        gen_params["request_id"] = f"img_gen_{int(time.time())}"

        logger.info(f"Generating {request.n} image(s) {size_str}")

        # Generate images using AsyncOmni (multi-stage mode)
        result = None
        stage_list = getattr(engine_client, "stage_list", None)
        if isinstance(stage_list, list):
            default_params_list = getattr(engine_client, "default_sampling_params_list", None)
            if not isinstance(default_params_list, list):
                default_params_list = [{} for _ in stage_types]
            else:
                default_params_list = list(default_params_list)
            if len(default_params_list) != len(stage_types):
                default_params_list = (default_params_list + [{} for _ in stage_types])[: len(stage_types)]

            sampling_params_list: list[dict[str, Any]] = []
            for idx, stage_type in enumerate(stage_types):
                if stage_type == "diffusion":
                    sampling_params_list.append(gen_params)
                else:
                    base_params = default_params_list[idx]
                    sampling_params_list.append(dict(base_params) if isinstance(base_params, dict) else base_params)

            async for output in engine_client.generate(
                prompt=gen_params["prompt"],
                request_id=gen_params["request_id"],
                sampling_params_list=sampling_params_list,
            ):
                result = output
        else:
            result = await engine_client.generate(**gen_params)

        if result is None:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No output generated from multi-stage pipeline.",
            )

        # Extract images from result
        images = []
        if hasattr(result, "images") and result.images:
            images = result.images
        elif hasattr(result, "request_output"):
            request_output = result.request_output
            if isinstance(request_output, dict) and request_output.get("images"):
                images = request_output["images"]
            elif hasattr(request_output, "images") and request_output.images:
                images = request_output.images

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

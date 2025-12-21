import json
import multiprocessing
import multiprocessing.forkserver as forkserver
import os
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import fields
from http import HTTPStatus
from typing import Any

import vllm.envs as envs
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import State
from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template, resolve_hf_chat_template, resolve_mistral_chat_template
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.api_server import (
    base,
    build_app,
    load_log_config,
    maybe_register_tokenizer_info_endpoint,
    router,
    setup_server,
    validate_json_request,
)
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_models import BaseModelPath, LoRAModulePath, OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParserManager

# yapf conflicts with isort for this block
# yapf: disable
# yapf: enable
from vllm.entrypoints.tool_server import DemoToolServer, MCPToolServer, ToolServer
from vllm.entrypoints.utils import load_aware_call, with_cancellation
from vllm.logger import init_logger
from vllm.tokenizers import MistralTokenizer
from vllm.utils.system_utils import decorate_logs

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model
from vllm_omni.entrypoints.async_diffusion import AsyncOmniDiffusion
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

logger = init_logger(__name__)


async def omni_run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server.

    Automatically detects if the model is a diffusion model and routes
    to the appropriate server implementation.
    """

    # Add process-specific prefix to stdout and stderr.
    decorate_logs("APIServer")

    listen_address, sock = setup_server(args)

    # Check if model is a diffusion model
    if is_diffusion_model(args.model):
        logger.info("Detected diffusion model, starting diffusion API server")
        await omni_run_diffusion_server_worker(listen_address, sock, args, **uvicorn_kwargs)
    else:
        await omni_run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def omni_run_diffusion_server(args, **uvicorn_kwargs) -> None:
    """Run a diffusion model API server."""

    # Add process-specific prefix to stdout and stderr.
    decorate_logs("DiffusionAPIServer")

    listen_address, sock = setup_server(args)
    await omni_run_diffusion_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def omni_run_diffusion_server_worker(listen_address, sock, args, **uvicorn_kwargs) -> None:
    """Run a diffusion model API server worker."""

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_diffusion(args) as diffusion_engine:
        app = build_app(args)

        await omni_diffusion_init_app_state(diffusion_engine, app.state, args)

        logger.info("Starting vLLM Diffusion API server on %s", listen_address)

        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=getattr(args, "enable_ssl_refresh", False),
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not getattr(args, "disable_uvicorn_access_log", False),
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=getattr(args, "ssl_keyfile", None),
            ssl_certfile=getattr(args, "ssl_certfile", None),
            ssl_ca_certs=getattr(args, "ssl_ca_certs", None),
            ssl_cert_reqs=getattr(args, "ssl_cert_reqs", 0),
            h11_max_incomplete_event_size=getattr(args, "h11_max_incomplete_event_size", None),
            h11_max_header_count=getattr(args, "h11_max_header_count", None),
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


async def omni_run_server_worker(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_omni(
        args,
        client_config=client_config,
    ) as engine_client:
        maybe_register_tokenizer_info_endpoint(args)
        app = build_app(args)

        vllm_config = await engine_client.get_vllm_config()
        await omni_init_app_state(engine_client, vllm_config, app.state, args)

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
async def build_async_diffusion(
    args: Namespace,
    **kwargs: Any,
) -> AsyncIterator[AsyncOmniDiffusion]:
    """Build an AsyncOmniDiffusion instance from command-line arguments.

    Creates an async context manager that yields an AsyncOmniDiffusion
    instance configured from the provided arguments.

    Args:
        args: Parsed command-line arguments containing model and configuration
        **kwargs: Additional keyword arguments passed to AsyncOmniDiffusion

    Yields:
        AsyncOmniDiffusion instance ready for use
    """
    diffusion_engine: AsyncOmniDiffusion | None = None

    try:
        # Build diffusion kwargs by extracting matching OmniDiffusionConfig fields from args
        config_field_names = {f.name for f in fields(OmniDiffusionConfig)}
        diffusion_kwargs: dict[str, Any] = {"model": args.model}

        # Diffusion parallelism configuration (e.g. `--usp 2`).
        parallel_config_kwargs: dict[str, Any] = {}
        for field in fields(DiffusionParallelConfig):
            if not hasattr(args, field.name):
                continue
            value = getattr(args, field.name)
            if value is None:
                continue
            parallel_config_kwargs[field.name] = value
        if parallel_config_kwargs:
            diffusion_kwargs["parallel_config"] = DiffusionParallelConfig(**parallel_config_kwargs)

        for field_name in config_field_names:
            if not hasattr(args, field_name):
                continue
            value = getattr(args, field_name)
            if value is None:
                continue
            # Special handling for cache_config JSON string
            if field_name == "cache_config" and isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse cache_config JSON: {e}")
                    continue
            diffusion_kwargs[field_name] = value

        diffusion_kwargs.update(kwargs)
        logger.info(f"diffusion_kwargs: {diffusion_kwargs}")
        logger.info(
            "Building AsyncOmniDiffusion with model=%s, num_gpus=%s",
            args.model,
            diffusion_kwargs.get("num_gpus", 1),
        )
        diffusion_engine = AsyncOmniDiffusion(**diffusion_kwargs)

        yield diffusion_engine
    finally:
        if diffusion_engine:
            diffusion_engine.shutdown()


@asynccontextmanager
async def build_async_omni_from_stage_config(
    args: Namespace,
    *,
    disable_frontend_multiprocessing: bool = False,
    client_config: dict[str, Any] | None = None,
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
        async_omni = AsyncOmni(model=args.model, cli_args=args)

        # # Don't keep the dummy data in memory
        # await async_llm.reset_mm_cache()

        yield async_omni
    finally:
        if async_omni:
            async_omni.shutdown()


async def omni_init_app_state(
    engine_client: EngineClient,
    vllm_config: VllmConfig,
    state: State,
    args: Namespace,
) -> None:
    """Initialize the FastAPI application state for omni API server.

    Sets up the application state with model information, request logger,
    and other server configuration needed for handling API requests.

    Args:
        engine_client: Engine client instance (AsyncOmni)
        vllm_config: vLLM configuration object
        state: FastAPI application state object to initialize
        args: Parsed command-line arguments
    """
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
    state.vllm_config = vllm_config
    _model_config = vllm_config.model_config
    state.log_stats = not args.disable_log_stats

    # For omni models
    state.stage_configs = engine_client.stage_configs

    resolved_chat_template = load_chat_template(args.chat_template)
    if resolved_chat_template is not None:
        # Get the tokenizer to check official template
        tokenizer = await engine_client.get_tokenizer()

        if isinstance(tokenizer, MistralTokenizer):
            # The warning is logged in resolve_mistral_chat_template.
            resolved_chat_template = resolve_mistral_chat_template(chat_template=resolved_chat_template)
        else:
            hf_chat_template = resolve_hf_chat_template(
                tokenizer=tokenizer,
                chat_template=None,
                tools=None,
                model_config=vllm_config.model_config,
            )

            if hf_chat_template != resolved_chat_template:
                logger.warning(
                    "Using supplied chat template: %s\nIt is different from official chat template '%s'. This discrepancy may lead to performance degradation.",  # noqa: E501
                    resolved_chat_template,
                    args.model,
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
    default_mm_loras = vllm_config.lora_config.default_mm_loras if vllm_config.lora_config is not None else {}

    lora_modules = args.lora_modules
    if default_mm_loras:
        default_mm_lora_paths = [
            LoRAModulePath(
                name=modality,
                path=lora_path,
            )
            for modality, lora_path in default_mm_loras.items()
        ]
        if args.lora_modules is None:
            lora_modules = default_mm_lora_paths
        else:
            lora_modules += default_mm_lora_paths

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=lora_modules,
    )
    await state.openai_serving_models.init_static_loras()
    state.openai_serving_chat = OmniOpenAIServingChat(
        engine_client,
        state.openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_force_include_usage=args.enable_force_include_usage,
        enable_log_outputs=args.enable_log_outputs,
        log_error_stack=args.log_error_stack,
    )

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


def Omnichat(request: Request) -> OmniOpenAIServingChat | None:
    return request.app.state.openai_serving_chat


async def omni_diffusion_init_app_state(
    diffusion_engine: AsyncOmniDiffusion,
    state: State,
    args: Namespace,
) -> None:
    """Initialize the FastAPI application state for diffusion model API server.

    Sets up the application state with diffusion model information and
    chat completion handler for image generation via /v1/chat/completions.

    Args:
        diffusion_engine: AsyncOmniDiffusion engine instance
        state: FastAPI application state object to initialize
        args: Parsed command-line arguments
    """
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    model_name = served_model_names[0] if served_model_names else args.model

    state.diffusion_engine = diffusion_engine
    state.log_stats = not getattr(args, "disable_log_stats", False)

    # Initialize chat handler with diffusion engine (uses /v1/chat/completions endpoint)
    # Note: Request-level parameters (num_inference_steps, guidance_scale, seed, height, width, etc.)
    # are passed per-request via the API, not as server defaults
    state.openai_serving_chat = OmniOpenAIServingChat.for_diffusion(
        diffusion_engine=diffusion_engine,
        model_name=model_name,
    )

    # Set other handlers to None for diffusion-only mode
    state.engine_client = None
    state.vllm_config = None

    state.enable_server_load_tracking = getattr(args, "enable_server_load_tracking", False)
    state.server_load_metrics = 0

    logger.info("Diffusion API server initialized for model: %s", model_name)


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
            content=generator.model_dump(), status_code=generator.code if hasattr(generator, "code") else 400
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")

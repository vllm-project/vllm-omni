# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import socket
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.usage.usage_lib import UsageContext
from vllm.utils.func_utils import deprecate_kwargs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager

from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.engine.output_processor import MultimodalOutputProcessor

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class AsyncOmniLLM(AsyncLLM):
    """Async single-stage LLM engine for use within a stage worker process.

    This class extends the base vLLM AsyncLLM class with omni-specific
    processors for handling multimodal inputs and outputs. It is used
    internally by AsyncOmniStage workers and should not be instantiated
    directly by users.

    Args:
        engine_args: AsyncOmniEngineArgs containing engine configuration
        vllm_config: Global vLLM configuration
        executor_class: Executor implementation class, e.g. MultiprocExecutor
        log_stats: Whether to log statistics
        usage_context: Usage context of the LLM (default: ENGINE_CONTEXT)
        mm_registry: Multi-modal registry for processing multimodal inputs
        use_cached_outputs: Whether to use cached outputs
        log_requests: Whether to log requests
        start_engine_loop: Whether to start the engine loop automatically
        stat_loggers: Customized stat loggers for the engine.
            If not provided, default stat loggers will be used.
            Note: Stat logger interface may change in V1.
        client_addresses: Optional dictionary mapping client names to addresses
        client_count: Total number of clients (default: 1)
        client_index: Index of this client (default: 0)
    """

    def __init__(
        self,
        engine_args: AsyncOmniEngineArgs,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        """
        Create an AsyncOmniLLM.

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            use_cached_outputs: Whether to use cached outputs.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()

        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.log_requests = log_requests

        self.log_stats = log_stats or (stat_loggers is not None)
        if not log_stats and stat_loggers is not None:
            logger.info(
                "AsyncLLM created with log_stats=False and non-empty custom logger list; "
                "enabling logging without default stat loggers"
            )

        if self.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            # Tokenizer (+ ensure liveness if running in another process).
            tokenizer = cached_tokenizer_from_config(model_config=vllm_config.model_config)

        # InputProcessor (converts Inputs --> EngineCoreRequests).
        self.input_processor = OmniInputProcessor(
            vllm_config=vllm_config,
            mm_registry=mm_registry,
        )

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        self.output_processor = MultimodalOutputProcessor(
            tokenizer=tokenizer,
            log_stats=self.log_stats,
            engine_core_output_type=engine_args.engine_output_type,
        )

        if self.observability_config.otlp_traces_endpoint is not None:
            tracer = init_tracer("vllm.llm_engine", self.observability_config.otlp_traces_endpoint)
            self.output_processor.tracer = tracer

        # Pause / resume state for async RL workflows.
        self._pause_cond = asyncio.Condition()
        self._paused = False

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        # Loggers.
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
            )
            self.logger_manager.log_engine_initialized()

        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

        # Use profiler_config from vllm_config (new way, aligned with vllm v1)
        if vllm_config.profiler_config.profiler == "torch" and not vllm_config.profiler_config.ignore_frontend:
            profiler_dir = vllm_config.profiler_config.torch_profiler_dir
            logger.info(
                "Torch profiler enabled. AsyncOmniLLM CPU traces will be collected under %s",
                profiler_dir,
            )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_omni_llm"
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=vllm_config.profiler_config.torch_profiler_with_stack,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profiler_dir,
                    worker_name=worker_name,
                    use_gzip=vllm_config.profiler_config.torch_profiler_use_gzip,
                ),
            )
        else:
            self.profiler = None

    @classmethod
    @deprecate_kwargs(
        "disable_log_requests",
        additional_message=("This argument will have no effect. Use `enable_log_requests` instead."),
    )
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        engine_args: AsyncOmniEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        disable_log_requests: bool = True,  # Deprecated, will be removed
    ) -> "AsyncLLM":
        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
            engine_args=engine_args,
        )

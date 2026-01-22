from collections.abc import Callable
from typing import Any

import cloudpickle
from pydantic import ValidationError
from tqdm import tqdm

# External library imports (vLLM)
from vllm.config import CompilationConfig, StructuredOutputsConfig, is_init_field
from vllm.entrypoints.llm import LLM
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.usage.usage_lib import UsageContext
from vllm.utils.counter import Counter
from vllm.v1.engine.llm_engine import LLMEngine

from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors

# Internal imports (our code)
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.entrypoints.utils import (
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)

logger = init_logger(__name__)


class OmniLLM(LLM):
    """Main entry point for vLLM-Omni inference.

    This class extends the base vLLM LLM class with omni-specific
    processors for handling multimodal inputs and outputs. It provides
    configuration loading for multi-stage pipelines, while stage management
    is handled by the Omni class.

    Args:
        model: Model name or path to load
        stage_configs_path: Optional path to YAML file containing stage
            configurations. If None, configurations are loaded from the model.
        log_stats: Whether to enable statistics logging
        compilation_config: Optional compilation configuration. Can be an
            integer (compilation level), dict, or CompilationConfig instance.
        hf_overrides: Optional HuggingFace model configuration overrides
        structured_outputs_config: Optional structured outputs configuration.
            Can be a dict or StructuredOutputsConfig instance.
        init_sleep_seconds: Number of seconds to sleep between starting
            each stage process during initialization (used by Omni class)
        shm_threshold_bytes: Threshold in bytes for using shared memory
            for IPC. Objects larger than this threshold will use shared memory.
        batch_timeout: Timeout in seconds for batching requests within a stage
        init_timeout: Timeout in seconds for waiting for all stages to initialize
        **kwargs: Additional keyword arguments passed to the base LLM class
            and engine

    Example:
        >>> llm = OmniLLM(model="Qwen/Qwen2.5-Omni-7B")
        >>> # Stage management is handled by Omni class
    """

    def __init__(
        self,
        model: str,
        stage_configs_path: str | None = None,
        log_stats: bool = False,
        compilation_config: int | dict[str, Any] | CompilationConfig | None = None,
        hf_overrides: dict[str, Any] | None = None,
        structured_outputs_config: dict[str, Any] | StructuredOutputsConfig | None = None,
        init_sleep_seconds: int = 20,
        shm_threshold_bytes: int = 65536,
        batch_timeout: int = 10,
        init_timeout: int = 300,
        **kwargs: Any,
    ):
        """LLM constructor with omni-specific configuration loading."""
        # Store stage management parameters (used by Omni class)
        self.worker_backend = kwargs.get("worker_backend", "multi_process")
        self.ray_address = kwargs.get("ray_address", None)
        self.batch_timeout = batch_timeout
        self._enable_stats: bool = bool(log_stats)

        # Load stage configurations
        if stage_configs_path is None:
            self.config_path = resolve_model_config_path(model)
            self.stage_configs = load_stage_configs_from_model(model)
        else:
            self.config_path = stage_configs_path
            self.stage_configs = load_stage_configs_from_yaml(stage_configs_path)

        # Initialize connectors
        self.omni_transfer_config, self.connectors = initialize_orchestrator_connectors(
            self.config_path, worker_backend=self.worker_backend, shm_threshold_bytes=shm_threshold_bytes
        )

        # Initialize LLM engine
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if "kv_transfer_config" in kwargs and isinstance(kwargs["kv_transfer_config"], dict):
            from vllm.config.kv_transfer import KVTransferConfig

            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(**raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict,
                    e,
                )
                raise ValueError(f"Invalid 'kv_transfer_config' provided: {e}") from e

        if compilation_config is not None:
            if isinstance(compilation_config, int):
                compilation_config_instance = CompilationConfig(level=compilation_config)
            elif isinstance(compilation_config, dict):
                compilation_config_instance = CompilationConfig(
                    **{k: v for k, v in compilation_config.items() if is_init_field(CompilationConfig, k)}
                )
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = CompilationConfig()

        if structured_outputs_config is not None:
            if isinstance(structured_outputs_config, dict):
                structured_outputs_instance = StructuredOutputsConfig(
                    **{k: v for k, v in structured_outputs_config.items() if is_init_field(StructuredOutputsConfig, k)}
                )
            else:
                structured_outputs_instance = structured_outputs_config
        else:
            structured_outputs_instance = StructuredOutputsConfig()

        engine_args = OmniEngineArgs(
            model=model,
            compilation_config=compilation_config_instance,
            structured_outputs_config=structured_outputs_instance,
            **kwargs,
        )

        # Create the Engine (autoselects V0 vs V1)
        self.llm_engine = LLMEngine.from_engine_args(engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)
        self.llm_engine.output_processor = MultimodalOutputProcessor(
            tokenizer=self.llm_engine.tokenizer,
            log_stats=self.llm_engine.log_stats,
            engine_core_output_type=engine_args.engine_output_type,
        )
        self.llm_engine.input_processor = OmniInputProcessor(
            vllm_config=self.llm_engine.vllm_config, tokenizer=self.llm_engine.tokenizer
        )
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: dict[str, Any] | None = None

        supported_tasks = self.llm_engine.get_supported_tasks()  # type: ignore

        logger.info("Supported_tasks: %s", supported_tasks)

        self.supported_tasks = supported_tasks

        # Load the Input/Output processor plugin if any
        io_processor_plugin = self.llm_engine.model_config.io_processor_plugin
        self.io_processor = get_io_processor(self.llm_engine.vllm_config, io_processor_plugin)
        self.model_config = self.llm_engine.model_config
        self.input_processor = self.llm_engine.input_processor

    def close(self) -> None:
        """Close resources.

        Note: Stage management is now handled by Omni class.
        This method closes the LLM engine but not stages.
        """
        # Close the LLM engine if it exists
        if hasattr(self, "llm_engine") and self.llm_engine is not None:
            if hasattr(self.llm_engine, "shutdown"):
                self.llm_engine.shutdown()

    def __del__(self) -> None:  # best-effort
        try:
            self.close()
        except Exception as e:
            logger.debug("[Orchestrator] __del__ close() raised: %s", e, exc_info=True)

    def _run_engine(self, *, use_tqdm: bool | Callable[..., tqdm] = True) -> list[RequestOutput | PoolingRequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: list[RequestOutput | PoolingRequestOutput] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            n = len(output.outputs)
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids) * n
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(len(stp.token_ids) for stp in output.outputs)
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = f"est. speed input: {in_spd:.2f} toks/s, output: {out_spd:.2f} toks/s"
                            pbar.update(n)
                        else:
                            pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

        if use_tqdm:
            pbar.close()
        # Sort the outputs by the int part of request ID which is in format of 'int-uuid'.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id.split("-")[0]))

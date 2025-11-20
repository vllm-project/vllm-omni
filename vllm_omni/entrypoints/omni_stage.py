"""
Stage manager for orchestrating multiple engines in vLLM-omni.

Enhanced to encapsulate per-stage process lifecycle and worker logic
(device setup, LLM init, batching, shared-memory IPC), while preserving
the original input processing utilities for cross-stage data wiring.

Enhanced to encapsulate per-stage process lifecycle and worker logic
(device setup, LLM init, batching, shared-memory IPC), while preserving
the original input processing utilities for cross-stage data wiring.
"""

import asyncio
import importlib
import logging
import multiprocessing as mp
from typing import Any, Optional, Union

from vllm.inputs import TextPrompt
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine

from vllm_omni.entrypoints.stage_utils import (
    _to_dict,
    maybe_dump_to_shm,
    maybe_load_from_ipc_with_metrics,
    set_stage_gpu_devices,
)
from vllm_omni.inputs.data import OmniTokensPrompt

from .omni_stage_worker import _stage_worker, _stage_worker_async_entry

logger = init_logger(__name__)


class OmniStage:
    def __init__(self, stage_config):
        self.stage_config = stage_config
        self.engine = None
        self.async_engine = None
        self.vllm_config = None
        self.tokenizer = None
        self.input_preprocessor = None
        self.is_tracing_enabled = False
        self.stage_id = stage_config.stage_id
        self.engine_args = stage_config.engine_args
        self.model_stage = stage_config.engine_args.model_stage
        self.engine_input_source = getattr(stage_config, "engine_input_source", [])
        self.engine_output_type = stage_config.engine_args.engine_output_type
        self.engine_outputs = None
        self.is_comprehension = getattr(stage_config, "is_comprehension", False)
        if hasattr(stage_config, "custom_process_input_func"):
            # Import the module specified in the config (already a full module path)
            module_path, func_name = stage_config.custom_process_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_input_func = getattr(module, func_name)
        else:
            self.custom_process_input_func = None

        self.final_output = getattr(stage_config, "final_output", False)
        self.final_output_type = getattr(stage_config, "final_output_type", None)
        default_sampling_params = getattr(stage_config, "default_sampling_params", {})
        self.default_sampling_params = SamplingParams(**_to_dict(default_sampling_params))
        # Runtime orchestration state (added)
        self._in_q: Optional[mp.Queue] = None
        self._out_q: Optional[mp.Queue] = None
        self._proc: Optional[mp.Process] = None
        self._log_file: Optional[str] = None
        self._shm_threshold_bytes: int = 65536
        self._logger = logging.getLogger(__name__)

    def set_engine(self, engine: LLMEngine) -> None:
        """Initialize the engine for the stage."""
        self.engine = engine

    def set_async_engine(self, async_engine: AsyncLLM) -> None:
        """Initialize the async engine for the stage."""
        self.async_engine = async_engine

    def set_vllm_config(self, vllm_config) -> None:
        """Set the vllm_config for the stage (received from worker process)."""
        self.vllm_config = vllm_config

    def set_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        """Set the tokenizer for the stage (received from worker process)."""
        self.tokenizer = tokenizer

    def set_input_preprocessor(self, input_preprocessor: InputPreprocessor) -> None:
        """Set the input preprocessor for the stage (received from worker process)."""
        self.input_preprocessor = input_preprocessor

    def set_is_tracing_enabled(self, is_tracing_enabled: bool) -> None:
        """Set the is_tracing_enabled for the stage (received from worker process)."""
        self.is_tracing_enabled = is_tracing_enabled

    def set_engine_outputs(self, engine_outputs: EngineCoreOutput) -> None:
        """Set the engine output for the stage."""
        self.engine_outputs = engine_outputs

    # ----------------- New Orchestration APIs -----------------
    def attach_queues(self, in_q: mp.Queue, out_q: mp.Queue) -> None:
        self._in_q = in_q
        self._out_q = out_q

    def init_stage_worker(
        self,
        model: str,
        *,
        is_async: bool = False,
        log_file: Optional[str] = None,
        shm_threshold_bytes: int = 65536,
        ctx: Optional[mp.context.BaseContext] = None,
        batch_timeout: int = 10,
    ) -> None:
        assert self._in_q is not None and self._out_q is not None, "Queues must be attached before start_process"
        self._log_file = log_file
        self._shm_threshold_bytes = shm_threshold_bytes
        ctx = ctx or mp.get_context("spawn")
        # Prepare lightweight dict config for worker
        engine_args = _to_dict(self.engine_args)
        runtime_cfg = _to_dict(getattr(self.stage_config, "runtime", {}))
        stage_payload: dict[str, Any] = {
            "stage_id": self.stage_id,
            "engine_args": engine_args,
            "runtime": runtime_cfg,
            "shm_threshold_bytes": self._shm_threshold_bytes,
        }
        if is_async:
            self._proc = ctx.Process(
                target=_stage_worker_async_entry,
                args=(
                    self,
                    model,
                    stage_payload,
                    self._in_q,
                    self._out_q,
                    self._log_file,
                    batch_timeout,
                ),
            )
        else:
            self._proc = ctx.Process(
                target=_stage_worker,
                args=(
                    model,
                    stage_payload,
                    self._in_q,
                    self._out_q,
                    self._log_file,
                    batch_timeout,
                ),
            )
        self._proc.start()

    def stop_stage_worker(self) -> None:
        if self._in_q is not None:
            try:
                self._in_q.put_nowait(None)
            except Exception as e:
                self._logger.warning("[Stage-%s] Failed to send shutdown to in_q: %s", self.stage_id, e)
        if self._proc is not None:
            try:
                self._proc.join(timeout=5)
            except Exception as e:
                self._logger.debug("[Stage-%s] join() failed: %s", self.stage_id, e, exc_info=True)
            if self._proc.is_alive():
                try:
                    self._proc.terminate()
                except Exception as e:
                    self._logger.warning("[Stage-%s] terminate() failed: %s", self.stage_id, e)

    def submit(self, payload: dict[str, Any]) -> None:
        assert self._in_q is not None
        self._in_q.put(payload)

    def try_collect(self) -> Optional[dict[str, Any]]:
        assert self._out_q is not None
        try:
            return self._out_q.get_nowait()
        except Exception:
            return None

    def process_engine_inputs(
        self, stage_list, prompt: Union[OmniTokensPrompt, TextPrompt] = None
    ) -> list[Union[OmniTokensPrompt, TextPrompt]]:
        """Process the engine input for the stage."""
        if self.custom_process_input_func is None:
            engine_inputs = []
            if len(self.engine_input_source) == 0:
                raise ValueError("engine_input_source is empty")
            source_stage_id = self.engine_input_source[0]
            source_outputs = stage_list[source_stage_id].engine_outputs
            if not isinstance(prompt, list):
                prompt = [prompt]
            multi_modal_data = {
                source_output.request_id: p.get("multi_modal_data", None)
                for source_output, p in zip(source_outputs, prompt)
            }

            for source_output in source_outputs:
                engine_input = OmniTokensPrompt(
                    prompt_token_ids=source_output.outputs[0].token_ids,
                    multi_modal_data=(multi_modal_data[source_output.request_id] if multi_modal_data else None),
                )
                engine_inputs.append(engine_input)
            return engine_inputs

        else:
            engine_input_source = self.engine_input_source
            return self.custom_process_input_func(stage_list, engine_input_source, prompt)

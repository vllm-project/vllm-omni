from typing import Any, Optional, Sequence, Union
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import queue
import sys
import time

import cloudpickle
from pydantic import ValidationError

# External library imports (vLLM)
from vllm.config import CompilationConfig, StructuredOutputsConfig, is_init_field
from vllm.engine.arg_utils import HfOverrides
from vllm.entrypoints.llm import LLM
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.plugins.io_processors import get_io_processor
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter
from vllm.v1.engine.llm_engine import LLMEngine

# Internal imports (our code)
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.engine.processor import OmniProcessor
from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.utils import load_stage_configs_from_model
from vllm_omni.entrypoints.log_utils import (
    remove_old_logs,
    configure_orchestrator_logger,
    init_stats_paths,
    OrchestratorMetrics,
)
from vllm_omni.entrypoints.pipeline_utils import (
    maybe_load_from_ipc as _load,
    encode_for_ipc as _encode,
    serialize_obj as _ser,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


class OmniLLM:
    """Multi-process pipelined OmniLLM.

    - Per-stage process with copy-based IPC (Queues)
    - window=-1 across stages (downstream starts when upstream finishes)
    - max_inflight=1 per stage (serial within a stage), but pipeline across stages
    """

    def __init__(self, model: str,
                 stage_configs=None,
                 log_stats: bool = False,
                 log_file: Optional[str] = None,
                 init_sleep_seconds: int = 20,
                 shm_threshold_bytes: int = 65536,
                 batch_timeout: int = 10,
                 init_timeout: int = 300,
                 **kwargs):
        self.batch_timeout = batch_timeout
        self._enable_stats: bool = bool(log_stats)
        # Do NOT call super().__init__ to avoid creating OmniStageLLM instances in parent.
        if stage_configs is None:
            self.stage_configs = load_stage_configs_from_model(model)
        else:
            self.stage_configs = stage_configs


        # Optional file handler for orchestrator
        self._log_file = log_file
        if self._log_file:
            remove_old_logs(self._log_file, len(self.stage_list))
            configure_orchestrator_logger(logger, self._log_file)

        self._stats_file, self._overall_stats_file = init_stats_paths(self._enable_stats, self._log_file)
        self._initialize_stages(model, init_sleep_seconds, shm_threshold_bytes, init_timeout)

    def _initialize_stages(self, model: str, init_sleep_seconds: int, shm_threshold_bytes: int, init_timeout: int) -> None:
        self.stage_list: list[OmniStage] = []
        # Build OmniStage instances in parallel, preserve original order
        def _build_stage(idx_cfg: tuple[int, Any]) -> tuple[int, OmniStage]:
            idx, cfg = idx_cfg
            return idx, OmniStage(cfg)

        with ThreadPoolExecutor(max_workers=min(len(self.stage_configs), max(1, os.cpu_count() or 1))) as executor:
            futures = [executor.submit(_build_stage, (idx, cfg)) for idx, cfg in enumerate(self.stage_configs)]
            results: list[tuple[int, OmniStage]] = []
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda x: x[0])
        self.stage_list = [st for _, st in results]
        logger.debug("[Orchestrator] Loaded %d stages", len(self.stage_list))

        self._ctx = mp.get_context("spawn")
        self._stage_in_queues: list[mp.Queue] = []
        self._stage_out_queues: list[mp.Queue] = []
        self._init_sleep_seconds = max(0, int(init_sleep_seconds))
        self._shm_threshold_bytes = max(0, int(shm_threshold_bytes))
        self._start_stage_processes(model)
        # Wait for all stages to report readiness before seeding
        self._stages_ready: set[int] = set()
        self._wait_for_stages_ready(timeout=init_timeout)
        

    def _start_stage_processes(self, model: str) -> None:
        for stage_id, stage in enumerate(self.stage_list):
            # Use unbounded queues to avoid deadlock when seeding many requests
            in_q: mp.Queue = self._ctx.Queue(maxsize=0)
            out_q: mp.Queue = self._ctx.Queue(maxsize=0)
            self._stage_in_queues.append(in_q)
            self._stage_out_queues.append(out_q)

            # Attach queues and start Stage-owned worker process
            stage.attach_queues(in_q, out_q)
            stage.init_stage_worker(
                model,
                log_file=self._log_file,
                shm_threshold_bytes=self._shm_threshold_bytes,
                ctx=self._ctx,
                batch_timeout=self.batch_timeout,
            )
            logger.debug("[Orchestrator] Stage-%s process started", stage_id)
            time.sleep(self._init_sleep_seconds)

    def close(self) -> None:
        for q in self._stage_in_queues:
            try:
                q.put_nowait(None)
            except Exception as e:
                logger.warning("[Orchestrator] Failed to send shutdown signal to stage input queue: %s", e)
        for stage in self.stage_list:
            try:
                stage.stop_stage_worker()
            except Exception as e:
                logger.warning("[Orchestrator] Failed to stop stage worker: %s", e)

    def __del__(self) -> None:  # best-effort
        try:
            self.close()
        except Exception as e:
            logger.debug("[Orchestrator] __del__ close() raised: %s", e, exc_info=True)

    def generate(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        sampling_params_list: Optional[
            Union[SamplingParams, Sequence[SamplingParams]]
        ] = None,
    ) -> list[OmniRequestOutput]:
        logger.debug("[Orchestrator] generate() called")
        if sampling_params_list is None:
            raise ValueError("sampling_params_list is required for pipelined generation")
        if len(sampling_params_list) != len(self.stage_list):
            raise ValueError(
                f"Expected {len(self.stage_list)} sampling params, got {len(sampling_params_list)}"
            )

        # Normalize prompts to a list for per-request iteration
        if not isinstance(prompts, (list, tuple)):
            request_prompts: list[PromptType] = [prompts]
        else:
            request_prompts = list(prompts)

        final_outputs: list[OmniRequestOutput] = []

        # Orchestrator keeps stage objects for input derivation
        num_stages = len(self.stage_list)

        # Map from request_id to original prompt
        request_id_to_prompt: dict[int, PromptType] = {i: p for i, p in enumerate(request_prompts)}

        # Track per-request start time for end-to-end timing
        _req_start_ts: dict[int, float] = {}
        _wall_start_ts: float = time.time()
        _last_finish_ts: float = _wall_start_ts

        # Determine the final stage for E2E stats (highest stage_id with final_output=True; fallback to last stage)
        final_stage_id_for_e2e = -1
        try:
            for _sid, _st in enumerate(self.stage_list):
                if getattr(_st, "final_output", False):
                    final_stage_id_for_e2e = max(final_stage_id_for_e2e, _sid)
            if final_stage_id_for_e2e < 0:
                final_stage_id_for_e2e = len(self.stage_list) - 1
        except Exception as e:
            logger.debug("[Orchestrator] Failed to determine final stage for E2E; falling back to last: %s", e, exc_info=True)
            final_stage_id_for_e2e = len(self.stage_list) - 1
        # Metrics/aggregation helper
        metrics = OrchestratorMetrics(num_stages, self._enable_stats, self._stats_file, self._overall_stats_file, _wall_start_ts)

        # Seed stage-0 queue with all requests
        logger.debug("[Orchestrator] Seeding %d requests into stage-0", len(request_prompts))
        # Mark first input time for stage-0
        metrics.stage_first_ts[0] = metrics.stage_first_ts[0] or time.time()

        for req_id, prompt in request_id_to_prompt.items():
            sp0: SamplingParams = sampling_params_list[0]  # type: ignore[index]
            task = {
                "request_id": req_id,
                "engine_inputs": prompt,
                "sampling_params": sp0,
            }
            self.stage_list[0].submit(task)
            _req_start_ts[req_id] = time.time()
            logger.debug("[Orchestrator] Enqueued request %s to stage-0", req_id)

        # For each stage, forward results to next stage; collect finals at the end
        # We pipeline by continually polling output queues in stage order
        remaining_by_stage: list[int] = [len(request_prompts)] + [0] * (num_stages - 1)
        completed_requests = 0
        total_requests = len(request_prompts)

        logger.debug("[Orchestrator] Entering scheduling loop: total_requests=%d, stages=%d", total_requests, num_stages)
        while completed_requests < total_requests:
            made_progress = False
            for stage_id, stage in enumerate(self.stage_list):
                result = stage.try_collect()
                if result is None:
                    continue

                made_progress = True
                req_id = result.get("request_id")
                if "error" in result:
                    logger.error("Stage %s error on request %s: %s", stage_id, req_id, result["error"])
                    continue
                
                if result.get("type") == "stage_ready":
                    #Only happens when stage is initialized slower than expected, so we wait for a short time and try again
                    time.sleep(0.05)
                    continue

                engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
                # Mark last output time for this stage whenever we receive outputs
                metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, time.time())
                try:
                    _m = result.get("metrics")
                    if _m is not None:
                        metrics.on_stage_metrics(stage_id, req_id, _m)
                except Exception as e:
                    logger.exception("[Orchestrator] Failed to process metrics for stage %s, req %s: %s", stage_id, req_id, e)
                logger.debug("[Orchestrator] Stage-%s completed request %s; forwarding or finalizing", stage_id, req_id)
                stage.set_engine_outputs(engine_outputs)

                if getattr(stage, "final_output", False):
                    final_outputs.append(
                        OmniRequestOutput(
                            stage_id=stage_id,
                            final_output_type=stage.final_output_type,  # type: ignore[attr-defined]
                            request_output=engine_outputs,
                        )
                    )
                    logger.debug("[Orchestrator] Request %s finalized at stage-%s", req_id, stage_id)

                    # End-to-end timing and time-per-token for final output (only once per request at the designated final stage)
                    try:
                        rid_int = int(req_id) if isinstance(req_id, (int, str)) and str(req_id).isdigit() else req_id
                        if stage_id == final_stage_id_for_e2e and rid_int not in metrics.e2e_done:
                            metrics.on_finalize_request(stage_id, req_id, engine_outputs, _req_start_ts.get(req_id, _wall_start_ts))
                    except Exception as e:
                        logger.exception("[Orchestrator] Finalize request handling error for req %s at stage %s: %s", req_id, stage_id, e)

                next_stage_id = stage_id + 1
                if next_stage_id < num_stages:
                    next_stage: OmniStage = self.stage_list[next_stage_id]
                    next_inputs = next_stage.process_engine_inputs(self.stage_list, [request_id_to_prompt[req_id]])
                    sp_next: SamplingParams = sampling_params_list[next_stage_id]  # type: ignore[index]
                    try:
                        # Measure transfer size and time (encode + enqueue)
                        size_bytes = 0
                        try:
                            size_bytes = len(_ser(next_inputs))
                        except Exception:
                            size_bytes = 0
                        t0 = time.time()
                        ipc_payload = _encode(
                            next_inputs,
                            getattr(self, "_shm_threshold_bytes", 65536),
                            obj_key="engine_inputs",
                            shm_key="engine_inputs_shm",
                        )
                        ipc_payload.update({
                            "request_id": req_id,
                            "sampling_params": sp_next,
                            "sent_ts": time.time(),
                        })
                        self.stage_list[next_stage_id].submit(ipc_payload)
                        t1 = time.time()
                        tx_ms = (t1 - t0) * 1000.0
                        metrics.on_forward(stage_id, next_stage_id, req_id, int(size_bytes), float(tx_ms), bool("engine_inputs_shm" in ipc_payload))
                    except Exception as e:
                        logger.warning("[Orchestrator] IPC encode failed for req %s: %s; falling back to inline payload", req_id, e)
                        self.stage_list[next_stage_id].submit({
                            "request_id": req_id,
                            "engine_inputs": next_inputs,
                            "sampling_params": sp_next,
                        })
                    logger.debug("[Orchestrator] Forwarded request %s to stage-%s", req_id, next_stage_id)
                    remaining_by_stage[next_stage_id] += 1
                else:
                    completed_requests += 1
                    logger.debug("[Orchestrator] Request %s fully completed (%d/%d)", req_id, completed_requests, total_requests)

            if not made_progress:
                time.sleep(0.005)
        logger.debug("[Orchestrator] All requests completed")

        # Summarize and print stats
        try:
            summary = metrics.build_and_log_summary(final_stage_id_for_e2e)
            logger.info("[Summary] %s", summary)
        except Exception as e:
            logger.exception("[Orchestrator] Failed to build/log summary: %s", e)

        return final_outputs

    def _wait_for_stages_ready(self, timeout: int = 120) -> None:
        deadline = time.time() + max(0, int(timeout))
        num_stages = len(self.stage_list)
        while len(self._stages_ready) < num_stages and time.time() < deadline:
            progressed = False
            for stage_id, stage in enumerate(self.stage_list):
                if stage_id in self._stages_ready:
                    continue
                result = stage.try_collect()
                if result is None:
                    continue
                progressed = True
                if result.get("type") == "stage_ready":
                    self._stages_ready.add(stage_id)
                    logger.debug("[Orchestrator] Stage-%s reported ready", stage_id)
                else:
                    # No user data should arrive before seeding; ignore other messages
                    pass
            if not progressed:
                time.sleep(0.01)
        if len(self._stages_ready) < num_stages:
            not_ready = sorted(set(range(num_stages)) - set(self._stages_ready))
            logger.warning(
                "[Orchestrator] Initialization timeout: only %s/%s stages are ready; not ready: %s",
                len(self._stages_ready), num_stages, not_ready,
            )
            # Provide actionable suggestions before shutdown
            try:
                suggestions = [
                    "Verify GPU/device assignment in config (runtime.devices) is correct.",
                    "Check GPU/host memory availability; reduce model or batch size if needed.",
                    "Check model weights path and network reachability (if loading remotely).",
                    "Increase initialization wait time (init_sleep_seconds or call-site timeout).",
                ]
                if getattr(self, "_log_file", None):
                    suggestions.append(
                        f"Inspect per-stage log files for details: {self._log_file}.stage<id>.log"
                    )
                logger.error(
                    "[Orchestrator] Stage initialization failed, shutting down. Suggestions:\n- %s",
                    "\n- ".join(suggestions),
                )
            except Exception:
                # Best-effort logging of suggestions
                logger.error(
                    "[Orchestrator] Stage initialization failed and an error occurred while logging suggestions",
                )

            # Attempt graceful shutdown of all stages before exiting
            try:
                self.close()
            except Exception:
                pass

            # Terminate the current process with non-zero exit code
            try:
                sys.exit(1)
            except SystemExit:
                raise
            except Exception:
                os._exit(1)


class OmniStageLLM(LLM):
    def __init__(
        self,
        model: str,
        compilation_config: Optional[
            Union[int, dict[str, Any], CompilationConfig]
        ] = None,
        hf_overrides: Optional[HfOverrides] = None,
        structured_outputs_config: Optional[
            Union[dict[str, Any], StructuredOutputsConfig]
        ] = None,
        **kwargs,
    ):
        """LLM constructor."""
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if "kv_transfer_config" in kwargs and isinstance(
            kwargs["kv_transfer_config"], dict
        ):
            from vllm.config.kv_transfer import KVTransferConfig

            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(**raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to "
                    "KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict,
                    e,
                )
                # Consider re-raising a more specific vLLM error or ValueError
                # to provide better context to the user.
                raise ValueError(f"Invalid 'kv_transfer_config' provided: {e}") from e

        if hf_overrides is None:
            hf_overrides = {}

        if compilation_config is not None:
            if isinstance(compilation_config, int):
                compilation_config_instance = CompilationConfig(
                    level=compilation_config
                )
            elif isinstance(compilation_config, dict):
                compilation_config_instance = CompilationConfig(
                    **{
                        k: v
                        for k, v in compilation_config.items()
                        if is_init_field(CompilationConfig, k)
                    }
                )
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = CompilationConfig()

        if structured_outputs_config is not None:
            if isinstance(structured_outputs_config, dict):
                structured_outputs_instance = StructuredOutputsConfig(
                    **{
                        k: v
                        for k, v in structured_outputs_config.items()
                        if is_init_field(StructuredOutputsConfig, k)
                    }
                )
            else:
                structured_outputs_instance = structured_outputs_config
        else:
            structured_outputs_instance = StructuredOutputsConfig()

        engine_args = OmniEngineArgs(
            model=model,
            hf_overrides=hf_overrides,
            compilation_config=compilation_config_instance,
            structured_outputs_config=structured_outputs_instance,
            **kwargs,
        )

        # Create the Engine (autoselects V0 vs V1)
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
        )
        self.llm_engine.output_processor = MultimodalOutputProcessor(
            tokenizer=self.llm_engine.tokenizer,
            log_stats=self.llm_engine.log_stats,
            engine_core_output_type=engine_args.engine_output_type,
        )
        self.llm_engine.processor = OmniProcessor(
            vllm_config=self.llm_engine.vllm_config, tokenizer=self.llm_engine.tokenizer
        )
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: Union[dict[str, Any], None] = None

        supported_tasks = self.llm_engine.get_supported_tasks()  # type: ignore

        logger.info("Supported_tasks: %s", supported_tasks)

        self.supported_tasks = supported_tasks

        # Load the Input/Output processor plugin if any
        io_processor_plugin = self.llm_engine.model_config.io_processor_plugin
        self.io_processor = get_io_processor(
            self.llm_engine.vllm_config, io_processor_plugin
        )

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import os
import sys
import queue
import time
import multiprocessing as mp
import logging

from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.omni_llm import OmniLLM, OmniStageLLM
from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.utils import load_stage_configs_from_model
from vllm_omni.entrypoints.log_utils import (
    remove_old_logs,
    configure_orchestrator_logger,
    init_stats_paths,
    record_stage_metrics,
    aggregate_rx_and_maybe_total,
    record_sender_transfer_agg,
    count_tokens_from_outputs,
    build_stage_summary,
    build_transfer_summary,
    log_transfer_tx,
    log_transfer_rx,
    log_transfer_total,
    log_orchestrator_e2e,
    log_orchestrator_summary,
    log_overall_summary,
    OrchestratorMetrics,
)
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.entrypoints.pipeline_utils import (
    maybe_load_from_ipc as _load,
    encode_for_ipc as _encode,
    serialize_obj as _ser,
    append_jsonl as _append_jsonl,
)

logger = init_logger(__name__)


class PipelinedOmniLLM(OmniLLM):
    """Multi-process pipelined OmniLLM.

    - Per-stage process with copy-based IPC (Queues)
    - window=-1 across stages (downstream starts when upstream finishes)
    - max_inflight=1 per stage (serial within a stage), but pipeline across stages
    """

    def __init__(self, model: str,
                 stage_configs=None,
                 log_stats: bool = True,
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

        self.stage_list: List[OmniStage] = []
        for stage_config in self.stage_configs:
            # Only construct lightweight OmniStage without setting engine in parent
            stage = OmniStage(stage_config)
            self.stage_list.append(stage)
        logger.debug("[Orchestrator] Loaded %d stages", len(self.stage_list))

        self._ctx = mp.get_context("spawn")
        self._stage_in_queues: List[mp.Queue] = []
        self._stage_out_queues: List[mp.Queue] = []
        # Optional file handler for orchestrator
        self._log_file = log_file
        if self._log_file:
            remove_old_logs(self._log_file, len(self.stage_list))
            configure_orchestrator_logger(logger, self._log_file)

        self._stats_file, self._overall_stats_file = init_stats_paths(self._enable_stats, self._log_file)

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
    ) -> List[OmniRequestOutput]:
        logger.debug("[Orchestrator] generate() called")
        if sampling_params_list is None:
            raise ValueError("sampling_params_list is required for pipelined generation")
        if len(sampling_params_list) != len(self.stage_list):
            raise ValueError(
                f"Expected {len(self.stage_list)} sampling params, got {len(sampling_params_list)}"
            )

        # Normalize prompts to a list for per-request iteration
        if not isinstance(prompts, (list, tuple)):
            request_prompts: List[PromptType] = [prompts]
        else:
            request_prompts = list(prompts)

        final_outputs: List[OmniRequestOutput] = []

        # Orchestrator keeps stage objects for input derivation
        num_stages = len(self.stage_list)

        # Map from request_id to original prompt
        request_id_to_prompt: Dict[int, PromptType] = {i: p for i, p in enumerate(request_prompts)}

        # Track per-request start time for end-to-end timing
        _req_start_ts: Dict[int, float] = {}
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
        remaining_by_stage: List[int] = [len(request_prompts)] + [0] * (num_stages - 1)
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
                    "Verify GPU/device assignment in config (mrs.devices) is correct.",
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



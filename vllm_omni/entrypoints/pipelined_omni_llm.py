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
            try:
                # Avoid duplicate handlers
                has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
                if not has_file_handler:
                    fh = logging.FileHandler(self._log_file)
                    fh.setLevel(logging.DEBUG)
                    fh.setFormatter(logging.Formatter("%(asctime)s [PID:%(process)d] %(levelname)s: %(message)s"))
                    logger.addHandler(fh)
                    logger.setLevel(logging.DEBUG)
            except Exception:
                pass

        # Orchestrator stats JSONL file
        self._stats_file: Optional[str] = None
        if self._enable_stats and self._log_file:
            try:
                self._stats_file = f"{self._log_file}.orchestrator.stats.jsonl"
            except Exception:
                self._stats_file = None
        # Overall stats JSONL file (per-request + summary)
        self._overall_stats_file: Optional[str] = None
        if self._enable_stats and self._log_file:
            try:
                self._overall_stats_file = f"{self._log_file}.overall.stats.jsonl"
            except Exception:
                self._overall_stats_file = None

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
            except Exception:
                pass
        for stage in self.stage_list:
            try:
                stage.stop_stage_worker()
            except Exception:
                pass

    def __del__(self) -> None:  # best-effort
        try:
            self.close()
        except Exception:
            pass

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
        except Exception:
            final_stage_id_for_e2e = len(self.stage_list) - 1
        # In-memory aggregators for this generate() call
        stage_total_time_ms: List[float] = [0.0 for _ in range(num_stages)]
        stage_total_tokens: List[int] = [0 for _ in range(num_stages)]
        stage_req_counts: List[int] = [0 for _ in range(num_stages)]
        transfer_agg: Dict[tuple[int, int], Dict[str, float]] = {}
        # Per-edge per-request sender timing to combine with receiver timing later
        transfer_edge_req: Dict[tuple[int, int, int], Dict[str, float]] = {}
        e2e_total_ms: float = 0.0
        e2e_total_tokens: int = 0
        e2e_count: int = 0
        e2e_done: set[int] = set()
        # Per-request overall aggregation
        per_request: Dict[int, Dict[str, Any]] = {}
        sum_per_request_transfer_ms: float = 0.0

        # Seed stage-0 queue with all requests
        logger.debug("[Orchestrator] Seeding %d requests into stage-0", len(request_prompts))

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
                # Aggregate per-request metrics from stage worker if present
                try:
                    _m = result.get("metrics")
                    if _m is not None:
                        stage_req_counts[stage_id] += 1
                        stage_total_time_ms[stage_id] += float(_m.get("stage_gen_time_ms", 0.0))
                        stage_total_tokens[stage_id] += int(_m.get("num_tokens_out", 0))
                        # record per-request stage metrics
                        try:
                            rid_int = int(req_id)
                            pr = per_request.setdefault(rid_int, {"stages": {}, "transfers_ms": 0.0, "transfers_bytes": 0})
                            pr_stages = pr["stages"]  # type: ignore[index]
                            pr_stages[stage_id] = {
                                "stage_gen_time_ms": float(_m.get("stage_gen_time_ms", 0.0)),
                                "num_tokens_out": int(_m.get("num_tokens_out", 0)),
                            }
                        except Exception:
                            pass
                        # Also aggregate receiver-side transfer decode time for (stage_id-1 -> stage_id)
                        try:
                            if stage_id > 0:
                                key = (stage_id - 1, stage_id)
                                agg = transfer_agg.get(key)
                                if agg is None:
                                    agg = {"sum_bytes": 0.0, "sum_ms": 0.0, "count": 0.0,
                                           "sum_rx_bytes": 0.0, "sum_rx_ms": 0.0, "rx_count": 0.0,
                                           "sum_total_ms": 0.0, "total_count": 0.0}
                                    transfer_agg[key] = agg
                                rx_b = float(_m.get("rx_transfer_bytes", 0.0))
                                rx_ms = float(_m.get("rx_decode_time_ms", 0.0))
                                in_flight_ms = float(_m.get("rx_in_flight_time_ms", 0.0))
                                agg["sum_rx_bytes"] += rx_b
                                agg["sum_rx_ms"] += rx_ms
                                agg["rx_count"] += 1.0
                                # If we have sender-side timing stored, compute combined (encode+enqueue + decode)
                                try:
                                    edge_req = (stage_id - 1, stage_id, int(req_id))
                                    s = transfer_edge_req.get(edge_req)
                                    if s is not None:
                                        total_ms = float(s.get("tx_ms", 0.0)) + in_flight_ms + rx_ms
                                        agg["sum_total_ms"] += total_ms
                                        agg["total_count"] += 1.0
                                        # accumulate per-request transfer totals
                                        try:
                                            rid_int = int(req_id)
                                            pr = per_request.setdefault(rid_int, {"stages": {}, "transfers_ms": 0.0, "transfers_bytes": 0})
                                            pr["transfers_ms"] = float(pr.get("transfers_ms", 0.0)) + total_ms  # type: ignore[index]
                                            pr["transfers_bytes"] = int(pr.get("transfers_bytes", 0)) + int(rx_b)  # type: ignore[index]
                                        except Exception:
                                            pass
                                        if getattr(self, "_stats_file", None):
                                            try:
                                                size_b = float(s.get("size_bytes", rx_b))
                                                _append_jsonl(self._stats_file, {  # type: ignore[arg-type]
                                                    "type": "transfer_total_stats",
                                                    "from_stage": stage_id - 1,
                                                    "to_stage": stage_id,
                                                    "request_id": req_id,
                                                    "size_bytes": int(size_b),
                                                    "tx_time_ms": float(s.get("tx_ms", 0.0)),
                                                    "in_flight_time_ms": in_flight_ms,
                                                    "rx_decode_time_ms": rx_ms,
                                                    "total_time_ms": total_ms,
                                                    "total_time_per_kb_ms": total_ms / max(size_b / 1024.0, 1e-6) if size_b > 0 else 0.0,
                                                })
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                                # Emit per-request RX stats to JSONL
                                if getattr(self, "_stats_file", None):
                                    try:
                                        _append_jsonl(self._stats_file, {  # type: ignore[arg-type]
                                            "type": "transfer_rx_stats",
                                            "from_stage": stage_id - 1,
                                            "to_stage": stage_id,
                                            "request_id": req_id,
                                            "rx_bytes": int(rx_b),
                                            "rx_decode_time_ms": rx_ms,
                                            "in_flight_time_ms": in_flight_ms,
                                            "rx_time_per_kb_ms": rx_ms / max(rx_b / 1024.0, 1e-6) if rx_b > 0 else 0.0,
                                        })
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                except Exception:
                    pass
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
                        if stage_id == final_stage_id_for_e2e and req_id not in e2e_done:
                            _t0 = _req_start_ts.get(req_id)
                            if _t0 is not None:
                                _t1 = time.time()
                                _last_finish_ts = max(_last_finish_ts, _t1)
                                _e2e_ms = (_t1 - _t0) * 1000.0
                                # Count tokens from final stage outputs
                                def _count_tokens(_ros: List[Any]) -> int:  # type: ignore[name-defined]
                                    total = 0
                                    for _ro in _ros:
                                        try:
                                            outs = getattr(_ro, "outputs", None)
                                            if outs and len(outs) > 0:
                                                tokens = getattr(outs[0], "token_ids", None)
                                                if tokens is not None:
                                                    total += len(tokens)
                                        except Exception:
                                            pass
                                    return total
                                _num_tokens = _count_tokens(engine_outputs)
                                _time_per_token_ms = (_e2e_ms / _num_tokens) if _num_tokens > 0 else 0.0
                                # Update E2E aggregators
                                e2e_total_ms += _e2e_ms
                                e2e_total_tokens += int(_num_tokens)
                                e2e_count += 1
                                e2e_done.add(req_id)
                                # Write per-request overall record
                                try:
                                    rid_int = int(req_id)
                                    pr = per_request.setdefault(rid_int, {"stages": {}, "transfers_ms": 0.0, "transfers_bytes": 0})
                                    per_req_record = {
                                        "type": "overall_request",
                                        "request_id": rid_int,
                                        "e2e_time_ms": _e2e_ms,
                                        "num_tokens_out": int(_num_tokens),
                                        "transfers_total_time_ms": float(pr.get("transfers_ms", 0.0)),
                                        "transfers_total_bytes": int(pr.get("transfers_bytes", 0)),
                                        "stages": pr.get("stages", {}),
                                    }
                                    sum_per_request_transfer_ms += float(pr.get("transfers_ms", 0.0))
                                    if getattr(self, "_overall_stats_file", None):
                                        try:
                                            _append_jsonl(self._overall_stats_file, per_req_record)  # type: ignore[arg-type]
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                if getattr(self, "_stats_file", None):
                                    try:
                                        _append_jsonl(self._stats_file, {  # type: ignore[arg-type]
                                            "type": "orchestrator_request_e2e",
                                            "request_id": req_id,
                                            "final_stage_id": stage_id,
                                            "e2e_time_ms": _e2e_ms,
                                            "num_tokens_out": int(_num_tokens),
                                            "e2e_time_per_token_ms": _time_per_token_ms,
                                        })
                                    except Exception:
                                        pass
                    except Exception:
                        pass

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
                        if self._enable_stats and getattr(self, "_stats_file", None):
                            try:
                                _append_jsonl(self._stats_file, {
                                    "type": "transfer_stats",
                                    "from_stage": stage_id,
                                    "to_stage": next_stage_id,
                                    "request_id": req_id,
                                    "size_bytes": int(size_bytes),
                                    "tx_time_ms": tx_ms,
                                    "tx_mbps": (float(size_bytes) * 8.0) / (max(tx_ms, 1e-6) * 1000.0),
                                    "used_shm": bool("engine_inputs_shm" in ipc_payload),
                                })
                            except Exception:
                                pass
                        # Update in-memory transfer aggregator
                        try:
                            key = (stage_id, next_stage_id)
                            agg = transfer_agg.get(key)
                            if agg is None:
                                agg = {"sum_bytes": 0.0, "sum_ms": 0.0, "count": 0.0,
                                       "sum_rx_bytes": 0.0, "sum_rx_ms": 0.0, "rx_count": 0.0,
                                       "sum_total_ms": 0.0, "total_count": 0.0}
                                transfer_agg[key] = agg
                            agg["sum_bytes"] += float(size_bytes)
                            agg["sum_ms"] += float(tx_ms)
                            agg["count"] += 1.0
                            # Store sender-side timing for per-request combination
                            transfer_edge_req[(stage_id, next_stage_id, int(req_id))] = {
                                "tx_ms": float(tx_ms),
                                "size_bytes": float(size_bytes),
                            }
                        except Exception:
                            pass
                    except Exception:
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
            stage_summary: List[Dict[str, Any]] = []
            for sid in range(num_stages):
                reqs = stage_req_counts[sid]
                tokens = stage_total_tokens[sid]
                total_ms = float(stage_total_time_ms[sid])
                avg_req = (total_ms / reqs) if reqs > 0 else 0.0
                avg_tok = (tokens * 1000.0 / total_ms) if total_ms > 0 else 0.0
                stage_summary.append({
                    "stage_id": sid,
                    "requests": int(reqs),
                    "tokens": int(tokens),
                    "total_time_ms": total_ms,
                    "avg_time_per_request_ms": avg_req,
                    "avg_tokens_per_s": avg_tok,
                })

            transfer_summary: List[Dict[str, Any]] = []
            for (src, dst), agg in transfer_agg.items():
                sum_bytes = float(agg.get("sum_bytes", 0.0))
                sum_ms = float(agg.get("sum_ms", 0.0))
                samples = int(agg.get("count", 0.0))
                tx_mbps = (sum_bytes * 8.0) / (max(sum_ms, 1e-6) * 1000.0) if sum_bytes > 0 else 0.0
                sum_rx_bytes = float(agg.get("sum_rx_bytes", 0.0))
                sum_rx_ms = float(agg.get("sum_rx_ms", 0.0))
                samples_rx = int(agg.get("rx_count", 0.0))
                rx_mbps = (sum_rx_bytes * 8.0) / (max(sum_rx_ms, 1e-6) * 1000.0) if sum_rx_bytes > 0 else 0.0
                sum_total_ms = float(agg.get("sum_total_ms", 0.0))
                samples_total = int(agg.get("total_count", 0.0))
                total_mbps = (sum_bytes * 8.0) / (max(sum_total_ms, 1e-6) * 1000.0) if sum_bytes > 0 else 0.0
                transfer_summary.append({
                    "from_stage": src,
                    "to_stage": dst,
                    "samples": samples,
                    "total_bytes": int(sum_bytes),
                    "total_time_ms": sum_ms,
                    "tx_mbps": tx_mbps,
                    "rx_samples": samples_rx,
                    "rx_total_bytes": int(sum_rx_bytes),
                    "rx_total_time_ms": sum_rx_ms,
                    "rx_mbps": rx_mbps,
                    "total_samples": samples_total,
                    "total_transfer_time_ms": sum_total_ms,
                    "total_mbps": total_mbps,
                })

            e2e_avg_req = (e2e_total_ms / e2e_count) if e2e_count > 0 else 0.0
            e2e_avg_tok = (e2e_total_tokens * 1000.0 / e2e_total_ms) if e2e_total_ms > 0 else 0.0
            wall_time_ms = max(0.0, (_last_finish_ts - _wall_start_ts) * 1000.0)
            summary: Dict[str, Any] = {
                "e2e_requests": int(e2e_count),
                # 按你的需求：e2e_total_time_ms 直接使用墙钟时间（非各请求之和）
                "e2e_total_time_ms": float(wall_time_ms),
                # 额外保留各请求 E2E 之和，供需要时参考
                "e2e_sum_time_ms": float(e2e_total_ms),
                "e2e_total_tokens": int(e2e_total_tokens),
                "e2e_avg_time_per_request_ms": e2e_avg_req,
                "e2e_avg_tokens_per_s": e2e_avg_tok,
                "wall_time_ms": wall_time_ms,
                "final_stage_id": final_stage_id_for_e2e,
                "stages": stage_summary,
                "transfers": transfer_summary,
            }
            logger.info("[Summary] %s", summary)
            if self._enable_stats and getattr(self, "_stats_file", None):
                try:
                    _append_jsonl(self._stats_file, {"type": "orchestrator_summary", **summary})  # type: ignore[arg-type]
                except Exception:
                    pass
            if self._enable_stats and getattr(self, "_overall_stats_file", None):
                try:
                    _append_jsonl(self._overall_stats_file, {"type": "overall_summary", **summary})  # type: ignore[arg-type]
                except Exception:
                    pass
        except Exception:
            pass

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



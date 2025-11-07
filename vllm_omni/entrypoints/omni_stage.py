"""
Stage manager for orchestrating multiple engines in vLLM-omni.

Enhanced to encapsulate per-stage process lifecycle and worker logic
(device setup, LLM init, batching, shared-memory IPC), while preserving
the original input processing utilities for cross-stage data wiring.
"""

import importlib
import logging
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Union

from vllm_omni.entrypoints.pipeline_utils import (
    _to_dict,
    set_stage_gpu_devices,
    maybe_load_from_ipc,
    maybe_dump_to_shm,
)

from vllm.inputs import TextPrompt
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm_omni.inputs.data import OmniTokensPrompt


class OmniStage:
    def __init__(self, stage_config):
        self.stage_config = stage_config
        self.engine = None
        self.async_engine = None
        self.stage_id = stage_config.stage_id
        self.engine_args = stage_config.engine_args
        self.model_stage = stage_config.engine_args.model_stage
        if hasattr(stage_config, "engine_input_source"):
            self.engine_input_source = stage_config.engine_input_source
        else:
            self.engine_input_source = []
        self.engine_output_type = stage_config.engine_args.engine_output_type
        self.engine_outputs = None
        if hasattr(stage_config, "custom_process_input_func"):
            # Import the module specified in the config (already a full module path)
            module_path, func_name = stage_config.custom_process_input_func.rsplit(
                ".", 1
            )
            module = importlib.import_module(module_path)
            self.custom_process_input_func = getattr(module, func_name)
        else:
            self.custom_process_input_func = None

        if hasattr(stage_config, "final_output"):
            self.final_output = stage_config.final_output
        else:
            self.final_output = False

        if hasattr(stage_config, "final_output_type"):
            self.final_output_type = stage_config.final_output_type
        else:
            self.final_output_type = None

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
        mrs_cfg = _to_dict(getattr(self.stage_config, "mrs", {}))
        stage_payload: Dict[str, Any] = {
            "stage_id": self.stage_id,
            "engine_args": engine_args,
            "mrs": mrs_cfg,
            "shm_threshold_bytes": self._shm_threshold_bytes,
        }
        self._proc = ctx.Process(
            target=_stage_worker,
            args=(model, stage_payload, self._in_q, self._out_q, self._log_file, batch_timeout),
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

    def submit(self, payload: Dict[str, Any]) -> None:
        assert self._in_q is not None
        self._in_q.put(payload)

    def try_collect(self) -> Optional[Dict[str, Any]]:
        assert self._out_q is not None
        try:
            return self._out_q.get_nowait()
        except Exception:
            return None

    def process_engine_inputs(
        self, stage_list, prompt: Union[OmniTokensPrompt, TextPrompt] = None
    ) -> List[Union[OmniTokensPrompt, TextPrompt]]:
        """Process the engine input for the stage."""
        if self.custom_process_input_func is None:
            engine_inputs = []
            if len(self.engine_input_source) == 0:
                raise ValueError("engine_input_source is empty")
            source_stage_id = self.engine_input_source[0]
            source_outputs = stage_list[source_stage_id].engine_outputs
            multi_modal_data = {
                source_output.request_id: prompt.get("multi_modal_data", None)
                for source_output, prompt in zip(source_outputs, prompt)
            }

            for source_output in source_outputs:
                engine_input = OmniTokensPrompt(
                    prompt_token_ids=source_output.outputs[0].token_ids,
                    multi_modal_data=(
                        multi_modal_data[source_output.request_id]
                        if multi_modal_data
                        else None
                    ),
                )
                engine_inputs.append(engine_input)
            return engine_inputs

        else:
            engine_input_source = self.engine_input_source
            return self.custom_process_input_func(
                stage_list, engine_input_source, prompt
            )


def _stage_worker(
    model: str,
    stage_payload: Dict[str, Any],
    in_q: mp.Queue,
    out_q: mp.Queue,
    log_file: Optional[str] = None,
    batch_timeout: int = 10,
) -> None:
    """Stage worker entry: device setup, LLM init, batching, SHM IPC."""
    import logging as _logging
    from vllm_omni.entrypoints.omni_llm import OmniStageLLM  # noqa: WPS433
    from vllm_omni.entrypoints.log_utils import (  # noqa: WPS433
        log_stage_request_stats,
        log_stage_running_avg,
        log_stage_batch_stats,
    )
    import queue as _queue
    import time as _time
    from vllm_omni.entrypoints.pipeline_utils import append_jsonl as _append_jsonl, serialize_obj as _ser

    stage_id = stage_payload["stage_id"]
    engine_args = stage_payload.get("engine_args", {})
    mrs_cfg = stage_payload.get("mrs", {})
    shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))

    # Per-stage file logger (optional)
    try:
        if log_file:
            stage_log = _logging.getLogger(__name__)
            stage_log.setLevel(_logging.DEBUG)
            fh = _logging.FileHandler(f"{log_file}.stage{stage_id}.log")
            fh.setLevel(_logging.DEBUG)
            fh.setFormatter(_logging.Formatter("%(asctime)s [PID:%(process)d] [Stage-%(stage)s] %(levelname)s: %(message)s"))
            class _StageFilter(_logging.Filter):
                def filter(self, record: _logging.LogRecord) -> bool:
                    setattr(record, "stage", stage_id)
                    return True
            fh.addFilter(_StageFilter())
            stage_log.addHandler(fh)
    except Exception:
        pass

    # Stage stats JSONL file
    _stats_file = f"{log_file}.stage{stage_id}.stats.jsonl" if log_file else None

    # Aggregates for running average
    _agg_total_tokens = 0
    _agg_total_gen_time_ms = 0.0
    # Monotonic batch id per stage process for orchestrator dedup on time aggregation
    _batch_seq = 0

    # Device mapping
    try:
        set_stage_gpu_devices(stage_id, mrs_cfg.get("devices"))
    except Exception as e:
        _logging.getLogger(__name__).warning("[Stage-%s] Device setup failed: %s", stage_id, e)

    # Init LLM
    _logging.getLogger(__name__).debug("[Stage-%s] Initializing engine with args keys=%s", stage_id, list(engine_args.keys()))
    stage_engine = OmniStageLLM(model=model, **engine_args)
    _logging.getLogger(__name__).debug("[Stage-%s] Engine initialized", stage_id)
    # Signal readiness to orchestrator
    try:
        out_q.put({"type": "stage_ready", "stage_id": stage_id})
    except Exception:
        pass

    # Batch processing loop
    while True:
        task = in_q.get()
        _recv_dequeue_ts = _time.time()
        if task is None:
            _logging.getLogger(__name__).debug("[Stage-%s] Received shutdown signal", stage_id)
            break

        max_batch_size = int(mrs_cfg.get("max_batch_size", 1) or 1)
        batch_tasks: List[Dict[str, Any]] = [task]
        if max_batch_size > 1:
            while len(batch_tasks) < max_batch_size:
                if not in_q.empty():
                    extra = in_q.get(timeout=batch_timeout)
                    if extra is None:
                        in_q.put(None)
                        break
                    batch_tasks.append(extra)
                else:
                    break

        batch_request_ids: List[Any] = []
        batch_engine_inputs: List[Any] = []
        _rx_bytes_by_rid: Dict[Any, int] = {}
        _rx_decode_ms_by_rid: Dict[Any, float] = {}
        _in_flight_ms_by_rid: Dict[Any, float] = {}
        for t in batch_tasks:
            rid = t["request_id"]
            try:
                sent_ts = float(t.get("sent_ts", None)) if isinstance(t, dict) else None
                if sent_ts is not None:
                    _in_flight_ms_by_rid[rid] = (_recv_dequeue_ts - sent_ts) * 1000.0
                else:
                    _in_flight_ms_by_rid[rid] = 0.0
            except Exception:
                _in_flight_ms_by_rid[rid] = 0.0
            _t0 = _time.time()
            ein = maybe_load_from_ipc(t, obj_key="engine_inputs", shm_key="engine_inputs_shm")
            _t1 = _time.time()
            _rx_decode_ms_by_rid[rid] = (_t1 - _t0) * 1000.0
            try:
                if "engine_inputs_shm" in t:
                    _rx_bytes_by_rid[rid] = int(t["engine_inputs_shm"]["size"])  # type: ignore[index]
                else:
                    _rx_bytes_by_rid[rid] = len(_ser(ein))
            except Exception:
                _rx_bytes_by_rid[rid] = 0
            batch_request_ids.append(rid)
            if isinstance(ein, list):
                batch_engine_inputs.extend(ein)
            elif isinstance(ein, dict):
                batch_engine_inputs.append(ein)
            else:
                _logging.getLogger(__name__).exception("[Stage-%s] Invalid engine input type: %s", stage_id, type(ein))
        sampling_params = batch_tasks[0]["sampling_params"]
        _logging.getLogger(__name__).debug("[Stage-%s] Received batch size=%d, request_ids=%s", stage_id, len(batch_tasks), batch_request_ids)
        print("--------------------------------", flush=True)
        print(f"[Stage-{stage_id}] Received batch size={len(batch_tasks)}, request_ids={batch_request_ids}", flush=True)
        print("--------------------------------", flush=True)
        try:
            _batch_seq += 1
            gen_outputs: List[Any] = []
            _gen_t0 = _time.time()
            for ro in stage_engine.generate(batch_engine_inputs, sampling_params, use_tqdm=False):
                gen_outputs.append(ro)
            _gen_t1 = _time.time()
            _gen_ms = (_gen_t1 - _gen_t0) * 1000.0

            # Group outputs per request id with fallback
            req_to_outputs: Dict[Any, List[Any]] = {rid: [] for rid in batch_request_ids}
            unmapped: List[Any] = []
            for ro in gen_outputs:
                rid = getattr(ro, "request_id", None)
                if rid in req_to_outputs:
                    req_to_outputs[rid].append(ro)
                else:
                    unmapped.append(ro)
            if unmapped:
                idx = 0
                for ro in unmapped:
                    target_rid = batch_request_ids[idx % len(batch_request_ids)]
                    req_to_outputs[target_rid].append(ro)
                    idx += 1

            # Per-request stats logging and aggregates
            def _count_tokens(_ros: List[Any]) -> int:
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

            for rid in batch_request_ids:
                _r_outputs = req_to_outputs.get(rid, [])
                _num_tokens = _count_tokens(_r_outputs)
                _agg_total_tokens += _num_tokens
                _agg_total_gen_time_ms += _gen_ms
                _tokens_per_s = (_num_tokens * 1000.0 / _gen_ms) if _gen_ms > 0 else 0.0
                if _stats_file:
                    try:
                        _append_jsonl(_stats_file, {
                            "type": "stage_request_stats",
                            "stage_id": stage_id,
                            "request_id": rid,
                            "batch_size": len(batch_tasks),
                            "num_tokens_out": int(_num_tokens),
                            "stage_gen_time_ms": _gen_ms,
                            "tokens_per_s": _tokens_per_s,
                            "rx_transfer_bytes": int(_rx_bytes_by_rid.get(rid, 0)),
                            "rx_decode_time_ms": float(_rx_decode_ms_by_rid.get(rid, 0.0)),
                            "rx_mbps": (float(_rx_bytes_by_rid.get(rid, 0)) * 8.0) / (max(float(_rx_decode_ms_by_rid.get(rid, 0.0)), 1e-6) * 1000.0),
                        })
                    except Exception:
                        pass

            if _stats_file:
                _avg_tokens_per_s = (_agg_total_tokens * 1000.0 / _agg_total_gen_time_ms) if _agg_total_gen_time_ms > 0 else 0.0
                log_stage_running_avg(_stats_file, stage_id, int(_agg_total_tokens), float(_agg_total_gen_time_ms), float(_avg_tokens_per_s))
                log_stage_batch_stats(_stats_file, stage_id, len(batch_tasks), float(_gen_ms), list(batch_request_ids))

            # Emit per-request results
            for rid in batch_request_ids:
                r_outputs = req_to_outputs.get(rid, [])
                try:
                    use_shm, payload = maybe_dump_to_shm(r_outputs, shm_threshold_bytes)
                    _metrics = {
                        "num_tokens_out": int(_count_tokens(r_outputs)),
                        "stage_gen_time_ms": _gen_ms,
                        "batch_id": int(_batch_seq),
                        "rx_decode_time_ms": float(_rx_decode_ms_by_rid.get(rid, 0.0)),
                        "rx_transfer_bytes": int(_rx_bytes_by_rid.get(rid, 0)),
                        "rx_in_flight_time_ms": float(_in_flight_ms_by_rid.get(rid, 0.0)),
                    }
                    if _stats_file:
                        _num_tokens = int(_metrics["num_tokens_out"])  # type: ignore[index]
                        _tokens_per_s = (_num_tokens * 1000.0 / _gen_ms) if _gen_ms > 0 else 0.0
                        _rx_b = int(_metrics["rx_transfer_bytes"])  # type: ignore[index]
                        _rx_ms = float(_metrics["rx_decode_time_ms"])  # type: ignore[index]
                        _rx_mbps = (float(_rx_b) * 8.0) / (max(float(_rx_ms), 1e-6) * 1000.0) if _rx_b > 0 else 0.0
                        log_stage_request_stats(
                            _stats_file,
                            stage_id,
                            rid,
                            len(batch_tasks),
                            _num_tokens,
                            float(_gen_ms),
                            float(_tokens_per_s),
                            _rx_b,
                            _rx_ms,
                            float(_rx_mbps),
                        )
                    if use_shm:
                        out_q.put({
                            "request_id": rid,
                            "stage_id": stage_id,
                            "engine_outputs_shm": payload,
                            "metrics": _metrics,
                        })
                    else:
                        out_q.put({
                            "request_id": rid,
                            "stage_id": stage_id,
                            "engine_outputs": payload,
                            "metrics": _metrics,
                        })
                except Exception:
                    out_q.put({
                        "request_id": rid,
                        "stage_id": stage_id,
                        "engine_outputs": r_outputs,
                        "metrics": {
                            "num_tokens_out": int(_count_tokens(r_outputs)),
                            "stage_gen_time_ms": _gen_ms,
                            "rx_decode_time_ms": float(_rx_decode_ms_by_rid.get(rid, 0.0)),
                            "rx_transfer_bytes": int(_rx_bytes_by_rid.get(rid, 0)),
                            "rx_in_flight_time_ms": float(_in_flight_ms_by_rid.get(rid, 0.0)),
                        },
                    })
                _logging.getLogger(__name__).debug("[Stage-%s] Enqueued result for request %s to downstream", stage_id, rid)
        except Exception as e:
            _logging.getLogger(__name__).exception("[Stage-%s] Failed on batch %s: %s", stage_id, batch_request_ids, e)
            for rid in batch_request_ids:
                out_q.put({
                    "request_id": rid,
                    "stage_id": stage_id,
                    "error": str(e),
                })

from vllm_omni.entrypoints.stage_utils import (
    _to_dict,
    maybe_dump_to_shm,
    maybe_load_from_ipc_with_metrics,
    set_stage_gpu_devices,
)


import asyncio
import logging
import multiprocessing as mp
from typing import Any, Optional, TYPE_CHECKING
from vllm.usage.usage_lib import UsageContext


from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.utils.diffusers_utils import is_diffusion_model

import logging as _logging
import time as _time
import enum

from vllm_omni.model_executor.models.qwen_image.text_encoder import QwenImageTextEncoder

from vllm_omni.entrypoints.log_utils import (  # noqa: WPS433
    compute_and_log_stage_request_stats,
    count_tokens_from_outputs,
    log_stage_batch_stats,
    log_stage_running_avg,
)
# Delay imports that would create circular dependencies. The heavy engines
# (OmniStageLLM / OmniStageDiffusion / transformers models) are imported
# lazily inside the worker functions so that module import time doesn't
# trigger a circular import with `omni_stage` / `omni_llm`.

if TYPE_CHECKING:
    from vllm_omni.entrypoints.omni_stage import OmniStage


class StageWorkerEngine(enum.Enum):
    VLLM = "vllm"
    TRANSFORMERS = "transformers"# temp for dev
    DIFFUSION = "diffusion"


    

# TODO: support multi engine
def _stage_worker(
    model: str,
    stage_payload: dict[str, Any],
    in_q: mp.Queue,
    out_q: mp.Queue,
    log_file: str | None = None,
    batch_timeout: int = 10,
    worker_engine: StageWorkerEngine = StageWorkerEngine.VLLM,
) -> None:
    """Stage worker entry: device setup, LLM init, batching, SHM IPC."""

    # no inline JSONL/serialization imports; logging handled by utilities

    stage_id = stage_payload["stage_id"]
    engine_args = stage_payload.get("engine_args", {})
    runtime_cfg = stage_payload.get("runtime", {})
    shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))

    # Per-stage file logger (optional)
    try:
        if log_file:
            stage_log = _logging.getLogger(__name__)
            stage_log.setLevel(_logging.DEBUG)
            fh = _logging.FileHandler(f"{log_file}.stage{stage_id}.log")
            fh.setLevel(_logging.DEBUG)
            fh.setFormatter(
                _logging.Formatter(
                    "%(asctime)s [PID:%(process)d] [Stage-%(stage)s] %(levelname)s: %(message)s"
                )
            )

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
        set_stage_gpu_devices(stage_id, runtime_cfg.get("devices"))
    except Exception as e:
        _logging.getLogger(__name__).warning(
            "[Stage-%s] Device setup failed: %s", stage_id, e
        )

    # Init LLM
    _logging.getLogger(__name__).debug(
        "[Stage-%s] Initializing engine with args keys=%s",
        stage_id,
        list(engine_args.keys()),
    )

    if is_diffusion_model(model):
        worker_engine = StageWorkerEngine.DIFFUSION
    if engine_args.get("model_stage") == "text_encoder":
        worker_engine = StageWorkerEngine.TRANSFORMERS

    print(model, worker_engine)
    print(stage_payload)

    if worker_engine == StageWorkerEngine.VLLM:
        # Lazy import to avoid circular import at module import time
        from vllm_omni.entrypoints.omni_llm import OmniStageLLM  # noqa: WPS433

        stage_engine = OmniStageLLM(model=model, **engine_args)
    elif worker_engine == StageWorkerEngine.TRANSFORMERS:
        #TODO replace this with text_encoder stage
        stage_engine = QwenImageTextEncoder()
    elif worker_engine == StageWorkerEngine.DIFFUSION:
        # Lazy import of diffusion stage implementation
        from vllm_omni.entrypoints.omni_diffusion import OmniStageDiffusion  # noqa: WPS433

        stage_engine = OmniStageDiffusion(model=model)
    else:
        raise ValueError(f"Unsupported worker engine: {worker_engine}")
    _logging.getLogger(__name__).debug("[Stage-%s] Engine initialized", stage_id)
    # Signal readiness to orchestrator
    try:
        out_q.put({"type": "stage_ready", "stage_id": stage_id})
    except Exception:
        pass

    # Batch processing loop
    while True:
        task = in_q.get()
        print("received task", task)
        _recv_dequeue_ts = _time.time()
        if task is None:
            _logging.getLogger(__name__).debug(
                "[Stage-%s] Received shutdown signal", stage_id
            )
            break

        max_batch_size = int(runtime_cfg.get("max_batch_size", 1) or 1)
        batch_tasks: list[dict[str, Any]] = [task]
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

        batch_request_ids: list[Any] = []
        batch_engine_inputs: list[Any] = []
        _rx_bytes_by_rid: dict[Any, int] = {}
        _rx_decode_ms_by_rid: dict[Any, float] = {}
        _in_flight_ms_by_rid: dict[Any, float] = {}
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
            ein, _rx_metrics = maybe_load_from_ipc_with_metrics(
                t, obj_key="engine_inputs", shm_key="engine_inputs_shm"
            )
            _rx_decode_ms_by_rid[rid] = float(_rx_metrics.get("rx_decode_time_ms", 0.0))
            _rx_bytes_by_rid[rid] = int(_rx_metrics.get("rx_transfer_bytes", 0))
            batch_request_ids.append(rid)
            print("ein: ", ein)
            if isinstance(ein, list):
                batch_engine_inputs.extend(ein)
            elif isinstance(ein, dict):
                batch_engine_inputs.append(ein)
            elif isinstance(ein, str):
                batch_engine_inputs.append(ein)
            else:
                _logging.getLogger(__name__).exception(
                    "[Stage-%s] Invalid engine input type: %s", stage_id, type(ein)
                )

        _logging.getLogger(__name__).debug(
            "[Stage-%s] Received batch size=%d, request_ids=%s",
            stage_id,
            len(batch_tasks),
            batch_request_ids,
        )
        print("--------------------------------", flush=True)
        print(
            f"[Stage-{stage_id}] Received batch size={len(batch_tasks)}, request_ids={batch_request_ids}",
            flush=True,
        )
        print("--------------------------------", flush=True)
        try:
            _batch_seq += 1
            gen_outputs: list[Any] = []
            _gen_t0 = _time.time()
            print("batch_engine_inputs", batch_engine_inputs)
            if not is_diffusion_model(model):
                sampling_params = batch_tasks[0]["sampling_params"]
                for ro in stage_engine.generate(
                    batch_engine_inputs, sampling_params, use_tqdm=False
                ):
                    gen_outputs.append(ro)
            else:
                print(f"inputs type: {type(batch_engine_inputs)}")
                for ro in stage_engine.generate(batch_engine_inputs):
                    gen_outputs.append(ro)
            _gen_t1 = _time.time()
            _gen_ms = (_gen_t1 - _gen_t0) * 1000.0

            # Group outputs per request id with fallback
            req_to_outputs: dict[Any, list[Any]] = {
                rid: [] for rid in batch_request_ids
            }
            unmapped: list[Any] = []
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
            for rid in batch_request_ids:
                _r_outputs = req_to_outputs.get(rid, [])
                _num_tokens = count_tokens_from_outputs(_r_outputs)
                _agg_total_tokens += _num_tokens
                _agg_total_gen_time_ms += _gen_ms

            if _stats_file:
                _avg_tokens_per_s = (
                    (_agg_total_tokens * 1000.0 / _agg_total_gen_time_ms)
                    if _agg_total_gen_time_ms > 0
                    else 0.0
                )
                log_stage_running_avg(
                    _stats_file,
                    stage_id,
                    int(_agg_total_tokens),
                    float(_agg_total_gen_time_ms),
                    float(_avg_tokens_per_s),
                )
                log_stage_batch_stats(
                    _stats_file,
                    stage_id,
                    len(batch_tasks),
                    float(_gen_ms),
                    list(batch_request_ids),
                )

            # Emit per-request results
            for rid in batch_request_ids:
                r_outputs = req_to_outputs.get(rid, [])
                try:
                    use_shm, payload = maybe_dump_to_shm(r_outputs, shm_threshold_bytes)
                    _metrics = {
                        "num_tokens_out": int(count_tokens_from_outputs(r_outputs)),
                        "stage_gen_time_ms": _gen_ms,
                        "batch_id": int(_batch_seq),
                        "rx_decode_time_ms": float(_rx_decode_ms_by_rid.get(rid, 0.0)),
                        "rx_transfer_bytes": int(_rx_bytes_by_rid.get(rid, 0)),
                        "rx_in_flight_time_ms": float(
                            _in_flight_ms_by_rid.get(rid, 0.0)
                        ),
                    }
                    if _stats_file:
                        compute_and_log_stage_request_stats(
                            _stats_file,
                            stage_id,
                            rid,
                            len(batch_tasks),
                            r_outputs,
                            float(_gen_ms),
                            int(_metrics["rx_transfer_bytes"]),  # type: ignore[index]
                            float(_metrics["rx_decode_time_ms"]),  # type: ignore[index]
                        )
                    if use_shm:
                        out_q.put(
                            {
                                "request_id": rid,
                                "stage_id": stage_id,
                                "engine_outputs_shm": payload,
                                "metrics": _metrics,
                            }
                        )
                    else:
                        out_q.put(
                            {
                                "request_id": rid,
                                "stage_id": stage_id,
                                "engine_outputs": payload,
                                "metrics": _metrics,
                            }
                        )
                except Exception:
                    out_q.put(
                        {
                            "request_id": rid,
                            "stage_id": stage_id,
                            "engine_outputs": r_outputs,
                            "metrics": {
                                "num_tokens_out": int(
                                    count_tokens_from_outputs(r_outputs)
                                ),
                                "stage_gen_time_ms": _gen_ms,
                                "rx_decode_time_ms": float(
                                    _rx_decode_ms_by_rid.get(rid, 0.0)
                                ),
                                "rx_transfer_bytes": int(_rx_bytes_by_rid.get(rid, 0)),
                                "rx_in_flight_time_ms": float(
                                    _in_flight_ms_by_rid.get(rid, 0.0)
                                ),
                            },
                        }
                    )
                _logging.getLogger(__name__).debug(
                    "[Stage-%s] Enqueued result for request %s to downstream",
                    stage_id,
                    rid,
                )
        except Exception as e:
            _logging.getLogger(__name__).exception(
                "[Stage-%s] Failed on batch %s: %s", stage_id, batch_request_ids, e
            )
            for rid in batch_request_ids:
                out_q.put(
                    {
                        "request_id": rid,
                        "stage_id": stage_id,
                        "error": str(e),
                    }
                )


def _stage_worker_async_entry(
    omni_stage: "OmniStage",
    model: str,
    stage_payload: dict[str, Any],
    in_q: mp.Queue,
    out_q: mp.Queue,
    log_file: Optional[str] = None,
    batch_timeout: int = 10,
) -> None:
    asyncio.run(
        _stage_worker_async(
            omni_stage, model, stage_payload, in_q, out_q, log_file, batch_timeout
        )
    )


async def _stage_worker_async(
    omni_stage: "OmniStage",
    model: str,
    stage_payload: dict[str, Any],
    in_q: mp.Queue,
    out_q: mp.Queue,
    log_file: str | None = None,
    batch_timeout: int = 10,
) -> None:
    """Stage worker entry: device setup, LLM init, batching, SHM IPC."""
    import logging as _logging
    import time as _time

    from vllm_omni.entrypoints.async_omni_llm import AsyncOmniStageLLM  # noqa: WPS433
    from vllm_omni.entrypoints.log_utils import (  # noqa: WPS433
        compute_and_log_stage_request_stats,
        count_tokens_from_outputs,
        log_stage_batch_stats,
        log_stage_running_avg,
    )

    # no inline JSONL/serialization imports; logging handled by utilities

    stage_id = stage_payload["stage_id"]
    engine_args = stage_payload.get("engine_args", {})
    runtime_cfg = stage_payload.get("runtime", {})
    shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))

    # Per-stage file logger (optional)
    try:
        if log_file:
            stage_log = _logging.getLogger(__name__)
            stage_log.setLevel(_logging.DEBUG)
            fh = _logging.FileHandler(f"{log_file}.stage{stage_id}.log")
            fh.setLevel(_logging.DEBUG)
            fh.setFormatter(
                _logging.Formatter(
                    "%(asctime)s [PID:%(process)d] [Stage-%(stage)s] %(levelname)s: %(message)s"
                )
            )  # noqa: E501

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
    # Monotonic batch id per stage process for orchestrator dedup on time
    # aggregation
    _batch_seq = 0

    # Device mapping
    try:
        set_stage_gpu_devices(stage_id, runtime_cfg.get("devices"))
    except Exception as e:
        _logging.getLogger(__name__).warning(
            "[Stage-%s] Device setup failed: %s", stage_id, e
        )

    # Init LLM
    _logging.getLogger(__name__).debug(
        "[Stage-%s] Initializing engine with args keys=%s",
        stage_id,
        list(engine_args.keys()),
    )
    omni_engine_args = AsyncOmniEngineArgs(model=model, **engine_args)
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = omni_engine_args.create_engine_config(usage_context=usage_context)
    stage_engine = AsyncOmniStageLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=usage_context,
        engine_args=omni_engine_args,
    )
    omni_stage.set_async_engine(stage_engine)
    # Don't keep the dummy data in memory
    await stage_engine.reset_mm_cache()
    _logging.getLogger(__name__).debug("[Stage-%s] Engine initialized", stage_id)
    # Signal readiness to orchestrator and send vllm_config back to main process
    try:
        # Send vllm_config back to main process so it can be accessed via
        # get_vllm_config(). This is needed because async_engine is only available
        # in the worker process

        # input_preprocessor = await stage_engine.get_input_preprocessor()
        out_q.put(
            {
                "type": "stage_ready",
                "stage_id": stage_id,
                "vllm_config": vllm_config,
                "tokenizer": getattr(stage_engine, "tokenizer", None),
                "is_tracing_enabled": await stage_engine.is_tracing_enabled(),
                # "input_preprocessor": input_preprocessor,
            }
        )
    except Exception as e:
        _logging.getLogger(__name__).warning(
            "[Stage-%s] Failed to send stage ready signal: %s", stage_id, e
        )

    # Batch processing loop
    while True:
        task = in_q.get()
        _recv_dequeue_ts = _time.time()
        if task is None:
            _logging.getLogger(__name__).debug(
                "[Stage-%s] Received shutdown signal", stage_id
            )
            break

        _rx_bytes_by_rid: dict[Any, int] = {}
        _rx_decode_ms_by_rid: dict[Any, float] = {}
        _in_flight_ms_by_rid: dict[Any, float] = {}

        rid = task["request_id"]
        try:
            sent_ts = (
                float(task.get("sent_ts", None)) if isinstance(task, dict) else None
            )
            if sent_ts is not None:
                _in_flight_ms_by_rid[rid] = (_recv_dequeue_ts - sent_ts) * 1000.0
            else:
                _in_flight_ms_by_rid[rid] = 0.0
        except Exception:
            _in_flight_ms_by_rid[rid] = 0.0
        ein, _rx_metrics = maybe_load_from_ipc_with_metrics(
            task, obj_key="engine_inputs", shm_key="engine_inputs_shm"
        )
        _rx_decode_ms_by_rid[rid] = float(_rx_metrics.get("rx_decode_time_ms", 0.0))
        _rx_bytes_by_rid[rid] = int(_rx_metrics.get("rx_transfer_bytes", 0))

        sampling_params = task["sampling_params"]
        _logging.getLogger(__name__).debug(
            "[Stage-%s] Received batch size=1, request_ids=%d", stage_id, rid
        )
        print("--------------------------------", flush=True)
        print(
            f"[Stage-{stage_id}] Received batch size=1, request_ids={rid}", flush=True
        )
        print("--------------------------------", flush=True)
        try:
            _batch_seq += 1
            _gen_t0 = _time.time()
            if isinstance(ein, list):
                ein = ein[0]

            async for res in stage_engine.generate(ein, sampling_params, rid):
                gen_output = res
            _gen_t1 = _time.time()
            _gen_ms = (_gen_t1 - _gen_t0) * 1000.0

            r_outputs = [gen_output]
            _num_tokens = count_tokens_from_outputs(r_outputs)
            _agg_total_tokens += _num_tokens
            _agg_total_gen_time_ms += _gen_ms

            if _stats_file:
                _avg_tokens_per_s = (
                    (_agg_total_tokens * 1000.0 / _agg_total_gen_time_ms)
                    if _agg_total_gen_time_ms > 0
                    else 0.0
                )
                log_stage_running_avg(
                    _stats_file,
                    stage_id,
                    int(_agg_total_tokens),
                    float(_agg_total_gen_time_ms),
                    float(_avg_tokens_per_s),
                )
                log_stage_batch_stats(_stats_file, stage_id, 1, float(_gen_ms), [rid])

            try:
                use_shm, payload = maybe_dump_to_shm(r_outputs, shm_threshold_bytes)
                _metrics = {
                    "num_tokens_out": int(count_tokens_from_outputs(r_outputs)),
                    "stage_gen_time_ms": _gen_ms,
                    "batch_id": int(_batch_seq),
                    "rx_decode_time_ms": float(_rx_decode_ms_by_rid.get(rid, 0.0)),
                    "rx_transfer_bytes": int(_rx_bytes_by_rid.get(rid, 0)),
                    "rx_in_flight_time_ms": float(_in_flight_ms_by_rid.get(rid, 0.0)),
                }
                if _stats_file:
                    compute_and_log_stage_request_stats(
                        _stats_file,
                        stage_id,
                        rid,
                        1,
                        r_outputs,
                        float(_gen_ms),
                        int(_metrics["rx_transfer_bytes"]),  # type: ignore[index]
                        float(_metrics["rx_decode_time_ms"]),  # type: ignore[index]
                    )
                if use_shm:
                    out_q.put(
                        {
                            "request_id": rid,
                            "stage_id": stage_id,
                            "engine_outputs_shm": payload,
                            "metrics": _metrics,
                        }
                    )
                else:
                    out_q.put(
                        {
                            "request_id": rid,
                            "stage_id": stage_id,
                            "engine_outputs": payload,
                            "metrics": _metrics,
                        }
                    )
            except Exception as e:
                _logging.getLogger(__name__).exception(
                    "[Stage-%s] Failed to enqueue result for request %s: %s",
                    stage_id,
                    rid,
                    e,
                )
                out_q.put(
                    {
                        "request_id": rid,
                        "stage_id": stage_id,
                        "engine_outputs": r_outputs,
                        "metrics": {
                            "num_tokens_out": int(count_tokens_from_outputs(r_outputs)),
                            "stage_gen_time_ms": _gen_ms,
                            "rx_decode_time_ms": float(
                                _rx_decode_ms_by_rid.get(rid, 0.0)
                            ),
                            "rx_transfer_bytes": int(_rx_bytes_by_rid.get(rid, 0)),
                            "rx_in_flight_time_ms": float(
                                _in_flight_ms_by_rid.get(rid, 0.0)
                            ),
                        },
                    }
                )
            _logging.getLogger(__name__).debug(
                "[Stage-%s] Enqueued result for request %s to downstream", stage_id, rid
            )

        except Exception as e:
            _logging.getLogger(__name__).exception(
                "[Stage-%s] Failed on request %s: %s", stage_id, rid, e
            )
            out_q.put(
                {
                    "request_id": rid,
                    "stage_id": stage_id,
                    "error": str(e),
                }
            )

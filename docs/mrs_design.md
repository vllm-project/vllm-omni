## Multi-Request Streaming (MRS) on a Single Machine

### 1. Background & Scope
- All processing runs on a single physical machine with multi-process, per-stage workers. No proxy or network transport involved.
- Current alignment with vllm-omni: `OmniLLM` supports multiple stages (`OmniStage`). GPU runners already expose streamable steps (prefill/decoding/diffusion), but the entry layer still collects lists and lacks intra-stage streaming and window scheduling.
- Goal: implement multi-stage, multi-request streaming (MRS) locally. Each stage outputs segments; downstream stages stitch and trigger compute based on configured windows. Shared memory and zero-copy strategies reduce data movement overhead.

### 2. Key Constraints
- Multi-process per stage: each stage is an independent process with a while loop; device visibility can be configured (`CUDA_VISIBLE_DEVICES`/`torch.cuda.set_device`).
- Simple IPC (copy-based): use `multiprocessing.Queue`/Pipe for inter-process communication with CPU copies/serialization; do not rely on CUDA IPC/SHM zero-copy in this version.
- Cross-stage pipeline: different stages can process different requests concurrently (e.g., stage A handles request 1 while stage B handles request 0).

### 3. Architecture Overview
- Processes & IPC queues
  - Each "sub-stage" is an OS process (worker). The loop: take from input_queue → compute → put to output_queue.
  - Inter-stage connection via IPC: copy-based `multiprocessing.Queue` passing dict payloads; use shared memory for large objects.
  - Each link is SPSC (single-producer/single-consumer): the upstream is the orchestrator and the downstream is a single stage process; queues are unbounded (maxsize=0) on the orchestrator side.
- Device visibility
  - Each stage sets `CUDA_VISIBLE_DEVICES` or calls `torch.cuda.set_device` to bind to GPU sets.
  - A stage may use multiple GPUs internally (TP/PP/DP) but presents as a single stage unit.
- Simplified IPC: copy-based queues/pipes for data transfer; zero-copy is future work.
- Pipeline progression: when a stage finishes a request, it enqueues outputs to the downstream stage; if downstream is idle, it starts immediately.
- Scheduling
  - A downstream stage triggers only after the upstream completes the request.
  - Windowed segmentation/stitched triggering is not implemented; intra-stage streaming is not provided.

### 7. IPC Implementation (simplified: copy-based)
- Use `multiprocessing.Queue`/Pipe for inter-process communication (control + data).
- Data is serialized/copied via CPU; no CUDA IPC/SHM zero-copy in this version.
- Backpressure: queues are unbounded; pressure manifests as compute-rate differences. Optional SHM reduces large-object transfer cost; RX/decoding overhead is recorded for observability.

### 8. Scheduling & Cancellation (simplified)
- Pipeline: when a stage finishes a request, it enqueues to the next stage; that stage immediately pulls the next request from its input queue, enabling cross-stage concurrency.
- Cancellation/timeout: explicit cancellation/timeouts are not provided; graceful shutdown uses a `None` sentinel sent to each stage input queue.

#### Short sequence example (req0/req1, stage A→B)
1) t0: stage A handles req0
2) t1: req0 completes on A → enters B; A immediately starts req1
3) t2: B handles req0 while A handles req1 (parallel across stages)

### 9. Integration Points (by file)
- `vllm_omni/entrypoints/omni_llm.py` (Orchestrator)
  - Class `OmniLLM` orchestrates multi-process stages; constructs `OmniStage` instances in parallel and spawns per-stage workers.
  - Spawns stage processes per config (set `CUDA_VISIBLE_DEVICES`/`torch.cuda.set_device`), creates control/data channels, builds simple full-trigger flow.
  - Stats/logging are disabled by default; per-stage and orchestrator stats are only written when explicitly enabled.
  - Manages process lifecycle: start/wait for readiness, graceful shutdown; forwards results between stages using copy-based IPC and optional SHM.
  - Stage readiness: each stage emits `{"type": "stage_ready"}` after initialization; the orchestrator waits for all stages or times out and logs diagnostic suggestions.

### 9.1 Process Device Visibility
- Device binding: set `CUDA_VISIBLE_DEVICES` before process start, or call `torch.cuda.set_device` early in init; configs can be single/multi-GPU.
- Cross-device transfer (simplified): via CPU copies; zero-copy paths are out of scope for now.

### 10. Worker Template (pseudo-code)
```python
def stage_worker(input_q, output_q, runtime, shm_threshold):
    # Device binding
    set_stage_gpu_devices(runtime.devices)
    engine = OmniStageLLM(model, **engine_args)
    batch_seq = 0
    while True:
        first = input_q.get()
        if first is None:
            break
        # Batch intake (up to runtime.max_batch_size)
        batch = [first]
        while len(batch) < int(runtime.max_batch_size or 1) and not input_q.empty():
            nxt = input_q.get()
            if nxt is None:
                input_q.put(None)
                break
            batch.append(nxt)
        # Decode IPC payload and sampling params
        request_ids = []
        engine_inputs = []
        rx_bytes = {}
        rx_decode_ms = {}
        for t in batch:
            ein, rx = maybe_load_from_ipc_with_metrics(t, "engine_inputs", "engine_inputs_shm")
            request_ids.append(t["request_id"])
            engine_inputs.extend(ein if isinstance(ein, list) else [ein])
            rx_bytes[t["request_id"]] = rx["rx_transfer_bytes"]
            rx_decode_ms[t["request_id"]] = rx["rx_decode_time_ms"]
        # Generate and dispatch (grouped by request_id)
        batch_seq += 1
        outputs = list(engine.generate(engine_inputs, batch[0]["sampling_params"], use_tqdm=False))
        grouped = group_by_request_id(outputs, request_ids)
        for rid in request_ids:
            r_out = grouped.get(rid, [])
            use_shm, payload = maybe_dump_to_shm(r_out, shm_threshold)
            msg = {
                "request_id": rid,
                "metrics": {
                    "batch_id": batch_seq,
                    "rx_transfer_bytes": int(rx_bytes.get(rid, 0)),
                    "rx_decode_time_ms": float(rx_decode_ms.get(rid, 0.0)),
                    # omitted: generation latency and token stats
                },
            }
            if use_shm:
                msg["engine_outputs_shm"] = payload
            else:
                msg["engine_outputs"] = payload
            output_q.put(msg)
```

### 12. Observability
- Metrics (as implemented):
  - per-request (emitted by stages): `num_tokens_out`, `stage_gen_time_ms`, `rx_transfer_bytes`, `rx_decode_time_ms`, `rx_in_flight_time_ms`, `batch_id`
  - orchestrator aggregates: E2E latency, tokens/s (written only when stats are enabled)
  - optional per-stage JSONL: `{log_file}.stage{stage_id}.stats.jsonl`
- Logs/tracing:
  - optional per-stage log files: `{log_file}.stage{stage_id}.log`
  - the orchestrator can log readiness, forward size/time, and summary information

### 15. Risks & Mitigations
— End-to-end latency: full-trigger sacrifices interactivity; start simple/correct, add windowed streaming later.
— CPU copy overhead: copy-based IPC for maintainability; add SHM/zero-copy later as an optimization path.

### 16. Refactor: Sink LLM init and process into Stage (diagram)

Objective: without changing external call patterns, encapsulate per-stage LLM init, process creation, and worker logic inside `Stage`, so `PipelinedOmniLLM` focuses on orchestration (seeding, polling, forwarding, collecting) while keeping multi-process, shared memory, and device mapping capabilities.

Key changes (vs. current):
- Before: Orchestrator created processes directly, passing `_stage_worker`; device setup, LLM init, batching, and SHM were implemented there.
- After: enhanced `OmniStage` owns its subprocess (`init_stage_worker`), with `_stage_worker` as the entry; device setup, LLM init, batching, and SHM live in `OmniStage`.

Classes & responsibilities
- OmniLLM (orchestrator)
  - Build `OmniStage` list in parallel (preserve `process_engine_inputs` wiring)
  - Connect adjacent `Stage` input/output queues; seed requests to stage 0
  - Poll stage outputs, decode results, call `process_engine_inputs`, then encode and forward to next stage
  - Termination & cleanup: distribute shutdown signals; join/terminate subprocesses
- Stage (stage unit)
  - Members: `stage_config` (with `engine_args` and `runtime`), `in_q/out_q`, subprocess handle, stats
  - API:
    - `init_stage_worker(...)`: spawn subprocess
    - `stop_stage_worker()`: graceful exit
    - `submit(ipc_payload)`: submit to input queue (may use shared memory)
    - `try_collect() -> Optional[payload]`: non-blocking get from output queue
    - `process_engine_inputs(stages, prompts)`: reuse/delegate existing logic
  - Subprocess entry `worker_main()`:
    - `set_stage_gpu_devices` (`CUDA_VISIBLE_DEVICES` & logical index mapping)
    - build `OmniStageLLM(model, **engine_args)`
    - while-loop:
      - take first task + up to `runtime.max_batch_size-1` non-blocking; support `max_batch_size>=1`
      - `maybe_load_from_ipc_with_metrics` per `engine_inputs` in batch
      - `stage_engine.generate(batched_inputs, sampling_params)` (window `-1`)
      - write per-request aggregated outputs to output queue after `maybe_dump_to_shm`
      - log start/finish; report `error` per request on exceptions

Data flow (sketch)
```
OmniLLM
  ├─ Stage[0].start_process(in0, out0)
  ├─ Stage[1].start_process(in1, out1)
  ├─ ...
  └─ Orchestrator loop:
       - stage0.submit(encode(engine_inputs))
       - for each stage:
           res = stage.try_collect()
           outputs = decode(res)
           stage.set_engine_outputs(outputs)
           if !final: next_inputs = next_stage.process_engine_inputs(stages, original_prompt)
                    next_stage.submit(encode(next_inputs))
```

Config (preserve & extend)
- `runtime.devices`:
  - Comma-separated visibility list (e.g., "2,5,7"): set visible set; default logical index 0 (first GPU)
  - Number/string: treat as logical index (0/1-based per implementation) mapping to physical GPUs
- `runtime.max_batch_size`: max requests taken per intake (default 1)
- `shm_threshold_bytes`: objects larger than this go via shared memory (default 64KB; injected into stages)
- Stage link and inputs:
  - `engine_input_source`: list of upstream stage_id(s) from which the downstream stage takes inputs (typically a single source)
  - `custom_process_input_func`: function to construct downstream engine_inputs from upstream outputs
- Output flags:
  - `final_output`, `final_output_type`: marks that a stage produces the final external output and its type

Kept & strengthened
- Multi-process: each stage in its own process; parallel across stages; serial/batching inside a stage controlled by `max_batch_size`
- Shared memory: large objects via SHM with strict `memoryview -> close -> unlink`
- Device mapping: unified `set_stage_gpu_devices`, support comma lists and logical indices

Migration (execution order)
1) Enhanced `OmniStage` with `init_stage_worker/stop_stage_worker/submit/try_collect/_stage_worker`.
2) `OmniLLM` holds `OmniStage` instances, replacing the older pipelined class; uses Stage APIs.
3) Reuse `entrypoints/stage_utils`: device/SHM/encode-decode utilities live here.
4) Regression: single/multi-request, various `max_batch_size`, SHM threshold, device mapping, multi-stage parallelism, error paths, and shutdown flows.

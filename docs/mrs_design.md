## Multi-Request Streaming (MRS) on a Single Machine — Design Doc

- Author: Team
- Date: 2025-10-27
- Version: v0.3 (simplified: copy-based IPC / serial per stage / full-trigger)

### 1. Background & Scope
- All processing runs on a single physical machine with multi-process, per-stage workers. No proxy or network transport involved.
- Current alignment with vllm-omni: `OmniLLM` supports multiple stages (`OmniStage`). GPU runners already expose streamable steps (prefill/decoding/diffusion), but the entry layer still collects lists and lacks intra-stage streaming and window scheduling.
- Goal: implement multi-stage, multi-request streaming (MRS) locally. Each stage outputs segments; downstream stages stitch and trigger compute based on configured windows. Shared memory and zero-copy strategies reduce data movement overhead.

### 2. Key Constraints
- Multi-process per stage: each stage is an independent process with a while loop; device visibility can be configured (`CUDA_VISIBLE_DEVICES`/`torch.cuda.set_device`).
- Simple IPC (copy-based): use `multiprocessing.Queue`/Pipe for inter-process communication with CPU copies/serialization; do not rely on CUDA IPC/SHM zero-copy in this version.
- Full-trigger across stages: unify `window_size = -1` for all edges — downstream starts only after upstream completes for that request.
- Serial within a stage: at most one in-flight request per stage (max_inflight=1). No batching/concurrency inside a stage in this version.
- Cross-stage pipeline: different stages can process different requests concurrently (e.g., stage A handles request 1 while stage B handles request 0).

### 3. Architecture Overview
- Processes & IPC queues
  - Each "sub-stage" is an OS process (worker). The loop: take from input_queue → compute → put to output_queue.
  - Inter-stage connection via IPC: control plane (Unix Domain Socket / Pipe) + data plane (shared memory ring buffer in future; copy-based in this version).
  - MPMC queue semantics; control plane passes light-weight pointers/metadata; data plane carries large payloads.
- Device visibility
  - Each stage sets `CUDA_VISIBLE_DEVICES` or calls `torch.cuda.set_device` to bind to GPU sets.
  - A stage may use multiple GPUs internally (TP/PP/DP) but presents as a single stage unit.
- Simplified IPC: copy-based queues/pipes for data transfer; zero-copy is future work.
- Pipeline progression: when a stage finishes a request, it enqueues outputs to the downstream stage; if downstream is idle, it starts immediately.
- Scheduling & windows
  - Each edge (stage→stage) defines a window policy; downstream triggers compute by consuming stitched window views.
  - `-1` mode triggers only after upstream emits `is_last=True` (full aggregation).
  - Backpressure propagates upstream via the control plane: once high-watermark is reached, upstream blocks or degrades behavior.

### 4. Stage Breakdown
- AR (autoregressive)
  - S0 Intake/Plan: parse request, assign request_id, output normalized tasks.
  - S1 Prefill: tokenize/encode/KV warmup; may micro-batch across requests.
  - S2 Decode Loop: step-by-step tokens; flush segments by `max_segment_tokens` and `min_flush_interval_ms`.
  - S3 Postprocess: detokenize, logprobs/struct wrapping (CPU if necessary).
  - S4 Consumer: local consumer callback/buffering.
- DiT/Diffusion
  - D0 Intake/Plan: scheduler/seed init.
  - D1 Sampling Loop: preview frames every k steps (downsample allowed).
  - D2 Decode/Encode: VAE decode and JPEG/WEBP encode (prefer GPU→pinned CPU).
  - D3 Postprocess/Consumer.

### 5. Unified Events & Segment Structure (in-process)
- Base event: `{ request_id, sequence, event_type, ts, payload, device_handles }`
- ARSegmentEvent (segment):
  - Fields: `tokens` (List[int] or GPU tensor view), optional `logprobs`, `is_last`, optional `hidden_states/aux` refs.
  - Semantics: strictly ordered by `sequence` for the same `request_id`; `is_last` marks the final segment.
- ImageProgressEvent: `{ step, image_tensor|jpeg_ref, is_last }` (diffusion preview/completion).
- Completed/Error: terminal or error events affect only the current request.

### 6. AR Segmentation & Windowing (simplified)
- Production (upstream stage)
  - Accumulate tokens during decoding; flush a segment if any of the following holds:
    - Reaches `max_segment_tokens` length
    - Exceeds `min_flush_interval_ms` since last flush
    - Request ended (`is_last=True`)
  - Flush immediately via `output_queue.put(segment)`; do not wait for entire request completion.
- Consumption (downstream stage)
  - Use unified `window_size = -1`: trigger only after receiving `is_last=True`; no window slicing.
- Backpressure & watermark
  - Must-deliver events (token.delta, Completed) are not dropped; block upstream briefly on high watermark.
  - For long sequences under `-1`, allow early segments to spill from GPU to pinned CPU to control VRAM pressure.

#### Example config (YAML, simplified)
```yaml
mrs:
  enabled: true
  defaults:
    window_size: -1           # Full-trigger across stages
    max_inflight: 1           # Serial per stage
```

### 7. IPC Implementation (simplified: copy-based)
- Use `multiprocessing.Queue`/Pipe for inter-process communication (control + data).
- Data is serialized/copied via CPU; no CUDA IPC/SHM zero-copy in this version.
- Backpressure: blocking reads/writes based on queue capacity, blocking upstream on high watermark.

### 8. Scheduling & Cancellation (simplified)
- Serial FIFO: each stage `max_inflight=1`; no cross-request micro-batching or concurrency.
- Cancel/timeout: ongoing work should exit early; remaining queued requests stay pending.
- Pipeline: when a stage finishes a request, it enqueues to the next stage; that stage immediately pulls the next request from its input queue, enabling cross-stage concurrency.

#### Short sequence example (req0/req1, stage A→B)
1) t0: stage A handles req0
2) t1: req0 completes on A → enters B; A immediately starts req1
3) t2: B handles req0 while A handles req1 (parallel across stages)

### 9. Integration Points (by file)
- `vllm_omni/entrypoints/omni_llm.py`
  - Introduce a ProcessOrchestrator (replace threaded version):
    - Spawn stage processes per config (set `CUDA_VISIBLE_DEVICES`/`torch.cuda.set_device`), create control/data channels, build window state machines.
    - Add `enable_mrs` in `generate`; when enabled, do not collect `stage.engine.generate(...)` into lists; stream segments to downstream via IPC instead.
  - Register `window_size` per edge; `-1` triggers downstream once `is_last` is observed.
  - Manage process lifecycle: start/heartbeat/graceful shutdown/restart on crash; propagate cancel/timeout.
- `vllm_omni/outputs.py`
  - Add in-process events: `ARSegmentEvent`, `ImageProgressEvent`, `CompletedEvent`, `ErrorEvent`.
  - Keep `OmniRequestOutput` as the external representation of final stage outputs.
- `vllm_omni/worker/gpu_ar_model_runner.py`
  - Near `valid_sampled_token_ids` parsing (after updating `req_state.output_token_ids`), aggregate new tokens into segments and emit; send final `is_last=True` and Completed.
  - Replace current `.to("cpu").contiguous()` sync path: by default keep GPU refs and let downstream copy asynchronously from a pinned pool when needed.
- `vllm_omni/worker/gpu_diffusion_model_runner.py`

### 9.1 Process Device Visibility
- Device binding: set `CUDA_VISIBLE_DEVICES` before process start, or call `torch.cuda.set_device` early in init; configs can be single/multi-GPU.
- Cross-device transfer (simplified): via CPU copies; zero-copy paths are out of scope for now.

### 10. Worker Template & Window State (pseudo-code)
```python
def stage_worker(input_q, output_q, ctx):
    while ctx.alive:
        item = input_q.get()  # blocking / with timeout
        if item.is_cancelled():
            continue
        with cuda_stream(ctx.stream):
            out_segments = run_stage_compute(item)  # produce segments
        for seg in out_segments:
            output_q.put(seg)

class WindowState:
    def __init__(self, window_size):
        self.window_size = window_size
        self.last_emitted = 0
        self.buffer = []  # references/paged views
    def on_segment(self, seg):
        self.buffer.append(seg)
        if self.window_size == -1:
            return self.emit_full_if_last(seg)
        return self.emit_windows()
    def emit_windows(self):
        emits = []
        while self._pending_len() - self.last_emitted >= self.window_size:
            emits.append(self._make_view(self.last_emitted, self.window_size))
            self.last_emitted += self.window_size
        return emits
    def emit_full_if_last(self, seg):
        return [self._make_full_view()] if seg.is_last else []
```

### 11. Configuration
- `mrs.enabled`, `mrs.ar.default_window_size`, `mrs.ar.min_flush_interval_ms`, `mrs.ar.max_segment_tokens`, `mrs.ar.scheduler_policy`.
- Devices/processes:
  - `mrs.stages`: [{ name, process: true, devices: "0" | "1,2", extra_env: {} }]
  - `mrs.edges`: [{ from, to, window_size }]
- Pinned/SHM pool sizing: estimate by concurrency and max sequence; tune watermarks and GC policies; configure CUDA IPC limits and cleanup thresholds if enabled.

### 12. Observability
- Metrics:
  - Stage-level: enqueue/dequeue rates, queue depth, blocking time on watermarks, window triggers/counts.
  - Request-level: TTFT, TPST, completion latency, cancel rate.
  - Resource-level: GPU utilization, VRAM usage/fragmentation, pinned usage.
- Logs/tracing: carry `request_id`, stage name, and `sequence`; sample decisions for downsampling/backpressure.

### 13. Test Plan
- Unit: queue correctness, window state machine, cancel/timeout, memory pool rent/return.
- Performance: AR QPS/TPST vs. window/segment sizes; diffusion preview frequency vs. throughput.
- Stability: long-run memory leaks; backpressure trigger/recovery paths.
- Compatibility: behavior parity when MRS is disabled.

### 14. Milestones
- M1: Process Orchestrator with simple IPC; serial per stage; full-trigger across stages (window=-1).
- M2: Hardening and observability; cancel/timeout and error paths.
- M3: Optional zero-copy and concurrency/micro-batching optimizations.

### 15. Risks & Mitigations
— End-to-end latency: full-trigger sacrifices interactivity; start simple/correct, add windowed streaming later.
— CPU copy overhead: copy-based IPC for maintainability; add SHM/zero-copy later as an optimization path.
— Serial throughput: `max_inflight=1` limits per-stage throughput; enable batching/concurrency later as needed.

### 16. Refactor: Sink LLM init and process into Stage (diagram)

Objective: without changing external call patterns, encapsulate per-stage LLM init, process creation, and worker logic inside `Stage`, so `PipelinedOmniLLM` focuses on orchestration (seeding, polling, forwarding, collecting) while keeping multi-process, shared memory, and device mapping capabilities.

Key changes (vs. current):
- Before: `PipelinedOmniLLM` created processes directly, passing `_stage_worker`; device setup, LLM init, batching, and SHM were implemented there.
- After: enhance `Stage` to start its own subprocess via `start_process()`, with `worker_main()` as the entry; device setup, LLM init, batching, and SHM live in `Stage`.

Classes & responsibilities
- PipelinedOmniLLM (orchestrator)
  - Build `Stage` list (preserve `OmniStage` init timing and `process_engine_inputs` flow)
  - Connect adjacent `Stage` input/output queues; seed requests to stage 0
  - Poll stage outputs, decode results, call `process_engine_inputs`, then encode and forward to next stage
  - Termination & cleanup: distribute shutdown signals; join/terminate subprocesses
- Stage (stage unit)
  - Members: `stage_config` (with `engine_args` and `mrs`), `in_q/out_q`, subprocess handle, stats
  - API:
    - `start_process(in_q, out_q)`: spawn subprocess
    - `stop_process()`: graceful exit
    - `submit(ipc_payload)`: submit to `in_q` (may use shared memory)
    - `try_collect() -> Optional[payload]`: non-blocking get from `out_q`
    - `process_engine_inputs(stages, prompts)`: reuse/delegate existing logic
  - Subprocess entry `worker_main()`:
    - set_stage_gpu_devices (`CUDA_VISIBLE_DEVICES` & logical index mapping)
    - build `OmniStageLLM(model, **engine_args)`
    - while-loop:
      - take first task + up to `mrs.max_batch_size-1` non-blocking; support `max_batch_size>=1`
      - `maybe_load_from_ipc` per `engine_inputs` in batch
      - `stage_engine.generate(batched_inputs, sampling_params)` (window `-1`)
      - write per-request aggregated outputs to `out_q` after `maybe_dump_to_shm`
      - log start/finish; report `error` per request on exceptions

Data flow (sketch)
```
PipelinedOmniLLM
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
- `mrs.devices`:
  - Comma-separated visibility list (e.g., "2,5,7"): set visible set; default logical index 0 (first GPU)
  - Number/string: treat as logical index (0/1-based per implementation) mapping to physical GPUs
- `mrs.max_batch_size`: max requests taken per intake (default 1)
- `shm_threshold_bytes`: objects larger than this go via shared memory (default 64KB; can be injected into stages)

Kept & strengthened
- Multi-process: each stage in its own process; parallel across stages; serial/batching inside a stage controlled by `max_batch_size`
- Shared memory: large objects via SHM with strict `memoryview -> close -> unlink`
- Device mapping: unified `set_stage_gpu_devices`, support comma lists and logical indices
- Windows: keep `window_size=-1` semantics (downstream triggers only after upstream finishes the request)

Migration (execution order)
1) Implement enhanced `Stage` (e.g., `entrypoints/stage.py`) with `start_process/stop_process/submit/try_collect/worker_main`.
2) Make `PipelinedOmniLLM` hold `Stage` instances, replacing direct queues and `_stage_worker` with `Stage` APIs.
3) Reuse `pipeline_utils`: device/SHM/encode-decode utilities stay in utils.
4) Regression: single/multi-request, various `max_batch_size`, SHM threshold, device mapping, multi-stage parallelism, error paths, and shutdown flows.

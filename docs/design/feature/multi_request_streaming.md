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

### 4. IPC Implementation (simplified: copy-based)
- Use `multiprocessing.Queue`/Pipe for inter-process communication (control + data).
- Data is serialized/copied via CPU; no CUDA IPC/SHM zero-copy in this version.
- Backpressure: queues are unbounded; pressure manifests as compute-rate differences. Optional SHM reduces large-object transfer cost; RX/decoding overhead is recorded for observability.

### 5. Scheduling & Cancellation (simplified)
- Pipeline: when a stage finishes a request, it enqueues to the next stage; that stage immediately pulls the next request from its input queue, enabling cross-stage concurrency.
- Cancellation/timeout: explicit cancellation/timeouts are not provided; graceful shutdown uses a `None` sentinel sent to each stage input queue.

#### Short sequence example (req0/req1, stage A→B)
1) t0: stage A handles req0
2) t1: req0 completes on A → enters B; A immediately starts req1
3) t2: B handles req0 while A handles req1 (parallel across stages)

### 6. Integration Points (by file)
- `vllm_omni/entrypoints/omni.py` (Orchestrator)
  - Class `Omni` orchestrates multi-process stages; constructs `OmniStage` instances in parallel and spawns per-stage workers.
  - Spawns stage processes per config (set `CUDA_VISIBLE_DEVICES`/`torch.cuda.set_device`), creates control/data channels, builds simple full-trigger flow.
  - Stats/logging are disabled by default; per-stage and orchestrator stats are only written when explicitly enabled.
  - Manages process lifecycle: start/wait for readiness, graceful shutdown; forwards results between stages using copy-based IPC and optional SHM.
  - Stage readiness: each stage emits `{"type": "stage_ready"}` after initialization; the orchestrator waits for all stages or times out and logs diagnostic suggestions.

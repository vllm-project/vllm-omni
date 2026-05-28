# vLLM-Omni Failure Mode Injection Scenarios and Expected Behavior Matrix

## Background and Goals

The online serving pipeline of vLLM-Omni can be abstracted as: `Serve` (service entry process) -> `Engine` (model orchestration unit) -> `Worker` (execution process) -> `GPU` (compute resource).

- `Serve` provides external APIs, accepts requests, and manages lifecycle.
- `Engine` handles request scheduling and inference orchestration; the number and organization of engines can vary by model type.
- `Worker` performs concrete model computation, usually bound to underlying GPU resources.
- `GPU` is the resource layer that ultimately carries inference compute and memory usage.

```mermaid
flowchart TD
    R(request)--> A(serve)
    A --> B(engine)
    A --> C(engine)
    A --> D(engine)
    B --> E(worker)
    C --> F(worker)
    D --> I(worker)
    E --> G(GPU)
    F --> H(GPU)
    I --> L(GPU)
```

The diagram above shows an abstract relationship. Omni models and Diffusion models share the same hierarchy; differences mainly come from engine decomposition, worker count, and scheduling strategy.

In real customer environments, the system will inevitably encounter accidental operations or environmental disturbances, such as unintended process termination, container/node jitter, or short-term resource contention. From the system perspective, many of these events map to two fault categories: "process receives abnormal signals" or "GPU memory is squeezed." For example, pressing `Ctrl+C` essentially sends a `SIGINT` signal to the `serve` process; running `kill -15 <pid>` maps to graceful termination via `SIGTERM`; running `kill -9 <pid>` maps to forced termination via `SIGKILL`; when `docker kill` is executed or a Pod is deleted, the container runtime usually sends `SIGKILL`.

This document currently covers two core failure modes, and the system is expected to show predictable and observable behavior under each mode:

- **Failure Mode 1: Process receives abnormal signals (`SIGINT` / `SIGTERM` / `SIGKILL`)**  
  Focus on three aspects: whether processes exit as expected and complete cleanup, whether requests fail fast or connections are interrupted, and whether GPU resources are eventually fully released with no residue.
- **Failure Mode 2: OOM (occupy all free GPU memory via an extra process)**  
  Focus on two aspects: whether service health degrades to the expected state (for example, `503`), and whether requests fail within an acceptable time without hanging.

More failure modes (for example, network jitter and network interruption) will be added in future iterations to improve end-to-end reliability validation coverage.

> Note: independent fault injection on the `engine` component is not covered in the current version and will be added gradually in future versions.

## Fault Injection Scenario Matrix

| Scenario | Fault Type | System Behavior | Current Status |
|------|----------------------------|----------|----------|
| No load | Send `SIGKILL` to Worker | Worker process is killed immediately; main service detects child-process loss and turns unavailable; API enters stable 5xx |  |
| No load | Send `SIGTERM` to Worker | Worker exits after receiving termination signal; main service is marked unavailable; API enters stable 5xx |  |
| No load | Send `SIGKILL` to serve main process | serve main process exits instantly; request connections are interrupted; related child processes are cleaned up with no residue; GPU memory is released quickly | [#3725](https://github.com/vllm-project/vllm-omni/issues/3725) <br>[#43060](https://github.com/vllm-project/vllm/issues/43060) |
| No load | Send `SIGTERM` to serve main process | serve enters graceful shutdown and stops serving; then exits and completes cleanup; GPU memory is released |  |
| No load | Send `SIGINT` to serve main process (equivalent to `Ctrl+C`) | Triggers serve shutdown path; service stops responding and becomes unavailable; related child processes exit and resources are released |  |
| No load | Send `SIGKILL` to all related processes | All related processes terminate immediately; service becomes unavailable at once; no residual processes remain; GPU memory is released quickly |  |
| No load | Send `SIGTERM` to all related processes | All processes enter exit flow and complete shutdown; service becomes unavailable; resource release is completed |  |
| Under load | Send `SIGKILL` to Worker | In-flight requests are hard interrupted (5xx/connection drop); main service becomes unavailable | [#3683](https://github.com/vllm-project/vllm-omni/issues/3683) |
| Under load | Send `SIGTERM` to Worker | In-flight requests are canceled or fail fast; main service becomes unavailable |  |
| Under load | Send `SIGKILL` to serve main process | serve is hard-killed and current connections are interrupted; in-flight requests fail; after cleanup there are no residual processes and GPU memory is released | [#3683](https://github.com/vllm-project/vllm-omni/issues/3683) |
| Under load | Send `SIGTERM` to serve main process | serve stops accepting new requests and executes shutdown flow; in-flight requests fail; no residue remains and GPU memory is released | [#3683](https://github.com/vllm-project/vllm-omni/issues/3683) |
| Under load | Send `SIGINT` to serve main process (equivalent to `Ctrl+C`) | `Ctrl+C`-style serve shutdown; in-flight requests fail (5xx/connection interruption); service unavailable; after exit there is no residue and GPU memory is released | [#3683](https://github.com/vllm-project/vllm-omni/issues/3683) |
| Under load | Send `SIGKILL` to all related processes | All processes terminate instantly; all in-flight requests fail; service becomes unavailable immediately; GPU memory is released quickly |  |
| Under load | Send `SIGTERM` to all related processes | All processes exit gracefully; in-flight requests fail; service unavailable; GPU memory is released | [#3683](https://github.com/vllm-project/vllm-omni/issues/3683) |
| OOM | Occupy all free GPU memory via an extra process | After OOM injection process starts, GPU memory is continuously saturated; service enters unavailable/degraded state and health check drops to 503; different request types (chat/speech, etc.) fail fast within a fixed time and return 500 (no hanging) |  |

## Source of Conclusions

The behaviors and conclusions above are summarized from current fault injection validation results on `Qwen3-Omni` and `Wan2.2`.

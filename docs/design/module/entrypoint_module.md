---
toc_depth: 3
---

# Entrypoint Module Architecture Design

The entrypoint module is the user-facing control plane of vLLM-Omni. It
accepts CLI commands, OpenAI-compatible HTTP requests, and offline Python API
calls, then maps them onto one shared multi-stage runtime.

In vLLM-Omni, the entrypoint layer is not just a thin wrapper around serving.
It also defines how runtime initialization is triggered, how requests are
normalized before stage-0 execution, and how final outputs are returned to a
simple request-oriented interface.

**Table of Content:**

- [1. Overview](#1-overview)
- [2. Design Goals](#2-design-goals)
- [3. Main Components](#3-main-components)
- [4. Key Component Design](#4-key-component-design)
- [5. Runtime Structure](#5-runtime-structure)
- [6. Initialization Design](#6-initialization-design)
- [7. Execution Logic](#7-execution-logic)
- [8. Lifecycle and Shutdown](#8-lifecycle-and-shutdown)

---

## 1. Overview

The entrypoint module covers the end-to-end serving path from external request
surfaces to internal stage execution. Its responsibilities include:

- CLI bootstrapping for `vllm serve ... --omni`
- online serving through the OpenAI-compatible API server
- offline inference through `Omni`
- asynchronous inference through `AsyncOmni`
- stage configuration resolution and runtime bootstrap
- request normalization before stage-0 submission
- per-request output routing back to the caller

This module is the boundary where vLLM-Omni turns a heterogeneous AR and
diffusion pipeline into one logical runtime.

## 2. Design Goals

The current entrypoint design is built around a small number of architectural
goals:

- **One runtime, multiple interfaces**: online serving and offline inference
  should reuse the same execution path instead of maintaining separate runtime
  stacks.
- **Centralized orchestration**: stage transitions should be decided in one
  place, not duplicated across API handlers, Python wrappers, or model
  adapters.
- **Minimal stage-0 preparation in the caller**: entrypoints should normalize
  requests and register request state locally, then hand execution to the
  runtime.
- **Compatibility with upstream vLLM**: the serving surface should remain close
  to vLLM where possible, while extending it for multi-stage omni-modality
  execution.

## 3. Main Components

| Component | Location | Responsibility |
| --- | --- | --- |
| Omni CLI entry | `vllm_omni/entrypoints/cli/main.py` | Detects `--omni` and switches from upstream CLI behavior to Omni-specific command registration |
| Serve command | `vllm_omni/entrypoints/cli/serve.py` | Parses Omni runtime arguments and starts the API server |
| API server | `vllm_omni/entrypoints/openai/api_server.py` | Builds the FastAPI app, initializes serving state, and exposes OpenAI-compatible routes |
| Offline runtime | `vllm_omni/entrypoints/omni.py` | Synchronous offline API that submits requests and polls final outputs |
| Async runtime | `vllm_omni/entrypoints/async_omni.py` | Asynchronous runtime client used by the API server and async Python callers |
| Shared runtime base | `vllm_omni/entrypoints/omni_base.py` | Shared lifecycle helpers, metrics handling, and output parsing |
| Request state | `vllm_omni/entrypoints/client_request_state.py` | Maintains per-request queues and runtime-facing bookkeeping at the entrypoint layer |
| Runtime proxy | `vllm_omni/engine/async_omni_engine.py` | Resolves stage configs, initializes stage clients, owns janus queues, and launches the orchestrator thread |
| Orchestrator | `vllm_omni/engine/orchestrator.py` | Owns stage-to-stage execution, routing, polling, forwarding, and control-plane fanout |
| Stage clients | `vllm_omni/engine/stage_engine_core_client.py`, `vllm_omni/diffusion/stage_diffusion_client.py` | Hide LLM and diffusion backends behind a uniform orchestrator-facing interface |

## 4. Key Component Design

The entrypoint module is centered on a few components that define the runtime
behavior.

### AsyncOmni and Omni

`AsyncOmni` and `Omni` provide the public Python APIs for online-style async
serving and offline batch inference. Their design goal is to keep the external
interface simple while hiding the complexity of multi-stage execution. They do
not implement stage transitions themselves. Instead, they submit requests,
track per-request state, and return final outputs in a request-oriented form.

### AsyncOmniEngine

`AsyncOmniEngine` is the runtime proxy between the public entrypoints and the
background orchestration logic. Its role is to resolve stage configuration,
bootstrap stage clients, prepare stage-0 requests, and own the queues that
connect the caller thread with the orchestrator thread.

This split is important because it keeps initialization and lifecycle control
close to the entrypoint layer, while moving execution decisions away from the
API-facing code.

### Orchestrator

`Orchestrator` is the central execution coordinator of the entrypoint design.
It owns request intake, stage polling, stage-to-stage forwarding, final-output
routing, and control-plane fanout. In architectural terms, this is the main
component that turns a collection of stage runtimes into one pipeline runtime.

This is also the main simplification in the current design: stage progression
is decided in one place instead of being scattered across individual entrypoint
surfaces.

### API Server Integration

The OpenAI-compatible API server is the HTTP-facing wrapper of the same runtime.
Its role is not to manage pipeline execution directly, but to construct the
FastAPI app, initialize the shared serving state, and bind `AsyncOmni` into the
OpenAI-compatible request/response layer.

This keeps the online serving stack aligned with the offline and Python runtime
paths instead of introducing a separate server-only execution model.

## 5. Runtime Structure

The current entrypoint stack includes the runtime control path used by both
offline inference and online serving. The diagram below shows the main
structure:

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                    API Layer                                    │
│  ┌─────────────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │ AsyncOmni (EngineClient)            │  │ Omni                             │  │
│  │ • generate() / abort() / shutdown() │  │ • generate()                     │  │
│  │ • _final_output_handler()           │  │                                  │  │
│  └─────────────────────────────────────┘  └──────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                              Engine Layer (Proxy)                               │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ AsyncOmniEngine                                                           │  │
│  │ • _bootstrap_orchestrator() & _initialize_stages()                        │  │
│  │ • add_request() / add_request_async() -> input_processor.process_inputs() │  │
│  │ • try_get_output() / try_get_output_async()                               │  │
│  └───────────────────┬─────────────────────────────────▲─────────────────────┘  │
│         request_queue (janus.Queue)        output_queue (janus.Queue)           │
├──────────────────────┼─────────────────────────────────┼────────────────────────┤
│                      ▼        Orchestration Layer      │                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ Orchestrator [background thread]                                          │  │
│  │ • _request_handler()                                                      │  │
│  │     - stage_client.add_request_async() & _prewarm_async_chunk_stages()    │  │
│  │ • _orchestration_output_handler()                                         │  │
│  │     - _process_stage_outputs() -> output_processors[i].process_outputs()  │  │
│  │     - _route_output() & _forward_to_next_stage()                          │  │
│  └──────────┬─────────────────────────┬────────────────────────┬─────────────┘  │
├─────────────┼─────────────────────────┼────────────────────────┼────────────────┤
│             │                 Communication Layer              │                │
│  ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐  │
│  │ StageEngineCoreClient │ │ StageEngineCoreClient │ │ StageDiffusionClient  │  │
│  │ • ZMQ ROUTER / PULL   │ │ • ZMQ ROUTER / PULL   │ │ • ZMQ ROUTER / PULL   │  │
│  │ • Msgpack codec       │ │ • Msgpack codec       │ │ • Msgpack codec       │  │
│  └──────────┬────────────┘ └──────────┬────────────┘ └──────────┬────────────┘  │
│             ▼ ZMQ IPC                 ▼ ZMQ IPC                 ▼ ZMQ IPC       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                 Execution Layer                                 │
│  ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐  │
│  │ StageCoreProc         │ │ StageCoreProc         │ │ DiffusionEngine       │  │
│  │ [background process]  │ │ [background process]  │ │ [background process]  │  │
│  └───────────────────────┘ └───────────────────────┘ └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

The entrypoint stack is intentionally split into three layers:

- the public interface layer (`Omni`, `AsyncOmni`, CLI, API server)
- the runtime proxy layer (`AsyncOmniEngine`)
- the orchestration layer (`Orchestrator`)

This separation keeps the user-facing API simple while moving multi-stage
execution decisions into one internal control point.

This architecture is organized around a strict separation of concerns.
`AsyncOmni` and `Omni` expose a single logical runtime to the caller, while
`AsyncOmniEngine` owns stage configuration, queue wiring, and lifecycle
management. The `Orchestrator` centralizes multi-stage execution semantics so
that stage transitions are not duplicated across the API server, offline
interfaces, and model-specific wrappers.

The stage backends are hidden behind a shared client abstraction.
Autoregressive stages use `StageEngineCoreClient`, which reuses the upstream
EngineCore communication model, while diffusion stages use
`StageDiffusionClient`, which wraps the diffusion runtime behind the same
orchestrator-facing contract. This allows vLLM-Omni to compose heterogeneous AR
and diffusion stages without forcing them into the same backend model.

## 6. Initialization Design

Initialization starts from one of the public entry surfaces, but the same
runtime bootstrap is reused underneath.

```text
vllm serve ... --omni
  -> entrypoints/cli/main.py
  -> entrypoints/cli/serve.py
  -> openai/api_server.py::omni_run_server()
  -> build_async_omni()
  -> AsyncOmni(...)
  -> AsyncOmniEngine(...)
```

For online serving, `entrypoints/cli/main.py` first detects `--omni` and
switches to the Omni-specific CLI path. `entrypoints/cli/serve.py` then extends
the upstream serve command with Omni runtime arguments such as stage
configuration, worker backend, and diffusion-related options before launching
the server.

The main bootstrap work happens inside `AsyncOmniEngine`. Its initialization
can be summarized as:

1. Resolve stage configuration from `stage_configs_path`, model defaults, or a
   generated single-stage diffusion configuration.
2. Derive runtime-wide metadata such as number of stages, stage types, default
   sampling parameters, and whether `async_chunk` mode is enabled.
3. Initialize stage clients for each configured stage. LLM stages are attached
   through `StageEngineCoreClient`, while diffusion stages are wrapped through a
   diffusion-specific client.
4. Create output processors, the stage-0 input processor, and the janus queues
   used between the caller thread and the orchestrator thread.
5. Start the `Orchestrator` in a background thread and wait until startup
   completes.

From the caller's perspective, this entire process produces one runtime object.
Internally, however, stage metadata, processors, queues, and worker-facing
clients are already prepared before the first request is accepted.

For API serving, `omni_init_app_state()` completes the final integration step.
It binds the initialized runtime into FastAPI state, constructs the serving
objects required by the OpenAI-compatible routes, and adapts behavior for pure
diffusion deployments versus multi-stage Omni deployments.

## 7. Execution Logic

The execution model is designed around a simple principle: entrypoints submit
requests, but the orchestrator owns the pipeline.

No matter whether a request comes from `Omni`, `AsyncOmni`, or the API server,
the path is the same at a high level:

1. The entrypoint resolves per-stage sampling parameters and creates
   request-level bookkeeping.
2. `AsyncOmniEngine` performs stage-0 preparation in the caller thread. When
   needed, it runs the upstream `InputProcessor`, restores omni-specific
   payloads such as prompt embeddings and additional information, and registers
   the request with the stage-0 output processor.
3. The prepared request is pushed into the request queue and consumed by the
   `Orchestrator`.
4. The `Orchestrator` submits the request to stage 0, polls stage outputs, and
   decides whether the result should be emitted as final output or forwarded to
   the next stage.
5. Final outputs are placed on the output queue and routed back to the
   request-specific state owned by `AsyncOmni` or `Omni`.

This is the key architectural choice of the current design. Stage forwarding,
multi-stage completion, async-chunk prewarming, and companion-request handling
are kept inside `Orchestrator`, so the public APIs do not need to know how
individual stages interact.

The return path is also intentionally simple. `AsyncOmni` runs a background
final-output dispatcher that reads from the orchestrator output queue and
demultiplexes messages by request ID. `Omni` uses the same queue but polls it
synchronously. In both cases, callers interact with request-oriented outputs
instead of stage-oriented internal events.

## 8. Lifecycle and Shutdown

Lifecycle management is centered on `AsyncOmniEngine.shutdown()`.

At shutdown time:

- the entrypoint stops its output handling loop
- the runtime proxy sends a shutdown message to the orchestrator
- the orchestrator shuts down all stage clients
- the background thread is joined
- janus queues are closed

This keeps cleanup symmetrical with initialization: the entrypoint owns the
public runtime object, while the orchestrator owns the internal stage runtime.
Weak finalizers are also used as a best-effort fallback to reduce resource
leaks when callers do not close the runtime explicitly.

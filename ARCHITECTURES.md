# vLLM-Omni Architecture Overview

> **Audience:** Contributors and advanced users who want to understand the internals of vLLM-Omni before submitting patches or new model integrations.

---

## Table of Contents

1. [High-Level Design](#1-high-level-design)
2. [Multi-Stage Pipeline](#2-multi-stage-pipeline)
3. [Engine Architecture & Communication](#3-engine-architecture--communication)
4. [AR Module](#4-ar-module)
5. [Diffusion Module](#5-diffusion-module)
6. [Omni Connector Layer](#6-omni-connector-layer)
7. [Scheduling](#7-scheduling)
8. [Distributed & Parallel Strategies](#8-distributed--parallel-strategies)
9. [Entrypoints & APIs](#9-entrypoints--apis)
10. [Adding a New Omni-Modality Model](#10-adding-a-new-omni-modality-model)

---

## 1. High-Level Design

vLLM-Omni extends vLLM to support **omni-modality generation** — producing heterogeneous outputs (text, audio, image, video) from a single request in a unified serving pipeline.

```
┌────────────────────────────────────────────────────────────┐
│                        Client Request                       │
│        /v1/chat/completions  |  /v1/audio/speech  | ...    │
└───────────────────────────────┬────────────────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │  OmniServeEntrypoint  │
                    │  (FastAPI / OpenAI    │
                    │   compatible API)     │
                    └───────────┬──────────┘
                                │  stage_configs_path
                    ┌───────────▼──────────┐
                    │   Multi-Stage Engine  │
                    │  (OmniStage × N)      │
                    └──┬──────────────┬────┘
                       │              │
              ┌────────▼───┐   ┌──────▼──────┐
              │  AR Stage   │   │ DiT Stage   │
              │ (LLM/TTS)  │   │ (Diffusion) │
              └────────────┘   └─────────────┘
```

Key design goals:

- **Modular stages** — each stage is an independent engine (AR or Diffusion) connected by a typed connector.
- **Heterogeneous outputs** — a single user request can fan out to text + audio + image simultaneously.
- **OpenAI-compatible surface** — drop-in replacement for standard vLLM serving.

---

## 2. Multi-Stage Pipeline

### Stage Configuration

Stages are described in a YAML file passed via `--stage_configs_path`. Each entry defines:

| Field | Description |
|---|---|
| `model` | HuggingFace model ID or local path |
| `stage_type` | `ar` (autoregressive) or `diffusion` |
| `engine_input_source` | Where this stage receives input (`api`, `kv_transfer`, etc.) |
| `custom_process_input_func` | Optional Python dotted path to a preprocessing function |
| `ulysses_degree` | Sequence parallelism degree for diffusion attention |
| `ring_degree` | Ring parallelism degree for diffusion attention |

### Stage Lifecycle

```
load stage_configs_path
       │
       ▼
OmniStage.__init__()   ← validates config, creates connector
       │
       ▼
OmniStage.start()      ← spawns engine process(es)
       │
       ▼
OmniConnector.setup()  ← establishes inter-stage channel
       │
  request loop
       │
       ▼
OmniStage.process()    ← schedules, executes, streams outputs
```

---

## 3. Engine Architecture & Communication

### Process Model

Each stage runs in its own set of processes:

```
  API Process
      │
      │  (asyncio queue)
      ▼
  Stage Process  ──── IPC/ZMQ ────►  Worker Process(es)
                                        │
                                     GPU Device(s)
```

- **API Process** — handles HTTP, parses requests, streams responses.
- **Stage Process** — runs the scheduler and output processor; orchestrates workers.
- **Worker Processes** — execute forward passes on GPU via `GPUWorker`.

### Inter-Stage Communication

Connectors carry typed payloads between stages:

| Connector Type | Transport | Use Case |
|---|---|---|
| `QueueConnector` | In-process asyncio queue | Single-host, same process |
| `ZMQConnector` | TCP/IPC socket | Multi-host or cross-process |
| `KVTransferConnector` | vLLM KV cache transfer protocol | AR→AR prefix caching |

> **Roadmap:** D2D (device-to-device) connectors via NCCL/UCX are planned to eliminate the current D2H2D copy overhead for large tensor payloads.

---

## 4. AR Module

The autoregressive module handles LLM and TTS model execution.

### Input Processing — `OmniInputProcessor`

Located in `vllm_omni/engine/input_processor.py`.

Responsibilities:
- Tokenizes text prompts
- Encodes multimodal inputs (images, audio) into prompt embeddings
- **Serializes embeddings** to pass across the process boundary (current limitation — streaming support pending)

### Output Processing — `MultimodalOutputProcessor`

Located in `vllm_omni/engine/output_processor.py`.

Responsibilities:
- Accumulates token logits and generated IDs
- Decodes token IDs → text or audio codec tokens
- Routes outputs to the next stage via the connector

> ⚠️ **Known Gap:** Streaming for audio generation is not yet supported. The processor accumulates tensors until generation is complete before flushing downstream. See [GitHub issue tracker] for the tracking issue.

### KV Cache Transfer — `KVCacheTransferData`

Defined in `vllm_omni/core/sched/omni_ar_scheduler.py`.

When two AR stages are chained (e.g., prefill → decode disaggregation), the scheduler attaches a `KVCacheTransferData` object to each request, which carries the KV block indices that should be migrated to the next stage's memory pool.

---

## 5. Diffusion Module

The diffusion module handles image and video generation using DiT (Diffusion Transformer) architectures.

### Execution Flow

```
Request arrives at DiT Stage
        │
        ▼
OmniGenerationScheduler   ← batches requests (single-step mode)
        │
        ▼
GPUWorker.execute_model() ← runs denoising loop
        │                   TODO: currently processes one request at a time
        ▼
   Output tensor          ← image pixels / latents
        │
        ▼
   OmniConnector          ← sends to next stage or API response
```

### Acceleration Backends

#### Attention

| Backend | Hardware | Notes |
|---|---|---|
| FlashAttention | CUDA | Default for NVIDIA GPUs |
| SDPA | CPU / fallback | PyTorch native |
| SageAttention | CUDA | Quantized sparse attention |
| AscendAttention | NPU | Huawei Ascend |

#### Cache / Step Reduction

| Backend | Description |
|---|---|
| TeaCache | Caches intermediate activations across denoising steps |
| Cache-DiT | Block-level caching for transformer layers |

### Tensor Parallelism Constraints

The `validate_zimage_tp_constraints()` function (in `vllm_omni/diffusion/models/z_image/z_image_transformer.py`) enforces that:

- **Number of attention heads** must be divisible by the TP degree
- **FFN hidden dimension** must be divisible by the TP degree

Violating either constraint raises a `ValueError` at startup. See `tests/diffusion/models/z_image/test_zimage_tp_constraints.py` for examples.

### CFG Parallelism *(Under Development)*

Classifier-Free Guidance can be parallelized by splitting the conditional and unconditional forward passes across devices. The `get_cfg_group()` function in `vllm_omni/diffusion/distributed/parallel_state.py` defines the device group; the actual parallel forward pass is not yet wired up.

---

## 6. Omni Connector Layer

Connectors are the typed I/O channels between stages.

### Class Hierarchy

```
OmniConnector (ABC)
├── QueueConnector
├── ZMQConnector
└── KVTransferConnector
        └── (future) NCCLConnector
        └── (future) UCXConnector
```

### Connector Interface

Every connector must implement:

```python
class OmniConnector(ABC):
    async def setup(self) -> None: ...
    async def send(self, data: OmniPayload) -> None: ...
    async def recv(self) -> OmniPayload: ...
    async def teardown(self) -> None: ...
```

### D2H2D vs D2D (Future)

Current connectors copy tensors **Device → Host → Device** (D2H2D). For large diffusion latents this is a significant bottleneck. The roadmap item is to implement direct **Device → Device** (D2D) transfer using:

- **NCCL** for same-cluster GPU-to-GPU
- **UCX** for cross-node RDMA
- **IPC** (shared memory) for same-host different-process

---

## 7. Scheduling

### AR Scheduling — `OmniARScheduler`

Extends vLLM's standard continuous batching scheduler with:

- **KV cache transfer coordination** — tracks which blocks need migration between stages
- **Multimodal token budgeting** — accounts for variable-length visual tokens in the sequence budget

### Diffusion Scheduling — `OmniGenerationScheduler`

A simpler scheduler designed for the non-autoregressive nature of diffusion:

- Collects requests up to a configurable batch size
- Dispatches the full batch to `GPUWorker.execute_model()` for the complete denoising loop
- Currently processes **one request at a time** (batching is a tracked TODO)

---

## 8. Distributed & Parallel Strategies

### AR Models

Inherits vLLM's existing strategies:

| Strategy | Config Key | Description |
|---|---|---|
| Tensor Parallelism | `tensor_parallel_size` | Split weight matrices across GPUs |
| Pipeline Parallelism | `pipeline_parallel_size` | Split layers across GPUs |
| Prefix Caching | `enable_prefix_caching` | Reuse KV cache for common prefixes |

### Diffusion Models

Additional strategies specific to DiT:

| Strategy | Config Key | Description |
|---|---|---|
| Ulysses SP | `ulysses_degree` | Sequence-parallel attention (heads split) |
| Ring SP | `ring_degree` | Ring-based sequence parallelism |
| CFG Parallelism | *(coming soon)* | Parallel conditional/unconditional passes |

---

## 9. Entrypoints & APIs

### CLI

```
vllm-omni serve  --stage_configs_path stages.yaml  [vLLM args]
vllm-omni bench  --config bench.yaml
```

Entry point: `vllm_omni/entrypoints/cli/main.py` → dispatches to `OmniServeCommand` or `OmniBenchCommand`.

### HTTP Endpoints

| Endpoint | Description | Extra Body Fields |
|---|---|---|
| `POST /v1/chat/completions` | Multimodal chat (text + image/audio input, text + audio output) | `extra_body` for omni-specific params |
| `POST /v1/audio/speech` | TTS synthesis (Qwen3-TTS and compatible models) | `voice`, `response_format`, model-specific fields |

> See `docs/cli/bench/serve.md` and `examples/online_serving/qwen3_tts/README.md` for full parameter lists.

---

## 10. Adding a New Omni-Modality Model

Follow these steps to register a new model with vLLM-Omni:

### Step 1 — Implement the model class

Place the model in the appropriate module:
- `vllm_omni/models/ar/` for autoregressive models
- `vllm_omni/diffusion/models/` for diffusion models

Inherit from the relevant base class and implement `forward()`.

### Step 2 — Register input/output processors

If the model requires custom tokenization or embedding logic, subclass `OmniInputProcessor` and register it.

For custom output decoding (e.g., audio codec), subclass `MultimodalOutputProcessor`.

### Step 3 — Add a stage config example

Add a YAML file under `examples/` demonstrating the `stage_configs_path` for the new model.

### Step 4 — Add tests

- Unit tests in `tests/models/<model_name>/`
- Integration tests that call `modify_stage_config()` from `tests/conftest.py` to parametrize hardware combinations

### Step 5 — Document it

Add a README under `examples/online_serving/<model_name>/` following the pattern of `examples/online_serving/qwen3_tts/README.md`.

---

## Appendix: Key Files at a Glance

| File | Purpose |
|---|---|
| `vllm_omni/entrypoints/omni.py` | Top-level engine entrypoint |
| `vllm_omni/entrypoints/cli/serve.py` | `OmniServeCommand` |
| `vllm_omni/engine/input_processor.py` | `OmniInputProcessor` |
| `vllm_omni/engine/output_processor.py` | `MultimodalOutputProcessor` |
| `vllm_omni/core/sched/omni_ar_scheduler.py` | `OmniARScheduler`, `KVCacheTransferData` |
| `vllm_omni/core/sched/omni_generation_scheduler.py` | `OmniGenerationScheduler` |
| `vllm_omni/distributed/omni_connectors/` | All connector implementations |
| `vllm_omni/diffusion/distributed/parallel_state.py` | `get_cfg_group()` |
| `vllm_omni/diffusion/data.py` | `ulysses_degree`, `ring_degree` config |
| `tests/conftest.py` | `modify_stage_config()`, test fixtures |

# Async Chunk

## Table of Contents

1. [Overview](#overview)
2. [Performance](#performance)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Stage Input Processor Contract](#stage-input-processor-contract)
6. [Related Files](#related-files)

## Overview

The `async_chunk` feature enables asynchronous, chunked processing of data across multiple stages in a multi-stage pipeline (e.g., Qwen3-Omni with Thinker → Talker → Code2Wav stages). Instead of waiting for a complete stage output before forwarding to the next stage, this feature allows stages to process and forward data in chunks as it becomes available, significantly reducing latency and improving throughput.

**Chunk Size Definition**

- **Prefill Phase**: `chunk_size = num_scheduled_tokens` for chunked prefill processing
- **Decode Phase**: `chunk_size = num_scheduled_tokens = 1 ` for per-token streaming

For qwen3-omni:
- **Thinker → Talker**: Per decode step (typically chunk_size=1)
- **Talker → Code2Wav**: Accumulated to `codec_chunk_frames` (default=25) before sending. During the initial phase, a dynamic initial chunk size (IC) is automatically selected based on server load to reduce TTFP. Use the per-request `initial_codec_chunk_frames` API field to override.
- **Code2Wav**: Streaming decode with code2wav chunk_size

With `async_chunk`:
- Stages can start processing as soon as chunks are available
- Overlapping execution across stages
- Reduced latency and improved throughput
- Better resource utilization
- Async scheduling: Chunk IO (get/put) overlaps with compute via background threads so the scheduler is not blocked waiting for chunks

## Performance
1. **Reduced Latency**: Next stage can start processing immediately
2. **Streaming Support**: Enables streaming for audio generation
3. **IO-Compute Overlap**: Chunk retrieval happens asynchronously while other requests compute
4. **Non-blocking Scheduler**: Requests waiting for chunks don't block the entire scheduler
5. **Code2Wav Batch Inference**: Supports batched processing in code2wav stage

| Input     | Output           | Async_chunk enabled | Code2Wav batch size | Max_Concurrency | Prompts | Mean E2E   | Mean TTFT  | Mean TPOT | Mean TTFP   | Mean RTF | Mean ITL |
|-----------|------------------|---------------------|---------------------|----------------|---------|------------|------------|-----------|-------------|----------|----------|
| text 100  | text 100+audio   | False               | 1                   | 1              | 50      | 6581.80    | 43.22      | 8.31      | 6459.34     | 0.24     | 8.22     |
| text 100  | text 100+audio   | False               | 1                   | 4              | 50      | 7398.63    | 67.57      | 9.14      | 7285.35     | 0.27     | 9.05     |
| text 100  | text 100+audio   | False               | 1                   | 10             | 50      | 13522.99   | 131.82     | 12.72     | 13410.44    | 0.49     | 12.60    |
| text 100  | text 100+audio   | False               | 64                  | 1              | 50      | 6505.13    | 43.14      | 8.52      | 6395.40     | 0.24     | 8.44     |
| text 100  | text 100+audio   | False               | 64                  | 4              | 50      | 7668.15    | 51.15      | 9.36      | 7562.37     | 0.28     | 9.27     |
| text 100  | text 100+audio   | False               | 64                  | 10             | 50      | 9516.18    | 138.06     | 14.75     | 9409.26     | 0.34     | 14.60    |
| text 100  | text 100+audio   | True                | 1                   | 1              | 50      | 6179.79    | 44.58      | 8.69      | 522.99      | 0.22     | 8.60     |
| text 100  | text 100+audio   | True                | 1                   | 4              | 50      | 7692.69    | 103.96     | 10.22     | 785.85      | 0.29     | 10.12    |
| text 100  | text 100+audio   | True                | 1                   | 10             | 50      | 11152.71   | 685.60     | 17.64     | 1628.88     | 0.41     | 17.62    |


Performance data collected on H800 GPUs through comprehensive benchmarking with cudagraph enabled. text input uses random dataset.

Enabling **async_chunk** (False→True) sharply reduces time-to-first-audio (TTFP)—e.g. ~92% at concurrency 1 (6.5s→0.52s)—and improves E2E latency (e.g. ~6% at conc 1, ~17% at conc 10). RTF (Real Time Factor) also improves with async_chunk on (e.g. ~8% at conc 1: 0.24→0.22, ~16% at conc 10: 0.49→0.41). Enabling **Code2Wav batch size 64** (vs 1) improves E2E and TTFP at higher concurrency when async_chunk is off (e.g. ~30% at conc 10: 13.5s→9.5s E2E, 13.4s→9.4s TTFP).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_ttfp_performance.png">
    <img alt="TTFP Performance Data Comparison" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_ttfp_performance.png" width=100%>
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_rtf_performance.png">
    <img alt="RTF Performance Data Comparison" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_rtf_performance.png" width=100%>
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_e2e_performance.png">
    <img alt="E2E Performance Data Comparison" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_e2e_performance.png" width=100%>
  </picture>
</p>

## Architecture

### Async Chunk Pipeline Overview

The following diagram illustrates the **Async Chunk Architecture** for multi-stage models (e.g., Qwen3-Omni with Thinker → Talker → Code2Wav), showing how data flows through the 4-stage pipeline with parallel processing and dual-stream output:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-async-chunk.png">
    <img alt="Async Chunk Pipeline Architecture" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-async-chunk.png" width=100%>
  </picture>
</p>

**Diagram Legend:**

| Step | Stage Type | Description |
|------|-----------|------------|
| `prefill` | Initialization | Context processing, KV cache initialization |
| `decode` | Autoregressive | Token-by-token generation in AR stages |
| `codes` | Audio Encoding | RVQ codec codes from Talker stage |
| `output` | Final Output | Text chunks or audio waveforms |

### Data Flow

#### Stage 0: Thinker (Multimodal Understanding + Text Generation)
- **Prefill**: Processes multimodal input (text/image/audio/video), initializes KV cache
- **Decode Loop**: Generates text tokens autoregressively
- **Chunk Triggers**: Each decode step (typically `chunk_size=1`) can trigger downstream processing
- **Dual Output**:
  - **Text Stream**: `text_0`, `text_1`, `text_2`... `text_n` streamed to output
  - **Hidden States**: Passed to Talker stage for audio synthesis

#### Stage 1: Talker (Text → RVQ Audio Codes)
- **Prefill**: Receives hidden states from Thinker as semantic condition
- **Decode Loop**: Generates RVQ codec codes autoregressively
- **Accumulation**: Codes accumulate to `codec_chunk_frames` (default=25) before forwarding
- **Dynamic IC**: Initial chunk size auto-selected based on server load to optimize TTFP
- **Output**: `codes` blocks (chunk 0, 1, ... n) sent to Code2Wav

#### Stage 2: Code2Wav (Vocoder Decoder)
- **Non-Autoregressive**: Processes RVQ codes in parallel batches
- **Streaming Decode**: Converts codes to audio waveforms chunk-by-chunk
- **Batching**: Supports batched inference for multiple concurrent requests
- **Output**: Audio segments `audio_0`, `audio_1`, ... `audio_n`

#### Stage 3: Output (Dual Stream)
- **Text Streaming**: `text_0` → `text_1` → `text_2` → ... (user sees response in real-time)
- **Audio Streaming**: `audio_0` → `audio_1` → ... (user hears audio progressively)

### Execution Timeline

```
Timeline: Parallel vs Sequential

Sequential (async_chunk=false):
[Thinker: ████████████████████]  (2.0s)
                            [Talker: ████████████████████]  (3.0s)
                                                        [Code2Wav: ████]  (1.0s)
Total: 6.0s, TTFP: 6.0s

Async Chunk (async_chunk=true):
[Thinker: ████░░░░████░░░░████]  (2.0s, streaming)
     [Talker: ░░████░░░░████░░]  (3.0s, parallel)
         [Code2Wav: ░░░░████░░]  (1.0s, batched)
Total: ~3.5s, TTFP: ~0.5s

█ = Active computation  ░ = Waiting/idle
```

#### Sequential Flow (for comparison)
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-non-async-chunk.png">
    <img alt="Sequential Data Flow" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-non-async-chunk.png" width=100%>
  </picture>
</p>

In sequential mode, each stage must wait for the previous stage to complete entirely before starting.

### Async Chunk System Architecture
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/async-chunk-architecture.png">
    <img alt="Async Chunk Architecture" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/async-chunk-architecture.png" width=100%>
  </picture>
</p>


### Key Components

1. **OmniConnector**: Inter-stage data transport only
   - Shared memory or other IPC mechanisms
   - **Transport-only API**: `put(from_stage, to_stage, put_key, data)` and `get(from_stage, to_stage, get_key)` (optionally with timeout)
   - **No request-specific state**: Connector does not track put_requests, get_requests, request_payload, finished_requests, or other request-bound metadata; it only performs put/get operations
   - Chunk keys and request/chunk lifecycle are managed by **OmniChunkTransferAdapter**

2. **Transfer Adapter Layer**: Extensible abstraction for managing data transfer via connectors
   - **OmniTransferAdapterBase**: Base class with background **recv_loop** and **save_loop** threads;
   - **OmniChunkTransferAdapter**: Chunk-specific implementation that owns the full chunk lifecycle when async_chunk is enabled
     - **Chunk ID and key construction**: Builds keys like `{req_id}_{stage_id}_{chunk_id}` for put/get
     - **Async get**: `load_async(request)` enqueues the request; background **recv_loop** polls the connector (non-blocking); when data is available, updates the request and marks it in `_finished_load_reqs`; scheduler calls `get_finished_requests()` to learn which requests have chunks ready
     - **Async put**: `save_async(pooling_output, request)` invokes `custom_process_next_stage_input_func` in the main thread to build the payload, then enqueues a save task; background **save_loop** performs `connector.put()`; payload processing and chunk accumulation (e.g. code2wav chunk_size) remain in the main thread

3. **Stage Input Processors**: Custom functions that process stage outputs into chunks for different models
   - Receive **transfer_manager** (OmniChunkTransferAdapter)
   - Qwen3-omni reference: `thinker2talker_async_chunk`, `talker2code2wav_async_chunk`

4. **Schedulers**: Modified to handle chunk-based scheduling with async IO-compute overlap
   - `OmniARScheduler`: For autoregressive stages
   - `OmniGenerationScheduler`: For generation stages
   - Both schedulers use **OmniChunkTransferAdapter** and **before/after** hooks around `super().schedule()`:
     - **Before** `super().schedule()`: `process_pending_chunks(waiting, running)` moves requests waiting for chunks to `WAITING_FOR_CHUNK`, enqueues load tasks for background polling
     - **After** `super().schedule()`: `restore_queues(waiting, running)` restores requests with ready chunks back to waiting/running, `postprocess_scheduler_output(scheduler_output)` attaches cached additional_information, clears chunk-ready flags
   - **put_chunk** `save_async(pooler_output, request)`; **get_chunk** / **get_chunk_for_generation** `load_async(request)`

5. **Model Runners**: Handle chunk processing
   - `OmniGPUModelRunner`: Processes chunks in AR stages
   - `GPUGenerationModelRunner`: Processes chunks in generation stages
     - Uses `ubatch_slices` from `get_forward_context()` to track per-request sequence lengths in batched inference
     - Reuses `ubatch_slices_padded` for code2wav batching to properly split batch outputs
     - Handles list-type multimodal outputs: iterates through requests and assigns corresponding tensor to each
     - Improved request state management: removes unscheduled and finished requests from input batch

6. **Model Implementation**: Model-specific chunk handling
   - `Qwen3OmniMoeForConditionalGeneration`: Main model with async_chunk support
     - **Code2Wav stage batching**: Uses `ubatch_slices` to construct batched codec codes tensor `[batch_size, 16, max_seq_len]`
     - **Batch output handling**: `generate_audio()` returns `list[torch.Tensor]`, one audio tensor per request
     - **Multimodal outputs**: Returns list of audio tensors for batch processing instead of single concatenated tensor
   - `Qwen3OmniCode2WavDecoder`: Audio generation model
     - `chunked_decode()` and `chunked_decode_streaming()`: Return `list[torch.Tensor]` (one per request)
     - Uses `ubatch_slices` to split batched waveform output into per-request audio chunks
     - Each request gets correctly sized audio based on its code sequence length: `waveform[:, :, :code_seq_len * total_upsample]`

7. **Request status**: `RequestStatus.WAITING_FOR_CHUNK` is added via patch (e.g. in `vllm_omni/patch.py`) so requests waiting for a chunk are not scheduled by the base vLLM scheduler until the chunk is ready.

## Configuration

Enable async chunk mode in the deploy YAML:

```yaml
async_chunk: true
```

The registered `PipelineConfig` owns the async handoff processor functions.
For Qwen3-Omni these are `thinker2talker_async_chunk` and
`talker2code2wav_async_chunk`; deploy YAML only selects the mode and runtime
sizing.

### Configuration ownership

- `async_chunk: bool`: Enable/disable async chunk mode
- `async_chunk_process_next_stage_input_func: str`: Pipeline-owned chunk processor path
- `execution_type: StageExecutionType`: Pipeline-owned runtime family
- `connectors: dict`: Deploy-owned connector definitions
- `max_num_seqs: int`: Maximum number of sequences for concurrent processing in the stage

### Connector Configuration

```yaml
connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector

stages:
  - stage_id: 0
    output_connectors:
      to_stage_1: connector_of_shared_memory
  - stage_id: 1
    input_connectors:
      from_stage_0: connector_of_shared_memory
```

### Code2Wav Batch Configuration

For optimal performance with async_chunk, the code2wav stage should be configured with batching:

```yaml
stages:
  - stage_id: 2  # code2wav stage
    devices: "1"
    max_num_seqs: 64  # Enables batched audio generation
```

## Stage Input Processor Contract

RFC #4872 standardized the per-model builders that convert one stage's outputs
into the next stage's inputs. Each builder belongs to exactly one *role*
(consumer-side vs producer-side) and is dispatched by the runtime according to
the `async_chunk` mode. The naming convention and the validation rules below
are load-bearing: the processor registry infers a builder's kind from its name
suffix and enforces the matching signature at startup.

### Naming convention and roles

| Suffix | Registry kind | Runs where | Role |
|--------|---------------|------------|------|
| `*_full_payload` | `producer_full_payload` | Worker (producer-side) | Packs the accumulated stage output into an `OmniPayload` and ships it through the connector (`FullPayloadProducer`) |
| `*_async_chunk` | `producer_async_chunk` | Scheduler (producer-side) | Streams one chunk per call while `async_chunk` is enabled (`AsyncChunkProducer`) |
| `*_token_only` | `placeholder_prompt_builder` | Orchestrator (consumer-side) | Allocates downstream KV slots only; bulk tensors arrive via the connector (`PlaceholderPromptBuilder`) |
| (no suffix, legacy) | `legacy_orchestrator_builder` | Orchestrator (consumer-side) | Legacy sync builder — a placeholder or diffusion input builder, adapted through `wrap_orchestrator_processor` |

Diffusion-stage edges use dedicated suffixes
(`ar2diffusion` / `ar2dit` / `thinker2imagegen` / ...) mapped to
`diffusion_input_builder`, and `moss_tts.talker2codec` keeps the legacy
`legacy_multi_source` shape. The canonical naming documentation lives in the
`vllm_omni/model_executor/stage_input_processors/__init__.py` module docstring.

### Minimal processor set per edge

A new model that must work in **both** modes implements, for each inter-stage
edge, the three processors below:

| Mode | Producer-side | Consumer-side (orchestrator) |
|------|---------------|------------------------------|
| `async_chunk=false` | `*_full_payload` (`FullPayloadProducer`) | `*_token_only` (`PlaceholderPromptBuilder`) |
| `async_chunk=true` | `*_async_chunk` (`AsyncChunkProducer`) | `*_token_only` (prewarm via `build_prewarm_placeholder`) |

The minimal union set is `*_full_payload`, `*_async_chunk` and `*_token_only`.
The `*_token_only` placeholder builder is required in both modes: in non-async
mode it builds the forward placeholder, and in async mode the orchestrator
reuses it to prewarm the downstream stage.

### OrchestratorInputContext and the C1 contract

Every orchestrator-facing builder is invoked under the fixed C1 contract
`(source_outputs, ctx)`:

```python
from vllm_omni.model_executor.stage_input_processors import OrchestratorInputContext


def my_token_only(
    source_outputs: list[Any],
    ctx: OrchestratorInputContext,
) -> list[OmniTokensPrompt]:
    """Upstream outputs -> next-stage token prompts (C1 contract)."""
    ...
```

`OrchestratorInputContext` carries the transition metadata and deliberately has
no `model_config` field (a processor that needs the model config reads it
through the upstream stage closure, never through this context):

```python
@dataclass(frozen=True)
class OrchestratorInputContext:
    prompt: Any | None = None
    requires_multimodal_data: bool = False
    streaming_context: Any | None = None
    sampling_params: Any | None = None
```

Processors that already accept `ctx` are used unchanged. Legacy positional
shapes (C0 3-arg, C2 placeholder with `streaming_context`, C3 diffusion with
`sampling_params`, C4 `moss_tts.talker2codec` multi-source) are adapted by
`wrap_orchestrator_processor` / `invoke_orchestrator_processor` and emit a
`DeprecationWarning`.

### Registry and startup validation

The registry performs **signature-level structural checks only** — it never
executes processor logic and never loads model weights. Kind inference is
name-driven (suffix-based):

- `register_processor(path, kind)` — manual kind override for names that do not
  follow the suffix convention (escape hatch; overrides are validated eagerly).
- `infer_kind(fn, *, path)` — suffix rules: `_token_only` ->
  `placeholder_prompt_builder`, `_full_payload` / `_batch` ->
  `producer_full_payload`, `_async_chunk` -> `producer_async_chunk`, diffusion
  suffixes -> `diffusion_input_builder`, no suffix ->
  `legacy_orchestrator_builder`, `moss_tts.talker2codec` ->
  `legacy_multi_source`.
- `validate_processor(fn, *, kind, path, stage_config=None)` — hard contract
  violations raise `ProcessorValidationError`; soft mismatches emit a
  `RuntimeWarning`.
- `resolve_processor(path, *, expected_kind=None, stage_config=None)` — the
  drop-in replacement for the legacy `getattr(importlib.import_module(...), ...)`
  lookups: imports, infers, validates, optionally checks `expected_kind`, and
  returns a `ProcessorSpec` whose `fn` is the same callable the legacy lookup
  produced.

### P8b dual entry: forward and prewarm placeholders

The `*_token_only` placeholder builder is exposed through two entry points so
the sync forward path and the async-chunk prewarm path share the same `_common`
length / packing helpers:

- `build_forward_placeholder(source_outputs, ctx)` — the non-async forward
  path. One placeholder `OmniTokensPrompt` per upstream output, sized by
  `_common.compute_placeholder_prompt_len(mode="full")`.
- `build_prewarm_placeholder(*, stage0_prompt, ctx, downstream_stage_id)` —
  async-chunk mode has no upstream `source_outputs` yet at prewarm time, so the
  length is a best-effort estimate from the stage-0 input prompt
  (`mode="stage0_only"`, i.e. `len(stage0_prompt)`). The connector fixup path
  (`adapter.construct_next_stage_streaming_input_prompt`) replaces the estimate
  with the real length once the upstream chunk arrives.

The orchestrator routes prewarm through `_prewarm_async_chunk_stages`, which
reads `build_prewarm_placeholder` off the resolved `*_token_only` function
object (both builders are attached as attributes on the exported function).

### Three-state async gate

The orchestrator only forwards via `process_engine_inputs` when the transition
is not served by the async-chunk data plane:

```python
if (
    (finished or segment_finished)
    and stage_id < req_state.final_stage_id
    and (not self.async_chunk or not self._stage_receives_async_chunks(stage_id + 1))
    and (not self._next_stage_already_submitted(stage_id, req_state) or req_state.streaming.enabled)
):
```

`_stage_receives_async_chunks(stage_id)` reports whether a stage's connector
supplies its runtime inputs. When `async_chunk=true` and the downstream stage
receives async chunks, the orchestrator skips `process_engine_inputs` for that
transition, so the corresponding input processor may be dead for the duration
of the request. `dead_processor_hint` is a pure decision helper (for warnings
and tests only); the runtime is warn-first in the M0 phase.

### Producer keyword contract

Producer-side builders never receive an `OrchestratorInputContext`; their
keyword-only parameters are load-bearing parts of the connector data plane and
must not be renamed or made positional:

```python
def my_full_payload(
    *,
    transfer_manager: Any,
    pooling_output: Any,
    request: Any,
    is_finished: bool = ...,  # optional: the worker retries without it
) -> Any: ...


def my_async_chunk(
    *,
    transfer_manager: Any,
    multimodal_output: Any,
    request: Any,
    is_finished: bool = False,  # required: the scheduler always passes it
) -> Any: ...
```

`pooling_output` (full payload) and `multimodal_output` (async chunk) are
cross-checked by `validate_processor`; using the wrong keyword name for a kind
is flagged as a structural warning.

## Related Files

- `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py`: Chunk processing functions (receive `transfer_manager` as first param)
- `vllm_omni/distributed/omni_connectors/transfer_adapter/base.py`: OmniTransferAdapterBase (recv_loop, save_loop, load_async, save_async)
- `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py`: OmniChunkTransferAdapter (process_pending_chunks, restore_queues, postprocess_scheduler_output)
- `vllm_omni/distributed/omni_connectors/connectors/shm_connector.py`: SharedMemoryConnector (transport-only put/get)
- `vllm_omni/core/sched/omni_ar_scheduler.py`: AR scheduler with chunk_transfer_adapter
- `vllm_omni/core/sched/omni_generation_scheduler.py`: Generation scheduler with same async chunk pattern
- `vllm_omni/worker/gpu_model_runner.py`: Model runner with chunk handling
- `vllm_omni/worker/gpu_generation_model_runner.py`: Generation model runner with batch output handling and ubatch_slices support
- `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py`: Model implementation with code2wav batching
- `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py`: Code2wav decoder with batch support
- `vllm_omni/engine/arg_utils.py`: Configuration definitions (async_chunk, worker_type)
- `vllm_omni/config/model.py`: Model config with async_chunk field

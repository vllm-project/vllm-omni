# Async Chunk Design

## Table of Contents

1. [Overview](#overview)
2. [Performance](#performance)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Related Files](#related-files)

## Overview

The `async_chunk` feature enables asynchronous, chunked processing of data across multiple stages in a multi-stage pipeline (e.g., Qwen3-Omni with Thinker → Talker → Code2Wav stages). Instead of waiting for a complete stage output before forwarding to the next stage, this feature allows stages to process and forward data in chunks as it becomes available, significantly reducing latency and improving throughput.

**Chunk Size Definition**

- **Prefill Phase**: `chunk_size = num_scheduled_tokens` for chunked prefill processing
- **Decode Phase**: `chunk_size = num_scheduled_tokens = 1 ` for per-token streaming

For qwen3-omni:
- **Thinker → Talker**: Per decode step (typically chunk_size=1)
- **Talker → Code2Wav**: Accumulated to code2wav chunk_size(default=25, current only support default, will support chunk_size soon) before sending
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

| Input/Output | Async_chunk enabled | cuda graph | Max_Concurrency | Prompts | Mean E2E | TTFT | TPOT | TTFP | RTF | ITL |
|--------------|--------------------|------------|-----------------|---------|----------|------|------|------|-----|-----|
| text 2500/ text 900 + Audio2048 | False | False | 1 | 10 | 169670.74 | 222.30 | 0.42 | 169468.94 | 1.04 | 40.77 |
| text 2500/ text 900 + Audio2048 | False | False | 4 | 40 | 179178.47 | 495.05 | 0.45 | 178973.77 | 1.09 | 44.92 |
| text 2500/ text 900 + Audio2048 | False | False | 8 | 80 | 202325.90 | 912.31 | 0.51 | 202122.80 | 1.24 | 50.40 |
| mix_modality/text 900 + Audio2048 | False | False | 1 | 10 | 174105.55 | 2708.65 | 0.42 | 173911.15 | 1.06 | 42.27 |
| mix_modality/text 900 + Audio2048 | False | False | 4 | 40 | 179145.45 | 563.60 | 0.44 | 178944.69 | 1.09 | 45.59 |
| mix_modality/text 900 + Audio2048 | False | False | 8 | 80 | 202338.01 | 1196.76 | 0.51 | 202137.49 | 1.24 | 51.05 |
| text 2500/ text 900 + Audio 2048 | True | False | 1 | 10 | 133524.14 | 230.06 | 0.33 | 2187.38 | 0.82 | 40.54 |
| text 2500/ text 900 + Audio 2048 | True | False | 4 | 40 | 169980.54 | 818.90 | 0.42 | 126714.64 | 1.05 | 41.84 |
| text 2500/ text 900 + Audio 2048 | True | False | 8 | 80 | 190142.91 | 1261.42 | 0.47 | 165481.64 | 1.17 | 44.06 |
| mix_modality/text 900 + Audio 2048 | True | False | 1 | 10 | 136287.53 | 2629.88 | 0.33 | 4577.68 | 0.84 | 41.22 |
| mix_modality/text 900 + Audio 2048 | True | False | 4 | 40 | 160287.43 | 981.72 | 0.39 | 119458.83 | 0.99 | 43.20 |
| mix_modality/text 900 + Audio 2048 | True | False | 8 | 80 | 190811.37 | 1432.88 | 0.47 | 166075.49 | 1.18 | 44.17 |
| text 2500/ text 900 + Audio 2048 | False | True | 1 | 10 | 30691.59 | 207.21 | 0.08 | 30490.14 | 0.19 | 8.86 |
| text 2500/ text 900 + Audio 2048 | False | True | 4 | 40 | 40214.43 | 431.38 | 0.10 | 40010.45 | 0.24 | 14.11 |
| text 2500/ text 900 + Audio 2048 | False | True | 8 | 80 | 52034.78 | 889.18 | 0.13 | 51840.75 | 0.32 | 20.61 |
| mix_modality/text 900 + Audio 2048 | False | True | 1 | 10 | 32894.34 | 2730.08 | 0.07 | 32689.03 | 0.20 | 8.78 |
| mix_modality/text 900 + Audio 2048 | False | True | 4 | 40 | 41223.70 | 570.17 | 0.10 | 41017.76 | 0.25 | 14.69 |
| mix_modality/text 900 + Audio 2048 | False | True | 8 | 80 | 52902.84 | 1143.91 | 0.14 | 52702.21 | 0.32 | 20.97 |
| text 2500/ text 900 + Audio 2048 | True | True | 1 | 10 | 30535.75 | 205.73 | 0.07 | 744.13 | 0.19 | 9.12 |
| text 2500/ text 900 + Audio2048 | True | True | 4 | 40 | 91392.10 | 792.43 | 0.22 | 67717.28 | 0.56 | 9.84 |
| text 2500/ text 900 + Audio2048 | True | True | 8 | 80 | 180537.21 | 1178.16 | 0.44 | 157043.02 | 1.11 | 10.59 |
| mix_modality/text 900 + Audio2048 | True | True | 1 | 10 | 33132.83 | 2668.88 | 0.08 | 3235.70 | 0.20 | 9.22 |
| mix_modality/text 900 + Audio2048 | True | True | 4 | 40 | 90883.91 | 859.90 | 0.22 | 67343.66 | 0.56 | 9.87 |
| mix_modality/text 900 + Audio2048 | True | True | 8 | 80 | 180765.18 | 1298.46 | 0.45 | 157264.30 | 1.12 | 10.54 |

Performance data collected on H800 GPUs through comprehensive benchmarking. text input uses random dataset. mix modality (1 image+1 video+1 audio) input uses random_mm dataset.

**async_chunk enables transformative results: CUDA Graph disabled achieves 98.7% TTFP reduction (169s→2.2s) + 21.3% E2E improvement; CUDA Graph enabled maintains 97.6% TTFP reduction (30s→0.7s)**

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_performance.png">
    <img alt="TTFP Performance Data Comparison" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_ttfp_performance.png" width=100%>
  </picture>
</p>

## Architecture
### Data Flow

#### Sequential Flow
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-non-async-chunk.png">
    <img alt="Data Flow between stages" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-non-async-chunk.png" width=100%>
  </picture>
</p>

#### Async Chunk Flow

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-async-chunk.png">
    <img alt="Data Flow between stages" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-async-chunk.png" width=100%>
  </picture>
</p>

### Async Chunk architecture
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

Enable async_chunk in stage configuration YAML:

```yaml
async_chunk: true
stage_args:
  - stage_id: 0
    engine_args:
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_async_chunk
  - stage_id: 1
    engine_args:
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_async_chunk
```

### Stage Configuration

- `async_chunk: bool`: Enable/disable async chunk mode
- `custom_process_next_stage_input_func: str`: Path to custom chunk processing function; receives `(transfer_manager, pooling_output, request)`. For qwen3-omni: `thinker2talker_async_chunk`, `talker2code2wav_async_chunk`
- `stage_connector_config: dict`: Connector configuration
- `worker_type: str`: Model type, e.g. `"ar"` or `"generation"` (used by OmniChunkTransferAdapter for mode-specific payload handling)
- `max_batch_size: int`: Maximum batch size for the stage


### Connector Configuration

```yaml
connectors:
  - from_stage: 0
    to_stage: 1
    spec:
      name: SharedMemoryConnector
      extra:
        stage_id: 0
```

### Code2Wav Batch Configuration

For optimal performance with async_chunk, the code2wav stage should be configured with batching:

```yaml
stage_args:
  - stage_id: 2  # code2wav stage
    runtime:
      devices: "1"
      max_batch_size: 64  # Enables batched audio generation
    engine_args:
      model_stage: code2wav
```

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

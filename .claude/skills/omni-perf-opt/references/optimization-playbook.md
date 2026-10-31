# Omni Optimization Playbook

Levers are grouped by layer, roughly in the order they are worth trying.
Every knob below exists in-tree at the time of writing; confirm the name in
the deploy YAML or model code before use, because model-specific connector
`extra` keys are only honored by the model that reads them.

## Contents
- Placement and memory
- Stage batching
- CUDA graphs
- Streaming hand-off (async_chunk and codec chunking)
- Model runner V2
- Talker and code predictor
- Code2wav decoder
- Orchestrator and transport
- Precision and quantization
- Candidate library

## Placement and memory

Knobs (deploy YAML, per stage): `devices`, `gpu_memory_utilization`
(fraction of total GPU memory; per-GPU sum ≤ 1.0), `num_replicas`,
`tensor_parallel_size`, `kv_cache_memory_bytes`, `max_model_len`.

- Start with the thinker alone on one GPU and talker plus code2wav on
  another (`qwen3_omni_moe.yaml`, `minicpmo_4_5_2gpu.yaml`).
- Scale the talker with `num_replicas` before raising thinker batch limits;
  device count must equal replicas × per-replica world size
  (`minicpmo_4_5_{3,4}gpu_stage1_replicas.yaml`).
- Single-GPU co-location is possible (`qwen3_omni_moe_mrv2_h200.yaml`:
  0.60 / 0.22 / 0.10) and should run under NVIDIA MPS.
- Smaller cards: thinker TP and a lower `max_model_len`
  (`minicpmo_4_5_8x4090.yaml`).

Evidence: per-stage utilization, OOM at load or first request, queueing at
one stage in `--log-stats`. Validate memory headroom at the target
concurrency, not just at startup.

## Stage batching

Knobs: per-stage `max_num_seqs`, `max_num_batched_tokens`,
`async_scheduling`.

- Thinker and talker use continuous batching; code2wav batches per step.
  Batching alone was the first large gain in the Qwen3-Omni study
  (`docs/design/qwen3_omni_tts_performance_optimization.md`).
- MiniCPM-o turn MRv2 sets tts and code2wav `max_num_seqs` to 8 (16 on
  H200); raise only with matching KV (`kv_cache_memory_bytes`) and graph
  sizes.
- Watch for RTF rising with concurrency while GPU time per step stays flat:
  that is under-batching or a single-replica bottleneck.

## CUDA graphs

- Outer graphs: `enforce_eager: false` per stage. Qwen3-Omni code2wav uses
  its inner graph on CUDA; ROCm forces stage 2 eager (MIOpen
  `conv_transpose1d` is not capture-safe).
- Thinker encoders:
  - Qwen3-Omni image encoder graph (`qwen3_omni/vision_encoder_cudagraph.py`)
    is enabled automatically when eligible (FA/Triton attention, first PP
    rank, no FP8 ViT attention, image limit > 0).
  - MiniCPM-o encoder graphs through `--hf-overrides`:
    `encoder_cuda_graph`, `encoder_cuda_graph_max_graphs` (default 4),
    `encoder_cuda_graph_min_capture_calls`, `encoder_cuda_graph_min_free_bytes`,
    `encoder_cuda_graph_share_pools` (`docs/design/minicpmo_encoder_cuda_graph.md`).
    Duplex H200 adds `vision_cuda_graph`, `vision_fused_layers`,
    `duplex_audio_encoder_cuda_graph`.
- Code2wav graph buckets: Qwen3-Omni `codec_graph_batch_sizes`; MiniCPM-o
  `hift_graph_capture_batch_sizes`, `cfm_encoder_cuda_graph`,
  `cfm_max_graphs`.

Graph caches have fixed budgets; when full, other shapes run eagerly. After
changing batch sizes or caps, read startup capture logs and the precapture
budget warning. Always compare with `--enforce-eager` for correctness.

## Streaming hand-off (async_chunk and codec chunking)

Knobs: top-level `async_chunk` (or `--async-chunk` / `--no-async-chunk`),
per-stage `async_chunk: false` opt-out, and connector `extra`:

| Key | Effect | Model |
|-----|--------|-------|
| `initial_codec_chunk_frames` | Size of the first code2wav chunk; smaller means earlier first audio | Qwen3-Omni |
| `codec_chunk_frames` | Steady-state chunk size; larger is more efficient, smaller is smoother | Qwen3-Omni, MiniCPM-o |
| `codec_left_context_frames` | Re-decoded context before each chunk; prevents boundary clicks, costs compute | Qwen3-Omni, MiniCPM-o |
| `codec_chunk_ramp` | Chunk size schedule, e.g. `[1, 2, 4, 8, 16, 25]` | Qwen3-Omni MRv2 |
| `talker_first_audio` | Talker emits the first frame itself (CUDA, single-rank talker, one-frame first chunk, prefix caching off); `VLLM_OMNI_TALKER_FIRST_AUDIO=0` disables | Qwen3-Omni MRv2 |
| `generation_first_chunk_express`, `generation_first_chunk_express_min_slack_s` | Decode waiting first chunks in their own step while other streams keep enough buffered audio | MiniCPM-o turn MRv2 |

Trade-off: `async_chunk` cuts AUDIO_TTFP by an order of magnitude but adds
chunk transport and left-context overhead, so E2EL and RTF can rise.
Validate smoothness with `audio_continuity_ok_rate`, not mean RTF.

## Model runner V2

Knob: `model_runner: v2` (top-level or per stage). Ready-made overlays:
`qwen3_omni_moe_mrv2.yaml`, `qwen3_omni_moe_mrv2_h200.yaml`,
`minicpmo_4_5_turn_mrv2.yaml` (+ `_h200`), `minicpmo_4_5_duplex_mrv2.yaml`.

- CUDA only; overlays fall back to V1 on NPU, XPU, ROCm, and MUSA.
- Gains come from keeping sampler and EOS state on device (no per-step D2H
  wait), native data plane hand-off, and FULL decode graphs.
- Output is not bitwise identical to V1. MRv2 downstream stages support turn
  mode; duplex needs the dedicated overlay and separate measurement.
- Enabling MRv2 for a new model is P2: it needs
  `supports_native_mrv2_data_plane=True` in the pipeline and a full parity
  pass (see `add-omni-model`).

## Talker and code predictor

Qwen3-Omni connector `extra` (H200 profile):
`code_predictor_kv_cache`, `code_predictor_short_kv`,
`code_predictor_fused_sampling`. Short KV and fused sampling change rounding;
keep a control with `code_predictor_fused_sampling: false` when P99 is the
primary requirement.

General patterns from the Qwen3-TTS study: re-prefill a tiny code predictor
statelessly instead of paging its KV, and `torch.compile` the predictor
forward. Remove `.item()`, `.cpu()`, and tensor-valued Python branches from
per-step loops.

## Code2wav decoder

| Knob | Model | Notes |
|------|-------|-------|
| `codec_fused_snake` | Qwen3-Omni | Fused decoder blocks; CUDA opt-in |
| `codec_cudnn_benchmark` | Qwen3-Omni | cuDNN algorithm search at warmup; startup cost, rounding changes |
| `token2wav_allow_tf32` / `MINICPMO_CODE2WAV_TF32` | MiniCPM-o | TF32 for CFM DiT GEMMs; HiFT stays FP32 |
| `cfm_fused_body` | MiniCPM-o | Defaults on under CUDA + TF32; set `false` to restore the original body |
| `cfm_slot_pool`, `cfm_row_offset_merge` | MiniCPM-o | Opt-in batching aids |

Graph replay must run under the same autocast context as capture (MiniCPM-o
regression test `test_decode_replays_encoder_graph_in_capture_autocast`).

## Orchestrator and transport

- Orchestrator: event-driven by default; `VLLM_OMNI_EVENT_DRIVEN_ORCH=0`
  selects the legacy 1 ms poll loop. Use `--enable-orch-monitor` when latency
  is high but GPUs idle.
- Connectors: `SharedMemoryConnector` on a single node; Mooncake, NIXL,
  Mori, and Yuanrong connectors for cross-device or cross-node transport
  (`distributed/omni_connectors/factory.py`). `--shm-threshold-bytes`
  (default 65536) sets the shared-memory threshold.
- Payload size: hand only the hidden layers and keys the next stage reads;
  oversized payloads show up as connector `tx_time_ms` and orchestrator
  saturation.
- Waits: `VLLM_OMNI_INPUT_WAIT_TIMEOUT_S`, `VLLM_OMNI_CONNECTOR_RECV_POLL_MS`.

## Precision and quantization

Thinker-only quantization (for example the ModelOpt Qwen3-Omni checkpoint)
keeps the audio stages in BF16. Route method selection, loading, and
quality checks to the `quantization` skill; treat any precision change as P2
with an accuracy and WER gate.

## Candidate library

| Priority | Layer | Candidate | Evidence | Validation |
|----------|-------|-----------|----------|------------|
| P0 | Measurement | Frozen scripts, warmup, concurrency sweep | Numbers drift between runs | Repeat-run spread |
| P0 | Placement | Thinker alone, talker + code2wav shared | Stage contention in `--log-stats` | Per-stage time, memory |
| P0 | Placement | Talker replicas | Talker queueing at high concurrency | RTF and P99 at target concurrency |
| P0 | Batching | Raise stage `max_num_seqs` with KV to match | RTF rises, per-step time flat | Throughput, memory, P99 |
| P0 | Streaming | Smaller first chunk / chunk ramp | High AUDIO_TTFP, good TTFT | TTFP, continuity, WER |
| P0 | Streaming | Chunk and left-context tuning | Gaps despite good RTF | `audio_continuity_ok_rate`, boundary clicks |
| P0/P1 | Runner | Existing MRv2 overlay | Host-bound decode, D2H waits | Parity versus V1, accuracy, P99 |
| P1 | Graphs | Encoder graphs, code2wav graph buckets | Launch-bound encoders or decoder | Graph vs eager parity, capture logs |
| P1 | Host | Remove per-step syncs and Python branches | Host-stack trace gaps | Same outputs, step time |
| P1 | Transport | Trim payload keys, co-locate | High connector in-flight time | Transfer timings, outputs unchanged |
| P2 | Kernels | Fused decoder, TF32, cuDNN search | Decoder dominates stage time | WER, waveform delta, P99 |
| P2 | Talker | Short KV, fused sampling | Code predictor dominates talker step | Codec ids, WER, P99 control |
| P2 | Precision | Quantized thinker | Thinker compute or memory bound | Accuracy, TTFT, memory |
| P2 | Runner | MRv2 for a new model | No existing overlay | Full parity ladder |

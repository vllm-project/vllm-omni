# MiniCPM-o 4.5 Stage 0 TTFT audit

Status: implemented for CPU/static validation; A3 profiling and performance
evidence are pending.

## Scored path

The competition Seed-TTS chat adapter constructs two text-only messages: a
system instruction and the target user text. `ref_audio` and `ref_text` stay in
the request's TTS extension fields for the downstream voice-conditioning path;
they are not placed in Stage 0 `multi_modal_data`.

Consequently, the scored concurrency-one TTFT path is:

```text
HTTP request
  -> chat-template rendering/tokenization
  -> Stage 0 input processing and queue
  -> Thinker text prefill
  -> logits/sampling/output processing
  -> first non-empty text SSE
```

It does not invoke the Stage 0 Whisper audio encoder, SigLIP vision encoder, or
resampler for the official Seed-TTS request. Optimizations to those components
remain important for Daily-Omni, Video-MME, speech input, and full-duplex
workloads, but they cannot be presented as scored Seed-TTS TTFT improvements.

The pinned challenge baseline already has:

- Stage 0 `PIECEWISE` Graph requested by the NPU deploy config;
- generic profiler ranges for runner preprocess, forward, postprocess, sample,
  bookkeeping, and asynchronous output;
- native `vllm_ttft_ms` in Stage 0 metrics;
- prefix caching explicitly disabled by the official deploy config.

This PR does not change Graph mode, prefix-cache policy, prompt semantics,
sampling, dtype, model math, API output, or formal metrics. No safe default-on
TTFT optimization is promoted without an A3 trace showing the dominant scored
hotspot.

## Opt-in host timeline

`VLLM_OMNI_ULTRA_TIMELINE=1` adds the following process-local Stage 0 events:

| Event | Boundary |
| --- | --- |
| `stage0_preprocess_begin/end/error` | Stage 0 input processor, including tokenization/MM input processing |
| `stage0_queue_enter/leave` | handoff to and pickup by the orchestrator queue |
| `stage0_submit_begin/end/error` | output-processor registration and EngineCore admission |
| `stage0_first_text_output` | first non-empty processed Stage 0 text output on the server |

Records contain only host timestamps and bounded metadata: prompt type,
presence of multimodal fields/features, prompt-token count, preprocessing and
queue durations, replica id, and error text. They do not include prompt text,
audio, token ids, tensor data, or device timing. The hooks do not call
`.item()`, `.cpu()`, `.numpy()`, or an accelerator synchronization API.

The flag is read once at engine/pool startup. When disabled, the request path
does not call the event writer or inspect request payload metadata. Timeline
mode remains diagnostic and must be disabled during formal scoring.

## A3 profiling gate

Start the server with a dedicated profiler directory and the timeline enabled,
then profile Stage 0 only around fixed Seed-TTS requests:

```bash
export VLLM_OMNI_ULTRA_TIMELINE=1
export VLLM_OMNI_ULTRA_TIMELINE_DIR=work/perf-evidence/<run-id>/timeline
export VLLM_TORCH_PROFILER_DIR=work/perf-evidence/<run-id>/profiles

curl -sS -X POST http://127.0.0.1:8000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"stages":[0]}'

# Run fixed Chinese Seed-TTS c=1 diagnostic requests here.

curl -sS -X POST http://127.0.0.1:8000/stop_profile \
  -H 'Content-Type: application/json' \
  -d '{"stages":[0]}'
```

Profiler runs are never score samples. A candidate may be added to this PR only
when the trace identifies a scored text-path hotspot and the smallest
value-equivalent change improves TTFT by at least 3% while RTF and TTFP remain
non-inferior under the frozen A/B protocol.

## Explicit exclusions

- The Whisper chunk-mask vectorization in upstream PR #5382 targets audio
  encoder work and is outside the scored text-only path.
- The resampler scalar-sync fix in upstream PR #5318 and SigLIP/vision changes
  in PRs #5130 and #5188 are likewise multimodal-path optimizations.
- Thinker Graph PR #5466 is CUDA/H20-specific and is not directly portable to
  the Ascend A3 runtime.
- Prefix caching, session KV leases, and Graph-mode changes remain separate
  hypotheses and stay disabled until target-hardware evidence supports them.

The Seed-TTS benchmark test now freezes the text-only request shape so a future
adapter change cannot silently route reference audio through the Thinker and
invalidate the TTFT workload contract.

# MiniCPM-o 4.5 MRv2 profiles

The turn profiles use vLLM's stage-level MRv2 contracts. Thinker publishes
live latent metadata outside graph replay, Talker keeps codec history and
EOS control on device, and Code2Wav reuses the shared Whole-Euler Flow
backend. The generic turn profile admits eight Talker requests with 2 GiB
of KV; the H200 turn profile admits sixteen with 4 GiB.

## Native duplex execution

`minicpmo_4_5_duplex_mrv2_h200.yaml` selects MRv2 for all three stages and
retains the shared chunk adapter for prompt replacement, segment boundaries,
and cancellation fencing. Pipelines must declare duplex MRv2 support before
selecting this session mode. The native MRv2 data plane remains turn-only.
The NPU, XPU, ROCm, and MUSA overrides select V1.

Thinker uses asynchronous scheduling. Replay reconstructs live token,
position, and attention metadata, while special-token metadata stays on the
host. Packed hidden-state handoff preserves FP32 values and their row
ownership. Accepted history and RNG state belong to requests rather than
GPU slots; identity fences prevent delayed output copies from updating a
replacement request or condition. Boundary sampling rolls back unused
candidate draws before the next accepted step.

Talker uses synchronous scheduling and FULL graph replay. Eligible native
duplex decode batches run up to four one-token steps per scheduler round
trip. The scheduler reserves matching KV lookahead; the model limits bursts
at token and context budgets, prefill boundaries, and terminal codec tokens.
Each step replays the ordinary one-token graph. Terminal rows stop accepting
output, and request-owned sampler checkpoints exclude discarded draws.
Turn mode and processors that do not declare a burst limit retain one step.

The seeded codec sampler captures auxiliary graphs on supported CUDA duplex
paths unless eager execution is requested. Unsupported samplers and devices
retain eager sampling. Warmup uses disposable seeded rows, before the runner's
JIT monitor and heap freeze, without consuming live or default RNG state.
The repetition penalty uses the last sixteen codec tokens across conditions
and replaces the upstream penalty processor in its ordered processor list.
Frequency and presence penalties continue through the upstream implementation.

The first codec window contains ten generated frames; later windows contain
twenty-five. After publishing the first nonterminal window, MRv2 Talker may
yield for 25 ms when it has one running request, allowing Code2Wav to start.
The scheduler expires this deadline and does not hold concurrent batches.
Cancellation keeps Thinker conversation history, fences interrupted output,
and starts the next response with fresh Code2Wav streaming context.
Reference audio is optional; sessions without it use the model's default
codec voice prompt.

## H200 deployment

The duplex profile admits sixteen sessions and sixteen slots per stage,
with 24 GiB Thinker KV and 4 GiB Talker KV. Longer unwindowed sessions need
a workload-specific KV budget. Window mode remains a client choice.

Code2Wav uses FP32 attention caches, a resident history slot pool, row-offset
merging, and encoder graphs through batch sixteen. It drops the duplicate
upstream attention buffer after assigning resident history. Whole-Euler
capture has a bounded capacity of 128 graphs, a 100-frame offset grid,
50/100-frame query buckets, and one eager warmup solve per new shape.
Masked padding preserves actual history offsets and streaming trims but
can change finite-precision attention results. Uncached shapes run eagerly.

HiFT captures exact widths from one through thirty-two codec frames at batch
sizes one through sixteen. Other widths keep their regular buckets and eager
fallback. Auxiliary capture finishes before service readiness; these graph
sets trade startup time and memory for serving latency.

The shared CFM backend defaults to the fused DiT body on CUDA when TF32 is
allowed and its layout and FP32 cache requirements are met. SM80+ uses tiled
TF32 attention; older CUDA devices use SDPA. Other platforms retain their
existing defaults. `cfm_fused_body: false` in connector `extra` restores the
original body. Disabling `token2wav_allow_tf32` or setting
`MINICPMO_CODE2WAV_TF32=off` disables default fusion; an explicit fusion setting
takes precedence. HiFT stays IEEE FP32. Fusion and TF32 can change rounding
and do not promise identical waveforms.

First-chunk express scheduling requires every ready continuation to have at
least 0.5 seconds of unplayed audio. Playback credit resets at each resumable
segment boundary, including empty terminal markers. In-flight chunks retain
their execution slots until output retirement, while stateful codecs retain
lifetime admission until decoder state is released.

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex_mrv2_h200.yaml
```

Validate duplex and turn paths with matched dependencies, input, and
concurrency. Duplex two-pass sampling does not support output logprobs.
Shared output snapshots and batched Talker preprocessing also affect V1
when async chunking or scheduling is enabled, so V1 remains part of regression
coverage. Numerical sampler parity does not establish audio quality, speaker
similarity, or audible playback latency; those require separate evaluation.

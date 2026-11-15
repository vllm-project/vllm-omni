# Qwen3-Omni MRv2 performance

This model adaptation builds on #8184's existing stage runtime and output
contracts. It adds Talker CPU metadata parsing, sampled-token embedding
handoff, an optional first-frame decoder, length-grouped Code2Wav graph
buckets, and opt-in incremental short-KV code prediction. The H200 profile
also enables conservative fused sampling and bounded cuDNN algorithm search.
The shared predictor keeps the upstream re-prefill helper and its default
sampling behavior; short KV is explicitly selected by the deployment profile.

Use `vllm_omni/deploy/qwen3_omni_moe_mrv2_h200.yaml` for the experimental H200
profile. Preserve a control with `code_predictor_fused_sampling=false` and
`codec_cudnn_benchmark=false` when C64 P99 is the primary requirement.
cuDNN search restores process settings on exit and has startup cost. Kernel
arithmetic is not bitwise-equivalent; the first-frame decoder uses additional
model memory. `vllm_omni/deploy/qwen3_omni_duplex_mrv2.yaml` runs these MRv2
stages for duplex. Qwen duplex submits each committed utterance as a new
request, so it uses the turn-mode data plane. Turn-mode measurements do not
establish duplex latency or interruption behavior.

First-frame delivery requires `talker_first_audio: true` in connector extra,
a one-frame initial codec chunk, CUDA, a local single-rank Talker and prefix caching disabled.
Other deployments retain the regular Code2Wav path.
`codec_fused_snake` explicitly opts CUDA profiles into fused decoder blocks;
the base V1 profile retains its original blocks and indexed text cache.

## Ascend generation stage

The default `qwen3_omni_moe.yaml` selects MRv2 for Thinker, Talker and
Code2Wav on NPU. The NPU generation runner combines the shared Omni V2
chunk lifecycle with Ascend request state, input batches and forward context.
Code2Wav does not build AR attention metadata or sample tokens, so it retains
the generic Omni model state instead of the Thinker/Talker Ascend attention
adapter. Startup skips language-model token warmup and retains model-owned
auxiliary graph capture.

The outer generation forward runs eagerly with exact codec input shapes.
The inner NPU decoder captures up to 32 exact tensor signatures while the
stage loads: one row for each start window (initial chunk or ramp), then the
steady chunk-plus-left-context window for every row count up to the
remaining budget (or `codec_graph_batch_sizes`). The runner is then frozen:
other signatures run eagerly, and no capture happens on a live request.
Replays use no frame or batch padding and return owned outputs, so later
replays cannot overwrite earlier audio. A capture failure fails stage
startup; `enforce_eager: true` on stage 2 disables inner graphs. NPU audio uses the synchronous
host materialization path before the native data plane publishes output and
releases each chunk slot. CUDA retains its asynchronous output path. This
adaptation alone does not establish a performance or audio-quality gain.
The duplex profile remains on MRv1.

## Ascend prefix cache

The default Thinker prefix cache is enabled on CUDA for V1 and V2. NPU keeps
it disabled by default until cold/hit Thinker token parity is established;
set `enable_prefix_caching: true` on stage 0 to opt in. Talker and Code2Wav
retain disabled prefix caching. When enabled, the NPU V2 AR runner
initializes the Omni stage-output cache after native Ascend KV setup; KV
reuse alone would omit the cached Thinker rows needed by the Talker. The
shared cache manager restores hidden and multimodal output rows on hits
using its synchronous non-CUDA copy path. Duplex and the separate
experimental profiles retain their explicit settings.

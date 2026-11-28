# Qwen3-Omni MRv2 performance

The default `qwen3_omni_moe.yaml` includes the streaming codec tuning: one-frame
first audio on eligible CUDA Talkers, a `[1, 2, 4, 8, 16, 25]` chunk ramp,
length-grouped codec graph buckets and CUDA fused Snake blocks. All three stages
keep prefix caching disabled. The duplex and H200 profiles inherit these
settings directly from the default; a separate codec overlay is no longer needed.
The ramp takes precedence over the four-frame initial chunk setting, which
remains the fallback when `codec_chunk_ramp: null` is configured.

This model adaptation builds on #8184's existing stage runtime and output
contracts. It adds Talker CPU metadata parsing, sampled-token embedding
handoff, an optional first-frame decoder, length-grouped Code2Wav graph
buckets, and opt-in incremental short-KV code prediction. The H200 profile
also enables conservative fused sampling and bounded cuDNN algorithm search.
The shared predictor keeps the upstream re-prefill helper and its default
sampling behavior; short KV is explicitly selected by the deployment profile.

Use `vllm_omni/deploy/qwen3_omni_moe_h200.yaml` for the experimental H200
profile. Preserve a control with `code_predictor_fused_sampling=false` and
`codec_cudnn_benchmark=false` when C64 P99 is the primary requirement.
cuDNN search restores process settings on exit and has startup cost. Kernel
arithmetic is not bitwise-equivalent; the first-frame decoder uses additional
model memory. `vllm_omni/deploy/qwen3_omni_duplex.yaml` runs these MRv2
stages for duplex. Qwen duplex submits each committed utterance as a new
request, so it uses the turn-mode data plane. Turn-mode measurements do not
establish duplex latency or interruption behavior.

First-frame delivery requires `talker_first_audio: true` in connector extra,
a one-frame initial codec chunk, CUDA, a local single-rank Talker and Talker prefix caching disabled.
The launcher must resolve both stages to the same model, subdirectory, revision and configured dtype.
Separately launched stages and stages configured for independent weight updates
use Code2Wav for the entire stream. When CUDA graph memory estimation is enabled,
the worker measures the Talker's first-frame graph pools before sizing the KV cache.
Other deployments retain the regular Code2Wav path.
Set `talker_first_audio: false` (or `VLLM_OMNI_TALKER_FIRST_AUDIO=0`) to disable
first-frame delivery. To restore the previous codec defaults, also set
`codec_chunk_ramp: null`, `codec_graph_batch_sizes: []` and
`codec_fused_snake: false` in connector extra. Thinker prefix caching can be
enabled separately on stage 0; it does not control Talker first-frame eligibility.

## ROCm generation stages

The default `qwen3_omni_moe.yaml` selects MRv2 for Thinker, Talker and
Code2Wav on ROCm. It preserves the ROCm stage-0 prefix-cache disable and keeps
Code2Wav eager because its MIOpen transposed convolutions are not safe inside
the outer graph capture. Async chunks, sampling defaults and the three-stage
payload contract are unchanged.

This runner migration does not enable the H200 profile's CUDA graphs, fused
sampling, cuDNN search, first-frame decoder or CUDA-only kernels on ROCm. It
does not by itself establish ROCm performance or audio-quality qualification;
the exact ROCm deployment remains subject to hardware CI.

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

Non-streaming NPU Code2Wav uses longer windows: 300 codec frames for the
first chunk and 325 for subsequent chunks including left context. It caps
each decoder batch at 8 rows and captures both windows for row counts 1–8
(or fewer when the stage's `max_num_seqs` is smaller). Larger batches are
split in row order before temporal chunking. The same limit applies with
`enforce_eager: true` and to uncaptured tail windows. This bounds decoder
activation pressure while preserving each request's existing chunk trimming;
it does not cap memory for model weights or accumulated output waveforms.
Changing the decoder batch shape can change BF16 numerical results, so this
split does not promise bitwise equality with an unsplit batch. Audio-quality
acceptance requires representative speech inputs in addition to shape and
trimming regressions.
The limit is separate from request concurrency and does not change the
streaming graph plan. Talker and Code2Wav share a device in the default NPU
deployment, so graph-count limits alone do not provide a memory budget for
these longer non-streaming windows (#10090).

## Ascend prefix cache

The default Thinker prefix cache is disabled on every platform. On NPU, cold/hit
Thinker token parity still needs to be established; set
`enable_prefix_caching: true` on stage 0 to opt in. Talker and Code2Wav
retain disabled prefix caching. When enabled, the NPU V2 AR runner
initializes the Omni stage-output cache after native Ascend KV setup; KV
reuse alone would omit the cached Thinker rows needed by the Talker. The
shared cache manager restores hidden and multimodal output rows on hits
using its synchronous non-CUDA copy path. Duplex and the separate
experimental profiles retain their explicit settings.

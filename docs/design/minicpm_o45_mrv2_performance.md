# MiniCPM-o 4.5 MRv2 profiles

The default `minicpmo_4_5.yaml` selects MRv2 for all three CUDA stages
and FlashAttention for the Talker. All three duplex stages keep
`async_chunk: true`. The CUDA turn profiles use the same runner and Talker
backend. They enable Thinker prefix caching and send its completed payload
to the Talker. Both V1 and MRv2 restore cached Thinker outputs for the Talker. MRv2 uses
the shared stage-output prefix cache to restore `latent`, `latent_input_ids`
and `latent_positions` together before publishing the llm2tts payload. Changing
the CUDA runner to V1 preserves the enabled cache. NPU, ROCm, XPU and MUSA keep
their existing disabled overrides; duplex profiles also keep caching disabled
because their mutable KV window cannot reuse prefix blocks.

The MRv2 output-cache opt-in requires an explicit `session_mode: turn` on
the Thinker model config. A missing mode does not advertise output-cache
support. The output cache refuses stage-0 speculative decoding, so an MRv2
Thinker with a `speculative_config` reuses native KV only and keeps prefix
caching on (see "Thinker prefix cache with drafts" below). A V1 Thinker with
speculation, including n-gram, is still refused while the cache is enabled;
such overlays must set stage 0 `enable_prefix_caching: false` under
`platforms.cuda`.

Turn mode uses the stage-level MRv2 contracts from #8184. Thinker emits live
latent metadata outside graph replay; Talker keeps codec history and EOS
control on device. Code2Wav reuses mainline Whole-Euler Flow graphs and shared
prompt state. The previous experimental block compilation, tiled attention,
channels-last and merged-CFM implementations are removed.

## Turn prefix-cache memory

The persistent stage-output cache is a pageable CPU mirror, not a second
GPU KV pool. `stage_prefix_cache_config` takes its `num_blocks` from the
Thinker KV configuration. `PrefixBlockPool` allocates one tensor per cached
key, shaped `[num_blocks, block_size, width]`; recycled KV slots overwrite
the corresponding output rows. There is no independent output-cache block
budget or deploy knob.

The Thinker caches hidden states and `latent` separately, plus
`latent_input_ids` and `latent_positions`. Mirror tensor storage is therefore
`num_blocks * block_size * sum(width * element_size)` across those keys.
For BF16 hidden states and latents of width `H`, with int64 ids and positions,
this is `num_blocks * block_size * (4 * H + 16)` bytes. Slot bookkeeping,
materialized hits and output payloads consume additional host memory.
Pinned host staging is separate: per key it uses
`staging_depth * staging_capacity_tokens * width * element_size` bytes.
Capacity follows `max_num_batched_tokens` (falling back to `max_model_len`);
MRv2 uses depth at least `max(4, 2 * max_concurrent_batches + 1)`.

`PrefixCacheConfig.gpu_staging_bytes` defaults to 512 MiB for device rows
awaiting a host copy. Budget pressure flushes pending copies; it is not a
hard peak-memory bound, since one reservation can exceed it when no tasks
remain pending. The Talker reserves its own 4 GiB CUDA KV cache.

## Runner and codec behavior

The CUDA turn profile and the opt-in duplex H200 profile enable
ordinary TF32 for CFM DiT GEMMs within Code2Wav forward/capture only
(`torch.backends.cuda.matmul.allow_tf32` on dense QKV/MLP, Triton
`input_precision="tf32"` on tiled attention). This is not compensated TF32x3.
The previous process matmul policy is restored afterwards. cuDNN's TF32 policy
is independent. HiFT stays IEEE FP32. TF32 changes rounding.
The CUDA duplex profile uses MRv2 on all three stages.

The shared Code2Wav backend defaults to the fused CFM body on both CUDA and
NPU, subject to architecture and attention-cache eligibility. CUDA accepts
matching FP32 or FP16 estimator/cache types; NPU retains FP32 eligibility. CUDA also
requires TF32 to be allowed; NPU fusion is independent of TF32. Set
`cfm_fused_body: false` in the stage connector extra to restore the original
body. On CUDA, disabling `token2wav_allow_tf32` or setting
`MINICPMO_CODE2WAV_TF32=off` also disables the default fusion choice; an explicit
`cfm_fused_body` setting takes precedence. Startup precapture includes query
buckets for final chunks with encoder lookahead, after steady-state buckets
and within `cfm_max_graphs`. Capture logs report host warmup and capture times.
With a reduced graph budget, steady-state graphs can consume all available
entries before final-chunk buckets are captured. Uncached final-chunk shapes
then execute eagerly, preserving correctness but losing the graph latency
benefit. The required budget depends on the configured batch sizes, query
widths and cache offsets; there is no universal minimum such as 30 graphs.
Check the startup precapture budget warning when tuning `cfm_max_graphs` for
memory-constrained deployments.

The shared asynchronous output snapshot and batched Talker preprocessing also
apply to V1 when async chunking/scheduling is enabled. Default V1 behavior must
therefore be included in end-to-end regression validation.

CFM graph execution defaults to the fused DiT body on CUDA (when TF32 is
allowed) and on NPU. NVIDIA SM80+ uses tiled TF32 attention; older CUDA devices
use the fused body's SDPA fallback. NPU uses packed QKV, in-place KV and ACLNN
AddLayerNorm, independent of TF32. Unsupported DiT layouts and cache types
retain the original body. CUDA supports matching FP32 or FP16 estimator/cache
types; NPU supports FP32. Other platforms retain their existing default.
Set `cfm_fused_body: false` in connector `extra` to restore the original body.
On CUDA, disabling TF32 with `token2wav_allow_tf32: false` or
`MINICPMO_CODE2WAV_TF32=off` also disables the default fusion choice; an explicit
`cfm_fused_body: true` overrides that choice and may use TF32 attention.
Slot pooling and its `cfm_row_offset_merge` option remain opt-in. Fusion changes
floating-point rounding and does not promise identical waveforms.

## Optional CFM precision and speaker replay

`cfm_float16: true` in connector `extra` opts CUDA Code2Wav into FP16 for the
CFM estimator, its random-noise table and attention caches. The flow encoder,
speaker extraction and HiFT retain their existing precision. This differs
from `token2wav_float16`, which affects the whole flow. FP16 CFM changes
rounding and requires speech-quality and service validation; it is off by
default and does not reduce the number of flow steps.

`token2wav_spk_cuda_graph: true` opts the CUDA Torch CAM++ backend into
exact-shape graph replay. It requires the Torch speaker backend and rejects
CPU/NPU or TensorRT speaker configurations. Up to four feature shapes are
captured without padding; additional shapes use the existing eager forward.
Autocast or a changed precision policy also uses the eager path. Each replay
returns an owned embedding. This option is off by default and does not expand
the reference-audio cache or batch reference extraction.

`token2wav_s3_cuda_graph: true` opts CUDA Code2Wav into exact-shape replay
of the S3 reference tokenizer. It requires the S3 `AudioEncoderV2` and keeps
the pinned `s3tokenizer==0.3.0` encoder and quantizer without reimplementing
their arithmetic. Up to two repeated shapes are
captured; additional shapes run eagerly. Long references retain the library's
sliding-window path. Replay owns its returned codes and lengths, and a changed
precision policy uses the original eager encoder. This option is off by
default; end-to-end performance and speech-quality qualification are pending.

## Turn-mode conditioning transport

Ordinary turn-mode `llm2tts` packs reference audio and thinker hidden states
with the existing Omni tensor serializer. The request envelope's
`model_intermediate_buffer` is untyped, so these values cross EngineCore as
owned bytes and are restored to CPU float32 tensors by the Talker and codec
bridge. This avoids scalar-list construction, worker list traversal and
list-to-tensor conversion. Values match the previous list-to-float32 path;
sampling and acoustic computation are unchanged.

Legacy lists and local tensors remain accepted by consumers. Native duplex
continues producing its existing lists. Each decode owns its storage, and the
packed snapshot cannot be changed by later producer mutations. Transport
microbenchmarks alone do not establish service throughput or speech quality.

## Thinker speculative decoding

The MRv2 Thinker accepts a stage-0 `speculative_config` whose drafter reads
Thinker layers (EAGLE-3, DFlash or DSpark). The MRv2 AR runner proposes drafts
after sampling and verifies them with the rejection sampler. Only the rows of
accepted tokens reach `llm2tts`, so the Talker still receives one Thinker row
per committed token. Example overlay:

```yaml
base_config: minicpmo_4_5_turn.yaml
stages:
  - stage_id: 0
    speculative_config:
      method: dspark
      model: <DSpark draft trained on the MiniCPM-o 4.5 Thinker>
      num_speculative_tokens: 7
```

vLLM 0.33 resolves a draft config with `architectures: ["DSparkDraftModel"]`
and `model_type: qwen3` to `Qwen3DSparkModel`. Drafters that read Thinker
layers require the Thinker on MRv2; the V1 Thinker rejects them at load.
Validation covers CUDA turn sessions with greedy text and speech output.
Duplex sessions keep their own sampler and are not covered, and platforms
whose profiles fall back to V1 (NPU, XPU, ROCm, MUSA) do not run this path.
Greedy text can differ from decoding without drafts at near-tied tokens.

## Thinker prefix cache with drafts

With a stage-0 `speculative_config` the MRv2 Thinker does not set
`supports_mrv2_prefix_cache_outputs`, so it reuses native KV alone: no output
rows are saved or merged and no host cache is allocated. `llm2tts` picks the
Talker's rows from the row ledger (`latent_input_ids`, `latent_positions`).
The reply starts after `<|tts_bos|>`, the last prompt token, and the prompt's
last token is always forwarded, so a new request's hit never covers a reply
row. A request resumed after preemption can hit on its earlier reply tokens,
whose rows are already in the ledger. A template that opens the reply earlier
in the prompt has no rows for the cached part of it: on a hit the request logs
the "Thinker rows stop" warning and produces no audio, so with such a template
a draft overlay also sets stage 0 `enable_prefix_caching: false`.

A hit changes which tokens share a prefill step, so greedy text on a hit can
differ from a miss at near-tied tokens in BF16, as it does without drafts.

## Talker speculative decoding

The MRv2 Talker also accepts a stage-1 `speculative_config` with a DSpark
drafter that reads Talker layers. Draft codes are verified against the
Talker's own sampling: each row's 16-code repetition window includes the
drafts before it, and codec EOS decisions (empty condition, length cap)
apply per row. Only rows of accepted codes reach Code2Wav. The CUDA platform
block selects the draft method. Set the DSpark config there and clear the
inherited K-step control stop ID. The pipeline supplies the codec EOS.

```yaml
base_config: minicpmo_4_5_turn.yaml
platforms:
  cuda:
    stages:
      - stage_id: 1
        async_scheduling: true
        default_sampling_params:
          stop_token_ids: []
        speculative_config:
          method: dspark
          model: <DSpark draft trained on the MiniCPM-o 4.5 Talker>
          num_speculative_tokens: 7
          draft_sample_method: probabilistic
```

The Talker samples its codes (temperature, top-k, top-p), so the drafts should
be sampled too: with vLLM's default argmax drafts, far fewer are accepted.
Rejection sampling keeps the Talker's output distribution either way.

The draft checkpoint must include the codec embedding (`embed_tokens.weight`)
and head (`lm_head.weight`). The Talker stage's config reports the text
vocabulary, so vLLM does not share the Talker's with a codec-vocabulary
draft. vLLM's `Qwen3DSparkModel` also reads `markov_rank` and the
confidence-head fields from the top level of the draft config. Talker drafters
cover turn sessions only; duplex sessions and the K-step Talker reject them at
load.

### CUDA MRv2 Talker K-step

Use `minicpmo_4_5_turn.yaml` for CUDA turn serving. It inherits FlashAttention,
synchronous scheduling, seven continuation drafts and the control stop ID from
`minicpmo_4_5.yaml`. The MRv2 Talker uses one FlashAttention query per codec
frame, advances the position and KV slot, and re-embeds the confirmed codec.
For a single-frame control, override CUDA stage 1 with
`speculative_config: null` and `default_sampling_params.stop_token_ids: []`.

Mixed prefill and decode steps use the single-frame codec path. A K-step batch
replays a CUDA graph when its sampling settings are supported, and falls back
to the same per-frame attention view for other settings. The connector receives
only valid codec frames, including the final short batch at EOS.

The profile enables CUDA MPS for the colocated stages. CUDA Graphs are captured
on demand for the batch sizes seen by the Talker. To prewarm a newly started
C16 replica, send several request waves at different concurrency levels before
measuring steady-state RTF.

## Code2Wav batching across reference audio

With CUDA ragged Whole-Euler enabled, ready requests with different reference
audio can share a Code2Wav decode batch. This also merges compatible tensor
histories of different lengths for a single reference; it does not enable slot
pooling. Other configurations retain reference-specific, exact-shape buckets,
preserving CFM step-graph and TRT dispatch. Mixed-history arena replay is
restricted to CUDA; the existing opt-in slot path is unchanged on other
platforms.

Reference extraction and initial cache setup remain separate for each
reference; speaker conditioning, attention lengths and cache trimming stay
per request. Cache epochs remain separate unless cross-turn batching is enabled.

The MiniCPM Code2Wav stage runs CAM++ speaker embedding on CUDA and NPU using
the in-tree torch model with FP32 weights and inputs, loaded from the
checkpoint's `campplus.onnx`. On CUDA, reference preparation follows the
stage's matmul policy (`token2wav_allow_tf32`, enabled by default); cuDNN's
convolution TF32 policy is independent. FP32 tensors therefore do not imply
strict FP32 arithmetic.
On other Torch accelerators, enable this path with
`token2wav_device_spk: true` in connector `extra`.
The CUDA path does not require TensorRT. Set `token2wav_device_spk: false`
in connector `extra` to use CPU ONNX Runtime for accuracy triage; explicitly
enabling `token2wav_trt` retains its TensorRT speaker embedding path.
This moves speaker embedding off the CPU but does not batch reference
extraction or initial cache setup. Device kernels can change floating-point
rounding, so this does not promise bit-identical embeddings or waveforms.

The ragged Whole-Euler path masks padded history and returns each request's
own cache length, reserving each row's retained length so it can reuse its
storage across chunks. Encoder work still groups by compatible cache shapes.
Unsupported ragged graph shapes retain the existing exact-shape fallback.
This changes compute batching, not the content-based reference cache identity
or reference normalization policy.

## Full-duplex MRv2

`minicpmo_4_5.yaml` selects MRv2 for the CUDA Thinker, Talker and Code2Wav.
The Thinker has no async producer, so the orchestrator hands each finished
segment to `llm2tts`.
The Talker reuses the streaming prompt recipe (full attention extends its KV
prefix until capacity; sliding recompute rebuilds the previous condition,
confirmed codec ids and current condition) and keeps the 16-frame codec
penalty history across conditions. A segment flushes only on an EOS sampled
in the current output, never on `turn_end` alone.

### Duplex allocation bounds and attention visibility

The duplex KV spec derives from `SlidingWindowSpec` to bound admission,
while its manager owns eviction, sink retention and position reanchoring.
The attention kernel must see the manager's complete dense block table,
including when the client selects windowing `off`.

On vLLM 0.33, MRv2 also uses `num_retained_tokens` to mask block-table entries
below an ordinary sliding window. Inheriting that policy incorrectly treated
the duplex allocation bound as an attention window. With the default geometry,
the bound is 8,112 tokens: a prefill starting at 8,121 retained all blocks, but
the following decode at 8,200 replaced the first five 16-token blocks with
zero. This discarded live instructions/context and caused later scene loss.
V1 did not apply this worker-side mask.

`MiniCPMO45DuplexWindowSpec.num_retained_tokens` returns zero, MRv2's sentinel
for disabling this mask. Admission sizing remains unchanged; only the duplex
manager may remove history. Ordinary sliding-window specs retain their mask.
The CUDA block-table regression covers both cache groups past the bound and
a compacted table with retained sink pages. A matched L20X/vLLM 0.33 long-video
run recovered all 12 scenes over 180 frames, with clean protocol checks and
the same transcript as V1; unfixed V2 covered only scenes 1–7.

### Proactive-reminder parity with V1

With the KV fix, an MRv2 Thinker reproduces V1 exactly. On one L20X at C1,
sample 140 gave identical append embeddings, decision log-probabilities and
sampled tokens on both runners. Sample 133 matched until unit 38. There, the
two runs received different appends: one carried a video frame and the other
did not. Whether a unit carries a frame depends on real-time arrival, so the
same runner can diverge between runs. At C8 on the 23 timed reminder IDs
(133–145, 261–270, soundtrack mix), Qwen2.5-VL-7B scored both runners 4/23,
repeatably. Terminator feedback matched on every unit, and no duplex row fell
back to the stock sampler.

The 0.28 floor needs at least 11 of 36 and equals the V1 reference of 11/36.
At that rate, one 36-sample run has a standard deviation of about 2.7 samples.
Earlier V1 runs scored 9 and 11; MRv2 runs scored 6 and 9. These spreads are
run-to-run variance, not a runner gap.

### Chunk-boundary probability precision

The shared V1/MRv2 Thinker sampler draws the chunk boundary before applying
text penalties and top-k/top-p filtering, as the released StreamDecoder does.
Its probability must come from `softmax(logits)`. Computing it as
`exp(logit_eos - logsumexp(logits))` materialized the intermediate reduction
in the model dtype. With three equal BF16 logits of 32, that expression gives
0.3671875 instead of the reference softmax's 0.333984375. A shared offset to
all logits could therefore change whether a speech chunk ended.

The boundary helper now selects the EOS probability from a batched softmax,
preserving the reference dtype and existing per-request random draws. The
regression checks equal-probability rows at several common offsets in BF16,
FP16 and FP32, including a draw whose boundary decision was previously wrong.
This is a shared sampling correction, not evidence by itself that MRv2 passes
the concurrent proactive-reminder quality floor.

## Code2Wav encoder shapes outside the startup arena

The shared encoder arena captures continuation chunks for the default prompt.
By default, startup then freezes admission: tail chunks and uncached prompt
initialization shapes run eagerly. In connector `extra`, opt in to
`code2wav_encoder_capture_after_precapture: true` to keep exact-shape runtime
capture enabled with `enable_code2wav_encoder_graph: true`.
`code2wav_encoder_capture_after` controls how often an exact signature must
appear before capture (minimum 2). The capture call still runs eager warmup;
later calls replay. No tail padding or continuation-graph substitution occurs:
`last_chunk`, cache presence, shapes, positional tables and stream identity
remain part of the graph signature.

The runtime cache retains at most `code2wav_encoder_max_graphs` additional
graphs, separately from the shared startup arena. At capacity, unseen shapes
remain eager and existing graphs stay replayable. Enabling this option costs
additional device memory and synchronous capture latency on admitted requests;
warm representative prompts and tail lengths before measuring steady-state
performance. A few warmup requests do not guarantee coverage of variable tails.

## NPU Talker decode graph experiment

Stage 1 can request `FULL_AND_PIECEWISE` through the NPU stage override's
`compilation_config.cudagraph_mode`. This requests full decode graphs and
piecewise prefill execution, subject to backend support and capture buckets.
The default deployment remains `PIECEWISE` until full-model correctness and
performance have been validated on the target Ascend stack.

Full replay skips Python `Talker.forward`. The model's batch-preprocess hook
therefore flushes deferred request cleanup before preprocessing reuses codec
slots. The forward cleanup remains a fallback for direct callers. The Omni
NPU forward adapter also passes the attention backend to the full graph
wrapper, matching Ascend's selection of the updatable attention graph path.
Dynamic attention parameters keep the upstream update/replay ordering.

Validate changing sequence lengths, request churn and padded decode batches,
then compare codec outputs and end-to-end Seed-TTS concurrency-4 performance
against an otherwise identical PIECEWISE deployment. A small NPUGraph
lifecycle test proves host cleanup across replay; it does not establish
Llama attention/KV correctness or a full-model speedup. Measure performance
without stack profiling and verify FULL replay separately from compile logs.

## NPU CFM DiT body schedule

The NPU fused DiT body (`cfm_fused_body`) runs the CUDA body's math on Ascend
graph-friendly kernels: packed QKV, `npu_add_layer_norm` for every modulated
residual + LayerNorm, and causal convolutions as one im2col GEMM over the
contiguous frames-major history (`[k * C_in + c]` columns). Both convolution
histories' causal caches are gathered with `index_select` into one frames-major
staging tensor and written to the caller's `(N, C, W)` cache in a single
transpose per body, instead of a per-block gather/transpose/copy or a per-block
Triton kernel. The whole-block CNN cache is transposed to frames-major once per
body rather than per block. CUDA and the unfused DiT body are unaffected.

## Reference prefetch at higher concurrency

`token2wav_ref_prefetch_max_waiters` (connector extra, default `1`, minimum `1`)
sets the largest pending-reference queue allowed to start idle prefetch. Setting
it to `4` permits a concurrency-4 experiment. Each idle call still performs
only one phase, never starts work while a Code2Wav stream is live, and observes
the existing cache capacity and pin limits. Reference features remain keyed by
content; different audio is never reused as if it were the same reference.

A larger limit can compete with Thinker/Talker on a shared device, so keep the
conservative default until cold and warm end-to-end TTFP/RTF measurements
justify a change. Setup prefetch remains a separate opt-in; warming prompt
features does not imply warming the CFM setup or achieving lower TTFP.

## CPU-reference CUDA speaker extraction

`token2wav_spk_cpu_reference: true` in connector `extra` selects a CUDA
TensorRT speaker backend compatible with the CPU ONNX reference graph.
`minicpmo_4_5.yaml` sets it by default, so a CUDA host with TensorRT installed
uses this speaker. If `tensorrt` cannot be imported, startup logs a warning and
keeps the Torch speaker, so CI images without TensorRT still load this profile.
Set the flag to `false` in an overlay to keep Torch even when TensorRT is
installed. Other platforms ignore the flag. An
explicit `token2wav_trt: true`,
`MINICPMO_TOKEN2WAV_TRT=1`, or `token2wav_device_spk: false` keeps the full
DiT TensorRT path or the CPU onnxruntime fallback and drops this flag.
The CAM++ TensorRT profile accepts 4 to 3000 fbank frames. The default 6 s
window is 598 frames and stays on this path. A wider
`ref_audio_max_seconds` is checked before the engine is installed: 31 s
normalizes to 3098 frames and is rejected at startup, rather than failing while
the first prompt is prepared. It does not change DiT, HiFT, sampling or chunk size.

`minicpmo_4_5.yaml` sets `cuda_mps: true` under `platforms.cuda`, so colocated
CUDA stages share a private MPS daemon when `nvidia-cuda-mps-control` is on
PATH and every stage is one numeric GPU with sequential initialization. NPU
does not enable it. A missing control binary, an MPS daemon that fails to
start or respond, `parallel_stage_init`, or a
non-numeric / multi-GPU stage disables MPS and continues. Set `cuda_mps: false`
in an overlay to disable it explicitly. Improved throughput can trade off
against first-audio tail latency and low-concurrency throughput.

Install a TensorRT build matching the CUDA runtime before enabling the speaker
flag on CUDA. For example, the CUDA 13 validation environment uses
`tensorrt-cu13==11.4.0.106`. ONNX Runtime remains required: startup exports
its BASIC-optimized graph before building the TensorRT engine. CAM++ uses
zero padding followed by ceil-mode average pooling. ONNX Runtime's PadFusion
changes the partial-window denominator; converting the original graph
directly would not reproduce the existing CPU reference. This path
uses the optimized graph and disables TensorRT TF32 for speaker extraction.
This does not change the independent CFM TF32 policy.

`COSYVOICE3_TRT_CACHE` controls the derived ONNX and engine cache directory
(default: `~/.cache/vllm_omni/cosyvoice3_trt`). Derived ONNX files are keyed
by source content and ONNX Runtime version and published under a file lock.
Derived graphs and engines are atomically published from uniquely named sibling
files using normal file-creation permissions (0666 filtered by the process
umask). Complete cache hits do not acquire a write lock. A prebuilt cache
can be mounted read-only only when its derived ONNX and engine keys still
match. Engine keys include the ONNX absolute path, size and integer-second
mtime, as well as precision policy, device name and TensorRT version. For
CPU-reference mode, this ONNX is the derived file inside the cache directory:
keep the same absolute mount path and preserve file mtimes when copying the
cache (for example, with `cp -a`). The source content and ONNX Runtime version
must also match to select the same derived ONNX. Relocating the cache or
changing mtimes produces a cache miss, even with identical model bytes.
Cold builds retain a lock and re-check the artifact under it; misses on a
read-only mount can raise `PermissionError` when acquiring the build lock.
Prebuild in a writable cache at the final path before mounting it read-only,
or provide a writable cache directory for rebuilding. The
first start builds an engine and can take substantially longer; measure
startup separately from steady-state serving. Source model files are not
modified. Revalidate speaker similarity and generated speech quality when
changing backend versions; floating-point waveform identity is not promised.

`minicpmo_4_5.yaml` enables both `token2wav_spk_cpu_reference` and
`platforms.cuda.cuda_mps`. Profiles that inherit it, including
`minicpmo_4_5_turn.yaml`, inherit both. Set either key to `false` in an
overlay to disable that part.

## MRv2 Talker seeded sampling

The MiniCPM-o MRv2 Talker uses V1's per-request exponential draws after
applying MRv2 logit processing. Its top-k/top-p adapter passes the actual
logit vocabulary size to vLLM's sampling states and forwards the returned
`has_topp_only_rows` predicate to native masking.

An unfinished chunked prefill produces a discarded sample. That draw must
not advance the request's seeded generator; the adapter saves and restores
its state using host-side prefill lengths, without reading device counters.
Completed prefill and decode rows advance normally, even in the same batch.
Generators follow request ids across preemption and streaming-input updates,
and are removed when requests finish. Batches requesting logprobs, sampling
masks or trace replay keep the upstream sampler and its RNG semantics.

Startup profiling also registers unseeded sampling slots before request ids
exist. These slots initialize upstream sampling state without creating V1
generators; seeded requests still require an admitted request id.

## V1 Talker decode bookkeeping

The CUDA V1 Talker uses the existing batch-preprocess row-fill hook for batches
containing only one-token decode requests with initialized audio state. It
checks the current request order, token offsets and prompt boundaries on every
step; it does not cache metadata by batch size. New admissions, preemption
resumes, mixed prefill/decode and K-step execution keep the generic path.
The hook preserves condition metadata and updates only the live decode fields
and the empty codec delta; the Talker still owns the sampled-id snapshot.

For a full CUDA decode batch, codec ids, validity and completion flags stay
batched until the runner's owned output snapshot has been copied. The output
finalizer then restores per-request tensors and scalar completion flags.
Mixed batches retain row-wise output. CPU placeholders are allocated only for
rows without a device result. Snapshot ownership remains necessary even when
all rows share one tensor: producer buffers may be overwritten on the next
step.

These shortcuts target V1. The default CUDA MRv2 Talker already skips identity
decode preprocessing and emits device batches, so a V1 improvement alone does
not establish a speedup for the default CUDA MRv2 deployment.

## Reference reuse across concurrent bursts

The Code2Wav backend keeps a runtime reference cache of 4 unless a deployment
sets `token2wav_runtime_prompt_cache_size`. The default `minicpmo_4_5.yaml` does
not raise that target. Each retained entry keeps its
temporary reference WAV and cached speech tokens, speaker embedding and mel
features. Unowned, unpinned entries are evicted when the total exceeds the
target. Active requests and prefetch pins keep their existing ownership rules
and may temporarily exceed it; the CFM setup cache is unchanged.

This targets repeated references across concurrent bursts. The first 16
English SeedTTS requests use nine different references, so a capacity of four
cannot retain the full warmed set. Fresh references still require extraction;
a warm-reference speedup does not establish cold-reference performance.
Override `token2wav_runtime_prompt_cache_size` in connector `extra` to trade
retention memory and temporary-file space for reuse. The backend default remains
four. A deployment overlay can raise it in connector `extra`.

## Warm-reference retention validation

Compared `4dc28a45f8` with only `token2wav_runtime_prompt_cache_size` changed
from 4 to 16. Same reserved NVIDIA L20X GPU 1, vLLM 0.32.0, Torch 2.13.0+cu130,
MPS and Torch speaker backend; all three stages stay on MRv2. The default
deployment YAML keeps the backend default of 4; this measurement used an overlay.

Fixed English SeedTTS order, C16/N16, 16 warmup requests per startup, five
measured rounds per startup, two independent starts per configuration.
Baseline starts bracketed the experiment session. Each configuration completed
160/160 measured requests without failures; profiling was inactive during
measurement. These 16 requests reuse nine references. This measures retention
across warmed bursts, not cold-reference latency or arbitrary reference mixes.

| Metric, mean of ten rounds | Cache 4 | Cache 16 | Change |
| --- | ---: | ---: | ---: |
| Mean E2E (ms) | 1479 | 1278 | -13.6% |
| Mean audio TTFP (ms) | 1008 | 789 | -21.7% |
| Request throughput (req/s) | 9.467 | 10.687 | +12.9% |
| Mean audio RTF | 0.3978 | 0.3415 | -14.2% |
| Generated audio per round (s) | 64.128 | 64.208 | +0.1% |

The same 128 utterances were evaluated twice per configuration with Whisper
large-v3 and WavLM base-plus pooled embedding cosine (not official UniSpeech
speaker-verification SIM). All 256 requests per configuration were scored;
there were no failed requests, empty/missing audio, ASR failures or SIM failures.
Mean utterance WER was **2.366% → 2.032%**; cosine similarity was
**0.8478 → 0.8479**. After averaging the two runs per utterance, paired
bootstrap 95% intervals for cache-16 minus cache-4 were
**[-0.836, +0.109] percentage points** for WER and
**[-0.00087, +0.00125]** for similarity. This does not prove
accuracy equivalence or an accuracy improvement.

Configuration and Code2Wav cache regressions: **1284 passed, 1 skipped**.
Local pre-commit and `git diff --check` passed. No NPU/ROCm hardware, CUDA C32,
or full repository suite was run for this retention change. Higher retention
keeps more idle reference features and temporary WAVs; active/pinned ownership
and the CFM setup-cache capacity are unchanged.

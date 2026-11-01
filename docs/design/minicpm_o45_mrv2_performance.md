# MiniCPM-o 4.5 MRv2 profiles

The default `minicpmo_4_5.yaml` selects MRv2 for all three stages on CUDA
and retains MRv1 on other platforms. CUDA stage 0 uses a full-payload
handoff; downstream stages retain streaming chunks. The ordinary turn profile
keeps Thinker prefix caching disabled with either runner.

Turn mode uses the stage-level MRv2 contracts from #8184. Thinker emits live
latent metadata outside graph replay; Talker keeps codec history and EOS
control on device. Code2Wav reuses mainline Whole-Euler Flow graphs and shared
prompt state. The previous experimental block compilation, tiled attention,
channels-last and merged-CFM implementations are removed.

`minicpmo_4_5_turn_mrv2_h200.yaml` selects Talker capacity 16 and 4 GiB KV;
the generic MRv2 profile selects capacity 8 and 2 GiB KV. These configurations
require new end-to-end performance measurements after the mainline integration.
Previously reported numbers do not describe this revised codec backend.

The generic MRv2 profile and the opt-in native V1 duplex H200 profile enable
ordinary TF32 for CFM DiT GEMMs within Code2Wav forward/capture only
(`torch.backends.cuda.matmul.allow_tf32` on dense QKV/MLP, Triton
`input_precision="tf32"` on tiled attention). This is not compensated TF32x3.
The previous process matmul policy is restored afterwards. cuDNN's TF32 policy
is independent. HiFT stays IEEE FP32. TF32 changes rounding.
Duplex defaults to MRv2 on CUDA; `minicpmo_4_5_duplex_mrv2.yaml` retains
an explicit MRv2 selection. Other platforms keep V1. Turn results
do not establish duplex performance or interruption correctness.

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

`minicpmo_4_5_duplex_mrv2.yaml` explicitly selects MRv2 for all three CUDA
stages, matching the default duplex profile. Non-CUDA platforms keep V1.
Stage 0 keeps `async_chunk: false` because the Thinker has no async producer:
as on V1, the orchestrator hands each finished segment to `llm2tts`.
The Talker reuses the streaming prompt recipe (full attention extends its KV
prefix until capacity; sliding recompute rebuilds the previous condition,
confirmed codec ids and current condition) and keeps the 16-frame codec
penalty history across conditions. A segment flushes only on an EOS sampled
in the current output, never on `turn_end` alone.

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

The default `minicpmo_4_5.yaml` sets the runtime reference cache retention target
to 16 instead of the backend default of 4. Each retained entry keeps its
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
four; overlays inheriting this profile also inherit its retention setting unless
they override it.

## Warm-reference retention validation

Compared `4dc28a45f8` with only `token2wav_runtime_prompt_cache_size` changed
from 4 to 16. Same reserved NVIDIA L20X GPU 1, vLLM 0.32.0, Torch 2.13.0+cu130,
MPS and Torch speaker backend; all three stages stay on MRv2. The measured
cache overlay is semantically identical to the updated deployment YAML.

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

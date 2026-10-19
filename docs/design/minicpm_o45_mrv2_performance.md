# MiniCPM-o 4.5 MRv2 profiles

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
Duplex defaults to the V1 session path; `minicpmo_4_5_duplex_mrv2.yaml` is the
opt-in CUDA overlay that runs all three duplex stages on MRv2. Turn results do
not establish duplex performance or interruption correctness.

The shared Code2Wav backend defaults to the fused CFM body on both CUDA and
NPU, subject to architecture and FP32 attention-cache eligibility. CUDA also
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
AddLayerNorm, independent of TF32. Unsupported DiT layouts or non-FP32 attention
caches retain the original body. Other platforms retain their existing default.
Set `cfm_fused_body: false` in connector `extra` to restore the original body.
On CUDA, disabling TF32 with `token2wav_allow_tf32: false` or
`MINICPMO_CODE2WAV_TF32=off` also disables the default fusion choice; an explicit
`cfm_fused_body: true` overrides that choice and may use TF32 attention.
Slot pooling and row-offset merging remain opt-in. Fusion changes floating-point
rounding and does not promise identical waveforms.

## Full-duplex MRv2

`minicpmo_4_5_duplex_mrv2.yaml` opts all three CUDA stages into MRv2 and
otherwise inherits the default duplex profile; non-CUDA platforms keep V1.
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

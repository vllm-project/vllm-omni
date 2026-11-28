# Higgs-Audio V3 Deploy Profiles

With vLLM 0.34, Model Runner V2 is the only supported runner. The default
profile is `higgs_multimodal_qwen3.yaml`.

## Legacy profile names

The `_high_throughput.yaml` and `_low_latency.yaml` names are retained for
compatibility, but both files are V2 profiles after #10800. The former keeps
the eager local MLP graph for throughput; the latter uses the external
`FULL_DECODE_ONLY` graph for low latency.

## Model Runner V2 (default)

The default profile uses Model Runner V2 for Stage 0, `FULL` CUDA graph capture,
`FLASH_ATTN`, and `async_scheduling: true`; Stage 1 stays eager with codec
graph windows `[8, 33]`. The connector emits 25 steady frames with 8 frames
of left context; its first 8-frame chunk holds about 320 ms of audio.

`higgs_multimodal_qwen3_high_throughput.yaml` is the V2 high-throughput
profile. It keeps Stage 0 `enforce_eager: true` and the Higgs local MLP graph.
`higgs_multimodal_qwen3_low_latency.yaml` is the V2 low-latency profile; it
uses the external `FULL_DECODE_ONLY` graph. Both profiles use
`async_scheduling: true` and the v2 async payload overrides required by the
talker.

Sampling surface under v2: **temperature / top_p / top_k / seed only** (the
self-owned sampler captures tensor-valued temperature/top-k/top-p; penalties,
logprobs, min_tokens, min_p, allowed/bad token ids, and logit_bias are
text-generation machinery it does not wire in). A non-neutral unsupported
value in `default_sampling_params` fails at startup with the remediation
named; per-request use raises. Remove the parameter or run the affected stage
with a compatible runner configuration to use them.

For a tuned large-card V2 profile see `higgs_multimodal_qwen3_h200.yaml`
(H200-validated: 64-sequence capacities, larger capture sizes, MPS).

## High Throughput

Use `higgs_multimodal_qwen3_high_throughput.yaml` for medium/high concurrency
serving. This profile keeps Stage 0 `enforce_eager: true`, which preserves the
Higgs-specific local MLP CUDA graph path. It is the throughput-oriented
production profile.

```bash
vllm-omni serve bosonai/higgs-audio-v3-tts-4b \
    --omni --trust-remote-code \
    --deploy-config vllm_omni/deploy/higgs_multimodal_qwen3_high_throughput.yaml
```

Non-CUDA overrides preserve eager execution and the previous connector
geometry. They keep V2 and async scheduling enabled, with the six CUDA-tuned
`audio_mrv2_*` optimizations disabled. They are retained for configuration and
test coverage only: ROCm, NPU, XPU, and MUSA are not supported for Higgs Audio
v3 on this vLLM 0.34 head until platform-specific quality and performance
validation is available. Do not treat these overrides as production support.

## Low Latency

Use `higgs_multimodal_qwen3_low_latency.yaml` for low-concurrency serving
(for example c1-c4) where Stage 0 decode launch overhead dominates. This profile
sets Stage 0 `enforce_eager: false` and explicitly enables vLLM
`FULL_DECODE_ONLY` CUDA graph:

```yaml
compilation_config:
  cudagraph_capture_sizes: [1, 2, 4, 8, 16]
  cudagraph_mode: FULL_DECODE_ONLY
  cudagraph_num_of_warmups: 1
```

FULL_DECODE is controlled by deploy configuration, not by an environment
variable. When this external decode graph is active, the Higgs talker disables
the local MLP CUDA graph automatically.

```bash
vllm-omni serve bosonai/higgs-audio-v3-tts-4b \
    --omni --trust-remote-code \
    --deploy-config vllm_omni/deploy/higgs_multimodal_qwen3_low_latency.yaml
```

## Why the High-Throughput and Low-Latency Profiles Pin Native FlashInfer

The two Stage 0 profiles set `attention_backend: FLASHINFER` and also set:

```yaml
attention_config:
  use_trtllm_attention: false
```

`FLASHINFER` names the overall vLLM attention backend. Starting with vLLM
0.25, that backend can internally route Hopper decode to the TRT-LLM XQA
kernel. The XQA eligibility check answers whether the kernel can run, not
whether it is the fastest kernel for the current workload. In particular, the
decode builder can select XQA statically when all of the following hold:

- the explicit `use_trtllm_attention` override is not `false`;
- the device is Hopper/SM90 and the XQA cubin is available; and
- the query-head count is divisible by the KV-head count.

Higgs has 32 query heads and 8 KV heads, so it passes the head-layout check.
The static gate does not include a model dtype, KV-cache dtype, context-length,
batch-shape, or measured-performance comparison. This matters because kernel
support is not a performance guarantee.

Higgs also has an unusual output shape that must not be confused with its
attention shape. One autoregressive audio step advances one token per request.
The resulting hidden row is then projected into eight codebook distributions
with shape `[1, 8, codebook_size]`. The eight codebooks therefore do not turn
the attention query into `q_len=8`; at concurrency one, XQA still receives a
single BF16 query row. At this small per-step shape, any XQA advantage for
other KV dtypes, batch sizes, or context lengths does not necessarily apply.
The difference is paid in every transformer layer and every generated audio
step, so even a small per-call disadvantage accumulates in TTFP and RTF.

The upstream XQA integration also showed that the result is dtype-sensitive:
its FP8 benchmark favored XQA, while its BF16 benchmark reported 21.16 ms mean
TPOT for XQA versus 19.47 ms for FA3. That is not a direct native-FlashInfer
comparison, but it demonstrates why SM90 compatibility alone is insufficient
to choose the fastest BF16 kernel. See the
[vLLM XQA benchmark](https://github.com/vllm-project/vllm/pull/43232#issuecomment-4501176335).

The end-to-end H800 A/B in
[issue #5584](https://github.com/vllm-project/vllm-omni/issues/5584#issuecomment-5188700218)
isolated the routing decision on the same vLLM 0.25 stack:

| Decode path | TTFP (ms) | RTF |
|---|---:|---:|
| XQA (automatic default) | 175.76 | 0.3405 |
| Native FlashInfer (`use_trtllm_attention: false`) | 166.39 | 0.3233 |
| vLLM 0.24 reference | 166.68 | 0.3209 |

This proves that XQA selection causes the observed local regression and that
native FlashInfer recovers the pre-v0.25 range. It does not by itself identify
one defective XQA instruction or memory transaction. Establishing that deeper
kernel-level cause requires an attention-only microbenchmark and GPU profiler
trace for the exact Higgs batch/context distribution. Until vLLM has a
workload-aware selector or XQA is faster for this BF16 shape, the model-level
pin is the deterministic choice.

## MRv2 INT8 drafting

`higgs_audio_int8` is opt-in. The INT8 copy drafts one eight-codebook row and the
BF16 Talker verifies it using the seeded single-step sampler's noise. Use one
CUDA SM80+ GPU, BF16 target weights, FlashAttention, async scheduling, and
`VLLM_BATCH_INVARIANT=1`. Keep prefix caching disabled. The extra weights reduce
the memory available for the Stage 0 KV cache.

Copy `higgs_multimodal_qwen3_h200.yaml` to `higgs-k1.yaml`. For Stage 0,
merge the fields below into the existing stage entry and remove
`compilation_config.max_cudagraph_capture_size`.

```yaml
max_num_seqs: 16
gpu_memory_utilization: 0.45
async_scheduling: true
enable_prefix_caching: false
attention_backend: FLASH_ATTN
max_num_batched_tokens: 2048
max_model_len: 2048
default_sampling_params:
  max_tokens: 512
compilation_config:
  mode: 0
  cudagraph_capture_sizes: [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
speculative_config:
  method: higgs_audio_int8
  num_speculative_tokens: 1
hf_overrides:
  audio_full_sample_graph: true
  enable_mlp_cudagraph: false
```

For Stage 1, set `max_num_seqs: 16`, `gpu_memory_utilization: 0.2`, and
`hf_overrides.codec_cuda_graph: false`. For the single-step baseline, copy
the resulting profile and remove Stage 0's `speculative_config`.

```bash
VLLM_BATCH_INVARIANT=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
  vllm-omni serve bosonai/higgs-tts-3-4b --omni --trust-remote-code \
  --revision 239f63fb7b02b1aa085f98d9efae5e35cc5523e8 --port 8098 \
  --deploy-config higgs-k1.yaml
```

Send the same seeded request to each server and compare the audio chunk sizes
and completion events.

```bash
curl --fail-with-body --no-buffer http://127.0.0.1:8098/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"bosonai/higgs-tts-3-4b","input":"The morning train arrived just as the lights came on.","seed":42,"max_new_tokens":512,"stream":true,"stream_format":"sse","response_format":"pcm"}'
```

Unseeded requests and the closing delay ramp use single-step sampling.

## Notes

- Stage 1 remains `enforce_eager: true` in every profile (the codec CUDA
  graphs under v2 are stage-local and independent of `compilation_config`).
- The MRV2 default runs FLASH_ATTN, where the XQA routing question below does
  not arise; the pin matters only for the FLASHINFER-shaped profiles.
- Keep `VLLM_USE_DEEP_GEMM=0` and `VLLM_MOE_USE_DEEP_GEMM=0` for this model
  unless DeepGEMM support is revalidated.
- Revalidate end-to-end throughput and audio quality before changing the default
  auto-discovered config.

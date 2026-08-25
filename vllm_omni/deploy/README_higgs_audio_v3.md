# Higgs-Audio V3 Deploy Profiles

Higgs-Audio V3 has two Stage 0 graph profiles. They are intentionally separate
because the graph paths are mutually exclusive.

## High Throughput

Use `higgs_multimodal_qwen3_high_throughput.yaml` for medium/high concurrency
serving. This profile keeps Stage 0 `enforce_eager: true`, which preserves the
Higgs-specific local MLP CUDA graph path. It is the default production profile
for throughput-oriented serving.

`higgs_multimodal_qwen3.yaml` is kept as the auto-discovered default deploy
config for `model_type=higgs_multimodal_qwen3`, and matches the high-throughput
profile.

```bash
vllm-omni serve bosonai/higgs-audio-v3-tts-4b \
    --omni --trust-remote-code \
    --deploy-config vllm_omni/deploy/higgs_multimodal_qwen3_high_throughput.yaml
```

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

## Why Stage 0 Pins Native FlashInfer

The three Stage 0 profiles set `attention_backend: FLASHINFER` and also set:

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

## Notes

- Stage 1 remains `enforce_eager: true` in both profiles.
- Keep `VLLM_USE_DEEP_GEMM=0` and `VLLM_MOE_USE_DEEP_GEMM=0` for this model
  unless DeepGEMM support is revalidated.
- Revalidate end-to-end throughput and audio quality before changing the default
  auto-discovered config.

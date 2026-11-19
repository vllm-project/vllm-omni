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

## MRv2 self-speculation (experimental)

The `higgs_audio_self` method drafts one row per step from a row the verifier
already sampled, so it needs no drafter checkpoint. It is off unless Stage 0
sets `speculative_config`, and seeded output matches single-step decoding only
when both runs set `VLLM_BATCH_INVARIANT=1`. The
[L20 comparison](https://github.com/vllm-project/vllm-omni/issues/9795#issuecomment-6998547374)
found no throughput gain on vLLM 0.33.0.

Copy `higgs_multimodal_qwen3_mrv2_h200.yaml` to `higgs-k1.yaml`. For Stage 0,
merge the fields below into the existing stage entry and remove
`compilation_config.max_cudagraph_capture_size`.

```yaml
max_num_seqs: 16
gpu_memory_utilization: 0.45
async_scheduling: false
max_num_batched_tokens: 2048
max_model_len: 2048
default_sampling_params:
  max_tokens: 512
compilation_config:
  mode: 0
  cudagraph_capture_sizes: [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
speculative_config:
  method: higgs_audio_self
  num_speculative_tokens: 1
hf_overrides:
  audio_full_sample_graph: false
  enable_mlp_cudagraph: false
```

For Stage 1, set `max_num_seqs: 16`, `gpu_memory_utilization: 0.2`, and
`hf_overrides.codec_cuda_graph: false`. For the single-step baseline, copy
the resulting profile, remove Stage 0's `speculative_config`, and set Stage 0
`async_scheduling: true`; only speculation runs this payload synchronously.

```bash
VLLM_BATCH_INVARIANT=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
  vllm-omni serve bosonai/higgs-tts-3-4b --omni --trust-remote-code \
  --revision 239f63fb7b02b1aa085f98d9efae5e35cc5523e8 --port 8098 \
  --deploy-config higgs-k1.yaml
```

Send the same seeded request to each server to check streaming.

```bash
curl --fail-with-body --no-buffer http://127.0.0.1:8098/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"bosonai/higgs-tts-3-4b","input":"The morning train arrived just as the lights came on.","seed":42,"max_new_tokens":512,"stream":true,"stream_format":"sse","response_format":"pcm"}'
```

Compare the `speech.audio.delta` chunk sizes and the final
`speech.audio.done` event. Unseeded requests and the closing delay ramp use
single-step sampling.
Keep enough KV capacity to avoid preempting requests after audio generation
starts, since replaying an audio prefix is unsupported.

## Notes

- Stage 1 remains `enforce_eager: true` in both profiles.
- Keep `VLLM_USE_DEEP_GEMM=0` and `VLLM_MOE_USE_DEEP_GEMM=0` for this model
  unless DeepGEMM support is revalidated.
- Revalidate end-to-end throughput and audio quality before changing the default
  auto-discovered config.

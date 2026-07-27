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

## Notes

- Stage 1 remains `enforce_eager: true` in both profiles.
- Keep `VLLM_USE_DEEP_GEMM=0` and `VLLM_MOE_USE_DEEP_GEMM=0` for this model
  unless DeepGEMM support is revalidated.
- Revalidate end-to-end throughput and audio quality before changing the default
  auto-discovered config.

# Higgs-Audio V3 Deploy Profiles

Higgs-Audio V3 has separate Stage 0 profiles for high-throughput and
low-latency serving. Both use full-decode CUDA graphs, but select different
capture buckets, audio runtimes, and codec chunk sizes for their target loads.

## High Throughput

Use `higgs_multimodal_qwen3_high_throughput.yaml` for medium/high concurrency
serving. This profile sets Stage 0 `enforce_eager: false` and uses vLLM
`FULL_DECODE_ONLY` CUDA graph with capture buckets through c64. It is the
default production profile for throughput-oriented serving.

The high-throughput and auto-discovered profiles select request-aware audio
fast paths directly in the deploy config. The high-throughput profile uses:

```yaml
enable_prefix_caching: false
async_scheduling: true
hf_overrides:
  audio_state_step_mode: compile
  audio_sampler_mode: torch
compilation_config:
  custom_ops: ["none"]
  cudagraph_capture_sizes: [1, 2, 4, 8, 16, 32, 64]
  cudagraph_mode: FULL_DECODE_ONLY
```

Prefix caching stays disabled because the pending postprocess overlap cannot
currently be combined with the Omni prefix-cache state path. Custom deploy
configs retain the compatibility defaults (`legacy` state updates and the
PyTorch sampler) unless they set these `hf_overrides` explicitly.

`higgs_multimodal_qwen3.yaml` is kept as the auto-discovered default deploy
config for `model_type=higgs_multimodal_qwen3`. It keeps the local MLP graph,
PyTorch audio sampler, 25-frame codec chunks, and a conservative
`max_num_seqs: 16`; the explicit high-throughput profile raises both stages to
64 and uses the full-decode graph, PyTorch sampler, and 75-frame codec chunks.

The explicit high-throughput profile emits 75 new codec frames per steady-state
chunk while retaining a one-frame initial chunk. The larger steady-state chunk
amortizes Stage 1 scheduling and codec decode overhead at c64; the small initial
chunk preserves the early streaming response. The auto-discovered c16 and
low-latency profiles retain 25-frame steady-state chunks.

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

The low-latency profile enables asynchronous scheduling and keeps prefix
caching disabled, but intentionally leaves the audio state and sampler at their
compatibility defaults to avoid compiled-state warm-up at low concurrency.

```bash
vllm-omni serve bosonai/higgs-audio-v3-tts-4b \
    --omni --trust-remote-code \
    --deploy-config vllm_omni/deploy/higgs_multimodal_qwen3_low_latency.yaml
```

## Notes

- Stage 1 remains `enforce_eager: true` in both profiles.
- `HIGGS_AUDIO_V3_AUDIO_STATE_STEP` and `HIGGS_AUDIO_V3_AUDIO_SAMPLER` override
  the corresponding model/deploy-config values for diagnostics and A/B tests.
  The startup log reports the resolved modes. Production deployments should
  prefer the explicit `hf_overrides` shown above.
- Keep `VLLM_USE_DEEP_GEMM=0` and `VLLM_MOE_USE_DEEP_GEMM=0` for this model
  unless DeepGEMM support is revalidated.
- Revalidate end-to-end throughput and audio quality before changing the default
  auto-discovered config.

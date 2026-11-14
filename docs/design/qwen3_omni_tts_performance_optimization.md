# Speech Generation on vLLM-Omni: Performance Optimizations for Qwen3-Omni and Qwen3-TTS

## Summary

vLLM-Omni supports end-to-end serving for speech-generating models, including both **Qwen3-Omni** (multimodal understanding + speech) and **Qwen3-TTS** (text-to-speech). Despite their different architectures, both models share the same multi-stage pipeline design and benefit from the same set of stacked optimizations:

1. **Batching** improves GPU utilization stage by stage and increases overall throughput.
2. **CUDA Graph** reduces CPU launch overhead and decode-time jitter on stable shapes.
3. **Async Chunk and Streaming Output** overlap compute and communication across stages and emit audio incrementally, improving both TTFP and E2E.

### Model architectures

**Qwen3-Omni** is a native multimodal model that understands text, audio, image, and video inputs, and generates both text and speech outputs. Its pipeline has three stages:

- **Thinker**: multimodal understanding and text generation
- **Talker (+ Talker-MTP / code predictor path)**: converts semantic/text representations into codec tokens
- **Code2Wav**: decodes codec tokens into waveform audio

**Qwen3-TTS** is a lightweight, high-quality text-to-speech model. Its pipeline has two stages:

- **Talker (AR decoder)**: auto-regressively generates codec tokens from text input
- **Code2Wav (vocoder)**: decodes codec tokens into waveform audio

The optimizations described in this post apply to both models. We present results for each side by side.

### vLLM-Omni vs HF Transformers

Compared with **HF Transformers** (offline, single request), vLLM-Omni with the full optimization stack delivers dramatically lower latency and higher efficiency for both models.

**Qwen3-Omni** (A100):

<table><tr>
<td><img src="figures/omni/E2EL_s_vllm_omni_vs_transformers.png" alt="Qwen3-Omni E2EL: vLLM vs HF" width="100%"/></td>
<td><img src="figures/omni/TTFP_s_vllm_omni_vs_transformers.png" alt="Qwen3-Omni TTFP: vLLM vs HF" width="100%"/></td>
<td><img src="figures/omni/RTF_vllm_omni_vs_transformers.png" alt="Qwen3-Omni RTF: vLLM vs HF" width="100%"/></td>
</tr></table>

| Metric | vLLM-Omni | HF Transformers | Improvement |
| --- | --- | --- | --- |
| E2E latency (s) | 23.78 | 336.10 | ~93% reduction |
| TTFP (s) | 0.934 | 336.10 | ~99.7% reduction |
| RTF | 0.32 | 3.776 | ~91% reduction (~12× faster) |

- **E2E latency**: 23.78 s vs 336.10 s - **~93%** reduction
- **TTFP**: 0.934 s vs 336.10 s - **~99.7%** reduction
- **RTF**: 0.32 vs 3.776 - **~91%** reduction (~12x faster)

**Qwen3-TTS** (H200, concurrency 1):

<table><tr>
<td><img src="figures/tts/Mean_E2EL_(ms)_vllm_omni_vs_transformers.png" alt="Qwen3-TTS E2EL: vLLM vs HF" width="100%"/></td>
<td><img src="figures/tts/Mean_AUDIO_TTFP_(ms)_vllm_omni_vs_transformers.png" alt="Qwen3-TTS TTFP: vLLM vs HF" width="100%"/></td>
<td><img src="figures/tts/Mean_AUDIO_RTF_vllm_omni_vs_transformers.png" alt="Qwen3-TTS RTF: vLLM vs HF" width="100%"/></td>
</tr></table>

| Metric | vLLM-Omni | HF Transformers | Improvement |
| --- | --- | --- | --- |
| E2E latency (ms) | 941 | 15,513 | ~94% reduction |
| TTFP (ms) | 64 | 15,513 | ~99.6% reduction (242× faster) |
| RTF | 0.16 | 2.64 | ~94% reduction (~16.5× faster) |

- **E2E latency**: 941 ms vs 15,513 ms - **~94%** reduction
- **TTFP**: 64 ms vs 15,513 ms - **~99.6%** reduction (242x faster)
- **RTF**: 0.16 vs 2.64 - **~94%** reduction (~16.5x faster)

### Stacked optimization summary

Each optimization stacks on the previous one. The summary plots below show the cumulative effect at each step, with one line per concurrency level (1, 4, 10).

**Qwen3-Omni** (A100):

<table><tr>
<td><img src="figures/omni/Summary_E2EL_ms_vs_features.png" alt="Qwen3-Omni E2EL: stacked optimization" width="100%"/></td>
<td><img src="figures/omni/Summary_TTFP_ms_vs_features.png" alt="Qwen3-Omni TTFP: stacked optimization" width="100%"/></td>
<td><img src="figures/omni/Summary_RTF_vs_features.png" alt="Qwen3-Omni RTF: stacked optimization" width="100%"/></td>
</tr></table>

- **E2EL reduction**: ~74% at concurrency 10 (410,054 ms -> 104,901 ms); ~90% at concurrency 1 (426,529 ms -> 41,216 ms)
- **TTFP reduction**: ~96% at concurrency 10 (409,705 ms -> 16,482 ms); ~99.7% at concurrency 1 (426,078 ms -> 1,164 ms)
- **RTF reduction**: ~74% at concurrency 10 (2.83 -> 0.74); ~90% at concurrency 1 (2.08 -> 0.21)

**Qwen3-TTS** (H200):

<table><tr>
<td><img src="figures/tts/Summary_mean_e2e_ms_vs_features.png" alt="Qwen3-TTS E2EL: stacked optimization" width="100%"/></td>
<td><img src="figures/tts/Summary_mean_ttfp_ms_vs_features.png" alt="Qwen3-TTS TTFP: stacked optimization" width="100%"/></td>
<td><img src="figures/tts/Summary_mean_rtf_vs_features.png" alt="Qwen3-TTS RTF: stacked optimization" width="100%"/></td>
</tr></table>

- **E2EL reduction**: ~85% at concurrency 10 (12,141 ms -> 1,767 ms); ~29% at concurrency 1 (1,323 ms -> 941 ms)
- **TTFP reduction**: ~96.5% at concurrency 10 (12,141 ms -> 425 ms); ~95% at concurrency 1 (1,323 ms -> 64 ms)
- **RTF reduction**: ~86% at concurrency 10 (2.19 -> 0.31); ~30% at concurrency 1 (0.23 -> 0.16)

**Benchmark environment:**

| | Qwen3-Omni | Qwen3-TTS |
| --- | ----------------------------- | --- |
| **GPU** | A100 | H200 |
| **Model** | Qwen3-Omni-30B-A3B-Instruct | Qwen3-TTS-12Hz-1.7B-CustomVoice |
| **vLLM** | v0.17.0 | v0.18.0 |
| **vllm-omni** | commit 199f7832 | v0.18.0rc2 |
| **CUDA** | 12.9 | 12.8 |

This post walks through each optimization in the same order they are typically enabled in practice, then ends with deployment playbooks for both models.

---

## Pipeline Batching

### How stage-wise batching works

For both Qwen3-Omni and Qwen3-TTS, batching is a pipeline-level optimization:

- Requests are grouped per stage using `stages[].max_num_seqs`
- Each stage executes batch inference with its own scheduler/worker
- Stage outputs are routed to downstream stages with per-request mapping preserved

**Batching strategy by stage:** The understanding and decode stages (Thinker for Omni, Talker for both) use **continuous batching**: requests can join and leave the batch over time. Code2Wav uses **static batching**: once a batch is formed, the stage runs the whole batch before starting the next. This matches the decode pattern of Code2Wav and keeps implementation simple while still improving throughput.

### Batching results (Baseline vs. Batch)

Batching alone greatly reduces E2EL and RTF across all concurrencies. The biggest gains appear at high concurrency where requests share GPU resources.

**Qwen3-Omni** (A100):

<table><tr>
<td><img src="figures/omni/Mean_E2EL_ms_Baseline_vs_Batch.png" alt="Qwen3-Omni E2EL: Baseline vs Batch" width="100%"/></td>
<td><img src="figures/omni/Mean_AUDIO_TTFP_ms_Baseline_vs_Batch.png" alt="Qwen3-Omni TTFP: Baseline vs Batch" width="100%"/></td>
<td><img src="figures/omni/Mean_AUDIO_RTF_Baseline_vs_Batch.png" alt="Qwen3-Omni RTF: Baseline vs Batch" width="100%"/></td>
</tr></table>

| Metric | Concurrency | Baseline | + Batch | Improvement |
| --- | --- | --- | --- | --- |
| E2EL (ms) | 1 | 426,529 | 307,719 | 1.4× |
| E2EL (ms) | 4 | 407,213 | 376,934 | 1.1× |
| E2EL (ms) | 10 | 410,054 | 234,844 | 1.7× |
| TTFP (ms) | 1 | 426,078 | 307,262 | 1.4× |
| TTFP (ms) | 4 | 406,843 | 376,466 | 1.1× |
| TTFP (ms) | 10 | 409,705 | 234,557 | 1.7× |
| RTF | 1 | 2.08 | 1.51 | 1.4× |
| RTF | 4 | 2.55 | 1.83 | 1.4× |
| RTF | 10 | 2.83 | 2.28 | 1.2× |

At concurrency 10, E2EL drops from ~410 s to ~235 s; at concurrency 1, from ~427 s to ~308 s.

**Qwen3-TTS** (H200):

<table><tr>
<td><img src="figures/tts/Mean_mean_e2e_ms_baseline_vs_batch.png" alt="Qwen3-TTS E2EL: Baseline vs Batch" width="100%"/></td>
<td><img src="figures/tts/Mean_mean_ttfp_ms_baseline_vs_batch.png" alt="Qwen3-TTS TTFP: Baseline vs Batch" width="100%"/></td>
<td><img src="figures/tts/Mean_mean_rtf_baseline_vs_batch.png" alt="Qwen3-TTS RTF: Baseline vs Batch" width="100%"/></td>
</tr></table>

| Metric | Concurrency | Baseline | + Batch | Improvement |
| --- | --- | --- | --- | --- |
| E2EL (ms) | 1 | 1,323 | 1,339 | 1.0× |
| E2EL (ms) | 4 | 5,171 | 1,471 | 3.5× |
| E2EL (ms) | 10 | 12,141 | 1,705 | 7.1× |
| RTF | 1 | 0.230 | 0.234 | 1.0× |
| RTF | 4 | 0.908 | 0.255 | 3.6× |
| RTF | 10 | 2.186 | 0.292 | 7.5× |
| Throughput (audio-s/wall-s) | 10 | 3.99 | 33.53 | 8.4× |

At concurrency 10, batching alone brings Qwen3-TTS RTF from 2.19 (slower than realtime) down to 0.29 (faster than realtime), and throughput from 4.0 to 33.5 audio-sec/wall-sec.

---

## CUDA Graph on the Critical Decode Path

### Why CUDA Graph helps here

In decode-heavy serving, repeatedly launching many small kernels from CPU can become a visible overhead. CUDA Graph reduces this overhead by capturing and replaying stable execution graphs.

In deploy configs, this is represented by `enforce_eager: false` for stages where graph capture is desired (Thinker/Talker), while Code2Wav keeps eager mode depending on stage behavior.

### CUDA Graph results on top of batching

**Qwen3-Omni** (A100):

<table><tr>
<td><img src="figures/omni/Mean_E2EL_ms_Batch_vs_Batch_CUDA_Graph.png" alt="Qwen3-Omni E2EL: Batch vs CUDA Graph" width="100%"/></td>
<td><img src="figures/omni/Mean_AUDIO_TTFP_ms_Batch_vs_Batch_CUDA_Graph.png" alt="Qwen3-Omni TTFP: Batch vs CUDA Graph" width="100%"/></td>
<td><img src="figures/omni/Mean_AUDIO_RTF_Batch_vs_Batch_CUDA_Graph.png" alt="Qwen3-Omni RTF: Batch vs CUDA Graph" width="100%"/></td>
</tr></table>

| Metric | Concurrency | Batch | + CUDA Graph | Improvement |
| --- | --- | --- | --- | --- |
| E2EL (ms) | 1 | 307,719 | 61,613 | 5.0× |
| E2EL (ms) | 4 | 376,934 | 79,019 | 4.8× |
| E2EL (ms) | 10 | 234,844 | 126,867 | 1.9× |
| TTFP (ms) | 1 | 307,262 | 61,257 | 5.0× |
| TTFP (ms) | 4 | 376,466 | 78,634 | 4.8× |
| TTFP (ms) | 10 | 234,557 | 126,534 | 1.9× |
| RTF | 1 | 1.51 | 0.32 | 4.7× |
| RTF | 4 | 1.83 | 0.43 | 4.3× |
| RTF | 10 | 2.28 | 0.90 | 2.5× |

For the larger Qwen3-Omni model (30B-A3B), CUDA Graph provides a significant improvement. At concurrency 1, E2EL drops from ~308 s to ~62 s; at concurrency 10, from ~235 s to ~127 s.

**Qwen3-TTS** (H200):

<table><tr>
<td><img src="figures/tts/Mean_mean_e2e_ms_batch_vs_cuda_graph.png" alt="TTS E2EL: Batch vs +CG" width="100%"/></td>
<td><img src="figures/tts/Mean_mean_ttfp_ms_batch_vs_cuda_graph.png" alt="TTS TTFP: Batch vs +CG" width="100%"/></td>
<td><img src="figures/tts/Mean_mean_rtf_batch_vs_cuda_graph.png" alt="TTS RTF: Batch vs +CG" width="100%"/></td>
</tr></table>

| Metric | Concurrency | Batch | + CUDA Graph | Improvement |
| --- | --- | --- | --- | --- |
| E2EL (ms) | 1 | 1,339 | 733 | 1.8× |
| E2EL (ms) | 4 | 1,471 | 987 | 1.5× |
| E2EL (ms) | 10 | 1,705 | 1,197 | 1.4× |
| RTF | 1 | 0.234 | 0.124 | 1.9× |
| RTF | 10 | 0.292 | 0.203 | 1.4× |
| Throughput (audio-s/wall-s) | 10 | 33.53 | 47.15 | 1.4× |

At concurrency 1, CUDA Graph reduces E2EL from 1,339 ms to 733 ms and RTF from 0.234 to 0.124 - nearly a 2x improvement. The benefit is consistent across all concurrency levels.

---

## Async Chunk and Streaming Output: Earlier Audio and Cross-Stage Overlap

### Why this step matters for first-packet latency

Two mechanisms work together to improve user-visible latency:

- **Streaming output**: audio streaming emits audio chunks as soon as they are decoded (lower **TTFP**). Without streaming, the client waits for larger buffers or end-of-sequence.
- **Async chunk** is the main enabler for *earlier* audio: instead of handing off whole-request results between stages, each stage forwards **chunks** so the next stage can start as soon as the first chunk is ready. For Omni: Thinker -> Talker forwards hidden-state chunks; for both: Talker -> Code2Wav forwards codec chunks; Code2Wav decodes and emits packets incrementally. This **overlaps compute and communication** across stages and directly reduces time-to-first-audio-packet (TTFP) and end-to-end latency (E2EL).

So in practice: streaming output defines *how* bytes are sent to the client; async chunk defines *when* the pipeline can produce the first bytes.

**Dependency between the two:** Async chunk and audio streaming output are mutually dependent. Without async chunk, **audio streaming output cannot truly take effect**. Without audio streaming output, async chunk's **TTFP advantage is not fully realized**: the client would still wait for larger buffers or end-of-sequence instead of hearing the first packet as soon as it is ready. We therefore recommend enabling **both** on top of batching + CUDA Graph; the benchmarks in this post use both.

### Results: Batch + CUDA Graph vs. Batch + CUDA Graph + Async Chunk + Streaming Output

**Qwen3-Omni** (A100):

<table><tr>
<td><img src="figures/omni/Mean_E2EL_ms_Batch_CUDA_Graph_vs_Async_Chunk.png" alt="Qwen3-Omni E2EL: CG vs Async Chunk" width="100%"/></td>
<td><img src="figures/omni/Mean_AUDIO_TTFP_ms_Batch_CUDA_Graph_vs_Async_Chunk.png" alt="Qwen3-Omni TTFP: CG vs Async Chunk" width="100%"/></td>
<td><img src="figures/omni/Mean_AUDIO_RTF_Batch_CUDA_Graph_vs_Async_Chunk.png" alt="Qwen3-Omni RTF: CG vs Async Chunk" width="100%"/></td>
</tr></table>

| Metric | Concurrency | Batch + CG | + Async Chunk | Improvement |
| --- | --- | --- | --- | --- |
| E2EL (ms) | 1 | 61,613 | 41,216 | 1.5× |
| E2EL (ms) | 4 | 79,019 | 67,584 | 1.2× |
| E2EL (ms) | 10 | 126,867 | 104,901 | 1.2× |
| TTFP (ms) | 1 | 61,257 | 1,164 | 53× |
| TTFP (ms) | 4 | 78,634 | 3,152 | 24.9× |
| TTFP (ms) | 10 | 126,534 | 16,482 | 7.7× |
| RTF | 1 | 0.32 | 0.21 | 1.5× |
| RTF | 4 | 0.43 | 0.34 | 1.3× |
| RTF | 10 | 0.90 | 0.74 | 1.2× |

Enabling both brings TTFP down sharply (concurrency 1: 61,257 ms -> 1,164 ms, **~98% reduction**; concurrency 4: 78,634 ms -> 3,152 ms, **~96% reduction**). E2EL and RTF also improve at every concurrency.

**Qwen3-TTS** (H200):

<table><tr>
<td><img src="figures/tts/Mean_mean_e2e_ms_cuda_graph_vs_async_chunk.png" alt="Qwen3-TTS E2EL: CG vs Async Chunk" width="100%"/></td>
<td><img src="figures/tts/Mean_mean_ttfp_ms_cuda_graph_vs_async_chunk.png" alt="Qwen3-TTS TTFP: CG vs Async Chunk" width="100%"/></td>
<td><img src="figures/tts/Mean_mean_rtf_cuda_graph_vs_async_chunk.png" alt="Qwen3-TTS RTF: CG vs Async Chunk" width="100%"/></td>
</tr></table>

| Metric | Concurrency | Batch + CG | + Async Chunk | Improvement |
| --- | --- | --- | --- | --- |
| TTFP (ms) | 1 | 733 | **64** | **11.5×** |
| TTFP (ms) | 4 | 987 | **119** | **8.3×** |
| TTFP (ms) | 10 | 1,197 | **425** | **2.8×** |
| E2EL (ms) | 1 | 733 | 941 | 0.8× |
| E2EL (ms) | 10 | 1,197 | 1,767 | 0.7× |
| RTF | 1 | 0.124 | 0.160 | 0.8× |
| RTF | 10 | 0.203 | 0.314 | 0.6× |

The TTFP improvement is the headline result for both models. For Qwen3-TTS at concurrency 1, users hear the first audio in **64 ms** instead of 733 ms - an **11.5x reduction**. For Qwen3-Omni at concurrency 1, TTFP drops from 61 s to 1.2 s - a **53x reduction**.

### Why E2EL and RTF are higher with async chunk (TTS)

The table above shows that enabling async chunk + streaming *increases* E2EL and RTF for TTS compared to CUDA Graph alone. This is expected - the two configurations optimize for fundamentally different metrics:

- **CUDA Graph (no async chunk)** generates the entire audio end-to-end before returning. No chunking overhead, so total compute is minimized.
- **Async Chunk + Streaming** splits the pipeline into incremental chunks, adding overhead from chunked transport, context overlap in Code2Wav (`codec_left_context_frames=25`), and smaller effective batch sizes per chunk.

**The tradeoff is intentional.** Async chunk trades ~30% higher total compute for **11x faster time-to-first-audio**. For interactive applications (voice assistants, chatbots), TTFP determines perceived responsiveness. For offline batch processing, CUDA Graph without async chunk is the better choice.

---

## TTS-Specific: Code Predictor Re-prefill + `torch.compile`

Qwen3-TTS has a **code predictor** - a small 5-layer transformer that generates residual codebook tokens (groups 1 through Q-1) autoregressively. Each AR step operates on very short sequences (2 to ~16 tokens).

The naive approach uses a KV cache for this small transformer, similar to the main Talker. But the KV cache machinery (block tables, slot mappings, paged attention) introduces significant overhead relative to the tiny model. Two optimizations replace that:

### Re-prefill (stateless forward, no KV cache)

Instead of maintaining a KV cache across steps, the code predictor **re-feeds the full growing sequence** at each AR step using `F.scaled_dot_product_attention`. With sequences of at most ~16 tokens through 5 layers, the O(T^2) attention cost is negligible - and removing the KV cache machinery (block table management, `set_forward_context`, slot mapping) saves far more time than it costs.

### `torch.compile` on the code predictor forward

The 5-layer transformer forward pass launches ~60 small CUDA kernels per step. `torch.compile(mode="default", dynamic=True)` fuses these into fewer kernels via Inductor:

```python
self._compiled_model_fwd = torch.compile(
    self.model.forward,
    mode="default",    # no Inductor CUDA graphs, avoids conflict with vLLM's CUDAGraphWrapper
    dynamic=True,      # sequence length grows each step (2, 3, ..., num_groups+1)
)
```

`mode="default"` is used instead of `mode="reduce-overhead"` to avoid conflicts with vLLM's own CUDA graph capture on the main Talker model. `dynamic=True` handles the growing sequence length without recompilation.

The historical benchmark results above use re-prefill. It remains the default
for Qwen3-TTS and the shared Qwen3-Omni predictor.

### Fused MRv2 serving boundaries

The single-stage `qwen3_tts_fused_single_gpu.yaml` profile decodes PCM in the
Talker process. Its CUDA MRv2 prefill path builds a projected text embedding
table before KV profiling (about 0.6 GB for the 1.7B checkpoint). V1,
two-stage deployments and non-NVIDIA platforms do not allocate that table.
Exact conditioned Talker inputs for preemption replay use additional storage
per active request after KV profiling; keep the headroom documented in the YAML.

The fused Talker sampler registers only when its platform and processor list
are supported. Installed logits-processor plugins keep the stock sampler and
immediate request-state flush ordering. Unsupported sampling settings retain
the stock registration, including rejection sampling. `compute_logits` keeps
the codec vocabulary mask for both sampling and prompt logprobs.

The PDL predictor requires NVIDIA Hopper or newer. A predictor with capacity
above 64 retains the cuBLAS weights and selects that path for batches above
64. These optimizations do not change reference/speaker encoder precision or
the default attention backend. Reference encoding uses the main BF16 path;
backend-specific throughput measurements should use explicit overrides.

### Experimental Qwen3-TTS single-GPU throughput profile

The opt-in `qwen3_tts_high_concurrency_mrv2_single_gpu.yaml` profile uses a
Qwen3-TTS-specific execution path on CUDA with BF16 weights:

- `code_predictor_kv_cache` reuses each layer's keys and values across the 15
  residual-codebook predictions within one audio frame. The first step writes
  two positions; subsequent steps write one. Scratch storage is bounded by the
  configured batch capacity, overwritten for every frame, and never shared
  between requests across frames. The worker's serialized CUDA stream owns it;
  it is separate from the Talker's paged KV cache.
- A short causal/GQA Triton kernel masks unwritten K/V loads. Dynamic-batch
  `torch.compile` warms the predictor before the runner captures whole-MTP
  CUDA graphs. This path creates no inner CUDA graphs. Unsupported devices and
  dtypes retain re-prefill; FP16 retains its existing precision safeguards.
- `code_predictor_fused_sampling` fuses temperature scaling, top-k filtering,
  and Gumbel argmax when the runner supplies FP32 uniforms. It preserves top-k
  ties and uses the supplied randomness. Explicit generator paths retain the
  existing sampler. Qwen3-Omni does not read either TTS option.
- `decode_cudnn_benchmark` selects convolution algorithms during Code2Wav
  warmup and graph capture. The process's cuDNN flags are restored afterwards,
  including on capture failure. Captured graphs retain the selected algorithms.

- `decode_time_major_conv` keeps codec convolution activations in time-major
  layout to reduce transposes and copies between layers.
- `talker_first_audio` completes residual prediction immediately after the
  first codebook sample and decodes the first frame in the Talker process.
  The codec advances its state without repeating the delivered frame; the
  orchestrator orders that frame before subsequent codec chunks. This path
  requires CUDA, MRV2, asynchronous chunks, a single-process executor, TP/PP 1,
  and disabled prefix caching. Reference-code requests retain codec delivery.
  Supported Qwen3-TTS profiles enable it explicitly; other models do not opt in.

The predictor, sampling, convolution-layout and autotuning options default to
false outside this throughput profile. It uses fixed 1/25-frame chunks,
72 frames of left context, Talker/codec capacities of 128/64, codec batches
through 8, and synchronous codec scheduling. Client concurrency is independent
of those capacities. This profile also sets `cuda_mps: true`; the runtime owns
a private MPS daemon unless an operator supplies an existing MPS pipe directory.
Ordinary profiles leave MPS disabled. See
[experimental MPS deployment](../configuration/stage_configs.md#experimental-mps-deployment)
for requirements and ownership. Startup includes compilation and graph capture.

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni \
  --deploy-config vllm_omni/deploy/qwen3_tts_high_concurrency_mrv2_single_gpu.yaml \
  --api-server-count 4 --stage-init-timeout 1800 --init-timeout 2400
```

KV reuse changes GEMM shapes and attention reduction order; cuDNN autotuning
can also change waveform numerics. This path is not bitwise equivalent to
re-prefill, and aggregate WER alone cannot establish perceptual or speaker
similarity equivalence. Validate quality on the intended workload before
adopting it. Qwen3-Omni keeps its existing predictor and audio-delivery behavior.

For a controlled ablation, copy the profile and disable individual options while keeping capacities, chunking, hardware, client concurrency,
and warmup identical. Report repeated runs, first-audio latency and failures
alongside throughput. Warm requests measure a ready service; they do not measure
cold-start latency.

---

## TTS-Specific: Dynamic Initial Chunk for Faster First Audio

In the async chunk pipeline, the standard `codec_chunk_frames` is 25 (each chunk = ~2 seconds of audio at 12 Hz). Waiting for 25 frames before forwarding the first chunk to Code2Wav adds unnecessary TTFP. The **initial codec chunk** optimization sends a smaller first chunk so Code2Wav can start decoding earlier.

**Dynamic initial chunk sizing (default behavior):**

Rather than using a fixed initial chunk size, vLLM-Omni dynamically selects it based on current server load. The initial chunk size is chosen from power-of-2 steps [2, 4, 8, 16] based on load factor (`active_requests / max_num_seqs`):

| Server load | Initial chunk frames | Rationale |
| --- | --- | --- |
| Low (e.g. 1/10 active) | **2** (~167 ms of audio) | Minimize TTFP when there's headroom |
| Medium (e.g. 5/10 active) | **4-8** | Balance TTFP vs decode efficiency |
| High (e.g. 10/10 active) | **16** | Larger first chunk to amortize decode cost |

After the initial chunk, all subsequent chunks use the standard `codec_chunk_frames` (25) size.

**How it works in the pipeline:**

1. Talker generates codec tokens auto-regressively
2. The stage input processor checks current load and picks an initial chunk size (e.g. **2 frames** at low load)
3. After that many frames, the first chunk is forwarded to Code2Wav
4. Code2Wav decodes this small chunk and emits the first audio packet
5. Subsequent chunks use the standard 25-frame size for efficient batch decoding

**Per-request override:** Clients can also set a fixed initial chunk size via the API:

```json
{"initial_codec_chunk_frames": 2}
```

This overrides the dynamic calculation for that request.

**Config (server-side):**

```yaml
connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      codec_streaming: true
      codec_chunk_frames: 25              # standard chunk size (~2s of audio)
      codec_left_context_frames: 25
      # initial chunk is computed dynamically by default
      # set initial_codec_chunk_frames: 2 to force a fixed value
```

The 64 ms TTFP result reported above for Qwen3-TTS at concurrency 1 uses the dynamic initial chunk, which picks `initial_codec_chunk_frames=2` at low load. At higher concurrency the dynamic sizing increases the initial chunk to maintain decode efficiency.

---

## Live Demo: Streaming TTS over WebSocket

vLLM-Omni supports real-time streaming audio output for Qwen3-TTS over WebSocket ([PR #1719](https://github.com/vllm-project/vllm-omni/pull/1719)). With `stream_audio: true`, the server sends chunked PCM audio frames as they are generated, so clients can start playback before full sentence synthesis completes.

The WebSocket protocol uses `audio.start` / binary PCM chunks / `audio.done` framing per sentence:

```json
// Client sends:
{"type":"session.config","voice":"Vivian","response_format":"pcm","stream_audio":true}
{"type":"input.text","text":"Hello world. This is a streaming demo."}
{"type":"input.done"}

// Server streams back per sentence:
{"type":"audio.start","sentence_index":0,"sentence_text":"Hello world.","format":"pcm","sample_rate":24000}
<binary PCM chunk 1>
<binary PCM chunk 2>
...
{"type":"audio.done","sentence_index":0,"total_bytes":96000,"error":false}
{"type":"audio.start","sentence_index":1,"sentence_text":"This is a streaming demo.","format":"pcm","sample_rate":24000}
<binary PCM chunk 1>
...
{"type":"audio.done","sentence_index":1,"total_bytes":72000,"error":false}
{"type":"session.done","total_sentences":2}
```

<iframe width="560" height="315" src="https://www.youtube.com/embed/O3IVniwwKNA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

## Deployment Playbook

### Qwen3-Omni

#### 1) Serve with the default 3-stage config

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091
```

Notes:

- `stages[].max_num_seqs` controls stage-level batching.
- Thinker/Talker commonly use `enforce_eager: false` for CUDA Graph paths.
- Code2Wav often remains eager (`enforce_eager: true`) depending on runtime behavior.
- The Qwen3-Omni deploy config pins `moe_backend: triton` for the Thinker and Talker stages. Triton has been more
 stable and faster than the FlashInfer CUTLASS unquantized MoE backend on recent vLLM rebases.

#### 2) Enable async chunk

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091
```

#### 3) Key config knobs

```yaml
async_chunk: true
stages:
  - stage_id: 0  # thinker
    max_num_seqs: 64
    enforce_eager: false
    max_num_batched_tokens: 32768

  - stage_id: 1  # talker
    max_num_seqs: 64
    enforce_eager: false
    max_num_batched_tokens: 32768

  - stage_id: 2  # code2wav
    max_num_seqs: 64
    enforce_eager: true
    max_num_batched_tokens: 51200
```

The async handoff processors are fixed by the registered Qwen3-Omni
`PipelineConfig`; deploy YAML only selects `async_chunk` and runtime sizing.

#### Reproduce Qwen3-Omni benchmarks

```bash
vllm bench serve \
  --dataset-name random \
  --port ${PORT} \
  --model ${MODEL_PATH} \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --max-concurrency ${MAX_CONCURRENCY} \
  --num-prompts ${NUM_PROMPTS} \
  --random-input-len 2500 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --random-output-len 900 \
  --extra_body '{"modalities": ["text","audio"]}'
```

### Qwen3-TTS

#### 1) Serve with async chunk (recommended)

```bash
vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --omni \
  --port 8000
```

The default config (`qwen3_tts.yaml`) enables the full optimization stack:

- Batching with `max_num_seqs: 10` on the Talker stage
- CUDA Graph on the Talker (`enforce_eager: false`)
- Async chunk with streaming transport

#### 2) Serve without async chunk (for comparison)

```bash
vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --omni \
  --port 8000 \
  --no-async-chunk
```

#### 3) Key config knobs

```yaml
async_chunk: true
stages:
  - stage_id: 0  # Talker (AR decoder)
    max_num_seqs: 10
    enforce_eager: false
    max_num_batched_tokens: 512

  - stage_id: 1  # Code2Wav (vocoder)
    max_num_seqs: 1
    enforce_eager: true
    max_num_batched_tokens: 8192

connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      codec_streaming: true
      codec_chunk_frames: 25
      codec_left_context_frames: 25
```

The async Talker-to-Code2Wav processor is defined by the registered Qwen3-TTS
pipeline rather than the deploy YAML.

#### Reproduce Qwen3-TTS benchmarks

```bash
GPU_DEVICE=0 \
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
NUM_PROMPTS=50 \
CONCURRENCY="1 4 10" \
bash benchmarks/qwen3-tts/vllm_omni/run_stacked_benchmark.sh
```

This cycles through four configs (Baseline -> + Batch -> + CUDA Graph -> + Async Chunk + Streaming), benchmarks each at the specified concurrency levels, and generates all comparison figures automatically.

## Qwen3-TTS NPU codec kernels

The NPU loader uses reference codebook normalization and clears any cached
normalized embeddings after loading weights and applying the final decoder dtype.
The optional codebook cache helper remains available to callers that explicitly
prepare it; it is not enabled by the Qwen3-TTS NPU loader.

Qwen3-TTS keeps its Ascend SnakeBeta kernel disabled by default. It supports contiguous, nonempty NCT
BF16 inference tensors within its validated grid size. The kernel preserves
BF16 rounding between multiply, sine, square, scale and addition. Unsupported
dtypes/layouts and training or gradient-enabled calls retain the existing path;
other models retain their existing behavior. Set `npu_fused_snake: true` in the
NPU Code2Wav stage `additional_config` or connector `extra` to opt in.
This deployment policy is not a performance conclusion:
shape-dependent JIT compilation and eager dispatch can outweigh reduced device
work. Shape dimensions are now runtime arguments without specialization; the
earlier eager probe does not qualify this kernel revision. Measure
your graph mode and workload when choosing it. If Triton is unavailable, the NPU
loader keeps the eager path. CUDA behavior is unchanged. Runtime compilation
errors are surfaced rather than silently hidden.

These changes reduce repeated codec work; they do not establish an end-to-end
throughput gain without matched model-level measurements and audio validation.

### NPU optimization defaults and overrides

The default NPU profile enables seven paths: Model Runner V2 with sampling
uniforms, Talker full decode graphs, Predictor frame graphs, frame-local KV,
fused RoPE/KV writes, incremental Codec, and fused SwiGLU. Fused sampling,
native GQA, dual Codec streams, stateless convolution graphs and fused Snake
require explicit opt-in. The loader leaves Codebook caching disabled; incremental
graphs use separate scratch pools and the selected-row kernel uses ordinary
Triton specialization. Existing non-NPU defaults are unchanged.

The Code2Wav connector `extra` can set `npu_conv_graph_shapes`, a list of exact
`[batch, hidden_frames]` shapes (for example `[[1, 1], [1, 25]]`). These are the
convolution segment inputs, not necessarily the new codec chunk lengths: suffix
context can increase the frame dimension. An empty list disables this path;
the NPU default is an empty list.
Incremental decode takes precedence and skips these redundant captures. Explicit shapes
apply when `npu_incremental_codec: false`; omitted shapes keep them disabled.
Selected shapes are captured at model load, after weight and coefficient
preparation; unseen shapes remain eager rather than capturing on a live request.
`enforce_eager=true` skips these captures with a warning. Explicit shape selection
bounds graph memory; include only shapes observed for the target workload.

The graph owns only scratch activations, not request IDs or streaming state.
Its output is copied before returning so a later replay cannot overwrite a
previous request's audio. The decoder keeps its existing per-request cache,
sliding-window and tail handling. Outer NPU graph capture uses the eager body
to avoid nested replay; reload or module conversion releases the inner graphs.
This is a stateless convolution graph optimization, not a replacement of the
streaming cache with fixed request slots.

On NPU BF16, `code_predictor_kv_cache: true` reuses per-layer K/V buffers
inside the predictor frame graph. It prefills the first two positions, then
computes only the new position for each remaining residual codebook. Fixed
capacity buffers are local to each graph batch size; causal masks exclude
unwritten positions, and the next frame overwrites its prefix. They do not
retain state across audio frames or requests. Qwen3-TTS enables this option
by default when NPU frame graphs are enabled. Set it to false to use full
re-prefill frame graphs. Disabling frame graphs also disables the implicit
KV default; unsupported calls still use the existing re-prefill fallback.
If frame capture fails or is skipped and no shared KV path is active, predictor setup
warns that the requested KV cache is unavailable. The warning follows capture
results, not just the requested graph flag.
`code_predictor_fused_sampling: true` opts in to a separate sampling kernel.
Qwen3-TTS leaves it disabled by default on NPU. CUDA and other models retain
their existing defaults; missing Triton uses reference sampling. Sampling retains the original per-request uniform
tensor, ATen scaling and logarithms, and top-k threshold. The NPU kernel combines
threshold masking, score subtraction and first-index argmax. It reads the top-k
threshold directly from its strided column without materializing a copy.
Unsupported input shapes/dtypes use the reference.

`code_predictor_frame_graph: true` captures all residual codebook
steps together and disables the outer MTP graph only on NPU. Other platforms
retain their outer graph when this option is set. It defaults to true on NPU
and requires matching temperature/top-k, top-p of 1, and supplied uniforms.
Unsupported calls retain the re-prefill fallback. Frame graphs also use the NPU fused sampler when enabled and supported, retaining
the same uniforms, ATen logarithms and top-k Gumbel semantics. Explicit false
uses reference sampling in both frame graphs and the per-step fallback.

The cached Predictor path combines RoPE and KV writes in a Triton kernel.
It preserves the eager BF16/FP16 multiply rounding before addition, with
floating-point contraction disabled. FP32 and environments without Triton
retain the eager path. Only the current frame's selected cache positions are
written; future positions remain masked by attention.

`npu_incremental_codec: true` retains convolution tails and
sliding-window KV in per-request slots. It is enabled by default on NPU.
Startup captures full chunks and single frames. Final partial chunks execute
without live graph capture. Each captured graph owns its scratch pool, and
outputs are copied before the next replay. Selected-row state stores retain
Triton's default stride and alignment specialization. Request completion releases and resets the slot; loading weights or
converting the decoder releases its graphs. This path changes the streaming
execution strategy and needs paired audio validation for the target workload.
Its additive attention mask selects eager attention only for incremental calls;
stateless requests retain their configured attention backend. The existing NPU
Code2Wav loader owns process-wide format and compile-mode setup; constructing
an incremental arena does not change those settings.

The Qwen3-TTS predictor and NPU codec loader own their respective defaults;
`qwen3_tts.yaml` does not duplicate these model options. Omitting an option uses
its code default. Override it via stage `additional_config` or connector `extra`.
Explicit connector values override stage values, and explicit false/empty values
remain authoritative. `code_predictor_npu_native_gqa` defaults to false and
`code_predictor_npu_fused_swiglu` defaults to true on NPU. `npu_codec_streams`
defaults to one. An explicit
`npu_codec_streams: 2` without either prerequisite raises a configuration error;
the single-stream default does not require Triton. By contrast,
`enforce_eager=true` deliberately skips stateless convolution graph capture,
including explicitly requested shapes, as the documented graph opt-out.
Predictor frame-local KV is enabled with NPU frame graphs; explicit false retains re-prefill.

Compare actual graph coverage, startup cost, memory, latency and audio quality
on the deployed model before changing these settings; synthetic tests alone do
not establish a service-level improvement.

### Independent Ascend engine migration

The [independent engine](https://github.com/InferMatrixLab/qwen3-tts-rsi-npu-engine)
uses its own scheduler and request arenas. Its performance results are evidence
for that engine and workload, not vLLM-Omni speedup claims. The integration uses
these existing Omni owners:

| Independent engine mechanism | vLLM-Omni integration |
| --- | --- |
| Packed QKV / gate-up projections, native normalization / rotary | Already provided by the shared code predictor and the Talker backend |
| Fixed predictor rotary coefficients (U002) | Predictor already uses precomputed cos/sin tables on NPU; no request-position cache or module replacement is needed |
| Model-fixed prompt embeddings | Existing prompt builder caches projected special tokens and CustomVoice speaker/language prefixes |
| Cached codec codebooks and fused Snake | Codebook cache helper and opt-in `npu_fused_snake`; both disabled by the default NPU loader |
| Prefill/decode graphs and continuous batching | Existing vLLM scheduler, paged KV and Talker graph runner |
| Incremental codec and convolution graphs | `npu_incremental_codec` and `npu_conv_graph_shapes` |
| Native grouped-query predictor attention | Opt-in `code_predictor_npu_native_gqa` |
| Fused predictor SwiGLU | NPU-default `code_predictor_npu_fused_swiglu` |
| Overlapping disjoint codec groups (U005) | Opt-in `npu_codec_streams: 2` |

Override these defaults in stage `additional_config` or connector `extra`.
On NPU fused SwiGLU defaults to `true`, native GQA defaults to `false`, and
codec decoding defaults to one stream. Two streams require an explicit opt-in,
the incremental decoder and Triton. Native predictor attention
passes the original grouped K/V heads to Ascend inference attention instead of
materializing repeated heads, with an explicit causal mask. The supported path
uses FP16/BF16; unsupported inputs retain the existing attention path. Native SwiGLU
uses the existing packed gate/up projection for BF16 NPU inputs and retains the
ordinary SiLU/multiply path on other devices or dtypes. These native operators
can change floating-point rounding and therefore sampled codec tokens. Qualify
paired audio quality for the deployment; operator tolerances are not a
waveform-equivalence guarantee. CUDA behavior and shared predictor defaults
are unchanged.

Two codec streams require `npu_incremental_codec: true` and Triton Ascend.
They overlap the existing groups without changing group sizes, request order,
chunk lengths or `decode_batch_max_size`. Each lane owns separate graph inputs,
scratch and output copies. A selected-row store kernel updates only each group's
disjoint convolution/KV slots; a native whole-tensor update is not safe for this
concurrency. Inputs are prepared on the owner stream, both lanes wait for it,
and the owner waits for both lanes before returning or propagating an exception.
Only then may request completion reset/reuse slots. Returned audio is independent
of graph scratch. A single group or outer graph capture stays on the owner
stream; later multi-group
calls wait for that work before reusing lane storage. Invalid or overlapping
request slots are rejected before any group is submitted.

Prewarming captures the same full-chunk and single-frame shapes for both lanes,
doubling the number of incremental codec graphs. Other lengths remain eager;
no live request captures a new graph. Request state remains one shared slot
arena with disjoint writers, while graph scratch and startup costs increase.
Reload and module conversion use the existing graph cleanup. Benchmark memory,
startup and steady-state latency separately.

The independent engine's packed prefill-KV admission and single-request borrowed
views belong to its contiguous arena and request loop. They are not copied over
vLLM's paged-KV/input-batch owners. Similarly, inverse-CDF sampling is not
substituted for Omni's Gumbel sampling: the latter retains its per-request
uniforms and existing token-selection semantics. A scheduler or sampler redesign
would require a separate semantic and service comparison.

### NPU Talker decode graph mode

The standard NPU Model Runner V2 profile uses `FULL_AND_PIECEWISE` for the
Talker: prefill retains piecewise graphs and decode replays a complete backbone
graph. Ascend's V2 graph manager owns attention metadata updates. Code2Wav
retains its `PIECEWISE` setting and its incremental decoder graphs.

The predictor frame graph remains independent of the backbone graph. Its
outer MTP wrapper stays disabled on NPU to avoid nested graph replay. Enabling
a complete MTP wrapper under a piecewise backbone did not improve the measured
service throughput and is not part of this configuration.

`FULL_DECODE_ONLY` also captures decode, but leaves prefill eager; the matched
CustomVoice experiment found higher first-audio latency than the combined mode.
Graph modes can change sampled codec tokens through numerical differences.
Validate normal termination, ASR error and speaker similarity for the deployed
model and workload rather than assuming byte-identical audio.

This combination was exercised with Qwen3-TTS 1.7B CustomVoice and Base on
Ascend 910B1. It does not qualify other Ascend SKUs or Model Runner V1. To use
V1 on NPU, also set stage 0 `cudagraph_mode: PIECEWISE` in the NPU platform
section of a copied deployment configuration. V1 full decode graphs can start
and replay but exhaust generation limits: a matched short-input smoke test
with `FULL_AND_PIECEWISE` reached the token ceiling, while `PIECEWISE` completed
normally. The Talker now rejects NPU V1 full decode modes before loading
weights. V1 `PIECEWISE`, `NONE` and eager execution remain available. This
configuration check does not qualify V1 performance or audio quality.

### NPU Model Runner V2 sampling inputs

The standard Qwen3-TTS NPU profile selects Model Runner V2. Sampling uniforms
are prepared from request-local generators independently of the outer MTP graph
wrapper on NPU. Other backends keep request-local per-step generators when
the outer MTP wrapper is absent; their graph paths still prepare uniform
tensors. This lets NPU predictor-owned frame graphs and fused sampling consume them
under PIECEWISE or when nested outer graphs are disabled. Graph padding retains
its own rows; seeded active rows keep request-local random streams. The changed
random draw granularity requires fresh end-to-end quality and performance
qualification; older V1 or V2 results do not qualify this combination.

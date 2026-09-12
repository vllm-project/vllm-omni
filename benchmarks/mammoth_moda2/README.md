# MammothModa2 AR prefix-cache benchmark

This benchmark compares three cache states for the MammothModa2 AR stage:

- **A**: automatic prefix caching disabled.
- **B1**: caching enabled, but reset after warmup and before every measured
  request. Every sample must be a verified cache miss.
- **B2**: caching enabled and warmed with the target request. Every measured
  sample must be a verified cache hit.

The harness records cache accounting, TTFT, end-to-end AR latency, generated
token throughput, and device-wide peak GPU memory. It uses the AR-only deploy
profiles and does not measure DiT execution.

## Prerequisites

- One isolated NVIDIA GPU with enough memory for MammothModa2-Preview.
- The model available locally or from Hugging Face.
- `pynvml`, PyTorch, vLLM, and vLLM-Omni installed.
- No other process using the measured GPU because memory is sampled
  device-wide through NVML.

## Reproduce A/B1/B2

Run each scenario in a separate process. The commands below use the 5,627-token
synthetic stress prompt from the reported result:

```bash
python benchmarks/mammoth_moda2/benchmark_prefix_cache.py \
  --scenario a \
  --model /root/models/MammothModa2-Preview \
  --prompt-repeat 800 \
  --iterations 10 \
  --output /tmp/mammoth_prefix_a.json

python benchmarks/mammoth_moda2/benchmark_prefix_cache.py \
  --scenario b1 \
  --model /root/models/MammothModa2-Preview \
  --prompt-repeat 800 \
  --iterations 10 \
  --output /tmp/mammoth_prefix_b1.json

python benchmarks/mammoth_moda2/benchmark_prefix_cache.py \
  --scenario b2 \
  --model /root/models/MammothModa2-Preview \
  --prompt-repeat 800 \
  --iterations 10 \
  --output /tmp/mammoth_prefix_b2.json
```

The harness fails rather than reporting a mislabeled scenario when cache
accounting does not satisfy these contracts:

- A: `cached_tokens == 0` and `cache_creation_tokens == 0`.
- B1: `cached_tokens == 0` and `cache_creation_tokens > 0`.
- B2: `cached_tokens > 0` and `cache_creation_tokens == 0`.

For B1, the model and kernels are warmed first. The scheduler is then paused,
its prefix cache is cleared, and it is resumed before each timed request. Cache
reset time is excluded from request latency.

## Diagnostic profiler

Profiler traces are diagnostic artifacts and must not be used as authoritative
latency measurements. Collect one request after warmup:

```bash
python benchmarks/mammoth_moda2/benchmark_prefix_cache.py \
  --scenario a \
  --model /root/models/MammothModa2-Preview \
  --prompt-repeat 80 \
  --iterations 1 \
  --profile-dir /tmp/mammoth_profile_a \
  --output /tmp/mammoth_profile_a.json

python benchmarks/mammoth_moda2/benchmark_prefix_cache.py \
  --scenario b2 \
  --model /root/models/MammothModa2-Preview \
  --prompt-repeat 80 \
  --iterations 1 \
  --profile-dir /tmp/mammoth_profile_b2 \
  --output /tmp/mammoth_profile_b2.json
```

Analyze the exported rank-0 traces:

```bash
python .claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py \
  /tmp/mammoth_profile_a/**/trace_rank0.json \
  --min-gap-ms 1 \
  --topn 20

python .claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py \
  /tmp/mammoth_profile_b2/**/trace_rank0.json \
  --min-gap-ms 1 \
  --topn 20
```

## Reported environment

- GPU: NVIDIA A800-SXM4-80GB
- vLLM: 0.28.0
- PyTorch: 2.13.0+cu129
- Precision: BF16
- Execution: eager
- Image grid: 256 x 256
- Prompt tokens: 5,627
- Generated tokens: 273
- Warmup: one request
- Measurements: 10 requests per scenario

The raw per-request measurements and profiler summary are in
`prefix_cache_results.ndjson` next to this document. It contains
one metadata record, one record per A/B1/B2 scenario, and one profiler-summary
record.

## Result summary

For B1, all 10 requests reported zero cached tokens and 5,616 cache-creation
tokens. For B2, all 10 requests reported 5,616 cached tokens and zero
cache-creation tokens.

- A mean TTFT: 383.02 ms; mean AR latency: 5,986.45 ms.
- B1 mean TTFT: 392.92 ms; mean AR latency: 5,997.09 ms.
- B2 mean TTFT: 60.72 ms; mean AR latency: 5,634.94 ms.
- B2 versus B1: TTFT decreased by 84.5%, AR latency decreased by 6.0%,
  and throughput increased by 6.4%.
- Peak device memory was 40,408.8 MiB for A and 40,410.8 MiB for B1/B2.

The shorter 587-token profiler workload reduced scheduled prefill from 587
tokens to 11. Prefill CUDA time decreased from 42.11 ms to 31.09 ms, while
total GEMM and GPU busy time remained nearly unchanged because autoregressive
decode was unaffected. D2H transfer volume increased by about 1.94 MiB and
transfer time increased by about 1.4 ms.

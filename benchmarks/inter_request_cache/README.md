# Inter-Request DiT Cache Benchmark

Cross-request cache reuse for diffusion models. On a cache hit, generation can
skip the full denoise loop (exact match) or resume from an intermediate step
(semantic match via CLIP similarity).

See `vllm_omni/diffusion/cache/inter_request/` for the implementation.

## T2I — Qwen-Image

7975 prompts, 1024x1024, 44 steps, seed=42. GeneVal spatial-relation prompts.

| stage | wallclock (s) | throughput (req/s) | mean latency (s) | p50 (s) | CLIP score |
|-------|---------------|--------------------|-------------------|---------|------------|
| baseline (cache_dit) | 69600.24 | 0.1146 | 8.73 | 8.35 | 27.90 |
| warmup (build cache)  | 66525.06 | 0.1199 | 8.34 | 8.40 | 26.35 |
| cache hit (reuse)     | 26347.26 | 0.3027 | 3.30 | 0.77 | 26.22 |

Overall hit rate was 44.7% across exact and semantic matches. The reuse pass
measured 2.64× end-to-end throughput, 0.77 s median latency, and a mean CLIP
score of 26.22. The baseline mean CLIP score was 27.90.

Raw metrics in `t2i_qwen_image_7975/summary.json`.

## T2V — Wan2.2

200 prompts, 832x480, 81 frames, 40 steps, seed=42. The prompts are a
CLIP-similarity cluster picked from T2V-CompBench (a seed prompt plus its
199 nearest neighbours), so semantic reuse kicks in during the warmup pass
itself.

| stage | wallclock (s) | throughput (req/s) | mean latency (s) | p50 (s) | ViCLIP |
|-------|---------------|--------------------|-------------------|---------|--------|
| baseline (cache_dit) | 24946.0 | 0.0080 | 124.73 | 121.9 | 0.2437 |
| warmup (build cache)  | 24277.3 | 0.0082 | 120.71 | 123.8 | 0.2328 |
| cache hit (reuse)     | 335.1   | 0.5968 | 0.99   | 0.86  | 0.2328 |

All 200 reuse-pass requests hit the cache on this prompt cluster. Using the
recorded wallclock duration for the same number of requests, the reuse pass
achieved 74.44× end-to-end throughput and a 126.27× reduction in mean request
latency. During warmup, 89 of the 199 requests after the first matched a
cached neighbour semantically and skipped an average of 8.1 of 40 steps.
Warmup wallclock duration was 2.68% lower than the baseline. Mean ViCLIP was
0.2437 at baseline and 0.2328 in both the warmup and reuse passes.

Raw metrics in `t2v_wan22_200/summary.json`.

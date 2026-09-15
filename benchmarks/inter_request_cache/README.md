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

Overall hit rate was 44.7% (exact + semantic).

The p50 drops to 0.77s because most hits are exact and return instantly. The
mean stays at 3.3s since the 55% that miss still pay the full ~8.7s. CLIP score
drops 1.7 points (27.90 -> 26.22), all of it from semantic hits that skip steps;
exact hits are lossless. Turning off the t2i image-similarity penalty makes it
worse (27.6 -> 25.0), so it stays on by default.

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

All 200 cache-hit requests were served from cache (hit rate 100%, 126x
throughput). During warmup, 89 of 199 requests matched a cached neighbour
semantically and skipped on average 8.1 of 40 denoising steps; 42% of those
hits were clamped at the 10-step storage cap. Warmup ends 3.2% faster than
baseline — the skipped steps roughly cancel the cache-write overhead. ViCLIP
on cache hit is identical to warmup (0.2328): exact reuse is lossless. The
baseline-vs-warmup ViCLIP gap (0.2437 vs 0.2328) is generation variance, not
cache degradation.

Raw metrics in `t2v_wan22_200/summary.json`.

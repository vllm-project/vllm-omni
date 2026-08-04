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

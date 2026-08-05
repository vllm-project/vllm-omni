## Quantization Quality Benchmark — Z-Image-Turbo-SVDQuant-NVFP4
Setup: 1024x1024, 20 steps, seed=42, LPIPS (alex)

### Summary

| Config | Avg Time | Speedup | Memory (GiB) | Mem Reduction | Mean LPIPS |
|--------|----------|---------|--------------|---------------|------------|
| BF16 baseline | 11.07s | 1.00x | 24.26 | — | (ref) |
| auto | 4.94s | 2.24x | 17.14 | 29% | 0.2324 |

> LPIPS < 0.01 = imperceptible, > 0.1 = clearly noticeable.

### Memory Profiling

First-prompt snapshot at 1024x1024, 20 steps. Weights = `memory_allocated()` before `generate()`; Peak = `max_memory_allocated()` during `generate()`; Activations = Peak − Weights.

| Config | Weights | Activations | Peak | Total Reduction |
|--------|---------|-------------|------|-----------------|
| BF16, TP=1 | 20.87 GiB | 3.39 GiB | 24.26 GiB | — |
| auto, TP=1 | 13.74 GiB | 3.40 GiB | 17.14 GiB | **29%** |

### Per-Prompt LPIPS

| Prompt | auto |
|--------|--------|
| a close-up portrait of an elderly fisherman with w... | 0.2760 |
| an aerial view of a coral reef with crystal clear ... | 0.2100 |
| extreme close-up of a dewdrop on a red rose petal,... | 0.3935 |
| a bustling night market in Tokyo with neon signs, ... | 0.3116 |
| a vintage bookstore storefront with the sign CLASS... | 0.1610 |
| a campfire in a dark forest with sparks rising int... | 0.1916 |
| a ballet dancer in mid-leap on an empty theater st... | 0.1597 |
| a cup of coffee on a wooden table, morning light | 0.1554 |

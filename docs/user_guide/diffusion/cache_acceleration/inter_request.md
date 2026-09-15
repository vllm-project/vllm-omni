# Inter-Request Cache Reuse Guide


## Table of Content

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Example Scripts](#example-scripts)
- [Configuration Parameters](#configuration-parameters)
- [Combined with Cache-DiT](#combined-with-cache-dit)
- [Persistent Cache](#persistent-cache)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The **inter-request cache** (Chorus Stage-1) accelerates text-to-image generation by reusing
DiT computation results across different requests. When multiple requests share the same or
semantically similar prompts, previously computed denoising latents can be reused, avoiding
redundant computation and providing significant end-to-end speedup.

### Three Cache Tiers

| Tier | Match Condition | Speedup | Quality |
|------|----------------|---------|---------|
| **Exact hit** | Identical prompt + seed + params | Skip all DiT steps (measured 11–142× latency on committed benchmarks; see `benchmarks/inter_request_cache/`) | Identical |
| **Semantic hit** | CLIP text similarity > threshold | Partial (skip first N steps) | Near-identical |
| **Miss** | No match | No speedup (compute & cache) | Normal |

Only **text-prompt** requests with an **explicit seed** (or a resolvable
generator seed) are cached. Image-conditioned requests (edit / i2v), embedding
or token prompts, and seedless requests bypass the cache — see *Cache Safety*
below.

---

## How It Works

1. **Exact matching**: Each request's full parameter set (prompt, seed, dimensions, guidance
   scale, steps, etc.) is hashed into a cache key. An exact match means the cached final latent
   can be returned directly — skipping the entire DiT forward.

2. **Semantic matching (optional)**: When CLIP is configured, a text embedding is computed for
   the incoming prompt and compared (via vectorized cosine similarity) against all cached prompt
   embeddings. If similarity exceeds the threshold, the cached entry's per-step latents are used
   to **resume** denoising from an intermediate step rather than starting from scratch.

3. **Hybrid similarity**: When image embeddings are available, the semantic score combines
   text-to-text similarity with a sigmoid penalty based on text-to-image similarity, improving
   match precision. This can be disabled with `inter_request_use_t2i_penalty=False`.

---

## Quick Start

### Basic Usage

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Qwen/Qwen-Image",
    mode="text-to-image",
    cache_backend="inter_request",
    cache_config={
        "inter_request_max_entries": 1000,
        "inter_request_max_memory_gb": 16.0,
        "inter_request_persistent_cache_dir": "./persistent_cache",
    },
)

# First request: cache miss, computes and stores
outputs = omni.generate(
    "a cup of coffee on the table",
    OmniDiffusionSamplingParams(num_inference_steps=50, seed=142),
)

# Same prompt + params: exact hit, skips all DiT computation
outputs = omni.generate(
    "a cup of coffee on the table",
    OmniDiffusionSamplingParams(num_inference_steps=50, seed=142),
)
```

### With Semantic Matching (CLIP)

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    mode="text-to-image",
    cache_backend="inter_request",
    cache_config={
        "inter_request_max_entries": 1000,
        "inter_request_max_memory_gb": 16.0,
        "inter_request_persistent_cache_dir": "./persistent_cache",
        "inter_request_clip_model_path": "/path/to/clip-vit-large-patch14",
        "inter_request_clip_threshold": 0.65,
        "inter_request_clip_min_skip": 5,
        "inter_request_clip_max_skip_ratio": 0.5,
    },
)
```

---

## Example Scripts

### Offline Inference

```bash
python3 examples/offline_inference/text_to_image/text_to_image.py \
    --model /path/to/Qwen-Image \
    --cache-backend inter_request \
    --persistent-cache-dir ./persistent_cache \
    --max-entries 1000 \
    --max-memory-gb 16.0 \
    --clip-model-path /path/to/clip-vit-large-patch14 \
    --clip-threshold 0.65
```

### Online Serving

```bash
python3 examples/online_serving/text_to_image/run_server_with_cache.py \
    --model /path/to/Qwen-Image \
    --cache-backend inter_request \
    --persistent-cache-dir ./persistent_cache \
    --max-entries 8000 \
    --max-memory-gb 800.0 \
    --clip-model-path /path/to/clip-vit-large-patch14 \
    --clip-threshold 0.65
```

---

## Configuration Parameters

All parameters use the `inter_request_` prefix in the `cache_config` dictionary:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inter_request_max_entries` | int | 100 | *(example scripts default to 8000)* | Maximum number of cached entries (LRU eviction) |
| `inter_request_max_memory_gb` | float | 4.0 | *(example scripts default to 800.0)* | Maximum total memory for cached latents (GB). Without LMCache, entries beyond the budget are LRU-evicted entirely; `0` is only meaningful in LMCache mode (embedding shells only, heavy tensors tiered to disk) |
| `inter_request_persistent_cache_dir` | str \| None | None | Directory for disk persistence; if set, cache survives restarts. Keys embed a model-checkpoint + TP-topology digest, so a dir written by a different model/topology will not match |
| `inter_request_clip_model_path` | str \| None | None | Path to CLIP model for semantic matching; if None, only exact matching is used |
| `inter_request_clip_threshold` | float | 0.75 | CLIP text similarity threshold (τ) for semantic hit |
| `inter_request_clip_min_skip` | int | 5 | Minimum denoising steps to skip on a semantic hit |
| `inter_request_clip_max_skip_ratio` | float | 0.5 | Maximum skip ratio of total steps (when similarity ≈ 1.0) |
| `inter_request_use_t2i_penalty` | bool | True | Enable t2i sigmoid penalty in hybrid similarity scoring |
| `inter_request_lmcache_disk_dir` | str \| None | None | Root dir for LMCache CPU→disk tiering (see below); None disables tiering |
| `inter_request_lmcache_max_cpu_gb` | float | 5.0 | LMCache CPU pool size (GB) |
| `inter_request_lmcache_max_disk_gb` | float | 100.0 | LMCache disk quota (GB) |
| `inter_request_max_stored_steps` | int | 0 | Max per-entry step latents stored (0 = all steps); semantic-hit resume is clamped to this |

### CLI Flags (example scripts)

| Flag | Maps to |
|------|---------|
| `--persistent-cache-dir` | `inter_request_persistent_cache_dir` |
| `--max-entries` | `inter_request_max_entries` |
| `--max-memory-gb` | `inter_request_max_memory_gb` |
| `--clip-model-path` | `inter_request_clip_model_path` |
| `--clip-threshold` | `inter_request_clip_threshold` |
| `--clip-min-skip` | `inter_request_clip_min_skip` |
| `--clip-max-skip-ratio` | `inter_request_clip_max_skip_ratio` |
| `--no-t2i-penalty` | Sets `inter_request_use_t2i_penalty=False` |

---

## Combined with Cache-DiT

For maximum acceleration, the inter-request cache can be combined with intra-request
[Cache-DiT](cache_dit.md) using the `inter_request+cache_dit` backend:

```bash
python3 examples/offline_inference/text_to_image/text_to_image.py \
    --model /path/to/Qwen-Image \
    --cache-backend inter_request+cache_dit \
    --persistent-cache-dir ./persistent_cache \
    --clip-model-path /path/to/clip-vit-large-patch14 \
    --clip-threshold 0.65
```

This applies cross-request reuse (exact/semantic hit) first, then intra-request
cache-dit acceleration (DBCache + TaylorSeer + SCM) for the remaining denoising steps.

---

## Persistent Cache

When `inter_request_persistent_cache_dir` is set, the cache is automatically:

1. **Loaded** on engine startup — all entries (latents, step latents, embeddings, cache keys)
   are restored and become immediately searchable.
2. **Saved** periodically and on shutdown — new entries are persisted to disk.

Cache keys embed a digest of the model checkpoint and TP world size, so a
persistent dir written by a different model or topology is never served.

## LMCache CPU→Disk Tiering (optional)

For caches larger than host memory, set `inter_request_lmcache_disk_dir` to
enable tiered storage via LMCache (`pip install 'vllm-omni[inter_request]'`):

- Final latents are persisted with `torch.save` into
  `<dir>/final_direct/` (async, off the request path).
- Step latents go through LMCache's `ECCacheEngine` into `<dir>/steps/`,
  which manages the CPU pool (`inter_request_lmcache_max_cpu_gb`) and spills
  to disk (`inter_request_lmcache_max_disk_gb`) with LRU eviction.
- Under sustained pool pressure, LMCache allocation waits at most 30s and
  then degrades to a no-store (the request is unaffected; the entry simply
  is not cached).

## Cache Safety

The inter-request cache deliberately **refuses** to serve or store:

- **Image-conditioned requests** (`multi_modal_data` in the prompt, e.g.
  Qwen-Image-Edit / Wan2.2-i2v) — the conditioning image is not part of the
  key, so reuse could return an edit of the wrong source image.
- **Non-text prompts** (token / embedding / custom prompt objects) — no
  stable text identity.
- **Seedless requests** — a random sample must stay random; caching would
  silently make identical prompts deterministic.

---

## Best Practices

- **Start without CLIP**: Exact matching alone provides large speedup for repeated prompts
  with zero quality loss. Add CLIP semantic matching only when prompt diversity is expected.
- **Tune the threshold**: Lower `clip_threshold` → more semantic hits but potentially lower
  quality. `0.60–0.75` is a good range for most use cases.
- **Set `max_memory_gb` appropriately**: Cached latents can be large. Monitor memory usage
  and adjust the limit to avoid OOM.
- **Use persistent cache in production**: Set `persistent_cache_dir` to a stable path so the
  cache survives restarts and warms up quickly.

---

## Troubleshooting

### "CLIP model path does not exist" warning

The CLIP model path must be a local directory (not a HuggingFace repo ID). Download the model
first:

```bash
huggingface-cli download openai/clip-vit-large-patch14 --local-dir /path/to/clip-vit-large-patch14
```

### Semantic hits never trigger

- Verify the CLIP model loaded successfully (check logs for "CLIP model loaded successfully").
- Lower the `clip_threshold` value.
- Ensure cached entries have `clip_embedding` (entries cached before CLIP was configured will
  not be semantically searchable; clear the cache and re-populate).

### Cache not persisting across restarts

Ensure `inter_request_persistent_cache_dir` points to a writable directory and that the
process has permission to create files there.

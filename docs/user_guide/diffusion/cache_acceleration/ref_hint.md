# Reference-Hint Cache Guide

## Overview

The reference-hint cache accelerates reference-conditioned diffusion models by
skipping part of the reference branch on selected denoising steps. It currently
supports Wan2.1-VACE. Other pipelines fail during initialization instead of
silently enabling an unsupported path.

This backend is approximate and opt-in. It keeps every denoising step and only
approximates the reference hints, but the approximation can change the output
and retained hints increase peak GPU memory. Enable it only after evaluating
the speed, quality, and memory tradeoff for your workload.

## Quick Start

Online serving:

```bash
vllm serve Wan-AI/Wan2.1-VACE-1.3B-diffusers --omni \
  --cache-backend ref_hint \
  --cache-config '{
    "ref_hint_refresh_interval": 2,
    "ref_hint_strategy": "forecast50",
    "ref_hint_acknowledge_lossy": true
  }'
```

Offline inference:

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    cache_backend="ref_hint",
    cache_config={
        "ref_hint_refresh_interval": 2,
        "ref_hint_strategy": "forecast50",
        "ref_hint_acknowledge_lossy": True,
    },
)
```

`ref_hint_acknowledge_lossy=True` is required whenever
`ref_hint_refresh_interval >= 2`. This prevents approximate reuse from being
enabled accidentally.

## Strategies

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ref_hint_refresh_interval` | `2` | Recompute hints every K denoising steps. K=1 always recomputes and bypasses hint storage. |
| `ref_hint_strategy` | `"forecast50"` | `"forecast50"` retains two fresh observations and applies a damped forecast. `"reuse"` retains and returns only the latest fresh observation. |
| `ref_hint_acknowledge_lossy` | `False` | Explicit opt-in required for K >= 2. |

K=1 is output-equivalent to direct computation and does not provide a speedup.
It bypasses cache storage, so it is useful only as a diagnostic setting.

## Measured Tradeoff

An independent B300 measurement used Wan2.1-VACE-1.3B with 20 denoising steps,
guidance 5, 480x832 output, 17 frames, three real references, and two seeds.
After one warmup, each case used three measured generations.

| Mode | Median E2E | Filtered mean | Peak VRAM |
|------|-----------:|--------------:|----------:|
| Cache off | 4.043 s | 4.044 +/- 0.012 s | 17,464 MiB |
| `forecast50`, K=2 | 3.692 s | 3.686 +/- 0.015 s | 19,450 MiB |
| Difference | 9.01% faster | 9.30% faster | +1,986 MiB / +11.37% |

Against the cache-off output, the same run measured a 6.91% mean DINOv2 drop
and a 12.16% worst-frame drop. These results passed the experiment's 8% mean
and 15% worst-frame gates, but they show that the backend is not free: about a
9% latency improvement cost about 11% more peak VRAM on this setup.

Measurements depend on the model, reference, seed, dimensions, frame count,
step count, device, and software versions. Benchmark all three axes on the
intended workload rather than assuming these numbers transfer unchanged.

## Memory Behavior

`forecast50` needs two full fresh hint sets for each active CFG branch and model
owner. `reuse` needs one. Hint size scales with the latent shape, VACE block
count, number of CFG branches, and number of transformer experts.

The backend releases retained hints in request-finally cleanup, including when
the pipeline forward raises. This prevents tensors from remaining referenced
between requests, but it does not remove the within-request peak shown above:
forecasting still needs its history during denoising.

If VRAM is the limiting resource, leave this backend disabled or compare the
single-history `reuse` strategy against `forecast50`; `reuse` has a different
quality profile and must be evaluated separately.

## Operational Notes

- Only one diffusion cache backend can be selected at a time.
- Step execution does not currently support cache backends.
- A request resets reference-hint state before generation and releases it after
  generation, so hints are not shared across requests.
- Sequential CFG branches and multiple transformer experts keep isolated state.

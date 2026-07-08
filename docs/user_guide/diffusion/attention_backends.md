# Diffusion Attention Backends

This document describes the diffusion attention backends available in vLLM-Omni, how to select them globally and per-role, the per-platform defaults, and how to use SageAttention.

## Overview

Diffusion attention backend selection is resolved in `vllm_omni.diffusion.attention.selector`. It looks up the backend from a structured `AttentionConfig` carried on `OmniDiffusionConfig` and falls back to the platform default when nothing is configured.

This backend is used by diffusion attention layers such as the DiT attention in video and image generation models. It does **not** affect autoregressive (LLM) attention paths — those go through vLLM's own attention backend selector.

The full set of backends and their platform defaults is in the **Backend Options** and **Platform Defaults** sections below. If no attention backend is configured, vLLM-Omni asks the current platform to choose the default.

## Backend Feature Support

### Legend

| Column | Description |
|---|---|
| Requires | Extra package needed beyond a stock vLLM-Omni install |
| Dtypes | Supported activation data types |
| Lossless | Bit-accurate attention math at the model dtype (❌ = quantized/lossy kernels) |
| Masks | Non-causal `attn_mask` / varlen support (needed by mask-heavy DiTs) |
| Head Sizes | Supported attention head dimensions ("Any" = no backend-side restriction) |
| Compute Cap. | Required CUDA compute capability |

Symbols: ✅ = supported, ❌ = not supported.

| Backend | Requires | Dtypes | Lossless | Masks | Head Sizes | Compute Cap. | Notes |
|---|---|---|---|---|---|---|---|
| `FLASH_ATTN` | `flash-attn` | fp16, bf16 | ✅ | ✅ | 64, 96, 128, 192, 256 | ≥ 8.0 | FlashAttention 2; auto-route default on pre-Blackwell |
| `FLASH_ATTN_4` | `flash-attn-4` (pre-release) | fp16, bf16 | ✅ | ✅ | 64, 96, 128, 256 | ≥ 8.0 | Opt-in, never auto-routed — see [FlashAttention-4](#flashattention-4-datacenter-blackwell) |
| `CUDNN_ATTN` | cuDNN ≥ 9.5 (in PyTorch 2.5+ wheels) | fp16, bf16 | ✅ | ✅ | Any (%8, ≤ 256) | Any | Pins `sdpa_kernel([CUDNN_ATTENTION])`; auto-route default on Blackwell; 2× e2e on mask-heavy DiTs (HV-1.5) |
| `FLASHINFER_ATTN` | `flashinfer` | fp16, bf16 | ✅ | ✅ (`custom_mask`) | 64, 128, 256 | ≥ 8.0 | Dense `single_prefill_with_kv_cache`; Blackwell fallback when cuDNN < 9.5 |
| `TORCH_SDPA` | — | fp16, bf16, fp32 | ✅ | ✅ | Any | Any | Default dispatcher; most conservative, always available; quality reference |
| `SAGE_ATTN` | `sageattention` | fp16, bf16 | ❌ (INT8, fp16 accum) | ❌ | 32–256 | ≥ 8.0 | Typically visually indistinguishable on diffusion outputs |
| `SAGE_ATTN_3` | `sageattn3` | fp16, bf16 | ❌ | ❌ | 64, 128, 256 | ≥ 10.0 | Blackwell only; GQA/MQA requests fall back to `TORCH_SDPA` |


## Setting the Diffusion Attention Backend

Diffusion attention backends can be configured three ways, in priority order:

1. **`--diffusion-attention-config`** — structured per-role config (highest priority).
2. **`--diffusion-attention-backend` / `DIFFUSION_ATTENTION_BACKEND` env var** — global shorthand that sets the default backend.
3. **Platform default** — used when nothing is configured.

`--diffusion-attention-backend` is shorthand for `--diffusion-attention-config.default.backend`. It may be combined with `--diffusion-attention-config.per_role.*` overrides, but is mutually exclusive with `--diffusion-attention-config.default.backend`.

### Global default

Set the default backend for every diffusion attention layer:

```bash
# CLI flag
vllm-omni serve <model> --diffusion-attention-backend SAGE_ATTN

# Environment variable (also recognized for backwards compatibility)
export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN
```

### Per-role configuration

Roles are free-form strings declared by each diffusion model. The two common categories are `"self"` and `"cross"`; model-specific roles (e.g. `"ltx2.audio_to_video"`) may also be declared. A role string is matched in this order:

1. Exact `per_role[role]` match
2. `per_role[role_category]` fallback (e.g. `"ltx2.audio_to_video"` → `"cross"`)
3. `default`
4. Platform default

Use vLLM-style dotted flags or one JSON blob:

```bash
# Dotted flags
vllm-omni serve <model> \
    --diffusion-attention-config.default.backend FLASH_ATTN \
    --diffusion-attention-config.per_role.cross.backend TORCH_SDPA

# JSON
vllm-omni serve <model> \
    --diffusion-attention-config '{"default":{"backend":"FLASH_ATTN"},"per_role":{"cross":{"backend":"TORCH_SDPA"}}}'
```

Backends may also accept backend-specific parameters via `extra`:

```bash
--diffusion-attention-config.per_role.self.backend SAGE_ATTN \
--diffusion-attention-config.per_role.self.extra.some_option value
```

### Programmatic API

When constructing `OmniDiffusionConfig` directly:

```python
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig

config = OmniDiffusionConfig(
    attention_config=AttentionConfig(
        default=AttentionSpec(backend="FLASH_ATTN"),
        per_role={
            "cross": AttentionSpec(backend="TORCH_SDPA"),
        },
    ),
    ...,
)
```

A plain dict is also accepted and normalized to `AttentionConfig`.

## Backend Selection Behavior

### Manual Selection

When you explicitly set a backend via `--diffusion-attention-backend` or `--diffusion-attention-config`:

1. The selection is validated against your environment (package installed, compute capability).
2. If the backend is unavailable or unsupported, vLLM-Omni logs a warning with the specific reason and **falls back to `TORCH_SDPA`** — unlike core vLLM, it does not raise.
3. If valid, the backend is used and the startup log prints `Using diffusion attention backend '<NAME>'`.

### Automatic Selection

When no backend is configured, the platform picks the first compatible backend from the priority tables below. The startup log line `Defaulting to diffusion attention backend CUDNN_ATTN (Blackwell sm_120, cuDNN 91002)` confirms the route. If neither the `Using ...` nor the `Defaulting to ...` line appears, the model didn't reach diffusion stage init — check earlier logs for failures.

`FLASH_ATTN_4`, `SAGE_ATTN`, and `SAGE_ATTN_3` are never auto-routed — they are opt-in only.

## Backend Priority (CUDA)

Priority is 1 = highest (tried first).

**Blackwell (sm_100 / sm_103 / sm_120 / sm_121):**

| Priority | Backend | Condition |
|---|---|---|
| 1 | `CUDNN_ATTN` | cuDNN ≥ 9.5 available (ships in PyTorch 2.5+ wheels) |
| 2 | `FLASHINFER_ATTN` | `flashinfer` installed, cuDNN < 9.5 |
| 3 | `FLASH_ATTN` | `flash-attn` installed with the Blackwell CUTE kernel |
| 4 | `TORCH_SDPA` | always available |

**Why CUDNN_ATTN by default**: on mask-heavy diffusion models (HunyuanVideo-1.5, Qwen-Image), cuDNN's pinned FMHA kernel sidesteps a PyTorch SDPA dispatch quirk where the unpinned dispatcher picks `EFFICIENT_ATTENTION` (~25 ms) for masked calls instead of cuDNN (~11 ms). The pin gives 2× e2e on HV-1.5 with no regression on lighter models.

**Hopper (sm_90) / Ada (sm_89) / Ampere (sm_80–sm_86):**

| Priority | Backend | Condition |
|---|---|---|
| 1 | `FLASH_ATTN` | `flash-attn` installed |
| 2 | `TORCH_SDPA` | always available |

`CUDNN_ATTN` and `FLASHINFER_ATTN` are still selectable explicitly on these GPUs but are not in the auto-route — FlashAttention 2 is the well-tuned path on pre-Blackwell hardware.

## End-to-End Benchmark (BF16, sm_120 RTX Pro 6000 Blackwell)

Same prompt and seed across runs. `Total generation time` from `text_to_video.py` / `text_to_image.py`.

| Model | Shape | TORCH_SDPA | CUDNN_ATTN | FLASHINFER_ATTN |
|---|---|---|---|---|
| HunyuanVideo-1.5 (T2V) | 480p / 33f / 50 steps | 147.05 s | **73.02 s** | 127.84 s |
| Wan 2.2 14B (T2V) | 480p / 33f / 40 steps | 117.75 s | 117.17 s | **115.07 s** |
| Qwen-Image (T2I) | 1024² / 50 steps | 17.41 s | **15.14 s** | 16.02 s |
| FLUX.2-dev (T2I) | 1024² / 50 steps, TP=2 | 53.62 s | **53.30 s** | 54.94 s |

Pattern: mask-heavy DiTs (HV-1.5, Qwen-Image) favor `CUDNN_ATTN`; lighter-mask DiTs and TP-saturated configs (Wan 2.2, FLUX.2 TP=2) tie within noise.

## SageAttention Installation

vLLM-Omni expects SageAttention to be installed into the same Python environment as vLLM-Omni.

Build from source:

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention

export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
pip install . --no-build-isolation
```

Quick check:

```bash
python -c "import sageattention; print(sageattention.__file__)"
```

## SageAttention3 Installation

vLLM-Omni expects SageAttention3 to be installed into the same Python environment as vLLM-Omni.

Build from source:

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention/sageattention3_blackwell
python setup.py install
```

Quick check:

```bash
python -c "import sageattn3; print(sageattn3.__file__)"
```

Notes:

- `SAGE_ATTN_3` is only selected on CUDA when `sageattn3` is importable and the GPU is Blackwell-class.
- SageAttention3's Blackwell kernel assumes `Hq == Hkv`. In vLLM-Omni, GQA/MQA diffusion requests fall back to PyTorch SDPA for correctness.

## Usage Examples

### Default (auto-route)

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
    --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
    --prompt "A dog running across a field of golden wheat." \
    --height 480 --width 832 --num-frames 33 \
    --num-inference-steps 50 --seed 42 --guidance-scale 6.0 \
    --output hv15.mp4
```

On Blackwell this picks `CUDNN_ATTN` automatically. Check the log for the `Defaulting to ...` line.

### Explicit backend selection

```bash
DIFFUSION_ATTENTION_BACKEND=FLASHINFER_ATTN python examples/offline_inference/text_to_video/text_to_video.py \
    --model Lightricks/LTX-2 \
    --prompt "A dog running across a field of golden wheat." \
    --height 480 --width 832 --num-frames 33 \
    --num-inference-steps 40 --seed 42 --guidance-scale 4.0 \
    --output ltx2.mp4
```

### FlashAttention-4 (datacenter Blackwell)

`FLASH_ATTN_4` is opt-in (never auto-routed). Requires `pip install --pre flash-attn-4` — the wheel publishes only the `flash_attn.cute` namespace; installing it alongside a `flash-attn` FA2 wheel in the same environment is not supported, since both own the `flash_attn` package directory. The backend covers dense, piecewise (mixed causal/full), and masked/varlen calls, works under torch.compile, and on sm_103 (B300) automatically selects the `sm_103a` compile target unless `CUTE_DSL_ARCH` is already set.

BF16 FA4 is a drop-in selection:

```bash
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN_4 python examples/offline_inference/text_to_video/text_to_video.py \
    --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
    --prompt "A dog running across a field of golden wheat." \
    --height 480 --width 832 --num-frames 33 \
    --num-inference-steps 50 --seed 42 --guidance-scale 6.0 \
    --output hv15_fa4.mp4
```

### SageAttention (lossy)

```bash
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN python examples/offline_inference/text_to_video/text_to_video.py \
    --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
    --prompt "A dog running across a field of golden wheat." \
    --height 480 --width 832 --num-frames 33 \
    --num-inference-steps 30 --seed 42 --guidance-scale 6.0 \
    --tensor-parallel-size 2 \
    --output hv15_sage.mp4
```

Example: Wan2.2 TI2V 5B

```bash
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN python examples/offline_inference/text_to_video/text_to_video.py \
    --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --prompt "A dog running across a field of golden wheat." \
    --height 704 --width 1280 --num-frames 49 \
    --num-inference-steps 30 --seed 42 --guidance-scale 5.0 \
    --tensor-parallel-size 2 \
    --output outputs/wan22_sage.mp4
```

### Enable SageAttention3

Example:

```bash
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN_3 python examples/offline_inference/text_to_video/text_to_video.py \
    --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
    --prompt "A dog running across a field of golden wheat." \
    --height 480 --width 832 --num-frames 33 \
    --num-inference-steps 30 --seed 42 --guidance-scale 6.0 \
    --tensor-parallel-size 2 \
    --output outputs/hv15_sage3.mp4
```

### Mixed backends across roles

Use `FLASH_ATTN` for self-attention and `TORCH_SDPA` for cross-attention:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
    --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --prompt "A dog running across a field of golden wheat." \
    --diffusion-attention-config.per_role.self.backend FLASH_ATTN \
    --diffusion-attention-config.per_role.cross.backend TORCH_SDPA \
    --tensor-parallel-size 2 \
    --output outputs/wan22_mixed.mp4
```

### Compare against FlashAttention

Unset the backend override, or explicitly use `FLASH_ATTN`:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
    --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --prompt "A dog running across a field of golden wheat." \
    --height 704 --width 1280 --num-frames 49 \
    --num-inference-steps 30 --seed 42 --guidance-scale 5.0 \
    --tensor-parallel-size 2 \
    --output outputs/wan22_fa2.mp4
```

## Validation Guidance

Don't assume a faster attention backend is numerically interchangeable with `TORCH_SDPA`.

Always compare:

- End-to-end runtime
- Diffusion-stage runtime (`add_req_and_wait` line in DiffusionEngine.step breakdown)
- Output quality against a known-good baseline (CLIP similarity, frame-level diff, or visual review)

At minimum, keep the same:

- model
- prompt
- seed
- resolution
- frame count / step count
- parallel config (TP / CFG-parallel / Ulysses degrees)

## Reproducing the Benchmark Table

The end-to-end numbers above were collected by running `text_to_video.py` /
`text_to_image.py` with the same prompt and seed while varying
`DIFFUSION_ATTENTION_BACKEND`. For a quick kernel-level comparison of the
backends without loading a model:

```bash
python benchmarks/diffusion/bench_attention_backends.py --preset hv15
```

It runs all three BF16 backends on representative DiT attention shapes and
prints a ranking table at the end.

# Diffusion Attention Backends

This document describes the diffusion attention backends available in vLLM-Omni, how to select them globally and per-role, the per-platform defaults, and how to use SageAttention.

## Overview

Diffusion attention backend selection is resolved in `vllm_omni.diffusion.attention.selector`. It looks up the backend from a structured `AttentionConfig` carried on `OmniDiffusionConfig` and falls back to the platform default when nothing is configured.

This backend is used by diffusion attention layers such as the DiT attention in video and image generation models. It does **not** affect autoregressive (LLM) attention paths — those go through vLLM's own attention backend selector.

The full set of backends and their platform defaults is in the **Backend Options** and **Platform Defaults** sections below. If no attention backend is configured, vLLM-Omni asks the current platform to choose the default.

## Backend Options

| Value | Notes |
|---|---|
| `TRTLLM_ATTN` | FlashInfer's trtllm-gen FMHA (TensorRT-LLM's generated kernels, vendored by FlashInfer). BF16, GQA native, `head_dim=128`. Datacenter Blackwell only (sm_100 / sm_103). Supports **Skip-Softmax** sparse attention and **SAGE** quantized attention — see below. Requires `flashinfer`. |
| `FLASH_ATTN` | Wraps FlashAttention 2. Default on Hopper / Ada / Ampere when `flash-attn` is installed. |
| `CUDNN_ATTN` | Pins `sdpa_kernel([CUDNN_ATTENTION])`. Default on Blackwell (sm_10x / sm_12x) with cuDNN ≥ 9.5. Wins on mask-heavy DiTs (HunyuanVideo-1.5: 2× e2e vs SDPA). |
| `FLASHINFER_ATTN` | Calls FlashInfer's dense `single_prefill_with_kv_cache` directly with `custom_mask` for non-causal masked attention. Used as Blackwell fallback when cuDNN is unavailable. Requires `flashinfer`. |
| `TORCH_SDPA` | PyTorch `scaled_dot_product_attention` with the default backend dispatcher. Most conservative; always available. |
| `SAGE_ATTN` | SageAttention 2.2 — INT8-quantized attention with FP16 accumulation. Lossy but typically visually indistinguishable on diffusion outputs. Requires `sageattention`. |
| `SAGE_ATTN_3` | Requires `sageattn3` from `SageAttention/sageattention3_blackwell`. CUDA only, intended for Blackwell GPUs, with GQA/MQA requests falling back to PyTorch SDPA. |
| `FLASH_ATTN_HUB` | FlashAttention 2 from HuggingFace `kernels` library. Useful for train/rollout alignment. |
| `FLASH_ATTN_3_HUB` | FlashAttention 3 from HuggingFace `kernels` library. CUDA Hopper (sm_90+) only; falls back to `FLASH_ATTN_HUB` on older GPUs. |


## Configuration

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

A backend that needs configuration exposes it as a typed field on the spec. Today the only
one is `TRTLLM_ATTN`'s Skip-Softmax (see [below](#trtllm_attn-backend-and-skip-softmax)):

```bash
--diffusion-attention-config.default.backend TRTLLM_ATTN \
--diffusion-attention-config.default.skip_softmax.target_sparsity 0.5
```

### Programmatic API

When constructing `OmniDiffusionConfig` directly:

```python
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig

config = OmniDiffusionConfig(
    diffusion_attention_config=AttentionConfig(
        default=AttentionSpec(backend="FLASH_ATTN"),
        per_role={
            "cross": AttentionSpec(backend="TORCH_SDPA"),
        },
    ),
    ...,
)
```

A plain dict is also accepted and normalized to `AttentionConfig`.

## Platform Defaults

### Blackwell (sm_100 / sm_103 / sm_120 / sm_121)

Auto-route preference, in order:

1. `TRTLLM_ATTN` — on **datacenter** Blackwell (sm_100 / sm_103) when `flashinfer` is installed, the model's `head_dim` is 128, **and the model is mask-free** (see below)
2. `CUDNN_ATTN` — when cuDNN ≥ 9.5 is available (ships in PyTorch 2.5+ wheels)
3. `FLASHINFER_ATTN` — when `flashinfer` is installed but cuDNN < 9.5
4. `FLASH_ATTN` — when `flash-attn` is installed with the Blackwell CUTE kernel
5. `TORCH_SDPA` — last resort

`TRTLLM_ATTN` is skipped on workstation Blackwell (sm_120 / sm_121) and for any `head_dim != 128`, so those GPUs keep the `CUDNN_ATTN` route described below.

`TRTLLM_ATTN` outranks `CUDNN_ATTN` on datacenter Blackwell because it is the only backend that can enable Skip-Softmax, not because its dense kernel is faster — dense, the two are comparable. It cannot honor an attention mask, so the auto-default only applies to pipelines verified mask-free (the Wan family); mask-using pipelines keep `CUDNN_ATTN`. `TRTLLM_ATTN` stays available everywhere via explicit `--diffusion-attention-backend TRTLLM_ATTN` (which raises clearly if a mask is received).

The startup log line `Defaulting to diffusion attention backend CUDNN_ATTN (Blackwell sm_120, cuDNN 91002)` confirms the route.

**Why CUDNN_ATTN by default**: on mask-heavy diffusion models (HunyuanVideo-1.5, Qwen-Image), cuDNN's pinned FMHA kernel sidesteps a PyTorch SDPA dispatch quirk where the unpinned dispatcher picks `EFFICIENT_ATTENTION` (~25 ms) for masked calls instead of cuDNN (~11 ms). The pin gives 2× e2e on HV-1.5 with no regression on lighter models.

### Hopper (sm_90) / Ada (sm_89) / Ampere (sm_80–sm_86)

Auto-route preference:

1. `FLASH_ATTN` — when `flash-attn` is installed
2. `TORCH_SDPA` — fallback

`CUDNN_ATTN` and `FLASHINFER_ATTN` are still selectable via env var on these GPUs but are not in the auto-route — FlashAttention 2 is the well-tuned path on pre-Blackwell hardware.

## TRTLLM_ATTN Backend and Skip-Softmax

`TRTLLM_ATTN` runs FlashInfer's trtllm-gen FMHA. Dense it is a BF16 backend like the others; on top of
that it can enable **Skip-Softmax**, a sparse-attention mode that trades a little fidelity for speed
(algorithm: [design doc](../../design/feature/skip_softmax.md)).

Enable it through the typed `skip_softmax` block on the attention spec:

| Key | Valid values | Meaning |
|---|---|---|
| `target_sparsity` | finite, `[0, 1]` | Operating point on a calibrated curve. Needs a calibration for the model. |
| `threshold` | finite, `≥ 0` | Direct threshold, no calibration needed. Mutually exclusive with `target_sparsity`. |
| `disabled_until_timestep` | finite, `[0, 1]` | Holds early, high-noise steps dense; skip turns on once the normalized timestep `t ≤ D`. |

```bash
vllm-omni serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --diffusion-attention-config '{"default": {"backend": "TRTLLM_ATTN",
      "skip_softmax": {"target_sparsity": 0.65, "disabled_until_timestep": 0.86}}}'
```

Programmatically the same block is a typed `SkipSoftmaxSpec` (values validated at construction):

```python
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, SkipSoftmaxSpec

AttentionConfig(
    default=AttentionSpec(
        backend="TRTLLM_ATTN",
        skip_softmax=SkipSoftmaxSpec(target_sparsity=0.65, disabled_until_timestep=0.86),
    ),
)
```

**Start at `target_sparsity=0.65, disabled_until_timestep=0.86`** and raise `target_sparsity` only
as far as your quality bar allows — the output diverges from dense as it climbs. The gain grows with
sequence length, since Skip-Softmax only accelerates attention. Dense BF16 is `TRTLLM_ATTN` with no
`skip_softmax` block.

Requires datacenter Blackwell with `head_dim == 128`; elsewhere, or without FlashInfer, selecting
`TRTLLM_ATTN` raises rather than silently degrading.

## TRTLLM_ATTN SAGE Quantization

On top of dense/Skip-Softmax, `TRTLLM_ATTN` can run **SAGE** (SageAttention) — the trtllm-gen
FMHA kernel with per-block quantized Q/K and per-channel fp8 V, so the two attention matmuls run
in low precision. Enable it through the typed `quant` block on the attention spec:

| Key | Valid values | Meaning |
|---|---|---|
| `dtype_qk` | `int8`, `fp8_e4m3` | Quantization dtype for Q and K. Absent ⇒ dense (SAGE off). |
| `q_block_size` | `1`, `4`, `16` | Per-token block size for Q scales (default `1`). |
| `k_block_size` | `1`, `4`, `16` | Per-token block size for K scales (default `16`). |

V is always quantized per-channel to fp8_e4m3 and K-smoothing is applied inside the routine, so
neither is a user knob. Only block sizes `1`, `4`, `16` have compiled kernels.

The `quant` block is a shared `AttnQuantSpec` consumed by more than one backend; each reads the
fields that apply and rejects the rest. `TRTLLM_ATTN` (SAGE) reads `dtype_qk` (`int8` / `fp8_e4m3`),
`q_block_size`, and `k_block_size`; `FLASHINFER_ATTN` reads `dtype_qk` (`float16` / `bfloat16`),
`dtype_vo`, and `flashinfer_backend` (see its section). A `dtype_qk` from the wrong set raises.

```bash
vllm-omni serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --diffusion-attention-config '{"default": {"backend": "TRTLLM_ATTN",
      "quant": {"dtype_qk": "fp8_e4m3", "q_block_size": 1, "k_block_size": 16}}}'
```

```python
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, AttnQuantSpec

AttentionConfig(
    default=AttentionSpec(
        backend="TRTLLM_ATTN",
        quant=AttnQuantSpec(dtype_qk="fp8_e4m3", q_block_size=1, k_block_size=16),
    ),
)
```


**Requirements.** Needs FlashInfer ≥ 0.6.16rc1. Kernel
availability is arch-dependent: `fp8_e4m3` QK has kernels on both **SM100** (B200) and **SM103**
(B300); `int8` QK kernels are compiled for **SM100 only**.

## End-to-End Benchmark (BF16, sm_120 RTX Pro 6000 Blackwell)

Same prompt and seed across runs. `Total generation time` from `text_to_video.py` / `text_to_image.py`.

| Model | Shape | TORCH_SDPA | CUDNN_ATTN | FLASHINFER_ATTN |
|---|---|---|---|---|
| HunyuanVideo-1.5 (T2V) | 480p / 33f / 50 steps | 147.05 s | **73.02 s** | 127.84 s |
| Wan 2.2 14B (T2V) | 480p / 33f / 40 steps | 117.75 s | 117.17 s | **115.07 s** |
| Qwen-Image (T2I) | 1024² / 50 steps | 17.41 s | **15.14 s** | 16.02 s |
| FLUX.2-dev (T2I) | 1024² / 50 steps, TP=2 | 53.62 s | **53.30 s** | 54.94 s |

Pattern: mask-heavy DiTs (HV-1.5, Qwen-Image) favor `CUDNN_ATTN`; lighter-mask DiTs and TP-saturated configs (Wan 2.2, FLUX.2 TP=2) tie within noise.


## Choosing a Backend Manually

### When to override the default

- **Quality validation**: compare a new backend against `TORCH_SDPA` as the reference, since SDPA's default dispatcher is the most extensively tested.
- **Lossy speedup hunting**: try `SAGE_ATTN` (INT8 quantized) on diffusion outputs — typically indistinguishable visually but always validate.
- **Workaround for known issues**: see Known Limitations above.

### Verifying which backend is in use

The startup log prints one of:

```
Using diffusion attention backend 'CUDNN_ATTN'           # explicit override
Defaulting to diffusion attention backend CUDNN_ATTN ... # auto-route
Defaulting to diffusion attention backend SDPA           # nothing else available
```

If you don't see one of these, the model didn't reach diffusion stage init — check earlier logs for failures.

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

## HuggingFace Kernels Hub Backends

To achieve perfect numerical consistency between **training** (typically using HuggingFace Diffusers with Hub kernels) and **serving/rollout** (using vLLM-Omni), you can use the Hub-based attention backends. This eliminates numerical drift / sampling divergence caused by executing different local kernel versions during rollout.

The following backend options are supported:
- `FLASH_ATTN_HUB` (HuggingFace `kernels-community/flash-attn2`)
- `FLASH_ATTN_3_HUB` (HuggingFace `kernels-community/flash-attn3`, Hopper sm_90+ only)

### Installation

To use these backends, you must install the `kernels` library:

```bash
pip install kernels==0.14.1
```

If the `kernels` library is not available in the environment, vLLM-Omni will log a warning and fall back gracefully to the corresponding local backend implementations (`FLASH_ATTN`). On CUDA GPUs below Hopper (compute capability < 9.0), `FLASH_ATTN_3_HUB` falls back to `FLASH_ATTN_HUB`.

### Usage

Select a Hub backend using the global CLI flag or environment variables:

```bash
# Environment variable
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN_HUB

# CLI flag
vllm-omni serve <model> --diffusion-attention-backend FLASH_ATTN_HUB
```

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

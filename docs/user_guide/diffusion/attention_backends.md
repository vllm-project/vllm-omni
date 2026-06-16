# Diffusion Attention Backends

This document describes the diffusion attention backends available in vLLM-Omni, how to select them globally and per-role, the per-platform defaults, and how to use SageAttention.

## Overview

Diffusion attention backend selection is resolved in `vllm_omni.diffusion.attention.selector`. It looks up the backend from a structured `AttentionConfig` carried on `OmniDiffusionConfig` and falls back to the platform default when nothing is configured.

This backend is used by diffusion attention layers such as the DiT attention in video and image generation models. It does **not** affect autoregressive (LLM) attention paths — those go through vLLM's own attention backend selector.

The full set of backends and their platform defaults is in the **Backend Options** and **Platform Defaults** sections below. If no attention backend is configured, vLLM-Omni asks the current platform to choose the default.

## Backend Options

| Value | Notes |
|---|---|
| `FLASH_ATTN` | Wraps FlashAttention 2. Default on Hopper / Ada / Ampere when `flash-attn` is installed. |
| `CUDNN_ATTN` | Pins `sdpa_kernel([CUDNN_ATTENTION])`. Default on Blackwell (sm_10x / sm_12x) with cuDNN ≥ 9.5. Wins on mask-heavy DiTs (HunyuanVideo-1.5: 2× e2e vs SDPA). |
| `FLASHINFER_ATTN` | Calls FlashInfer's dense `single_prefill_with_kv_cache` directly with `custom_mask` for non-causal masked attention. Used as Blackwell fallback when cuDNN is unavailable. Requires `flashinfer`. |
| `TORCH_SDPA` | PyTorch `scaled_dot_product_attention` with the default backend dispatcher. Most conservative; always available. |
| `SAGE_ATTN` | SageAttention 2.2 — INT8-quantized attention with FP16 accumulation. Lossy but typically visually indistinguishable on diffusion outputs. Requires `sageattention`. |
| `SAGE_ATTN_3` | Requires `sageattn3` from `SageAttention/sageattention3_blackwell`. CUDA only, intended for Blackwell GPUs, with GQA/MQA requests falling back to PyTorch SDPA. |
| `SPARSE_ATTN` | In-tree dispatcher to **external** sparse-attention kernels registered via the `vllm_omni.sparse_attn` entry-point group (no kernels are vendored). Opt-in per role; maskless self-attention only. See [Sparse attention (plugin)](#sparse-attention-plugin). |


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

Backends may also accept backend-specific parameters via `extra`:

```bash
--diffusion-attention-config.per_role.self.backend SPARSE_BLOCK \
--diffusion-attention-config.per_role.self.extra.block_size 128
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

## Sparse attention (plugin)

`SPARSE_ATTN` is an in-tree *dispatcher*: vLLM-Omni owns only the slot, while
the actual sparse kernels ship as **external, pip-installable packages**. A kernel
package registers a callable via the `vllm_omni.sparse_attn` entry-point group
(`dummy_sparse` below is a placeholder name — use your kernel's):

```toml
# pyproject.toml of the external kernel package
[project.entry-points."vllm_omni.sparse_attn"]
dummy_sparse = "dummy_sparse_attn:attn_fn"
```

Plugin contract (a callable; device-agnostic self-attention):

```python
def attn_fn(query, key, value, params: dict, is_causal: bool, softmax_scale: float) -> torch.Tensor: ...
```

`query`/`key`/`value` use the **BSND** layout `[batch, seq_len, num_heads, head_dim]`,
and the callable returns the same layout. `is_causal` and `softmax_scale` are the
standard attention scalars set by the framework (`softmax_scale` defaults to
`head_dim ** -0.5`); kernel-specific knobs and per-forward state arrive in `params`.

### Selecting a sparse kernel

Opt-in per role. The shorthand routes the plugin name through the dispatcher:

```bash
# Shorthand: backend = <plugin name>
vllm-omni serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --diffusion-attention-config.per_role.self.backend dummy_sparse \
    --diffusion-attention-config.per_role.self.extra.topk 0.3

# Explicit dispatcher form (equivalent)
vllm-omni serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --diffusion-attention-config '{"per_role":{"self":{"backend":"SPARSE_ATTN","extra":{"backend":"dummy_sparse","params":{"topk":0.3}}}}}'
```

Target `self` only — cross-attention keeps its dense default (sparsifying text
cross-attention degrades quality).

### The `params` dict

The dispatcher hands the kernel one flat `params` dict, merged from three tiers
(later wins, `None` dropped, so use `params.get(key, default)`):

1. **static** — the kernel's configured params (`extra.params`, or flat `extra` keys).
2. **per-forward** — canonical framework fields (`PerForwardState`), each a per-sample
   `[num_samples]` tensor: `denoise_step_idx` / `total_denoise_steps` (e.g.
   `progress = denoise_step_idx / total_denoise_steps`), plus video geometry
   `total_latent_frames` / `patches_per_frame` when the model publishes its patch
   grid (`seq_len == total_latent_frames * patches_per_frame`) — used by
   structure-aware kernels to map the flat sequence to (frame, patch) coordinates.
3. **dynamic** — the opaque `AttentionMetadata.extra`.

vLLM-Omni interprets no kernel-level parameter; the plugin reads what it needs.

### Well-known keys and capabilities

Some `params` keys carry attention inputs that change the result and are forwarded
only when the model sets them:

- `attn_mask` — validity / padding mask.
- `full_attn_spans` — per-sample `[start, end)` spans using full (bidirectional)
  attention within an otherwise causal sequence.

A plugin that consumes one of these must advertise it through an optional
`capabilities` dict on the callable. If a model provides such an input and the
plugin has not advertised support, the dispatcher raises rather than letting the
kernel ignore it:

```python
def attn_fn(query, key, value, params, is_causal, softmax_scale):
    mask = params.get("attn_mask")   # present only when advertised below
    ...

attn_fn.capabilities = {"supports_attention_mask": True}
```

### Writing a plugin

```python
# dummy_sparse_attn/__init__.py  (external package)
def attn_fn(query, key, value, params, is_causal, softmax_scale):
    # query/key/value are BSND: [batch, seq_len, num_heads, head_dim]
    topk = params.get("topk", 1.0)
    step = params.get("denoise_step_idx")        # per-sample tensor, or None
    total = params.get("total_denoise_steps")    # per-sample tensor, or None
    ...                                          # call the sparse kernel (returns BSND)
    return out
```

Install it into the same environment (`pip install dummy-sparse-attn`); the
dispatcher resolves it eagerly at model init, so a missing or broken plugin fails
loudly there rather than silently falling back to dense.

## Platform Defaults

### Blackwell (sm_100 / sm_103 / sm_120 / sm_121)

Auto-route preference, in order:

1. `CUDNN_ATTN` — when cuDNN ≥ 9.5 is available (ships in PyTorch 2.5+ wheels)
2. `FLASHINFER_ATTN` — when `flashinfer` is installed but cuDNN < 9.5
3. `FLASH_ATTN` — when `flash-attn` is installed with the Blackwell CUTE kernel
4. `TORCH_SDPA` — last resort

The startup log line `Defaulting to diffusion attention backend CUDNN_ATTN (Blackwell sm_120, cuDNN 91002)` confirms the route.

**Why CUDNN_ATTN by default**: on mask-heavy diffusion models (HunyuanVideo-1.5, Qwen-Image), cuDNN's pinned FMHA kernel sidesteps a PyTorch SDPA dispatch quirk where the unpinned dispatcher picks `EFFICIENT_ATTENTION` (~25 ms) for masked calls instead of cuDNN (~11 ms). The pin gives 2× e2e on HV-1.5 with no regression on lighter models.

### Hopper (sm_90) / Ada (sm_89) / Ampere (sm_80–sm_86)

Auto-route preference:

1. `FLASH_ATTN` — when `flash-attn` is installed
2. `TORCH_SDPA` — fallback

`CUDNN_ATTN` and `FLASHINFER_ATTN` are still selectable via env var on these GPUs but are not in the auto-route — FlashAttention 2 is the well-tuned path on pre-Blackwell hardware.

## End-to-End Benchmark (BF16, sm_120 RTX Pro 6000 Blackwell)

Same prompt and seed across runs. `Total generation time` from `text_to_video.py` / `text_to_image.py`.

| Model | Shape | TORCH_SDPA | CUDNN_ATTN | FLASHINFER_ATTN |
|---|---|---|---|---|
| HunyuanVideo-1.5 (T2V) | 480p / 33f / 50 steps | 147.05 s | **73.02 s** | 127.84 s |
| Wan 2.2 14B (T2V) | 480p / 33f / 40 steps | 117.75 s | 117.17 s | **115.07 s** |
| Qwen-Image (T2I) | 1024² / 50 steps | 17.41 s | **15.14 s** | 16.02 s |
| FLUX.2-dev (T2I) | 1024² / 50 steps, TP=2 | 53.62 s | **53.30 s** | 54.94 s |

Pattern: mask-heavy DiTs (HV-1.5, Qwen-Image) favor `CUDNN_ATTN`; lighter-mask DiTs and TP-saturated configs (Wan 2.2, FLUX.2 TP=2) tie within noise.

## Known Limitations

### LTX-2.0: `CUDNN_ATTN` crashes under torch.compile

LTX-2's audio attention has a symbolic head_dim under torch.compile tracing. cuDNN's SDPA backend selector rejects symbolic dims and Dynamo aborts compilation. Tracked in [#3121](https://github.com/vllm-project/vllm-omni/issues/3121).

**Workaround**: explicitly select `FLASHINFER_ATTN` or `TORCH_SDPA` for LTX-2.0:

```bash
DIFFUSION_ATTENTION_BACKEND=FLASHINFER_ATTN python examples/offline_inference/text_to_video/text_to_video.py \
    --model Lightricks/LTX-2 ...
```

### FA4 not yet integrated

FlashAttention-4 (released March 2026) targets Blackwell natively and reportedly beats cuDNN by ~20% on B200. As of this writing the `flash-attn-4 4.0.0b10` wheel crashes with `AttributeError: 'NoneType' object has no attribute '_trait'` during JIT on sm_120. Not yet wired into vLLM-Omni; revisit when stable lands.

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
    --output outputs/wan22_fa3.mp4
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

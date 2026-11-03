# Diffusion Attention Backends

Use this page to select and configure a diffusion attention backend. Backend-
specific installation, tuning, and compatibility details live in separate
guides so that the selection contract stays easy to scan.

Diffusion attention backend selection applies to DiT attention in image and
video generation models. It does **not** change autoregressive LLM attention,
which uses vLLM's attention selector.

For the internal selector, registry, and platform contract, see
[Attention Backend Selection](../../design/feature/attention_backend_selection.md).

## Choose a guide

| Need | Guide |
| --- | --- |
| Select a conservative or platform-native dense kernel | [Dense Backends](attention_backends/dense_backends.md) |
| Use FlashInfer trtllm-gen FMHA, Skip-Softmax, or TRTLLM SAGE quantization | [TRTLLM Attention](attention_backends/trtllm.md) |
| Install and use SageAttention 2.2 or SageAttention3 | [SageAttention](attention_backends/sage.md) |
| Use INT8 SPARGE top-k sparse attention on Intel XPU | [SPARGE Attention](attention_backends/sparge.md) |
| Match training or rollout kernels loaded from Hugging Face | [Hugging Face Hub Backends](attention_backends/huggingface_hub.md) |
| Use block-sparse video attention on Ascend NPU | [RainFusion](attention_backends/rainfusion.md) |
| Use FastVideo VSA with FastWan2.2-TI2V-5B or FastH3 MiniMax-H3 on CUDA | [FastVideo VSA](attention_backends/fastvideo_vsa.md) |
| Run VDN-H3 (Video DeltaNet MiniMax-H3) hybrid-attention checkpoints on CUDA | [VDN-H3 Hybrid Attention](attention_backends/vdnh3_attn.md) |

## Backend options

| Value | Family | Primary use | Detail |
| --- | --- | --- | --- |
| `TORCH_SDPA` | Dense | Conservative reference; always available | [Dense Backends](attention_backends/dense_backends.md#torch_sdpa) |
| `FLASH_ATTN` | Dense | FlashAttention 4/3/2 depending on the installed package and GPU | [Dense Backends](attention_backends/dense_backends.md#flash_attn) |
| `CUDNN_ATTN` | Dense | Mask-heavy DiTs on Blackwell with cuDNN 9.5 or newer | [Dense Backends](attention_backends/dense_backends.md#cudnn_attn) |
| `FLASHINFER_ATTN` | Dense or quantized | FlashInfer batch prefill; optional mixed Q/K and V dtypes | [Dense Backends](attention_backends/dense_backends.md#flashinfer_attn) |
| `TRTLLM_ATTN` | Dense, sparse, or quantized | Datacenter Blackwell with `head_dim=128` and compatible packed paths | [TRTLLM Attention](attention_backends/trtllm.md) |
| `SAGE_ATTN` | Quantized | SageAttention 2.2 INT8 attention | [SageAttention](attention_backends/sage.md#sage_attn) |
| `SAGE_ATTN_3` | Quantized | SageAttention3 on Blackwell | [SageAttention](attention_backends/sage.md#sage_attn_3) |
| `SPARGE_ATTN` | Block sparse + quantized | INT8 Q/K SPARGE top-k attention on Intel XPU | [SPARGE Attention](attention_backends/sparge.md) |
| `FLASH_ATTN_HUB` | Hub kernel | FlashAttention 2 from Hugging Face `kernels` | [Hugging Face Hub Backends](attention_backends/huggingface_hub.md) |
| `FLASH_ATTN_3_HUB` | Hub kernel | FlashAttention 3 from Hugging Face `kernels` on Hopper or newer | [Hugging Face Hub Backends](attention_backends/huggingface_hub.md) |
| `RAINFUSION_ATTN` | Block sparse | MindIE-SD RainFusion video attention on Ascend NPU | [RainFusion](attention_backends/rainfusion.md) |
| `FASTVIDEO_VSA` | Block sparse | FastVideo variable sparse self-attention for FastWan2.2-TI2V-5B and FastH3 MiniMax-H3 on CUDA | [FastVideo VSA](attention_backends/fastvideo_vsa.md) |
| `VDNH3_ATTN` | Hybrid window | Frame-window softmax of VDN-H3 hybrid attention; the model adds the linear branch | [VDN-H3 Hybrid Attention](attention_backends/vdnh3_attn.md) |

## Configuration

Backend selection is resolved in this order:

1. `--diffusion-attention-config` per-role configuration.
2. `--diffusion-attention-backend` or `DIFFUSION_ATTENTION_BACKEND` as a
   global default.
3. The current platform's default.

`--diffusion-attention-backend` is shorthand for
`--diffusion-attention-config.default.backend`. Do not pass it together with an
explicit `default.backend` in the structured configuration.

### Global default

```bash
vllm-omni serve <model> --diffusion-attention-backend FLASH_ATTN

# Backwards-compatible environment variable
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
```

### Per-role configuration

Roles are declared by each diffusion model. Common categories are `self` and
`cross`; a model may also use a more specific role such as
`ltx2.audio_to_video`. Resolution order is:

1. Exact `per_role[role]` match.
2. Category `per_role[role_category]` match.
3. `default`.
4. Platform default.

```bash
# Dotted flags
vllm-omni serve <model> \
  --diffusion-attention-config.default.backend FLASH_ATTN \
  --diffusion-attention-config.per_role.cross.backend TORCH_SDPA

# Equivalent JSON
vllm-omni serve <model> \
  --diffusion-attention-config \
  '{"default":{"backend":"FLASH_ATTN"},"per_role":{"cross":{"backend":"TORCH_SDPA"}}}'
```

### Python API

```python
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionSpec,
    OmniDiffusionConfig,
)

config = OmniDiffusionConfig(
    diffusion_attention_config=AttentionConfig(
        default=AttentionSpec(backend="FLASH_ATTN"),
        per_role={"cross": AttentionSpec(backend="TORCH_SDPA")},
    ),
    ...,
)
```

Backend-specific typed blocks are documented with their consumers:

- `quant`: [FlashInfer](attention_backends/dense_backends.md#flashinfer-quantized-attention)
  and [TRTLLM SAGE](attention_backends/trtllm.md#sage-quantization).
- `skip_softmax`: [TRTLLM Skip-Softmax](attention_backends/trtllm.md#skip-softmax).
- `sparge`: [SPARGE Attention](attention_backends/sparge.md#configuration).
- `block_sparse`: [RainFusion](attention_backends/rainfusion.md#configuration).
- `fastvideo_vsa_topk`: [FastVideo VSA](attention_backends/fastvideo_vsa.md#choose-top-k).

## Platform defaults

### Blackwell (sm_100 / sm_103 / sm_120 / sm_121)

The CUDA auto-route preference is:

1. `TRTLLM_ATTN` on datacenter Blackwell (sm_100/sm_103) when FlashInfer is
   available, `head_dim=128`, and the model declares a compatible packed or
   mask-free path.
2. `CUDNN_ATTN` when cuDNN 9.5 or newer is available.
3. `FLASHINFER_ATTN` when FlashInfer is available but cuDNN is too old.
4. `FLASH_ATTN` when a compatible package is installed.
5. `TORCH_SDPA`.

`TRTLLM_ATTN` is not auto-selected on workstation Blackwell
(sm_120/sm_121), for other head dimensions, or for paths that require an
unsupported mask.

### Hopper, Ada, and Ampere

The CUDA auto-route uses `FLASH_ATTN` when available and otherwise falls back
to `TORCH_SDPA`. `CUDNN_ATTN` and `FLASHINFER_ATTN` remain explicit options.

Other platforms validate an explicit backend and choose their own default
through the platform implementation. Check the startup log to confirm the
resolved backend.

## Choosing a backend manually

Override the platform default when you need:

- a correctness reference (`TORCH_SDPA`);
- a backend-specific workaround;
- training/rollout kernel alignment (Hub backends); or
- an explicitly validated sparse or quantized speedup.

The startup log prints the resolved backend and whether it came from explicit
configuration or platform defaulting. If no resolution message appears, check
earlier logs for diffusion-stage initialization failures.

## Reference benchmark

The following BF16 results were measured on an sm_120 RTX Pro 6000 Blackwell
with the same prompt and seed across runs. Treat them as reference results, not
portable guarantees.

| Model | Shape | `TORCH_SDPA` | `CUDNN_ATTN` | `FLASHINFER_ATTN` |
| --- | --- | ---: | ---: | ---: |
| HunyuanVideo-1.5 (T2V) | 480p / 33f / 50 steps | 147.05 s | **73.02 s** | 127.84 s |
| Wan 2.2 14B (T2V) | 480p / 33f / 40 steps | 117.75 s | 117.17 s | **115.07 s** |
| Qwen-Image (T2I) | 1024² / 50 steps | 17.41 s | **15.14 s** | 16.02 s |
| FLUX.2-dev (T2I) | 1024² / 50 steps, TP=2 | 53.62 s | **53.30 s** | 54.94 s |

Mask-heavy DiTs favored `CUDNN_ATTN`; lighter-mask or TP-saturated workloads
were close enough that users should benchmark their exact model and shape.

## Unified attention quantization (`diffusion_kv_cache_dtype`)

`diffusion_kv_cache_dtype` is a historically named Omni option. It requests
quantized attention **compute** for every eligible DiT attention layer; it does
not select the dtype of a persistent KV cache. It is independent of the
upstream vLLM `--kv-cache-dtype` option, which belongs to autoregressive KV
storage:

```bash
vllm-omni serve <model> --diffusion-kv-cache-dtype fp8
```

- Leaving the option unset or using `auto` keeps native dtype execution.
- `fp8` quantizes Q/K/V for the current attention call and dequantizes inside
  the attention kernel. Persistent model-owned KV storage is unaffected.
- CUDA FP8 attention uses FA3. Before quantization, vLLM's normalized Hadamard
  transform rotates Q/K within each head to reduce outlier-induced rounding
  error; V is unchanged. Non-power-of-two head dimensions use equal
  power-of-two blocks within the head.
- Ascend NPU passes BF16/FP16 Q/K/V and the selected precision to MindIE-SD's
  `quant_attention`, which performs the vendor-side quantization and attention.
- `diffusion_kv_cache_skip_steps` / `diffusion_kv_cache_skip_layers` exempt
  individual denoise steps or transformer blocks (format: `"0-9,20,25-30"`).

Backend support for `fp8`:

| Backend | Platform | Notes |
| --- | --- | --- |
| `FLASH_ATTN` | CUDA (SM90 Hopper) | FA3 FP8 |
| `FLASH_ATTN` | Ascend NPU | MindIE-SD `quant_attention` with Hadamard Q/K rotation; also `mxfp8`, `mxfp4` |
| `RAINFUSION_ATTN` | Ascend NPU | Block-sparse fp8/mxfp4 |

Backends not listed here reject the knob at startup with the list of compatible
options. Quantization is lossy; validate image/video quality against the
unquantized baseline before enabling it in production.

CUDA FP8 attention returns the input dtype (BF16 or FP16), casting FA3's BF16
output when needed. This path currently accepts dense inputs without caller
masks, packed-sequence metadata, or piecewise spans; incompatible requests fail explicitly.
FP8 attention requires FA3 on SM90 and a head dimension divisible by 16 and at
most 256; unsupported FA implementations are rejected.

Dynamic quantization adds Q/K rotation, reductions, FP8 conversion, and kernel launches.
FP8 attention can therefore be slower than BF16, especially for short
sequences. This option does not promise lower peak memory use or faster
end-to-end generation.

An explicit `diffusion_attention_config.*.quant` spec conflicts with the knob
and is rejected at config load; drop one of them. A pure
`flashinfer_backend` variant pin without dtypes is allowed to compose.

## Compatibility anchors

The following headings preserve links to sections that moved into dedicated
guides.

## TRTLLM_ATTN Backend and Skip-Softmax

See [TRTLLM Attention](attention_backends/trtllm.md#skip-softmax).

## TRTLLM_ATTN SAGE Quantization

See [TRTLLM Attention](attention_backends/trtllm.md#sage-quantization).

## RAINFUSION_ATTN Backend and Block-Sparse Video Attention

See [RainFusion](attention_backends/rainfusion.md).

## SageAttention Installation

See [SageAttention](attention_backends/sage.md#installation).

## SageAttention3 Installation

See [SageAttention](attention_backends/sage.md#sageattention3-installation).

## HuggingFace Kernels Hub Backends

See [Hugging Face Hub Backends](attention_backends/huggingface_hub.md).

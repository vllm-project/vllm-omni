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
| Match training or rollout kernels loaded from Hugging Face | [Hugging Face Hub Backends](attention_backends/huggingface_hub.md) |
| Use block-sparse video attention on Ascend NPU | [RainFusion](attention_backends/rainfusion.md) |
| Use FastVideo VSA with FastWan2.2-TI2V-5B on CUDA | [FastVideo VSA](attention_backends/fastvideo_vsa.md) |

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
| `FLASH_ATTN_HUB` | Hub kernel | FlashAttention 2 from Hugging Face `kernels` | [Hugging Face Hub Backends](attention_backends/huggingface_hub.md) |
| `FLASH_ATTN_3_HUB` | Hub kernel | FlashAttention 3 from Hugging Face `kernels` on Hopper or newer | [Hugging Face Hub Backends](attention_backends/huggingface_hub.md) |
| `RAINFUSION_ATTN` | Block sparse | MindIE-SD RainFusion video attention on Ascend NPU | [RainFusion](attention_backends/rainfusion.md) |
| `FASTVIDEO_VSA` | Block sparse | FastVideo variable sparse self-attention for FastWan2.2-TI2V-5B on CUDA | [FastVideo VSA](attention_backends/fastvideo_vsa.md) |

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

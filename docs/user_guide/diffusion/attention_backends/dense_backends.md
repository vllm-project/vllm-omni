# Dense Attention Backends

Use these backends to run dense attention. Optional quantized or sparse modes
remain inactive unless they are explicitly configured. Start with
`TORCH_SDPA` for correctness comparisons, then benchmark the fastest compatible
kernel for your model, shape, and hardware.

For selection precedence and per-role configuration, see the
[attention backend overview](../attention_backends.md).

## `TORCH_SDPA`

`TORCH_SDPA` calls PyTorch `scaled_dot_product_attention` and lets PyTorch's
dispatcher choose the implementation. It is always available and is the most
conservative reference when validating another backend.

```bash
vllm-omni serve <model> --diffusion-attention-backend TORCH_SDPA
```

## `FLASH_ATTN`

`FLASH_ATTN` uses the installed FlashAttention implementation. On Blackwell it
is FA4 only (`flash_attn.cute` / `vllm-omni[fa4]`). If FA4 is unavailable,
explicit `FLASH_ATTN` is rejected and automatic selection continues to another
compatible Blackwell backend. Hopper-only FA2/FA3 wheels are not used, even if
they import. On Hopper, Ada, and Ampere it is the preferred automatic route
when a compatible package is installed.

```bash
vllm-omni serve <model> --diffusion-attention-backend FLASH_ATTN
```

### FlashAttention 4 on Blackwell

Install the optional CUDA 13 extra:

```bash
pip install 'vllm-omni[fa4]'
```

Version `4.0.0b18` is required; earlier beta wheels had known JIT failures on
Blackwell. On Blackwell, `FLASH_ATTN` is FA4 only: Hopper-only FA2/FA3
wheels are not used, even if they import. If FA4 is missing, explicit
`FLASH_ATTN` is rejected and automatic selection continues to other
Blackwell backends.

## `TRTLLM_ATTN`

`TRTLLM_ATTN` runs FlashInfer's trtllm-gen FMHA kernels and is the platform
default on datacenter Blackwell for models that declare a compatible attention
path. Selected without a `quant` or `skip_softmax` block, it runs dense BF16
attention at FA4-level performance.

```bash
vllm serve <model> --omni \
  --diffusion-attention-backend TRTLLM_ATTN
```

See [TRTLLM Attention](trtllm.md) for its requirements and for the optional
SAGE quantization and Skip-Softmax modes.

## `CUDNN_ATTN`

`CUDNN_ATTN` pins PyTorch SDPA to `CUDNN_ATTENTION`. It is particularly useful
for mask-heavy DiTs and is automatically preferred on Blackwell when cuDNN
9.5 or newer is available and the higher-priority TRTLLM route is not
compatible.

```bash
vllm-omni serve <model> --diffusion-attention-backend CUDNN_ATTN
```

### LTX-2.0 limitation

LTX-2 audio attention has a symbolic head dimension during `torch.compile`
tracing. The cuDNN SDPA selector rejects that symbolic dimension and Dynamo
aborts compilation. This is tracked in
[issue #3121](https://github.com/vllm-project/vllm-omni/issues/3121).

Use `FLASHINFER_ATTN` or `TORCH_SDPA` as a workaround:

```bash
DIFFUSION_ATTENTION_BACKEND=FLASHINFER_ATTN \
  python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2 ...
```

## `FLASHINFER_ATTN`

`FLASHINFER_ATTN` uses FlashInfer's batch-prefill wrapper. It is an explicit
option on CUDA platforms and an automatic Blackwell fallback when FlashInfer
is installed but cuDNN is too old for `CUDNN_ATTN`.

On Blackwell, `auto` resolves to FlashInfer cute-dsl, which cannot run a
nontrivial custom mask. Automatic selection may fall back to SDPA for those
masks. An explicit `FLASHINFER_ATTN` selection does not: sequence-parallel
auto-padding that would create a padding mask is rejected at capability
preflight. Use `TORCH_SDPA`, pin `quant.flashinfer_backend` to `fa2`/`fa3`
where those kernels exist, or choose a mask-capable backend.

```bash
vllm-omni serve <model> --diffusion-attention-backend FLASHINFER_ATTN
```

### FlashInfer quantized attention

The backend accepts an `AttentionSpec.quant` block. For QK16/V8, keep Q and K
in FP16 or BF16 and use FP8 E4M3 for V:

```python
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionSpec,
    AttnQuantSpec,
    OmniDiffusionConfig,
)

config = OmniDiffusionConfig(
    diffusion_attention_config=AttentionConfig(
        default=AttentionSpec(
            backend="FLASHINFER_ATTN",
            quant=AttnQuantSpec(
                dtype_qk="bfloat16",
                dtype_vo="fp8_e4m3",
            ),
        ),
    ),
    ...,
)
```

`dtype_qk` controls Q and K; `dtype_vo` controls V. Mixed-dtype
configurations require FlashInfer 0.6.16rc1 or newer. The shared quantization
schema is also consumed by TRTLLM, but each backend validates its own allowed
fields and values; see [TRTLLM SAGE quantization](trtllm.md#sage-quantization).

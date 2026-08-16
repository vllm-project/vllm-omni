# Dense Attention Backends

Use these backends when you want dense attention without backend-specific
sparsity. Start with `TORCH_SDPA` for correctness comparisons, then benchmark
the fastest compatible kernel for your model, shape, and hardware.

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
prefers the CuTe-based FlashAttention 4 path and falls back to
FlashAttention 3 or 2 when that path is unavailable. On Hopper, Ada, and
Ampere it is the preferred automatic route when a compatible package is
installed.

```bash
vllm-omni serve <model> --diffusion-attention-backend FLASH_ATTN
```

### FlashAttention 4 on Blackwell

Install the optional CUDA 13 extra:

```bash
pip install 'vllm-omni[fa4]'
```

Version `4.0.0b18` is required; earlier beta wheels had known JIT failures on
Blackwell. If the CuTe path is unavailable, the backend falls back to the
compatible FlashAttention 3 or 2 path.

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

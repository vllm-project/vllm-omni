# SVDQuant W4A4

## Overview

[SVDQuant](https://arxiv.org/abs/2411.05007) combines four-bit weights
and activations with a small low-rank branch that corrects part of the
quantization error. vLLM-Omni consumes an offline-quantized checkpoint;
it does not calibrate the model while loading.

## Checkpoint contract

Place the following entry in the diffusion transformer's `config.json`:

```json
{
  "quantization_config": {
    "quant_method": "svdquant",
    "rank": 32,
    "precision": "nvfp4",
    "act_unsigned": false,
    "modules_to_not_convert": []
  }
}
```

For a quantized linear with input size `K`, output size `N`, and correction
rank `R`, the checkpoint stores:

| Suffix | Shape | dtype |
| --- | --- | --- |
| `qweight` | `(N, K / 2)` | `int8` (two packed FP4 values per byte) |
| `wscales` | `(K / 16, N)` | `float8_e4m3fn` |
| `proj_down` | `(K, R)` | `bfloat16` |
| `proj_up` | `(N, R)` | `bfloat16` |
| `smooth_factor` | `(K,)` | `bfloat16` |
| `wcscales` | `(N,)` | `bfloat16` |
| `wtscale` | `(1,)` | `bfloat16` |

`K` must be divisible by 16 on every tensor-parallel rank. Set `wcscales` to
ones when no per-output correction is needed. Modules listed in
`modules_to_not_convert` keep their checkpoint precision.

## Runtime support

The compatibility path accepts BF16 inputs and executes an NVFP4 GEMM followed
by the BF16 rank correction. It supports vLLM's FlashInfer, CUTLASS, and FBGEMM
NVFP4 tensor layouts; incompatible forced backends fail during model loading.
The loader accepts SM100, SM103 and SM120, subject to the selected backend's
hardware support. V1 model-level validation was performed on SM103; accepting
a capability is not evidence of model quality or E2E performance on it.

Set `linear_backend: "flashinfer"` in the transformer's serialized quantization
configuration to opt into native fused rank correction. The default is
`"compatibility"`. Native execution uses FlashInfer's public `svdquant_linear`
API with `"cute-dsl"` on SM120 or `"cutlass"` on SM100/SM103. It requires the
reviewed backend parameter from [FlashInfer PR #4420](https://github.com/flashinfer-ai/flashinfer/pull/4420),
W4A4, rank 32/64/96/128, N and K divisible by 128, and finite positive smoothing
and output scales. Unsupported contracts fail explicitly; choose compatibility
for those checkpoints. Inspect the installed API as well as its package version.

The loader retains the canonical packed residual, interleaves its block scales,
and compensates the native up projection for scalar and output channel scales.
The down projection continues to consume the original input. Native and
compatibility paths can differ in BF16 rounding, so validate the full model
using the selected path. W4A16 remains on its separate reference implementation.

## Low-memory candidate

The [MiniMax-H3 low-memory recipe](../../../recipes/MiniMaxAI/MiniMax-H3-SVDQuant.md)
exports a finite-schedule BF16 AdaLN table and an encoder-local NVFP4 W4A16
checkpoint. W4A16 declares `activation_bits: 16` in the text encoder's own
`config.json`; the transformer defaults to `activation_bits: 4`. A global
transformer SVDQuant config does not quantize a BF16 encoder implicitly.

The weight-only reference path keeps packed weights and dequantizes one linear
per call for BF16 GEMM. It is distinct from native W4A4, and does not claim a
fused W4A16 operator. The supplied offline recipe is uncalibrated; output quality,
full E2E latency and 32 GB deployment require independent model validation.

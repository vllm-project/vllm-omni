# SVDQuant W4A4

## Overview

[SVDQuant](https://arxiv.org/abs/2411.05007) combines four-bit weights
and activations with a small low-rank branch that corrects part of the
quantization error. vLLM-Omni consumes an offline-quantized NVFP4 or MXFP4
checkpoint; it does not calibrate the model while loading.

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
| `wscales` (NVFP4) | `(K / 16, N)` | `float8_e4m3fn` |
| `wscales` (MXFP4) | `(K / 32, N)` | `uint8` (E8M0 scales) |
| `proj_down` | `(K, R)` | `bfloat16` |
| `proj_up` | `(N, R)` | `bfloat16` |
| `smooth_factor` | `(K,)` | `bfloat16` |
| `wcscales` (NVFP4 only) | `(N,)` | `bfloat16` |
| `wtscale` (NVFP4 only) | `(1,)` | `bfloat16` |

`K` must be divisible by 16 for NVFP4 or 32 for MXFP4 on every
tensor-parallel rank. Set `wcscales` to ones when no NVFP4 per-output
correction is needed. MXFP4 checkpoints do not contain `wcscales` or
`wtscale`. Modules listed in `modules_to_not_convert` keep their checkpoint
precision.

## Runtime support

The compatibility path accepts BF16 inputs and executes a four-bit GEMM followed
by the BF16 rank correction. NVFP4 is enabled on SM103 and supports vLLM's
FlashInfer, CUTLASS, and FBGEMM tensor layouts; incompatible forced backends
fail during model loading.

MXFP4 is enabled on SM100 and SM103 and uses vLLM's standard MXFP4 kernel
selection. FlashInfer provides W4A4 execution when its CuTe-DSL backend is
available; otherwise Marlin uses W4A16. Native fusion of the quantized GEMM and
rank correction is separate from this checkpoint-loading contract.

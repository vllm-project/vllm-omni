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
SM103 is the currently validated and enabled hardware target. Native fusion of
the NVFP4 GEMM and rank correction is separate from this checkpoint-loading
contract.

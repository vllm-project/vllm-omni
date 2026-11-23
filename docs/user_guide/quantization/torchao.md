# TorchAO Quantization

## Overview

[TorchAO](https://github.com/pytorch/ao) provides quantization tools for
PyTorch and supports both pre-quantized checkpoint loading and runtime
quantization.

## Hardware Support

| Device | Support |
| -------- | --------- |
| NVIDIA Blackwell GPU (SM 100+) | ⭕ |
| NVIDIA Ada GPU (SM 89) | ✅ |
| NVIDIA Hopper GPU (SM 90) | ⭕ |
| NVIDIA Ampere GPU (SM 80+) | ⭕ |
| AMD ROCm | ⭕ |
| Intel XPU | ⭕ |
| Ascend NPU | ⭕ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this
guide.

## Model Type Support

### Diffusion Model (Boogu-Image)

| Model                 | Checkpoint                                                         | Scope                      | Scheme                  |
| --------------------- | ------------------------------------------------------------------ | -------------------------- | ----------------------- |
| Boogu-Image Base/Edit | `Boogu/Boogu-Image-0.1-Base-fp8`, `Boogu/Boogu-Image-0.1-Edit-fp8` | Diffusion transformer only | FP8 weight-only (W8A16) |

## Configuration

Using Boogu-Image as an example:

```bash
vllm serve Boogu/Boogu-Image-0.1-Base-fp8 \
  --omni \
  --port 8091 \
  --diffusion-quantization-config \
  '{"transformer":{"method":"torchao_float8_weight_only"}}'
```

The equivalent command using the complete serialized TorchAO configuration is:

```bash
vllm serve Boogu/Boogu-Image-0.1-Base-fp8 \
  --omni \
  --port 8091 \
  --diffusion-quantization-config \
  '{
    "transformer": {
      "method": "torchao",
      "quant_type": {
        "default": {
          "_type": "Float8WeightOnlyConfig",
          "_version": 2,
          "_data": {
            "weight_dtype": {
              "_type": "torch.dtype",
              "_data": "float8_e4m3fn"
            },
            "set_inductor_config": false
          }
        }
      }
    }
  }'
```

## Parameters

| Parameter    | Type | Default | Description                                                                                                                         |
| ------------ | ---- | ------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `method`     | str  | -       | Use `torchao_float8_weight_only` for the serialized FP8 weight-only shorthand, or `torchao` with a full `quant_type` configuration. |
| `quant_type` | dict | -       | Complete serialized TorchAO configuration used with `method: "torchao"`.                                                            |

See TorchAO's [vLLM integration guide](https://docs.pytorch.org/ao/0.17/eager_tutorials/torchao_vllm_integration.html#configuration-system)
for the serialized form and [workflow configs](https://docs.pytorch.org/ao/0.17/api_reference/api_ref_quantization.html#workflow-configs)
for available configuration classes and parameters.

## Validation and Notes

At load time, vLLM-Omni builds a `TorchAOConfig` from
`--diffusion-quantization-config`. For the Boogu-Image checkpoints documented
here, indexed PyTorch `.bin` shards are loaded through `pt_weights_iterator`.

When a component map is used, only the components included in the
configuration use TorchAO. Other components keep their own checkpoint and
runtime settings.

Use either the `torchao_float8_weight_only` shorthand or the equivalent full
`quant_type` configuration shown above. The checkpoint must already contain
weights quantized with TorchAO.

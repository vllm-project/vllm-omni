# INT8 ConvRot Quantization

## Overview

INT8 ConvRot loads ComfyUI tensor-wise W8A8 checkpoints whose linear weights
were rotated and quantized offline. At runtime, `comfy-kitchen` applies the
matching Hadamard rotation to each activation, dynamically quantizes it to
INT8, and executes the INT8 GEMM.

This path currently supports the pruned FL2VA MiniMax-H3 checkpoint:

`Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`

The base `MiniMaxAI/MiniMax-H3` repository still supplies the model config,
tokenizer, text encoder, and VAEs. The Comfy checkpoint replaces only the
FL2VA diffusion transformer.

## Installation

Install the optional native kernel dependency:

```bash
pip install "vllm-omni[int8-convrot]"
```

The published `comfy-kitchen` CUDA wheel requires a CUDA 13 PyTorch runtime
and an R580-or-newer NVIDIA driver. INT8 ConvRot fails at startup or execution
when the native CUDA backend is unavailable; it does not silently fall back to
an eager or dequantized implementation.

## Configuration

Download the checkpoint locally, or use its Hugging Face `resolve` URL as the
transformer override. For the standard single-stage diffusion path, construct
`Omni` with both the replacement path and its quantization method:

```python
from vllm_omni import Omni

checkpoint = (
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
)

omni = Omni(
    model="MiniMaxAI/MiniMax-H3",
    task_type="fl2va",
    model_paths={"transformer": checkpoint},
    diffusion_quantization_config={
        "transformer": {"method": "int8_convrot"},
    },
    trust_remote_code=True,
)
```

For the opt-in disaggregated topology, stage 0 is the text encoder and stage 1
is diffusion. Scope both settings to stage 1:

```bash
export CHECKPOINT="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"

vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --task-type fl2va \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated.yaml \
  --stage-overrides \
  "{\"1\":{\"model_paths\":{\"transformer\":\"${CHECKPOINT}\"}}}" \
  --diffusion-quantization-config \
  '{"transformer":{"method":"int8_convrot"}}'
```

Adjust the disaggregated deploy's stage devices and parallel sizes for your
hardware. Do not apply a global tensor-parallel override: its text-encoder and
diffusion stages intentionally have different topologies.

For a local checkpoint, `model_paths["transformer"]` may point to the
`.safetensors` file itself or to a directory containing exactly one
`.safetensors` file. Because the published files do not carry partition
metadata and FL2VA/Ref2VA have the same tensor schema, the filename must contain
exactly one standalone `fl2va` or `ref2va` token. The loader rejects a missing,
ambiguous, or serving-partition-mismatched token.

The loader reads each layer's `.comfy_quant` marker and configures only the
marked linears. Unmarked token-refiner and input/output layers keep their
checkpoint dtype. The pruned checkpoint's `adaln_t_table` is also detected
automatically, so the model uses the matching low-rank AdaLN curve path instead
of constructing the dense time embedder.

## Tensor Parallelism

ConvRot operates independently on fixed-size groups along the input dimension.
For row-parallel linears, every tensor-parallel shard boundary must therefore
remain aligned to the checkpoint's `convrot_groupsize` (256 for this
checkpoint). vLLM-Omni validates the TP-local width and rejects incompatible
parallel sizes instead of producing incorrect output.

The loader rejects any tensor-parallel degree that splits a ConvRot group.
Validate end-to-end output quality and memory for the exact deploy topology,
resolution, and clip length used in production.

HSDP/FSDP is rejected for this checkpoint. Its mixed INT8 weights, FP32
per-row scales, and FP32 AdaLN curve projections require a dedicated sharding
policy that is not yet implemented.

## Checkpoint Contract

Each quantized linear uses three tensors:

```text
<layer>.weight        I8  [out_features, in_features]
<layer>.weight_scale  F32 [out_features] or [out_features, 1]
<layer>.comfy_quant   U8  JSON bytes
```

The marker must identify the `int8_tensorwise` format and its ConvRot group
size. The loader validates marker JSON, tensor dtypes and shapes, per-output-row
scales, and group alignment before model construction.

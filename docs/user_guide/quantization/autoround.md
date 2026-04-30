# AutoRound Quantization

## Overview

[AutoRound](https://github.com/intel/auto-round) produces pre-quantized
checkpoints for LLMs, VLMs, and diffusion models. vLLM-Omni reads the
checkpoint's `config.json` and auto-detects
`quantization_config.quant_method = "auto-round"`.

AutoRound is static quantization: no `--quantization` flag is needed at
inference time when the checkpoint already contains the quantization config.

## Hardware Support

| Device | Support |
|--------|---------|
| NVIDIA Blackwell GPU (SM 100+) | ✅ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ |
| NVIDIA Ampere GPU (SM 80+) | ✅ |
| AMD ROCm | ⭕ |
| Intel XPU | ✅ |
| Ascend NPU | ❌ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this
guide. AutoRound is Intel-supported.

## Model Type Support

### Diffusion Model (Qwen-Image, Wan2.2)

| Model | Checkpoint | Scope | Scheme | Backend |
|-------|------------|-------|--------|---------|
| FLUX.1-dev | `vllm-project-org/FLUX.1-dev-AutoRound-w4a16` | Diffusion transformer | W4A16 | GPTQ-Marlin or Intel-supported AutoRound backend |
| Qwen-Image | Not listed | Diffusion transformer | W4A16 | Not validated |
| Wan2.2 | Not listed | Diffusion transformer | W4A16 | Not validated |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Model | Checkpoint | Scope | Scheme | Backend |
|-------|------------|-------|--------|---------|
| Qwen2.5-Omni-7B | `Intel/Qwen2.5-Omni-7B-int4-AutoRound` | Language-model stage | W4A16 | AutoRound |
| Qwen3-Omni-30B-A3B-Instruct | `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` | Thinker language-model stage | W4A16 | AutoRound |
| Qwen3-TTS | Not listed | TTS language-model stage | W4A16 | Not validated |

AutoRound support is checkpoint-driven. A model is supported when its
checkpoint uses a compatible INC/AutoRound config and the target stage maps to
vLLM-Omni's runtime module names.

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

| Model | Scope | Status | Notes |
|-------|-------|--------|-------|
| BAGEL | Checkpoint-defined diffusion or transformer stage | Not validated | Requires a compatible AutoRound checkpoint |
| GLM-Image | Checkpoint-defined diffusion or transformer stage | Not validated | Requires a compatible AutoRound checkpoint |

## Configuration

Python API:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="vllm-project-org/FLUX.1-dev-AutoRound-w4a16")

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=28),
)
outputs[0].save_images("output.png")
```

CLI:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model vllm-project-org/FLUX.1-dev-AutoRound-w4a16 \
  --prompt "A cat sitting on a windowsill" \
  --num-inference-steps 28 \
  --output outputs/flux_w4a16.png
```

## Parameters

| Field | Type | Description |
|-------|------|-------------|
| `quant_method` | str | Must be `"auto-round"` |
| `bits` | int | Quantized weight bit width, usually `4` |
| `group_size` | int | Quantization group size |
| `packing_format` | str | AutoRound packing format, for example `auto_round:auto_gptq` |
| `block_name_to_quantize` | str | Checkpoint block names that should map to runtime module names |

The checkpoint should contain a config like:

```json
{
  "quantization_config": {
    "quant_method": "auto-round",
    "bits": 4,
    "group_size": 128,
    "sym": true,
    "packing_format": "auto_round:auto_gptq",
    "block_name_to_quantize": "transformer_blocks,single_transformer_blocks"
  }
}
```

## Validation and Notes

At load time, vLLM-Omni builds an `OmniINCConfig`, remaps checkpoint block names
to runtime module names, and selects the matching vLLM compute backend.

Example checkpoint creation:

```bash
auto-round \
  --model black-forest-labs/FLUX.1-dev \
  --scheme W4A16 \
  --batch_size 1 \
  --disable_opt_rtn \
  --dataset coco2014 \
  --iters 0
```

Use the generated output directory directly as the `model` argument. See the
[AutoRound documentation](https://github.com/intel/auto-round) for all
available schemes and options.

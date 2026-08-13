# AutoRound Quantization

## Overview

[AutoRound](https://github.com/intel/auto-round) produces pre-quantized
checkpoints for LLMs, VLMs, diffusion models, and world models. vLLM-Omni reads the
checkpoint's `config.json` and auto-detects
`quantization_config.quant_method = "auto-round"`.

AutoRound is static quantization: no `--quantization` flag is needed at
inference time when the checkpoint already contains the quantization config.

The validation terms on this page follow the
[support levels](overview.md#support-levels) defined in the overview. A listed
checkpoint is not enough by itself to establish end-to-end model support.

## Hardware Support

| Device | Support |
|--------|---------|
| NVIDIA Blackwell GPU (SM 100+) | ✅ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ |
| NVIDIA Ampere GPU (SM 80+) | ✅ |
| AMD ROCm | ⭕ |
| Intel XPU | ✅ |
| Ascend NPU | ❌ |

Legend: `✅` backend available, `❌` unsupported, `⭕` not verified in this
guide. This table is method-level backend availability. AutoRound is an Intel
project; vLLM selects a compatible CUDA or XPU compute backend at load time.

## Model Type Support

### Diffusion Model (Qwen-Image, Wan2.2)

| Model | Checkpoint | Scope | Scheme | Validation |
|-------|------------|-------|--------|------------|
| FLUX.1-dev | `vllm-project-org/FLUX.1-dev-AutoRound-w4a16` | Diffusion transformer | W4A16 | Validated |
| Qwen-Image | `INC4AI/Qwen-Image-AutoRound-W4A16` | Diffusion transformer | W4A16 | Validated with end-to-end generation and quality comparison |
| Wan2.2-I2V | `Intel/Wan2.2-I2V-A14B-Diffusers-int4-AutoRound` | Both diffusion transformers | W4A16 | CI-backed |
| Wan2.2-T2V | `Intel/Wan2.2-T2V-A14B-Diffusers-int4-AutoRound` | Both diffusion transformers | W4A16 | CI-backed |
| Wan2.2-TI2V | `Intel/Wan2.2-TI2V-5B-Diffusers-int4-AutoRound` | Diffusion transformer | W4A16 | Validated; not in the scheduled Wan AutoRound job |

CUDA execution uses GPTQ-Marlin or another compatible vLLM AutoRound backend;
Intel XPU uses the Intel-supported backend. The text encoder and VAE stay in
BF16 for all entries above.

### World Model (Cosmos3)

| Model | Checkpoint | Scope | Scheme | Validation |
|-------|------------|-------|--------|------------|
| Cosmos3-Nano | `Intel/Cosmos3-Nano-int4-AutoRound` | World-model transformer | W4A16 | Integrated; no model-specific end-to-end test is maintained in-tree |
| Cosmos3-Super | `Intel/Cosmos3-Super-int4-AutoRound` | World-model transformer | W4A16 | Validated with a manual serving run |

The Cosmos VAE and guardrail components stay in BF16.

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Model | Checkpoint | Scope | Scheme | Validation |
|-------|------------|-------|--------|------------|
| Qwen2.5-Omni-7B | `Intel/Qwen2.5-Omni-7B-int4-AutoRound` | Thinker language-model stage | W4A16 | Validated outside scheduled CI |
| Qwen3-Omni-30B-A3B-Instruct | `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` | Thinker language-model stage | W4A16 | Validated outside scheduled CI |
| Qwen3-TTS | Not listed | TTS language-model stage | W4A16 | Not validated |

AutoRound support is checkpoint-driven. A model is supported when its
checkpoint uses a compatible INC/AutoRound config and the target stage maps to
vLLM-Omni's runtime module names.

For Qwen-Omni, only the thinker language model uses the AutoRound checkpoint.
Audio and vision encoders, the talker, and waveform-decoder stages stay in
BF16.

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

| Model | Checkpoint | Scope | Validation |
|-------|------------|-------|------------|
| GLM-Image | `Intel/GLM-Image-int4-AutoRound` | Diffusion transformer | CI-backed |
| BAGEL | Not listed | Checkpoint-defined diffusion or transformer stage | Not validated |

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

Checkpoint auto-detection and successful layer construction establish the
**Integrated** level. Promote a model to **Validated** only after the named
checkpoint completes an end-to-end generation or multimodal request and its
output passes a model-appropriate sanity or quality check. Compare against the
BF16 baseline for quality-sensitive image and video paths. Promote it to
**CI-backed** only when that named checkpoint is selected by scheduled hardware
CI.

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

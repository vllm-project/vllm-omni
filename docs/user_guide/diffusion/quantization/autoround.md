# AutoRound Quantization

## Overview

[AutoRound](https://github.com/intel/auto-round) is an advanced quantization toolkit designed for Large Language Models (LLMs), Vision-Language Models (VLMs), and diffusion models. It achieves high accuracy at ultra-low bit widths (2–4 bits) with minimal tuning by leveraging sign-gradient descent, while providing broad hardware compatibility with multi-datatype support.

The quantization config is auto-detected from the checkpoint's `config.json` (`quantization_config.quant_method = "auto-round"`). No extra CLI flags are needed.

### Supported Schemes

| Scheme | Bits | Status |
|--------|------|--------|
| W4A16 | 4 | ✅ Supported |
| W8A16 | 8 | Planned |

W4A16 is the first supported scheme. Additional schemes will be added in future releases.

## Configuration

1. **Python API**: point `model` at a pre-quantized checkpoint. The quantization is detected automatically.

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

2. **CLI**: pass the quantized model path directly.

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model vllm-project-org/FLUX.1-dev-AutoRound-w4a16 \
  --prompt "A cat sitting on a windowsill" \
  --num-inference-steps 28 \
  --output outputs/flux_w4a16.png
```

No `--quantization` flag is needed — the quantization method is read from the checkpoint.

## How It Works

The checkpoint's `config.json` contains:

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

At load time:

1. `TransformerConfig.from_dict()` parses the `quantization_config` section and builds a vLLM `INCConfig` via `build_quant_config("auto-round", ...)`.
2. `OmniDiffusionConfig.set_tf_model_config()` propagates the detected config to the engine.
3. The appropriate compute kernel (e.g. GPTQ-Marlin for W4A16) is selected automatically based on the checkpoint's bit-width and packing format.

## Supported Models

| Model | HF Checkpoint | Scheme | Group Size | Backend |
|-------|--------------|--------|------------|---------|
| FLUX.1-dev | `vllm-project-org/FLUX.1-dev-AutoRound-w4a16` | W4A16 | 128 | GPTQ-Marlin |

## Creating a Quantized Checkpoint

Use [AutoRound](https://github.com/intel/auto-round) to quantize a BF16 model. The `--scheme` flag selects the quantization scheme:

```bash
# W4A16 (4-bit weight, 16-bit activation)
auto-round \
    --model black-forest-labs/FLUX.1-dev \
    --scheme W4A16 \
    --batch_size 1 \
    --disable_opt_rtn \
    --dataset coco2014 \
    --iters 0
```

The output directory can be used directly as the `model` argument. See the [AutoRound documentation](https://github.com/intel/auto-round) for all available schemes and options.

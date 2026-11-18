# Stable Diffusion 3.5

> Text-to-image serving with Stability AI's SD 3.5 family

## Summary

- Vendor: Stability AI
- Model: `stabilityai/stable-diffusion-3.5-medium`, `stabilityai/stable-diffusion-3.5-large`
- Task: Text-to-image generation
- Mode: Online serving with the OpenAI-compatible images API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a lightweight, fast text-to-image model that
fits on a single consumer or workstation GPU. SD 3.5-medium requires only
~17 GiB VRAM and generates 1024×1024 images in ~4 seconds, making it one of
the most accessible diffusion models in vLLM-Omni. SD 3.5-large produces
noticeably sharper images at the cost of ~34 GiB VRAM and ~21 seconds per
image.

## References

- Model (medium): <https://huggingface.co/stabilityai/stable-diffusion-3.5-medium>
- Model (large): <https://huggingface.co/stabilityai/stable-diffusion-3.5-large>
- Related example under `examples/`:
  [`examples/online_serving/text_to_image/README.md`](../../examples/online_serving/text_to_image/README.md)
- User guide:
  [`docs/user_guide/examples/online_serving/text_to_image.md`](../../docs/user_guide/examples/online_serving/text_to_image.md)

## FP8 DiT and INT8 T5 for SD3 / SD3.5 medium

The native pipeline supports FP8 for the image-stream FFN down-projection
and INT8 weight-only quantization for the T5 encoder (`text_encoder_3`).
Attention, the context stream, both CLIP encoders, and the VAE retain the
configured compute dtype. Use the existing component configuration to enable
the combination:

```bash
uv pip install torchao==0.17.0
vllm serve stabilityai/stable-diffusion-3.5-medium --omni \
  --diffusion-quantization-config \
  '{"transformer":{"method":"fp8"},"text_encoder_3":{"method":"int8"}}' \
  --enforce-eager --port 8091
```

For offline inference:

```python
from vllm_omni import Omni

omni = Omni(
    model="stabilityai/stable-diffusion-3.5-medium",
    quantization_config={
        "transformer": {"method": "fp8"},
        "text_encoder_3": {"method": "int8"},
    },
    enforce_eager=True,
)
```

The same configuration applies to
`stabilityai/stable-diffusion-3-medium-diffusers`. T5 weights are quantized
during CPU checkpoint loading before GPU placement. Encoders remain on GPU
between requests, avoiding repeated CPU/GPU transfers. A plain `fp8` setting
quantizes only the DiT; T5 INT8 must be selected through the component map.
The T5 path requires a BF16/FP16 source checkpoint and the optional `torchao`
package. It does not load serialized INT8 T5 checkpoints. This INT8 T5 path
requires models to remain on GPU: diffusion CPU offload is rejected before
loading because its hooks do not support these TorchAO tensor subclasses.
Omit T5 quantization when using the existing CPU offload paths.

Run the BF16-versus-quantized quality gate with:

```bash
pytest tests/diffusion/quantization/test_quantization_quality.py \
  -v -m "" -k "fp8_int8_sd3"
```

Compare memory, latency, and LPIPS with the BF16 baseline:

```bash
python benchmarks/diffusion/quantization_quality.py \
  --model stabilityai/stable-diffusion-3.5-medium --task t2i \
  --quantization-config \
  '{"transformer":{"method":"fp8"},"text_encoder_3":{"method":"int8"}}' \
  --prompts "watercolor painting of a mountain lake at sunrise, soft pastel colors" \
  --height 768 --width 768 --num-inference-steps 28 --seed 42 \
  --enforce-eager --output-dir sd3-quantization-benchmark
```

### Validation on RTX 4090 D

Measured on one 24 GiB RTX 4090 D, batch size one, in eager mode with
PyTorch 2.13.0+cu129, vLLM 0.31.0+cu129, Diffusers 0.40.0, Transformers
5.14.1, and TorchAO 0.17.0. The quality gate used 1024×1024, 20 steps,
seed 42, and `a cup of coffee on a wooden table, morning light`.

| Model | BF16 peak | FP8 DiT + INT8 T5 peak | Reduction | LPIPS vs BF16 |
| --- | ---: | ---: | ---: | ---: |
| SD3.5 medium | 17.98 GiB | 13.46 GiB | 25% | 0.0475 |
| SD3 medium | 17.00 GiB | 12.48 GiB | 27% | 0.0096 |

At 768×768 and 28 steps, median single-image generation time over three
repeats was 1.81 s versus 1.79 s for SD3.5 and 1.51 s versus 1.49 s for
SD3. Two additional 1024×1024 prompts per model gave LPIPS below 0.05.
These measurements cover three prompts/seeds per model, not general image
quality or concurrent serving performance.

CPU-side quantization also lowered the peak during startup and its dummy
warmup: 16.19 to 11.68 GiB for SD3.5, and 15.21 to 10.70 GiB for SD3.
It increased startup time from 7.61 to 25.22 s and 5.99 to 20.95 s,
respectively, because weights are quantized on each load. Peak values use
PyTorch allocated memory; CUDA context and allocator reserve are additional.

## Hardware Support

This recipe documents two CUDA GPU configurations on a single RTX A6000 48 GB,
one for each model variant. Extend it with more hardware sections as community
validation lands.

## GPU

### 1x NVIDIA RTX A6000 48GB — stable-diffusion-3.5-medium

#### Environment

- OS: Ubuntu 22.04
- Python: 3.12
- Driver / runtime: NVIDIA driver 580.126.09, CUDA 13.0
- vLLM version: 0.22.0
- vLLM-Omni version or commit: 0.22.0rc2.dev8+g41a795b5d

#### Command

```bash
vllm serve stabilityai/stable-diffusion-3.5-medium --omni --port 8091
```

#### Verification

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-3.5-medium",
    "prompt": "a cute cat sitting on a windowsill, highly detailed, sharp focus",
    "negative_prompt": "blurry, out of focus",
    "guidance_scale": 4.5,
    "n": 1,
    "size": "1024x1024"
  }' | python3 -c "
import json, base64, sys
data = json.load(sys.stdin)
with open('output.png', 'wb') as f:
    f.write(base64.b64decode(data['data'][0]['b64_json']))
print('saved output.png')
"
```

Or use the example client:

```bash
python examples/online_serving/text_to_image/openai_chat_client.py \
  --server http://localhost:8091 \
  --prompt "a cute cat sitting on a windowsill" \
  --output output.png
```

#### Notes

- Memory usage: Model weights ~15.6 GiB, peak VRAM ~17.0 GiB (no CFG), ~20.9 GiB with `guidance_scale > 1` (CFG enabled).
- Generation time: ~4 seconds without CFG, ~8 seconds with `guidance_scale=4.5`.
- Model loading time: ~55 seconds (including weight download on first run).
- `negative_prompt` requires `guidance_scale > 1` to take effect; SD3 defaults to 1.0 (no CFG).

### 1x NVIDIA RTX A6000 48GB — stable-diffusion-3.5-large

#### Environment

- OS: Ubuntu 22.04
- Python: 3.12
- Driver / runtime: NVIDIA driver 580.126.09, CUDA 13.0
- vLLM version: 0.22.0
- vLLM-Omni version or commit: 0.22.0rc2.dev8+g41a795b5d

#### Command

```bash
vllm serve stabilityai/stable-diffusion-3.5-large --omni --port 8091
```

> **Note**: `stable-diffusion-3.5-large` is a gated model. Accept the license
> on [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
> and set `HF_TOKEN` before running.

#### Verification

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-3.5-large",
    "prompt": "a cute cat sitting on a windowsill, highly detailed, sharp focus",
    "negative_prompt": "blurry, out of focus",
    "guidance_scale": 4.5,
    "n": 1,
    "size": "1024x1024"
  }' | python3 -c "
import json, base64, sys
data = json.load(sys.stdin)
with open('output.png', 'wb') as f:
    f.write(base64.b64decode(data['data'][0]['b64_json']))
print('saved output.png')
"
```

#### Notes

- Memory usage: Peak VRAM ~31.6 GiB at 1024×1024, ~45.5 GiB at 2048×2048 with `guidance_scale=4.5`.
- Generation time: ~19 seconds at 1024×1024, ~102 seconds at 2048×2048.
- Image quality is noticeably better than medium, with sharper details.
- `negative_prompt` requires `guidance_scale > 1` to take effect; SD3 defaults to 1.0 (no CFG).
- 2048×2048 fits within 48 GB but leaves little headroom (~45.5 GiB used out of 49 GiB).

## XPU

### 1x Intel Arc Pro B70 (32 GB) — stable-diffusion-3.5-medium

Offline text-to-image at 1024x1024 with BF16 weights fully resident.

#### Environment

- OS: Linux
- Python: 3.10+
- torch: 2.13.0+xpu
- vLLM: 0.29.0 (`98dff2a8`)
- vLLM-Omni: `main` at `4c7a98c2`

#### Command

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model stabilityai/stable-diffusion-3.5-medium \
  --prompt "a sunset over mountains, photorealistic" \
  --num-inference-steps 28 \
  --vae-use-tiling \
  --output sd35_medium_output.png
```

#### Verification

Confirm `sd35_medium_output.png` is written and matches the prompt.

#### Notes

- Memory usage: peak 19.5 GiB against 29.8 GiB free.
- Known limitations: only `stable-diffusion-3.5-medium` offline generation was
  qualified. `stable-diffusion-3.5-large` and online serving are out of scope
  for this profile.

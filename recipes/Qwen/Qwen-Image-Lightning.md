# Qwen-Image-Lightning

> 4/8-step text-to-image serving with the lightx2v step-distillation LoRA

## Summary

- Vendor: lightx2v (adapter) / Qwen (base model)
- Model: `Qwen/Qwen-Image` + `lightx2v/Qwen-Image-Lightning`
- Task: Text-to-image generation, accelerated to 4 or 8 denoising steps
- Mode: Online serving with a per-request LoRA
- Maintainer: Community

## When to use this recipe

Use this recipe when you want Qwen-Image latency reduced by roughly an order
of magnitude: the Lightning adapter is a step-distillation LoRA that produces
fully formed images in 4 or 8 steps with CFG disabled, instead of the ~50-step
CFG schedule of the base model.

The Lightning checkpoints are published as single Kohya-style safetensors
files (`lora_down/lora_up` keys plus per-module `alpha` scalars, no
`adapter_config.json`). vLLM-Omni converts this format to its PEFT layout
in memory at load time, so the files can be used as downloaded.

## References

- Adapter weights: [`lightx2v/Qwen-Image-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-Lightning)
- Base model recipe: [`recipes/Qwen/Qwen-Image.md`](Qwen-Image.md)
- LoRA guide: [`docs/user_guide/diffusion/lora.md`](../../docs/user_guide/diffusion/lora.md)

## Hardware Support

This recipe documents one CUDA GPU serving configuration. Extend it with more
hardware sections as community validation lands.

## GPU

### 1x A100/A800 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with an 80 GB GPU
- vLLM / vLLM-Omni: match the repository requirements for your checkout

#### Assets

Download one Lightning checkpoint next to the server, e.g.:

```bash
huggingface-cli download lightx2v/Qwen-Image-Lightning \
  Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors \
  --local-dir /path/to/qwen-image-lightning
```

#### Command

Start the baseline server:

```bash
vllm serve Qwen/Qwen-Image --omni --port 8091
```

Request an image with the Lightning LoRA, 4 steps, CFG disabled:

```bash
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image",
    "prompt": "a corgi wearing sunglasses on a beach, golden hour, photo",
    "num_inference_steps": 4,
    "true_cfg_scale": 1.0,
    "seed": 42,
    "lora": {
      "name": "lightning-4steps",
      "path": "/path/to/qwen-image-lightning/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors",
      "scale": 1.0
    }
  }'
```

For offline inference, set the LoRA on the sampling params:

```python
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

lora_path = "/path/to/qwen-image-lightning/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
omni = Omni(model="Qwen/Qwen-Image", lora_path=lora_path)
params = OmniDiffusionSamplingParams(
    num_inference_steps=4,
    true_cfg_scale=1.0,
    seed=42,
    lora_request=LoRARequest(
        lora_name="lightning-4steps",
        lora_int_id=stable_lora_int_id(lora_path),
        lora_path=lora_path,
    ),
    lora_scale=1.0,
)
```

#### Verification

- With the LoRA: a 4-step request returns a clean, fully formed image.
- Without the LoRA (same request minus the `lora` field): a 4-step request
  returns a blurry, washed-out image — the undistilled base model cannot
  converge in 4 steps. This contrast confirms the adapter is applied.
- The server log shows the in-memory conversion and the adapter size:
  `Detected single-file (non-PEFT) LoRA ...` and
  `Loaded LoRA model: ... num_modules=720`.

## Notes

- Use `num_inference_steps: 8` with the 8-step checkpoint variants.
- Keep `true_cfg_scale` at `1.0`: guidance is distilled into the adapter.
  True CFG engages only when a negative prompt is also supplied; when it
  does, it doubles per-step compute and visibly degrades output
  (oversaturated, color-shifted results in our tests).
- Extra steps buy nothing: in our tests a 50-step run produced a comparable
  image at >10x the latency of the 4-step configuration — the point of the
  adapter is that the 4/8-step schedule is sufficient.
- The sibling repositories (`Qwen-Image-2512-Lightning`,
  `Qwen-Image-Edit-*-Lightning`) publish the same file format for the newer
  base models and should load the same way; validation results welcome.

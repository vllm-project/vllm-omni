# Anima

> Native single-file diffusion text-to-image

## Summary

- Vendor: circlestone-labs
- Model: [`circlestone-labs/Anima`](https://huggingface.co/circlestone-labs/Anima)
- Task: text2img
- Mode: Offline inference, Online serving (OpenAI-compatible API)
- Maintainer: Community

## When to use this recipe

Use this recipe to run Anima via vLLM-Omni's native `AnimaPipeline`. Anima is a
Cosmos-style diffusion transformer text-to-image model distributed as a
single-file transformer checkpoint, not as a normal Hugging Face model
directory.

The native path reads the Anima transformer checkpoint directly, converts
original Cosmos transformer keys when needed, and loads the Cosmos transformer
and text conditioner into native vLLM-Omni modules. Non-denoiser components
such as `text_encoder`, `tokenizer`, `t5_tokenizer`, `vae`, and optionally
`scheduler` must be supplied through a Diffusers-layout components directory.

Native Anima currently supports baseline single-GPU execution. Cache-DiT,
TeaCache, CPU offload, layer-wise offload, quantization, TP/SP, CFG parallel,
HSDP, and step execution are not supported by `AnimaPipeline` yet.

## References

- Offline example:
  [`examples/offline_inference/text_to_image/README.md`](../../examples/offline_inference/text_to_image/README.md)
- Supported model entry:
  [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- HuggingFace model page:
  [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)
- Diffusers-layout components:
  [circlestone-labs/Anima-Base-v1.0-Diffusers](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers)

## Hardware Support

## ROCm

### 1x AMD MI300X

#### Environment

- OS: Ubuntu 22.04.5 LTS (x86_64)
- Python: 3.12.13
- Driver / runtime: ROCm 7.2.53211, HIP runtime 7.2.53211, MIOpen 3.5.1
- PyTorch: 2.10.0+git8514f05 built with ROCm 7.2.53211
- GPU: 1x AMD MI300X
- vLLM version: 0.23.0
- vLLM-Omni version or commit: 0.1.dev2002+g704724675.rocm

Recommended environment variables:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ROCM_USE_AITER=0
export PYTORCH_ROCM_ARCH=gfx942
export PYTORCH_NVML_BASED_CUDA_CHECK=1
```

#### Command - prepare assets

Download the official transformer checkpoint and the Diffusers-layout component
directory:

```bash
mkdir -p /path/to/models/anima-official
mkdir -p /path/to/models/anima-components

hf download circlestone-labs/Anima \
    split_files/diffusion_models/anima-base-v1.0.safetensors \
    --local-dir /path/to/models/anima-official

hf download circlestone-labs/Anima-Base-v1.0-Diffusers \
    --local-dir /path/to/models/anima-components

export ANIMA_CHECKPOINT=/path/to/models/anima-official/split_files/diffusion_models/anima-base-v1.0.safetensors
export ANIMA_COMPONENTS=/path/to/models/anima-components
```

The `ANIMA_COMPONENTS` directory must be in Diffusers `from_pretrained()`
layout. Raw auxiliary files such as `qwen_3_06b_base.safetensors` and
`qwen_image_vae.safetensors` are converter inputs; they are not accepted
directly as `components_path`.

#### Command - text-to-image

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model "$ANIMA_CHECKPOINT" \
    --model-class-name AnimaPipeline \
    --custom-pipeline-args "{\"components_path\":\"$ANIMA_COMPONENTS\"}" \
    --prompt "A cinematic close-up of a glass teapot on a wooden table." \
    --seed 42 \
    --guidance-scale 4.0 \
    --num-inference-steps 50 \
    --height 1024 \
    --width 1024 \
    --output /tmp/anima_output.png
```

Use 1024x1024, 50 denoising steps, `max_sequence_length=512`, one image per
prompt, empty negative prompt, and CFG scale 4.0 when matching Anima's default
Diffusers settings.

#### Verification

Check that `/tmp/anima_output.png` exists and contains a generated image.

#### Notes

- Key flags: `--model-class-name AnimaPipeline` selects the native Anima path;
  `--custom-pipeline-args` supplies `components_path`.
- Keep the default diffusion load format. No deploy config is required when a
  local checkpoint and `--model-class-name AnimaPipeline` are provided.
- Start with `max-concurrency=1` for correctness and latency validation.
- Keep requests at the same resolution when comparing runs.
- Do not enable parallelism, cache acceleration, offload, or quantized
  checkpoint flags for Anima until support is added to `AnimaPipeline`.

## Online Serving

Anima supports text-to-image generation through the OpenAI-compatible image
generation API.

### Launch

```bash
vllm serve "$ANIMA_CHECKPOINT" \
    --omni \
    --port 8099 \
    --model-class-name AnimaPipeline \
    --custom-pipeline-args "{\"components_path\":\"$ANIMA_COMPONENTS\"}"
```

### Send requests

```bash
curl http://localhost:8099/v1/images/generations \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$ANIMA_CHECKPOINT\",
      \"prompt\": \"A cinematic close-up of a glass teapot on a wooden table.\",
      \"size\": \"1024x1024\",
      \"num_inference_steps\": 50,
      \"guidance_scale\": 4.0,
      \"seed\": 42
    }"
```

The same generation knobs used by other text-to-image recipes apply:
`num_inference_steps`, `seed`, `height` / `width` through `size`, and optional
negative prompting.
